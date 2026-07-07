import argparse
import logging
import os

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from hawp.base.utils.checkpoint import DetectronCheckpointer
from hawp.base.utils.logger import setup_logger
from hawp.fsl.config import cfg
from hawp.fsl.dataset import build_train_dataset
from hawp.fsl.model.build import build_model
from hawp.fsl.utils import reached_debug_limit


def set_requires_grad(module, requires_grad):
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def get_point_detector(model):
    return model.backbone.point_detector if hasattr(model, "backbone") else model.point_detector


def rgb_to_grayscale_tensor(images):
    if images.ndim != 4:
        raise ValueError("teacher images must have shape N x C x H x W")
    if images.shape[1] == 1:
        return images
    if images.shape[1] != 3:
        raise ValueError("teacher RGB images must have 1 or 3 channels")
    weights = images.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (images * weights).sum(dim=1, keepdim=True)


def run_adapt_step(teacher, student, rgb_images, raw_images, optimizer):
    teacher.eval()
    student.train()
    set_requires_grad(teacher, False)

    teacher_detector = get_point_detector(teacher)
    student_detector = get_point_detector(student)
    with torch.no_grad():
        teacher_outputs = teacher_detector.forward_dense(rgb_to_grayscale_tensor(rgb_images))
        teacher_labels = teacher_outputs["detector_logits"].argmax(dim=1)
        teacher_descriptors = teacher_outputs["dense_descriptors"]

    student_outputs = student_detector.forward_dense(raw_images[:, :1])
    loss_detector = F.cross_entropy(student_outputs["detector_logits"], teacher_labels)
    loss_descriptor = 1.0 - F.cosine_similarity(
        student_outputs["dense_descriptors"],
        teacher_descriptors,
        dim=1,
    ).mean()
    loss_total = loss_detector + loss_descriptor

    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()

    return {
        "loss_detector": float(loss_detector.detach().cpu()),
        "loss_descriptor": float(loss_descriptor.detach().cpu()),
        "loss_total": float(loss_total.detach().cpu()),
    }


class DelayedCosineAnnealingLR:
    def __init__(self, optimizer, noise_warmup_steps, total_steps, min_lr):
        self.optimizer = optimizer
        self.noise_warmup_steps = max(0, int(noise_warmup_steps))
        self.total_steps = max(1, int(total_steps))
        self.step_count = 0
        cosine_steps = max(1, self.total_steps - self.noise_warmup_steps)
        self.cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=float(min_lr))

    def step(self):
        self.step_count += 1
        if self.step_count < self.noise_warmup_steps:
            return
        if self.step_count >= self.total_steps:
            for group in self.optimizer.param_groups:
                group["lr"] = self.cosine.eta_min
            return
        self.cosine.step()

    def state_dict(self):
        return {
            "noise_warmup_steps": self.noise_warmup_steps,
            "total_steps": self.total_steps,
            "step_count": self.step_count,
            "cosine": self.cosine.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.noise_warmup_steps = state_dict["noise_warmup_steps"]
        self.total_steps = state_dict["total_steps"]
        self.step_count = state_dict["step_count"]
        self.cosine.load_state_dict(state_dict["cosine"])


def build_adapt_lr_scheduler(optimizer, scheduler_name, total_steps, noise_warmup_steps, min_lr):
    if scheduler_name == "none":
        return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    if scheduler_name != "cosine_after_noise":
        raise ValueError(f"Unsupported adapt LR scheduler: {scheduler_name}")
    return DelayedCosineAnnealingLR(
        optimizer,
        noise_warmup_steps=noise_warmup_steps,
        total_steps=total_steps,
        min_lr=min_lr,
    )


def maybe_save_noise_warmup_checkpoint(
    checkpointer,
    was_saved,
    previous_step,
    current_step,
    noise_warmup_steps,
):
    if was_saved or noise_warmup_steps <= 0:
        return was_saved
    if previous_step < noise_warmup_steps <= current_step:
        checkpointer.save("adapt_raw_noise_warmup_done")
        return True
    return False


def noise_ratio_decay(iteration, noise_maxstep):
    return min(int(iteration), int(noise_maxstep)) / float(noise_maxstep)


def _load_model(config, checkpoint_path, device):
    model = build_model(config).to(device)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state, strict=False)
    return model


def main():
    parser = argparse.ArgumentParser(description="RawPLNet point detector adaptation")
    parser.add_argument("config", type=str)
    parser.add_argument("--teacher-ckpt", required=True, type=str)
    parser.add_argument("--student-ckpt", default=None, type=str)
    parser.add_argument("--logdir", required=True, type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument(
        "--lr-scheduler",
        default="cosine_after_noise",
        choices=["none", "cosine_after_noise"],
        help="adapt LR schedule; cosine_after_noise keeps LR fixed until DATASETS.RAW.NOISE_MAXSTEP",
    )
    parser.add_argument("--lr-min", default=1e-5, type=float, help="minimum LR for cosine_after_noise")
    parser.add_argument(
        "--lr-decay-start",
        default=None,
        type=int,
        help="step to start cosine decay; defaults to DATASETS.RAW.NOISE_MAXSTEP",
    )
    parser.add_argument("--max-iters", default=None, type=int, help="stop after this many adapt iterations for local debug")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    os.makedirs(args.logdir, exist_ok=True)
    logger = setup_logger("hawp.adapt_raw", args.logdir, out_file="adapt_raw.log")
    device = cfg.MODEL.DEVICE

    teacher = _load_model(cfg, args.teacher_ckpt, device)
    student = _load_model(cfg, args.student_ckpt or args.teacher_ckpt, device)
    set_requires_grad(teacher, False)
    set_requires_grad(student, False)
    set_requires_grad(get_point_detector(student), True)

    optimizer = torch.optim.Adam(
        [p for p in get_point_detector(student).parameters() if p.requires_grad],
        lr=args.lr,
    )
    loader = build_train_dataset(cfg, return_adapt_pair=True)
    total_steps = args.epochs * len(loader)
    if args.max_iters is not None:
        total_steps = min(total_steps, args.max_iters)
    lr_decay_start = cfg.DATASETS.RAW.NOISE_MAXSTEP if args.lr_decay_start is None else args.lr_decay_start
    scheduler = build_adapt_lr_scheduler(
        optimizer,
        scheduler_name=args.lr_scheduler,
        total_steps=total_steps,
        noise_warmup_steps=lr_decay_start,
        min_lr=args.lr_min,
    )
    logger.info(
        "adapt lr schedule=%s base_lr=%g min_lr=%g decay_start=%d total_steps=%d",
        args.lr_scheduler,
        args.lr,
        args.lr_min,
        lr_decay_start,
        total_steps,
    )

    checkpointer = DetectronCheckpointer(
        cfg,
        student,
        optimizer,
        scheduler,
        save_dir=args.logdir,
        save_to_disk=True,
        logger=logging.getLogger("hawp.adapt_raw"),
    )

    iteration = 0
    saved_noise_warmup = False
    for epoch in range(1, args.epochs + 1):
        for (teacher_images, raw_images), annotations in loader:
            teacher_images = teacher_images.to(device)
            raw_images = raw_images.to(device)
            current_lr = optimizer.param_groups[0]["lr"]
            ratio_dec = noise_ratio_decay(iteration, cfg.DATASETS.RAW.NOISE_MAXSTEP)
            losses = run_adapt_step(teacher, student, teacher_images, raw_images, optimizer)
            previous_iteration = iteration
            if iteration % 20 == 0:
                logger.info(
                    "epoch=%d iter=%d lr=%.8g ratio_dec=%.6f losses=%s",
                    epoch,
                    iteration,
                    current_lr,
                    ratio_dec,
                    losses,
                )
            iteration += 1
            scheduler.step()
            saved_noise_warmup = maybe_save_noise_warmup_checkpoint(
                checkpointer,
                was_saved=saved_noise_warmup,
                previous_step=previous_iteration,
                current_step=iteration,
                noise_warmup_steps=cfg.DATASETS.RAW.NOISE_MAXSTEP,
            )
            if reached_debug_limit(args.max_iters, iteration):
                logger.info("Stopping early after %d iterations for debug run", iteration)
                break
        checkpointer.save("adapt_raw_epoch_{:05d}".format(epoch))
        if reached_debug_limit(args.max_iters, iteration):
            break

    checkpointer.save("adapt_raw_final")


if __name__ == "__main__":
    main()
