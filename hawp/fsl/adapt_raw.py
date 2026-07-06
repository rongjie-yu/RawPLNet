import argparse
import logging
import os

import torch
import torch.nn.functional as F

from hawp.base.utils.checkpoint import DetectronCheckpointer
from hawp.base.utils.logger import setup_logger
from hawp.fsl.config import cfg
from hawp.fsl.dataset import build_train_dataset
from hawp.fsl.model.build import build_model
from hawp.fsl.raw import RawSynthesisConfig, RawSynthesizer
from hawp.fsl.utils import reached_debug_limit


def set_requires_grad(module, requires_grad):
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def get_point_detector(model):
    return model.backbone.point_detector if hasattr(model, "backbone") else model.point_detector

def run_adapt_step(teacher, student, rgb_images, raw_images, optimizer):
    teacher.eval()
    student.train()
    set_requires_grad(teacher, False)

    teacher_detector = get_point_detector(teacher)
    student_detector = get_point_detector(student)
    with torch.no_grad():
        teacher_outputs = teacher_detector.forward_dense(rgb_images[:, :1])
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
    loader = build_train_dataset(cfg, return_rgb=True)
    raw_synthesizer = RawSynthesizer(RawSynthesisConfig.from_cfg(cfg.DATASETS.RAW), device=device)

    iteration = 0
    for epoch in range(1, args.epochs + 1):
        for rgb_images, annotations in loader:
            rgb_images = rgb_images.to(device)
            raw_np = [
                raw_synthesizer.synthesize_rgb(
                    image.permute(1, 2, 0).detach().cpu().numpy(),
                    iter_idx=iteration,
                )
                for image in rgb_images
            ]
            raw_images = torch.stack([
                torch.from_numpy(image).permute(2, 0, 1)
                for image in raw_np
            ]).float().to(device)
            losses = run_adapt_step(teacher, student, rgb_images, raw_images, optimizer)
            if iteration % 20 == 0:
                logger.info("epoch=%d iter=%d losses=%s", epoch, iteration, losses)
            iteration += 1
            if reached_debug_limit(args.max_iters, iteration):
                logger.info("Stopping early after %d iterations for debug run", iteration)
                break
        if reached_debug_limit(args.max_iters, iteration):
            break

    checkpointer = DetectronCheckpointer(
        cfg,
        student,
        optimizer,
        save_dir=args.logdir,
        save_to_disk=True,
        logger=logging.getLogger("hawp.adapt_raw"),
    )
    checkpointer.save("adapt_raw_final")


if __name__ == "__main__":
    main()
