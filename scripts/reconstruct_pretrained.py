from pathlib import Path
import hashlib


ROOT = Path(__file__).resolve().parents[1]
PRETRAINED = ROOT / "pretrained"
OUTPUT = PRETRAINED / "plnet.pth"
PARTS = sorted(PRETRAINED.glob("plnet.pth.part-*"))
EXPECTED_SHA256 = "99fd3261d754e19018c61fe464368d19faed4dbe45f4025764c2dc0d6272408c"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if not PARTS:
        raise FileNotFoundError("No pretrained/plnet.pth.part-* files found")

    with OUTPUT.open("wb") as output:
        for part in PARTS:
            with part.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    output.write(chunk)

    actual = sha256(OUTPUT)
    if actual != EXPECTED_SHA256:
        OUTPUT.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {OUTPUT}: expected {EXPECTED_SHA256}, got {actual}"
        )

    print(f"Reconstructed {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
