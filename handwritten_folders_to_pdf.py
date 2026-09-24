from __future__ import annotations

import argparse
import io
import logging
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageOps
from tqdm import tqdm

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

LOG = logging.getLogger("handwritten_pdf")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif"}
A4_W, A4_H = 2480, 3508
A4_W_PT, A4_H_PT = 595.275590551, 841.88976378
JPEG_QUALITY = 95

def natural_key(name: str):
    return [int(x) if x.isdigit() else x.casefold() for x in re.split(r"(\d+)", name)]

def list_images(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.casefold() in EXTENSIONS], key=lambda p: natural_key(p.name))

def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image, dtype=np.uint8).copy()

def landscape_to_portrait(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    return cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE) if w > h else rgb

def resize_for_processing(rgb: np.ndarray, max_edge: int = 3200) -> np.ndarray:
    h, w = rgb.shape[:2]
    edge = max(h, w)
    if edge <= max_edge:
        return rgb
    scale = max_edge / float(edge)
    return cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)

def estimate_skew(gray: np.ndarray) -> float:
    scale = min(1.0, 1000.0 / max(gray.shape[:2]))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    blur = cv2.GaussianBlur(small, (3, 3), 0)
    dark = 255 - cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)
    threshold = np.percentile(dark, 82)
    binary = (dark > max(20, threshold * 0.75)).astype(np.uint8) * 255
    best_angle, best_score = 0.0, -1.0
    h, w = binary.shape
    for angle in np.arange(-4.0, 4.01, 0.5):
        if abs(float(angle)) < 1e-9:
            rotated = binary
        else:
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), float(angle), 1.0)
            rotated = cv2.warpAffine(binary, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        energy = rotated.sum(axis=1).astype(np.float64)
        score = float(np.mean(np.sort(energy)[-max(3, len(energy) // 10):])) / (float(energy.mean()) + 1e-6)
        if score > best_score:
            best_score, best_angle = score, float(angle)
    return 0.0 if abs(best_angle) < 0.5 else best_angle

def deskew(rgb: np.ndarray) -> np.ndarray:
    angle = estimate_skew(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if abs(angle) < 0.5:
        return rgb
    h, w = rgb.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    return cv2.warpAffine(rgb, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

def enhance(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8)).apply(l)
    background = cv2.GaussianBlur(l, (0, 0), sigmaX=21, sigmaY=21)
    normalized = np.clip((l.astype(np.float32) / np.maximum(background.astype(np.float32), 1.0)) * 210.0, 0, 255).astype(np.uint8)
    l = cv2.addWeighted(l, 0.70, normalized, 0.30, 0)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

def fit_a4(rgb: np.ndarray) -> np.ndarray:
    margin = 60
    h, w = rgb.shape[:2]
    scale = min((A4_W - 2 * margin) / w, (A4_H - 2 * margin) / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    page = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.full((A4_H, A4_W, 3), 255, np.uint8)
    x, y = (A4_W - nw) // 2, (A4_H - nh) // 2
    canvas[y:y + nh, x:x + nw] = page
    return canvas

def cpu_sharpen(rgb: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(rgb, (0, 0), 1.0)
    return cv2.addWeighted(rgb, 1.18, blur, -0.18, 0)

def gpu_sharpen(batch: list[np.ndarray]) -> list[np.ndarray]:
    if torch is None or F is None or not torch.cuda.is_available():
        return [cpu_sharpen(x) for x in batch]
    try:
        x = torch.stack([torch.from_numpy(x).permute(2, 0, 1).float().div(255) for x in batch]).cuda()
        blur = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        y = torch.clamp(x + 0.18 * (x - blur), 0, 1)
        y = y.mul(255).byte().permute(0, 2, 3, 1).cpu().numpy()
        torch.cuda.synchronize()
        return list(y)
    except RuntimeError:
        return [cpu_sharpen(x) for x in batch]

def encode_jpeg(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True, subsampling=0)
    return buf.getvalue()

def write_pdf(pages: list[np.ndarray], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".tmp.pdf")
    doc = pymupdf.open()
    rect = pymupdf.Rect(0, 0, A4_W_PT, A4_H_PT)
    for rgb in pages:
        page = doc.new_page(width=A4_W_PT, height=A4_H_PT)
        page.insert_image(rect, stream=encode_jpeg(rgb), keep_proportion=False)
    doc.save(temp, garbage=4, deflate=True, clean=True)
    doc.close()
    temp.replace(output)

def discover_folders(root: Path, output: Path) -> list[Path]:
    folders = []
    output = output.resolve()
    for current, dirs, files in os.walk(root):
        current_path = Path(current).resolve()
        dirs[:] = [d for d in dirs if not (current_path / d).resolve().is_relative_to(output)]
        if any(Path(name).suffix.casefold() in EXTENSIONS for name in files):
            folders.append(current_path)
    return sorted(folders, key=lambda p: natural_key(str(p.relative_to(root))))

def gpu_status() -> tuple[bool, str]:
    if torch is None:
        return False, "PyTorch is not installed"
    try:
        return (True, torch.cuda.get_device_name(0)) if torch.cuda.is_available() else (False, "PyTorch is installed, but CUDA is unavailable")
    except Exception as exc:
        return False, f"CUDA check failed: {exc}"

def process_folder(folder: Path, output: Path, batch_size: int, do_deskew: bool) -> tuple[int, int]:
    paths = list_images(folder)
    pages, pending, failed = [], [], 0
    for path in tqdm(paths, desc=folder.name, unit="img"):
        try:
            rgb = landscape_to_portrait(resize_for_processing(load_image(path)))
            if do_deskew:
                rgb = deskew(rgb)
            pending.append(fit_a4(enhance(rgb)))
            if len(pending) >= batch_size:
                pages.extend(gpu_sharpen(pending))
                pending.clear()
        except Exception as exc:
            failed += 1
            LOG.exception("Failed to process %s: %s", path, exc)
    if pending:
        pages.extend(gpu_sharpen(pending))
    if pages:
        write_pdf(pages, output)
        LOG.info("Created: %s | pages=%d | failed=%d", output, len(pages), failed)
    else:
        LOG.error("No pages were successfully processed for %s", folder)
    return len(pages), failed

def parse_args():
    p = argparse.ArgumentParser(description="Recursively convert image folders into A4 PDFs.")
    p.add_argument("--input", type=Path, help="Input root containing image folders")
    p.add_argument("--output", type=Path, help="Output directory for PDFs")
    p.add_argument("--gpu-batch", type=int, default=1, help="GPU batch size (default: 1)")
    p.add_argument("--no-deskew", action="store_true", help="Disable small-angle deskew")
    return p.parse_args()

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    input_root = args.input or Path(input("Enter input root folder: ").strip().strip('"'))
    output_root = args.output or Path(input("Enter output folder: ").strip().strip('"'))
    input_root, output_root = input_root.expanduser().resolve(), output_root.expanduser().resolve()
    if not input_root.is_dir():
        LOG.error("Input folder does not exist: %s", input_root)
        return 1
    output_root.mkdir(parents=True, exist_ok=True)
    enabled, message = gpu_status()
    LOG.info("CUDA GPU enabled: %s", message if enabled else f"No GPU acceleration ({message})")
    LOG.info("Tesseract: NOT USED")
    folders = discover_folders(input_root, output_root)
    LOG.info("Found %d image-containing folder(s).", len(folders))
    total_pages = total_failed = pdf_count = 0
    for folder in folders:
        rel = folder.relative_to(input_root)
        output = output_root / rel.parent / f"{rel.name}.pdf"
        pages, failed = process_folder(folder, output, max(1, args.gpu_batch), not args.no_deskew)
        total_pages += pages
        total_failed += failed
        pdf_count += int(pages > 0)
    LOG.info("Completed | PDFs=%d | pages=%d | failed=%d", pdf_count, total_pages, total_failed)
    return 0 if pdf_count else 1

if __name__ == "__main__":
    sys.exit(main())
