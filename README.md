# Handwritten Image → PDF

[![CI](https://github.com/Arhanpg/Handwritten-Image-to-PDF/actions/workflows/ci.yml/badge.svg)](https://github.com/Arhanpg/Handwritten-Image-to-PDF/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-optional-76B900.svg)](https://pytorch.org/)

A local-first, OCR-free batch converter that turns nested folders of handwritten images into clean, readable A4 PDFs, with optional NVIDIA CUDA acceleration.

## Why?

Handwritten datasets often arrive as hundreds of phone photos or scans split across folders. This tool converts that structure directly:

```text
input/
├── s1/
│   ├── page_001.jpg
│   └── page_002.jpg
├── s2/
│   └── ...
└── s37/
    └── ...

output/
├── s1.pdf
├── s2.pdf
└── s37.pdf
```

Each image-containing folder becomes one PDF, and filenames are sorted naturally.

## Features

- Recursive image-folder discovery
- One PDF per folder
- Natural filename ordering
- EXIF orientation correction
- Landscape → portrait normalization
- Small-angle deskew
- CLAHE contrast enhancement
- Gentle illumination/background normalization
- A4 pages on a 2480×3508 working canvas
- High-quality JPEG embedding
- Optional CUDA sharpening through PyTorch
- CPU fallback when CUDA is unavailable
- No Tesseract or OCR runtime dependency
- Output directory is excluded from input discovery
- Progress bars, logging, failure counts, and safe PDF replacement
- Automated tests and GitHub Actions CI

## Verified dataset run

The project was tested on a real handwritten-image dataset with **37 folders, 195 pages, 37 PDFs, and 0 failed images** using an NVIDIA GeForce RTX 3050 Laptop GPU. The complete run took about two minutes on that machine. This is an example result, not a universal benchmark.

## Orientation note

The project intentionally does not guess 180° upside-down orientation for portrait-shaped handwritten pages. Without OCR or a dedicated orientation model, such a page can look valid in both directions.

The converter therefore uses EXIF orientation, landscape-to-portrait normalization, and small-angle deskewing rather than making an unreliable 180° guess.

## Installation

```bash
git clone https://github.com/Arhanpg/Handwritten-Image-to-PDF.git
cd Handwritten-Image-to-PDF
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install core dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Optional NVIDIA GPU acceleration

Install a CUDA-enabled PyTorch build appropriate for your machine from the official selector:

https://pytorch.org/get-started/locally/

Verify:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

For a 4 GB GPU, start with `--gpu-batch 1`.

## Usage

Interactive:

```bash
python handwritten_folders_to_pdf.py
```

Explicit paths:

```bash
python handwritten_folders_to_pdf.py --input ./input --output ./output
```

For a 4 GB GPU:

```bash
python handwritten_folders_to_pdf.py --input ./input --output ./output --gpu-batch 1
```

Try `--gpu-batch 2` only if your GPU has enough VRAM.

Disable deskew for maximum throughput:

```bash
python handwritten_folders_to_pdf.py --input ./input --output ./output --no-deskew
```

Supported images: JPG, JPEG, PNG, BMP, TIFF, TIF, WEBP, JFIF.

## Processing pipeline

```text
Image
  ↓
EXIF correction
  ↓
Safe resize
  ↓
Landscape → portrait
  ↓
Small-angle deskew
  ↓
CLAHE + illumination normalization
  ↓
A4 canvas
  ↓
Optional CUDA sharpening
  ↓
JPEG → PDF
```

## Development

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and [docs/design.md](docs/design.md) for architecture notes.

## Privacy

Processing is local. The project does not require a cloud API and does not upload document images.

## License

Apache License 2.0. See [LICENSE](LICENSE).
