# Design

## Goal

Convert nested folders of handwritten images into independent, readable A4 PDFs while keeping document processing local.

## Stages

1. Discover image-containing folders recursively.
2. Sort pages naturally by filename.
3. Load images and apply EXIF orientation.
4. Resize oversized camera images to bound memory usage.
5. Rotate physically landscape pages to portrait.
6. Estimate small page skew using projection-profile search.
7. Enhance local contrast and compensate gently for uneven illumination.
8. Fit the page proportionally to an A4 canvas.
9. Optionally sharpen batches with CUDA through PyTorch.
10. Encode pages as high-quality JPEG streams and embed them into PDF pages.
11. Replace the output atomically only after successful page processing.

## GPU design

CUDA is optional. The CPU pipeline remains the source of truth for document preparation. GPU work is deliberately limited to the final sharpening operation so the application remains portable and useful on machines without NVIDIA hardware.

## Orientation policy

No OCR is required. EXIF metadata is honored and landscape pages are normalized to portrait. Exact 180-degree orientation for portrait-shaped handwriting is intentionally not guessed.

## Safety

Generated PDFs are written only after at least one page succeeds. The output directory is excluded from recursive discovery to prevent reprocessing generated artifacts.
