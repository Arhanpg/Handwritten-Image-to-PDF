from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from handwritten_folders_to_pdf import (
    fit_a4,
    landscape_to_portrait,
    list_images,
    natural_key,
)


def test_natural_key():
    names = ["page10.jpg", "page2.jpg", "page1.jpg"]
    assert sorted(names, key=natural_key) == ["page1.jpg", "page2.jpg", "page10.jpg"]


def test_landscape_to_portrait():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    result = landscape_to_portrait(image)
    assert result.shape[:2] == (200, 100)


def test_fit_a4():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    result = fit_a4(image)
    assert result.shape == (3508, 2480, 3)


def test_list_images(tmp_path: Path):
    for name in ["page10.jpg", "page2.jpg", "ignore.txt", "page1.png"]:
        (tmp_path / name).write_bytes(b"test")
    paths = list_images(tmp_path)
    assert [p.name for p in paths] == ["page1.png", "page2.jpg", "page10.jpg"]


def test_pdf_pipeline_smoke(tmp_path: Path):
    from handwritten_folders_to_pdf import process_folder

    folder = tmp_path / "s1"
    folder.mkdir()
    for i in range(2):
        image = np.full((500, 350, 3), 255, dtype=np.uint8)
        cv2.putText(image, f"Page {i + 1}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2)
        Image.fromarray(image).save(folder / f"page{i + 1}.png")

    output = tmp_path / "s1.pdf"
    pages, failed = process_folder(folder, output, batch_size=1, do_deskew=False)
    assert pages == 2
    assert failed == 0
    assert output.exists()
