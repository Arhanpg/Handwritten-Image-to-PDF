# Contributing

Thanks for helping improve Handwritten Image → PDF!

## Development setup

```bash
python -m venv .venv
pip install -r requirements-dev.txt
pytest -q
ruff check .
```

## Pull requests

- Keep changes focused.
- Add or update tests when behavior changes.
- Do not commit private handwritten datasets or generated PDFs.
- Keep CPU-only operation working unless the change specifically targets optional CUDA support.
- Explain performance or image-quality trade-offs in the PR description.

## Bug reports

Include Python version, operating system, OpenCV/PyMuPDF versions, command used, relevant logs, and a small synthetic or non-sensitive reproduction when possible.

Never upload private exam papers, personal documents, or other sensitive images to an issue.
