"""Serve P2P's official DocLayout-YOLO detector locally.

This is a transport-only replacement for the Modal deployment in
``third_party/P2P/figure_detection.py``. Model, class mapping, caption
matching, box union, and response schema are identical to upstream.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="path2doclayout_yolo_ft.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8019)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_app(model_path: Path) -> FastAPI:
    from doclayout_yolo import YOLOv10
    import torch

    resolved = model_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"P2P detector weight not found: {resolved}")
    model_sha256 = _sha256(resolved)
    model = YOLOv10(resolved)
    app = FastAPI(title="P2P official DocLayout detector")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": str(resolved),
            "sha256": model_sha256,
            "model_size_bytes": resolved.stat().st_size,
            "doclayout_yolo_version": importlib.metadata.version("doclayout-yolo"),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "upstream": "third_party/P2P/figure_detection.py",
        }

    @app.post("/")
    async def predict(
        img: UploadFile = File(...),
        task: str = Form(...),
    ) -> list[dict[str, Any]]:
        image = Image.open(BytesIO(await img.read()))
        results = model.predict(image)
        figs: list[dict[str, Any]] = []
        for result in results:
            boxes = result.__dict__["boxes"].xyxy.cpu().tolist()
            classes = result.__dict__["boxes"].cls.cpu().tolist()
            scores = result.__dict__["boxes"].conf.cpu().tolist()
            targets: list[dict[str, Any]] = []
            captions: list[dict[str, Any]] = []
            for box, cls, score in zip(boxes, classes, scores):
                if task == "figure":
                    if cls == 3:
                        targets.append({"box": box, "score": score})
                elif task == "table":
                    if cls == 5:
                        targets.append({"box": box, "score": score})
                elif task == "figurecaption":
                    if cls == 3:
                        targets.append({"box": box, "score": score})
                    elif cls == 4:
                        captions.append({"box": box, "score": score})
                elif task == "tablecaption":
                    if cls == 5:
                        targets.append({"box": box, "score": score})
                    elif cls == 6 or cls == 7:
                        captions.append({"box": box, "score": score})
            if not captions:
                figs = targets
            else:
                matches = []
                for target in targets:
                    min_distance = float("inf")
                    correct_match = None
                    for caption in captions:
                        target_box = target["box"]
                        caption_box = caption["box"]
                        distance = abs(target_box[0] - caption_box[0]) + abs(
                            target_box[3] - caption_box[1]
                        )
                        if distance < min_distance:
                            min_distance = distance
                            correct_match = (target, caption)
                    if correct_match is not None:
                        matches.append(correct_match)
                for target, caption in matches:
                    target_box = target["box"]
                    caption_box = caption["box"]
                    union_box = [
                        min(target_box[0], caption_box[0]),
                        min(target_box[1], caption_box[1]),
                        max(target_box[2], caption_box[2]),
                        max(target_box[3], caption_box[3]),
                    ]
                    figs.append({"box": union_box, "score": 1.0})
        return figs

    return app


def main() -> None:
    args = _parser().parse_args()
    import uvicorn

    uvicorn.run(create_app(args.model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
