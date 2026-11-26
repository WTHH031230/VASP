"""Expose local MinerU 3.x through the cloud-v4 contract used by PosterMELD.

PosterMELD's upstream parser expects MinerU's hosted upload/poll/download API.
This localhost-only bridge preserves that client unchanged while forwarding
each uploaded PDF to a persistent local ``mineru-api`` VLM task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", default="http://127.0.0.1:8010")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    return parser


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def create_app(upstream: str, state_dir: Path) -> FastAPI:
    upstream = upstream.rstrip("/")
    state_dir = state_dir.expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = state_dir / "content-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="Local MinerU cloud-v4 compatibility bridge")

    def local_model_metadata() -> dict[str, Any]:
        config_value = os.getenv("MINERU_TOOLS_CONFIG_JSON", "")
        config_path = Path(config_value).expanduser() if config_value else None
        payload: dict[str, Any] = {}
        if config_path and config_path.is_file():
            try:
                parsed = json.loads(config_path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    payload = parsed
            except (OSError, json.JSONDecodeError):
                pass
        models_dir = payload.get("models-dir") if isinstance(payload, dict) else None
        return {
            "config": str(config_path.resolve()) if config_path and config_path.is_file() else None,
            "model_source": payload.get("model-source") if isinstance(payload, dict) else None,
            "vlm_model_dir": models_dir.get("vlm") if isinstance(models_dir, dict) else None,
        }

    def record_path(batch_id: str) -> Path:
        return state_dir / batch_id / "batch.json"

    def cached_zip(record: dict[str, Any]) -> Path | None:
        cache_key = str(record.get("cache_key") or "")
        if not cache_key:
            return None
        candidate = cache_dir / f"{cache_key}.zip"
        return candidate if candidate.is_file() and candidate.stat().st_size else None

    def read_record(batch_id: str) -> dict[str, Any]:
        path = record_path(batch_id)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="batch not found")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail="invalid batch state") from exc
        if not isinstance(value, dict):
            raise HTTPException(status_code=500, detail="invalid batch state")
        return value

    @app.get("/health")
    async def health() -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
                response = await client.get(f"{upstream}/health")
            response.raise_for_status()
            upstream_health = response.json()
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"local MinerU unavailable: {exc}",
            ) from exc
        return {
            "status": "ok",
            "mode": "local_vlm_cloud_v4_bridge",
            "upstream": upstream,
            "upstream_health": upstream_health,
            "local_model": local_model_metadata(),
        }

    @app.post("/api/v4/file-urls/batch")
    async def create_batch(request: Request) -> dict[str, Any]:
        payload = await request.json()
        files = payload.get("files") if isinstance(payload, dict) else None
        if not isinstance(files, list) or len(files) != 1:
            raise HTTPException(
                status_code=400,
                detail="PosterMELD local bridge accepts one PDF per batch",
            )
        file_spec = files[0] if isinstance(files[0], dict) else {}
        filename = Path(str(file_spec.get("name") or "paper.pdf")).name
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="only PDF input is supported")
        batch_id = uuid.uuid4().hex
        batch_dir = state_dir / batch_id
        batch_dir.mkdir(parents=True)
        record = {
            "batch_id": batch_id,
            "filename": filename,
            "data_id": str(file_spec.get("data_id") or Path(filename).stem),
            "model_version": str(payload.get("model_version") or "vlm"),
            "language": str(payload.get("language") or "en"),
            "enable_formula": bool(payload.get("enable_formula", True)),
            "enable_table": bool(payload.get("enable_table", True)),
            "is_ocr": bool(file_spec.get("is_ocr", False)),
            "status": "awaiting_upload",
            "mineru_task_id": None,
        }
        _atomic_json(record_path(batch_id), record)
        base = str(request.base_url).rstrip("/")
        return {
            "code": 0,
            "msg": "ok",
            "data": {
                "batch_id": batch_id,
                "file_urls": [
                    {
                        "name": filename,
                        "url": f"{base}/upload/{batch_id}",
                    }
                ],
            },
        }

    @app.put("/upload/{batch_id}")
    async def upload(batch_id: str, request: Request) -> Response:
        record = read_record(batch_id)
        batch_dir = state_dir / batch_id
        pdf_path = batch_dir / record["filename"]
        pdf_path.write_bytes(await request.body())
        if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="empty PDF upload")
        digest = hashlib.sha256()
        digest.update(pdf_path.read_bytes())
        digest.update(
            json.dumps(
                {
                    "model_version": record["model_version"],
                    "language": record["language"],
                    "enable_formula": record["enable_formula"],
                    "enable_table": record["enable_table"],
                    "is_ocr": record["is_ocr"],
                },
                sort_keys=True,
            ).encode("utf-8")
        )
        record["cache_key"] = digest.hexdigest()
        if cached_zip(record) is not None:
            record["status"] = "completed_cache_hit"
            _atomic_json(record_path(batch_id), record)
            return Response(status_code=200)
        form = {
            "backend": "vlm-engine",
            "parse_method": "ocr" if record["is_ocr"] else "auto",
            "formula_enable": str(record["enable_formula"]).lower(),
            "table_enable": str(record["enable_table"]).lower(),
            "image_analysis": "true",
            "return_md": "true",
            "return_middle_json": "true",
            "return_model_output": "true",
            "return_content_list": "true",
            "return_images": "true",
            "response_format_zip": "true",
            "return_original_file": "false",
            "client_side_output_generation": "false",
            "start_page_id": "0",
            "end_page_id": "99999",
        }
        try:
            async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
                with pdf_path.open("rb") as handle:
                    response = await client.post(
                        f"{upstream}/tasks",
                        data=form,
                        files={"files": (pdf_path.name, handle, "application/pdf")},
                    )
            response.raise_for_status()
            task = response.json()
            task_id = str(task["task_id"])
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"local MinerU task submission failed: {exc}"
            _atomic_json(record_path(batch_id), record)
            raise HTTPException(status_code=502, detail=record["error"]) from exc
        record["status"] = "submitted"
        record["mineru_task_id"] = task_id
        _atomic_json(record_path(batch_id), record)
        return Response(status_code=200)

    @app.get("/api/v4/extract-results/batch/{batch_id}")
    async def batch_result(batch_id: str, request: Request) -> dict[str, Any]:
        record = read_record(batch_id)
        task_id = record.get("mineru_task_id")
        cache_path = cached_zip(record)
        if cache_path is not None:
            state = "done"
            error = ""
        elif not task_id:
            state = "failed" if record.get("status") == "failed" else "pending"
            error = str(record.get("error") or "")
        else:
            try:
                async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
                    response = await client.get(f"{upstream}/tasks/{task_id}")
                response.raise_for_status()
                task = response.json()
                local_status = str(task.get("status") or "").lower()
                state = {
                    "completed": "done",
                    "failed": "failed",
                    "processing": "running",
                    "pending": "pending",
                }.get(local_status, local_status or "pending")
                error = str(task.get("error") or "")
            except Exception as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"local MinerU status request failed: {exc}",
                ) from exc
        base = str(request.base_url).rstrip("/")
        result: dict[str, Any] = {
            "file_name": record["filename"],
            "data_id": record["data_id"],
            "state": state,
            "err_msg": error,
        }
        if state == "done":
            result["full_zip_url"] = f"{base}/download/{batch_id}"
        return {
            "code": 0,
            "msg": "ok",
            "data": {"batch_id": batch_id, "extract_result": [result]},
        }

    @app.get("/download/{batch_id}")
    async def download(batch_id: str) -> Response:
        record = read_record(batch_id)
        cache_path = cached_zip(record)
        if cache_path is not None:
            return Response(
                content=cache_path.read_bytes(),
                media_type="application/zip",
                headers={"Content-Disposition": f'attachment; filename="{batch_id}.zip"'},
            )
        task_id = record.get("mineru_task_id")
        if not task_id:
            raise HTTPException(status_code=409, detail="batch was not submitted")
        try:
            async with httpx.AsyncClient(timeout=600, trust_env=False) as client:
                response = await client.get(f"{upstream}/tasks/{task_id}/result")
            if response.status_code == 202:
                raise HTTPException(status_code=409, detail="result is not ready")
            response.raise_for_status()
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"local MinerU result download failed: {exc}",
            ) from exc
        cache_key = str(record.get("cache_key") or "")
        if cache_key:
            temporary = cache_dir / f".{cache_key}.{uuid.uuid4().hex}.tmp"
            temporary.write_bytes(response.content)
            target = cache_dir / f"{cache_key}.zip"
            temporary.replace(target)
            record["cache_zip"] = str(target)
            _atomic_json(record_path(batch_id), record)
        return Response(
            content=response.content,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{batch_id}.zip"'
            },
        )

    return app


def main() -> None:
    args = _parser().parse_args()
    import uvicorn

    uvicorn.run(
        create_app(args.upstream, args.state_dir),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
