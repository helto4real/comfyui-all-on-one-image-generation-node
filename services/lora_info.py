"""LoRA model info and Civitai lookup routes for the AIO node pack."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


INFO_SUFFIX = ".aio-lora-info.json"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def _get_nested(data: dict[str, Any] | None, path: str, default=None):
    current: Any = data or {}
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _json_response(data: dict[str, Any]):
    from aiohttp import web  # type: ignore

    return web.json_response(data)


def _get_param(request, name: str, default: str | None = None) -> str | None:
    return request.rel_url.query.get(name, default)


def _is_truthy(value: str | None) -> bool:
    return value not in (None, "", "0", "false", "False", "no", "No")


def _lora_path(file: str) -> Path | None:
    import folder_paths  # type: ignore

    path = folder_paths.get_full_path("loras", file)
    if path and Path(path).exists():
        return Path(path)
    absolute = Path(file).expanduser()
    if absolute.exists():
        return absolute
    return None


def _sidecar_path(path: Path) -> Path:
    return Path(f"{path}{INFO_SUFFIX}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_image(file: str, path: Path) -> str | None:
    stem = path.with_suffix("")
    for extension in IMAGE_EXTENSIONS:
        candidate = stem.with_suffix(extension)
        if candidate.exists():
            return f"/aio-image-gen/api/loras/img?file={urllib.parse.quote(file)}"
    return None


def _read_safetensors_metadata(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".safetensors":
        return {}
    try:
        with path.open("rb") as handle:
            header_size = int.from_bytes(handle.read(8), "little", signed=False)
            if header_size <= 0:
                return {}
            header = json.loads(handle.read(header_size))
        metadata = header.get("__metadata__", {}) or {}
        for key, value in list(metadata.items()):
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                try:
                    metadata[key] = json.loads(value)
                except Exception:
                    pass
        return metadata
    except Exception:
        return {}


def _file_info(file: str) -> dict[str, Any] | None:
    path = _lora_path(file)
    if path is None:
        return None
    return {
        "file": file,
        "path": str(path),
        "modified": path.stat().st_mtime * 1000,
        "imageLocal": _local_image(file, path),
        "hasInfoFile": _sidecar_path(path).exists(),
    }


def _merge_metadata(info: dict[str, Any], metadata: dict[str, Any]) -> None:
    if not metadata:
        return
    info.setdefault("raw", {})["metadata"] = metadata
    if metadata.get("ss_output_name") and not info.get("name"):
        info["name"] = metadata["ss_output_name"]
    if metadata.get("ss_sd_model_name"):
        info["baseModelFile"] = metadata["ss_sd_model_name"]
    if metadata.get("ss_clip_skip") and metadata.get("ss_clip_skip") != "None":
        info["clipSkip"] = metadata["ss_clip_skip"]

    tag_frequency = metadata.get("ss_tag_frequency")
    if isinstance(tag_frequency, dict):
        words: dict[str, dict[str, Any]] = {
            item.get("word"): item for item in info.get("trainedWords", []) if item.get("word")
        }
        for bucket in tag_frequency.values():
            if not isinstance(bucket, dict):
                continue
            for word, count in bucket.items():
                item = words.setdefault(word, {"word": word, "count": 0})
                item["count"] = item.get("count", 0) + count
                item["metadata"] = True
        info["trainedWords"] = sorted(words.values(), key=lambda item: item.get("count", 0), reverse=True)


def _fetch_civitai(file_hash: str) -> dict[str, Any]:
    url = f"https://civitai.com/api/v1/model-versions/by-hash/{file_hash}"
    request = urllib.request.Request(url, headers={"User-Agent": "AIO-ComfyUI-LoRA-Info/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        data = {"error": f"Civitai returned HTTP {exc.code}"}
    except Exception as exc:
        data = {"error": str(exc)}
    data["_sha256"] = file_hash
    data["_civitai_api"] = url
    return data


def _merge_civitai(info: dict[str, Any], data: dict[str, Any]) -> None:
    info.setdefault("raw", {})["civitai"] = data
    if data.get("error"):
        return
    version_name = data.get("name")
    model_name = _get_nested(data, "model.name", "")
    if not info.get("name"):
        info["name"] = f"{model_name} - {version_name}" if model_name and version_name else model_name or version_name
    if not info.get("type"):
        info["type"] = _get_nested(data, "model.type")
    if not info.get("baseModel"):
        info["baseModel"] = data.get("baseModel")

    words = {item.get("word"): item for item in info.get("trainedWords", []) if item.get("word")}
    raw_words = ",".join((data.get("triggerWords") or []) + (data.get("trainedWords") or []))
    for word in [part.strip() for part in raw_words.split(",") if part.strip()]:
        item = words.setdefault(word, {"word": word})
        item["civitai"] = True
    if words:
        info["trainedWords"] = list(words.values())

    model_id = data.get("modelId")
    if model_id:
        link = f"https://civitai.com/models/{model_id}"
        if data.get("id"):
            link = f"{link}?modelVersionId={data['id']}"
        links = info.setdefault("links", [])
        for value in (link, data.get("_civitai_api")):
            if value and value not in links:
                links.append(value)

    existing_images = {image.get("url") for image in info.get("images", [])}
    images = info.setdefault("images", [])
    for image in data.get("images", []) or []:
        url = image.get("url")
        if not url or url in existing_images:
            continue
        image_id = Path(urllib.parse.urlparse(url).path).stem
        images.append(
            {
                "url": url,
                "civitaiUrl": f"https://civitai.com/images/{image_id}" if image_id else None,
                "width": image.get("width"),
                "height": image.get("height"),
                "type": image.get("type"),
                "nsfwLevel": image.get("nsfwLevel"),
                "seed": _get_nested(image, "meta.seed"),
                "positive": _get_nested(image, "meta.prompt"),
                "negative": _get_nested(image, "meta.negativePrompt"),
                "steps": _get_nested(image, "meta.steps"),
                "sampler": _get_nested(image, "meta.sampler"),
                "cfg": _get_nested(image, "meta.cfgScale"),
                "model": _get_nested(image, "meta.Model"),
                "resources": _get_nested(image, "meta.resources"),
            }
        )


def get_lora_info(
    file: str,
    *,
    light: bool = False,
    fetch_metadata: bool = False,
    fetch_civitai: bool = False,
) -> dict[str, Any] | None:
    path = _lora_path(file)
    if path is None:
        return None

    info = _read_json(_sidecar_path(path))
    info.update({key: value for key, value in (_file_info(file) or {}).items() if value is not None})
    info.setdefault("images", [])
    if info.get("imageLocal") and not any(image.get("url") == info["imageLocal"] for image in info["images"]):
        info["images"].insert(0, {"url": info["imageLocal"]})

    if light:
        return info

    file_hash = info.get("sha256") or _sha256(path)
    info["sha256"] = file_hash
    info.setdefault("raw", {})
    if fetch_metadata or "metadata" not in info["raw"]:
        _merge_metadata(info, _read_safetensors_metadata(path))
    if fetch_civitai or "civitai" not in info["raw"]:
        _merge_civitai(info, _fetch_civitai(file_hash))

    _write_json(_sidecar_path(path), info)
    return info


def save_lora_info_partial(file: str, partial: dict[str, Any]) -> dict[str, Any] | None:
    path = _lora_path(file)
    if path is None:
        return None
    info = get_lora_info(file, fetch_metadata=True) or {}
    info.update(partial)
    _write_json(_sidecar_path(path), info)
    return info


def register_routes() -> None:
    from aiohttp import web  # type: ignore
    from server import PromptServer  # type: ignore
    import folder_paths  # type: ignore

    routes = PromptServer.instance.routes

    @routes.get("/aio-image-gen/api/loras")
    async def list_loras(request):
        files = folder_paths.get_filename_list("loras")
        if _get_param(request, "format") == "details":
            return web.json_response([info for file in files if (info := _file_info(file)) is not None])
        return web.json_response(list(files))

    @routes.get("/aio-image-gen/api/loras/info")
    async def lora_info(request):
        files = (_get_param(request, "files") or "").split(",")
        if not files or files == [""]:
            files = folder_paths.get_filename_list("loras")
        light = _is_truthy(_get_param(request, "light"))
        data = [
            info
            for file in files
            if (info := get_lora_info(file, light=light, fetch_metadata=not light)) is not None
        ]
        return web.json_response({"status": 200, "data": data})

    @routes.get("/aio-image-gen/api/loras/info/refresh")
    async def lora_info_refresh(request):
        files = (_get_param(request, "files") or "").split(",")
        if not files or files == [""]:
            return _json_response({"status": 404, "error": "No LoRA file provided."})
        data = [
            info
            for file in files
            if (info := get_lora_info(file, fetch_metadata=True, fetch_civitai=True)) is not None
        ]
        return web.json_response({"status": 200, "data": data})

    @routes.post("/aio-image-gen/api/loras/info")
    async def lora_info_save(request):
        file = _get_param(request, "file")
        if not file:
            return _json_response({"status": 404, "error": "No LoRA file provided."})
        post = await request.post()
        partial = json.loads(post.get("json") or "{}")
        info = save_lora_info_partial(file, partial)
        if info is None:
            return _json_response({"status": 404, "error": "LoRA file not found."})
        return web.json_response({"status": 200, "data": info})

    @routes.get("/aio-image-gen/api/loras/img")
    async def lora_image(request):
        file = _get_param(request, "file")
        if not file:
            return _json_response({"status": 404, "error": "No LoRA file provided."})
        path = _lora_path(file)
        if path is None:
            return _json_response({"status": 404, "error": "LoRA file not found."})
        stem = path.with_suffix("")
        for extension in IMAGE_EXTENSIONS:
            candidate = stem.with_suffix(extension)
            if candidate.exists():
                return web.FileResponse(candidate)
        return _json_response({"status": 404, "error": "No preview image found."})


try:
    register_routes()
except Exception:
    # Tests and direct imports often run outside ComfyUI's server process.
    pass
