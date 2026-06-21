"""Routes for the local Ideogram 4 prompt library."""

from __future__ import annotations

import logging
from typing import Any

try:
    from ..services.ideogram4_prompt_library import (
        Ideogram4PromptLibraryError,
        create_prompt,
        delete_prompt,
        duplicate_prompt,
        list_items,
        patch_prompt,
        replace_prompt,
        use_prompt,
    )
except ImportError:  # pragma: no cover - direct test imports
    from services.ideogram4_prompt_library import (
        Ideogram4PromptLibraryError,
        create_prompt,
        delete_prompt,
        duplicate_prompt,
        list_items,
        patch_prompt,
        replace_prompt,
        use_prompt,
    )


ROUTE_PREFIX = "/aio_image_generate/ideogram4_prompt_library"
_ROUTES_REGISTERED = False


def register_ideogram4_prompt_library_routes() -> bool:
    """Register prompt library routes when ComfyUI's PromptServer is available."""

    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return True

    try:
        from aiohttp import web  # type: ignore
        import server  # type: ignore

        prompt_server = getattr(server.PromptServer, "instance", None)
    except Exception as exc:  # pragma: no cover - direct tests run outside ComfyUI
        logging.debug("AIO Ideogram prompt library routes unavailable: %s", exc)
        return False

    if prompt_server is None:
        return False

    routes = prompt_server.routes

    @routes.get(f"{ROUTE_PREFIX}/items")
    async def get_items(_request):
        try:
            return web.json_response({"ok": True, **list_items()})
        except Exception as exc:
            return _error_response(web, exc)

    @routes.post(f"{ROUTE_PREFIX}/prompts")
    async def post_prompt(request):
        try:
            data = await _json_payload(request)
            item = create_prompt(_entry_payload(data), metadata=data)
            return web.json_response({"ok": True, "item": item})
        except Exception as exc:
            return _error_response(web, exc)

    @routes.put(f"{ROUTE_PREFIX}/prompts" + "/{item_id}")
    async def put_prompt(request):
        try:
            data = await _json_payload(request)
            item = replace_prompt(request.match_info["item_id"], _entry_payload(data), metadata=data)
            return web.json_response({"ok": True, "item": item})
        except Exception as exc:
            return _error_response(web, exc)

    @routes.patch(f"{ROUTE_PREFIX}/prompts" + "/{item_id}")
    async def patch_prompt_route(request):
        try:
            data = await _json_payload(request)
            item = patch_prompt(
                request.match_info["item_id"],
                metadata=data,
                payload=_optional_entry_payload(data),
            )
            return web.json_response({"ok": True, "item": item})
        except Exception as exc:
            return _error_response(web, exc)

    @routes.post(f"{ROUTE_PREFIX}/prompts" + "/{item_id}/duplicate")
    async def duplicate_prompt_route(request):
        try:
            data = await _json_payload(request, empty_ok=True)
            item = duplicate_prompt(request.match_info["item_id"], metadata=data)
            return web.json_response({"ok": True, "item": item})
        except Exception as exc:
            return _error_response(web, exc)

    @routes.delete(f"{ROUTE_PREFIX}/prompts" + "/{item_id}")
    async def delete_prompt_route(request):
        try:
            deleted = delete_prompt(request.match_info["item_id"])
            return web.json_response({"ok": True, **deleted})
        except Exception as exc:
            return _error_response(web, exc)

    @routes.post(f"{ROUTE_PREFIX}/prompts" + "/{item_id}/use")
    async def use_prompt_route(request):
        try:
            item = use_prompt(request.match_info["item_id"])
            return web.json_response({"ok": True, "item": item, "prompt": item["payload"]})
        except Exception as exc:
            return _error_response(web, exc)

    _ROUTES_REGISTERED = True
    return True


async def _json_payload(request, *, empty_ok: bool = False) -> dict[str, Any]:
    try:
        data = await request.json()
    except Exception:
        if empty_ok:
            return {}
        raise Ideogram4PromptLibraryError("Request body must be JSON.")
    if not isinstance(data, dict):
        raise Ideogram4PromptLibraryError("Request JSON body must be an object.")
    return data


def _entry_payload(data: dict[str, Any]) -> dict[str, Any]:
    payload = _optional_entry_payload(data)
    if payload is None:
        raise Ideogram4PromptLibraryError("Request must include a prompt payload.")
    return payload


def _optional_entry_payload(data: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("prompt", "payload"):
        value = data.get(key)
        if isinstance(value, dict):
            return value
    return None


def _error_response(web, exc: Exception):
    status = 400 if isinstance(exc, Ideogram4PromptLibraryError) else 500
    return web.json_response({"ok": False, "error": str(exc)}, status=status)


__all__ = ["ROUTE_PREFIX", "register_ideogram4_prompt_library_routes"]
