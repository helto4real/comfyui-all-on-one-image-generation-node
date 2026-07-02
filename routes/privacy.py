"""Local privacy encryption routes for AIO Image Generate."""

from __future__ import annotations

import logging

try:
    from helto_privacy import aiohttp_check_privacy_token
except Exception:  # pragma: no cover - dependency errors surface through privacy helpers
    aiohttp_check_privacy_token = None  # type: ignore[assignment]

try:
    from ..services.privacy import crypto_status, decrypt_state, encrypt_state
except ImportError:  # pragma: no cover - direct test imports
    from services.privacy import crypto_status, decrypt_state, encrypt_state


ROUTE_PREFIX = "/aio_image_generate/privacy"
_ROUTES_REGISTERED = False


def register_privacy_routes() -> bool:
    """Register privacy routes when ComfyUI's PromptServer is available."""

    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return True

    try:
        from aiohttp import web  # type: ignore
        import server  # type: ignore

        prompt_server = getattr(server.PromptServer, "instance", None)
    except Exception as exc:  # pragma: no cover - direct tests run outside ComfyUI
        logging.debug("AIO Image Generate privacy routes unavailable: %s", exc)
        return False

    if prompt_server is None:
        return False

    routes = prompt_server.routes

    @routes.get(f"{ROUTE_PREFIX}/status")
    async def get_privacy_status(_request):
        return web.json_response({"ok": True, **crypto_status()})

    @routes.post(f"{ROUTE_PREFIX}/encrypt")
    async def post_privacy_encrypt(request):
        denied = _privacy_token_denied(request)
        if denied is not None:
            return denied
        try:
            payload = await request.json()
            envelope = encrypt_state(payload.get("state", {}))
            return web.json_response({"ok": True, "envelope": envelope})
        except Exception as exc:  # noqa: BLE001 - route should report readable privacy errors
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @routes.post(f"{ROUTE_PREFIX}/decrypt")
    async def post_privacy_decrypt(request):
        denied = _privacy_token_denied(request)
        if denied is not None:
            return denied
        try:
            payload = await request.json()
            state = decrypt_state(payload.get("payload", {}))
            return web.json_response({"ok": True, "state": state})
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    _ROUTES_REGISTERED = True
    return True


def _privacy_token_denied(request):
    if aiohttp_check_privacy_token is None:
        return None
    return aiohttp_check_privacy_token(request)
