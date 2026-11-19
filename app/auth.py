"""Who is calling? Resolves the request to an owner id (or None for anonymous).

Two modes, switched by AUTH_REQUIRED:
  false (default, local dev): the browser's self-minted UUID in `X-Owner-Id` is trusted. Convenient,
        but it is only an unguessable link, not a credential.
  true  (anything public): only a Supabase access token (`Authorization: Bearer <jwt>`) counts. The
        server asks Supabase Auth to validate it, so the user id comes from a signed token and
        cannot be chosen by the client. `X-Owner-Id` is ignored entirely.
Anonymous callers can still search the shared library; uploading/deleting needs an identity.
"""
import time
import uuid

from fastapi import HTTPException

from app import db
from app.config import get_settings

_TTL_S = 300
_cache: dict[str, tuple[float, str]] = {}  # token -> (expires_at, user_id): saves a network call per request


def _verify_token(token: str) -> str:
    now = time.monotonic()
    hit = _cache.get(token)
    if hit and hit[0] > now:
        return hit[1]
    try:
        user = db.get_client().auth.get_user(token).user
        user_id = str(uuid.UUID(user.id))
    except Exception:  # noqa: BLE001 - any failure (expired, forged, Supabase down) means "not authenticated"
        raise HTTPException(401, "Invalid or expired login. Please sign in again.") from None
    if len(_cache) > 5000:
        _cache.clear()
    _cache[token] = (now + _TTL_S, user_id)
    return user_id


def resolve_owner(x_owner_id: str | None, authorization: str | None) -> str | None:
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise HTTPException(401, "Authorization header must be 'Bearer <token>'.")
        return _verify_token(token.strip())
    if get_settings().auth_required:
        return None  # no token -> anonymous, whatever X-Owner-Id says
    if x_owner_id is None:
        return None
    try:
        return str(uuid.UUID(x_owner_id))
    except ValueError:
        raise HTTPException(400, "X-Owner-Id must be a UUID.") from None
