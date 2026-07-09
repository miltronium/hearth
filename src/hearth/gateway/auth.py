"""Bearer-token auth for the gateway (ARCHITECTURE §2).

A single local token (generated on first run, stored ``0600``) gates every route except
the unauthenticated liveness probe ``/v1/hearth/admin/health``. Loopback-only binding
keeps this low-stakes, but the token stops other local users from driving the daemon.
"""

from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..config import get_or_create_token

# auto_error=False so we can return an OpenAI-style 401 rather than FastAPI's default.
_bearer = HTTPBearer(auto_error=False)


def require_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """FastAPI dependency: 401 unless ``Authorization: Bearer <token>`` matches.

    A no-op when ``settings.require_auth`` is False (tests, trusted single-user setups).
    """
    settings = request.app.state.settings
    if not settings.require_auth:
        return
    expected = get_or_create_token(settings)
    if credentials is None or not secrets.compare_digest(credentials.credentials, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "missing or invalid bearer token",
                    "type": "invalid_request_error",
                    "code": "hearth.auth.unauthorized",
                }
            },
            headers={"WWW-Authenticate": "Bearer"},
        )
