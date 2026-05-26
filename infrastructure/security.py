"""Enterprise-grade API security: OIDC/JWT, API key, RBAC, rate limiting.

Authentication modes (REEFTWIN_AUTH_MODE):
    - "none"    — no auth (dev mode, all endpoints open)
    - "apikey"  — X-API-Key header (default)
    - "oidc"    — JWT Bearer token validated against OIDC/Keycloak provider

RBAC roles (from JWT ``reeftwin_roles`` claim or ``realm_access.roles``):
    - reef_admin     — full access to all resources
    - scientist      — upload datasets, run simulations, view all reef states
    - analyst        — view reef states and dashboards (read-only)
    - public_viewer  — read-only access to public reef summaries

Deny-by-default: if no recognised role is present, access is denied.

Environment variables:
    REEFTWIN_AUTH_MODE       — none | apikey | oidc
    REEFTWIN_API_KEYS        — comma-separated valid API keys (apikey mode)
    OIDC_ISSUER_URL          — OIDC issuer (e.g. https://keycloak.example.com/realms/reeftwin)
    OIDC_AUDIENCE            — expected JWT audience (default: reeftwin-api)
    OIDC_JWKS_URL            — JWKS endpoint (auto-derived from issuer if not set)
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from infrastructure.logging import get_logger

logger = get_logger("security")


# ==========================================================================
# RBAC — Roles, Permissions, and Mappings
# ==========================================================================

class Role(str, Enum):
    REEF_ADMIN = "reef_admin"
    SCIENTIST = "scientist"
    ANALYST = "analyst"
    PUBLIC_VIEWER = "public_viewer"


class Permission(str, Enum):
    SIMULATE = "simulate"
    UPLOAD_DATASET = "upload_dataset"
    VIEW_REEF_STATE = "view_reef_state"
    VIEW_PUBLIC_REEFS = "view_public_reefs"
    VIEW_DASHBOARDS = "view_dashboards"
    RAG_QUERY = "rag_query"
    AGENT_QUERY = "agent_query"
    INTERPRET = "interpret"
    SMART_QUERY = "smart_query"
    MANAGE_ALL = "manage_all"


ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.REEF_ADMIN: frozenset(Permission),
    Role.SCIENTIST: frozenset({
        Permission.SIMULATE,
        Permission.UPLOAD_DATASET,
        Permission.VIEW_REEF_STATE,
        Permission.VIEW_PUBLIC_REEFS,
        Permission.VIEW_DASHBOARDS,
        Permission.RAG_QUERY,
        Permission.AGENT_QUERY,
        Permission.INTERPRET,
        Permission.SMART_QUERY,
    }),
    Role.ANALYST: frozenset({
        Permission.VIEW_REEF_STATE,
        Permission.VIEW_PUBLIC_REEFS,
        Permission.VIEW_DASHBOARDS,
        Permission.RAG_QUERY,
        Permission.INTERPRET,
        Permission.SMART_QUERY,
    }),
    Role.PUBLIC_VIEWER: frozenset({
        Permission.VIEW_PUBLIC_REEFS,
    }),
}

# Object-level access: maps user sub → set of reef_ids they may access.
# Empty dict means "use role-based defaults" (admin/scientist see all).
_reef_acl: dict[str, set[str]] = {}


def set_reef_acl(acl: dict[str, set[str]]) -> None:
    """Configure per-user reef access lists (for testing or dynamic policy)."""
    global _reef_acl
    _reef_acl = acl


def clear_reef_acl() -> None:
    global _reef_acl
    _reef_acl = {}


def extract_roles(payload: dict[str, Any]) -> list[Role]:
    """Extract RBAC roles from a JWT payload.

    Checks (in order):
        1. ``reeftwin_roles`` claim (list of role strings)
        2. ``realm_access.roles`` (Keycloak convention)

    Returns only recognised Role values; unknown strings are ignored.
    """
    raw: list[str] = []
    if "reeftwin_roles" in payload:
        raw = payload["reeftwin_roles"]
    elif "realm_access" in payload and "roles" in payload["realm_access"]:
        raw = payload["realm_access"]["roles"]

    roles: list[Role] = []
    for r in raw:
        try:
            roles.append(Role(r))
        except ValueError:
            pass
    return roles


def user_has_permission(roles: list[Role], permission: Permission) -> bool:
    """Return True if any of the given roles grant the requested permission."""
    return any(permission in ROLE_PERMISSIONS.get(role, frozenset()) for role in roles)


def check_reef_access(
    roles: list[Role], sub: str, reef_id: str,
) -> None:
    """Object-level authorisation for a specific reef_id.

    Rules:
        - reef_admin can access any reef.
        - If an explicit ACL entry exists for the user, enforce it.
        - scientist can access all reefs (unless restricted by ACL).
        - analyst can access all reefs (unless restricted by ACL).
        - public_viewer is denied (must use /public/reefs).
    """
    if Role.REEF_ADMIN in roles:
        return

    if sub in _reef_acl:
        if reef_id not in _reef_acl[sub]:
            raise HTTPException(status_code=403, detail=f"Access denied for reef {reef_id}")
        return

    if Role.SCIENTIST in roles or Role.ANALYST in roles:
        return

    raise HTTPException(status_code=403, detail=f"Access denied for reef {reef_id}")


# ==========================================================================
# OIDC / JWT Authentication
# ==========================================================================

@dataclass
class OIDCConfig:
    issuer_url: str = ""
    audience: str = "reeftwin-api"
    jwks_url: str = ""
    algorithms: list[str] = field(default_factory=lambda: ["EdDSA", "RS256"])
    jwks_cache_ttl: int = 3600

    @classmethod
    def from_env(cls) -> OIDCConfig:
        issuer = os.getenv("OIDC_ISSUER_URL", "")
        return cls(
            issuer_url=issuer,
            audience=os.getenv("OIDC_AUDIENCE", "reeftwin-api"),
            jwks_url=os.getenv("OIDC_JWKS_URL", ""),
        )

    @property
    def effective_jwks_url(self) -> str:
        if self.jwks_url:
            return self.jwks_url
        if self.issuer_url:
            return f"{self.issuer_url.rstrip('/')}/.well-known/openid-configuration"
        return ""


class JWKSClient:
    """Cached JWKS key fetcher with automatic refresh on key rotation."""

    def __init__(self, jwks_url: str, cache_ttl: int = 3600) -> None:
        self._jwks_url = jwks_url
        self._cache_ttl = cache_ttl
        self._keys: dict[str, Any] = {}
        self._last_fetch: float = 0

    def _fetch_jwks(self) -> None:
        url = self._jwks_url
        if "openid-configuration" in url:
            try:
                req = urllib.request.Request(url, headers={"Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    discovery = json.loads(resp.read())
                url = discovery["jwks_uri"]
                self._jwks_url = url
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"OIDC discovery failed: {e}")

        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                jwks = json.loads(resp.read())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"JWKS fetch failed: {e}")

        self._keys = {}
        for key_data in jwks.get("keys", []):
            kid = key_data.get("kid")
            if kid:
                self._keys[kid] = key_data
        self._last_fetch = time.time()
        logger.info("JWKS fetched: %d keys from %s", len(self._keys), url)

    def get_signing_key(self, kid: str) -> Any:
        """Resolve the public key for a given kid. Supports RSA and OKP (Ed25519)."""
        import jwt as pyjwt
        if time.time() - self._last_fetch > self._cache_ttl or kid not in self._keys:
            self._fetch_jwks()
        if kid not in self._keys:
            raise HTTPException(status_code=401, detail=f"Signing key not found: {kid}")

        key_data = self._keys[kid]
        kty = key_data.get("kty", "RSA")
        jwk_json = json.dumps(key_data)

        if kty == "OKP":
            # Ed25519 / Ed448 (EdDSA)
            return pyjwt.algorithms.OKPAlgorithm.from_jwk(jwk_json)
        elif kty == "RSA":
            return pyjwt.algorithms.RSAAlgorithm.from_jwk(jwk_json)
        elif kty == "EC":
            return pyjwt.algorithms.ECAlgorithm.from_jwk(jwk_json)
        else:
            raise HTTPException(status_code=401, detail=f"Unsupported key type: {kty}")


_oidc_config: OIDCConfig | None = None
_jwks_client: JWKSClient | None = None


def _get_oidc_setup() -> tuple[OIDCConfig, JWKSClient | None]:
    global _oidc_config, _jwks_client
    if _oidc_config is None:
        _oidc_config = OIDCConfig.from_env()
    if _jwks_client is None and _oidc_config.effective_jwks_url:
        _jwks_client = JWKSClient(_oidc_config.effective_jwks_url, _oidc_config.jwks_cache_ttl)
    return _oidc_config, _jwks_client


def reset_oidc_state() -> None:
    """Reset cached OIDC state (for testing)."""
    global _oidc_config, _jwks_client
    _oidc_config = None
    _jwks_client = None


def validate_jwt(token: str) -> dict[str, Any]:
    """Validate JWT: signature (RS256/JWKS), issuer, audience, expiry.

    Returns decoded payload or raises HTTPException(401).
    """
    import jwt as pyjwt

    config, jwks_client = _get_oidc_setup()

    if not config.issuer_url:
        raise HTTPException(status_code=500, detail="OIDC_ISSUER_URL not configured")
    if jwks_client is None:
        raise HTTPException(status_code=500, detail="JWKS client not initialised")

    try:
        unverified_header = pyjwt.get_unverified_header(token)
    except pyjwt.exceptions.DecodeError:
        raise HTTPException(status_code=401, detail="Malformed JWT")

    kid = unverified_header.get("kid")
    if not kid:
        raise HTTPException(status_code=401, detail="JWT missing kid in header")

    alg = unverified_header.get("alg", "RS256")
    if alg not in config.algorithms:
        raise HTTPException(status_code=401, detail=f"Unsupported algorithm: {alg}")

    signing_key = jwks_client.get_signing_key(kid)

    try:
        payload = pyjwt.decode(
            token,
            signing_key,
            algorithms=config.algorithms,
            audience=config.audience,
            issuer=config.issuer_url,
            options={"verify_exp": True, "verify_iss": True, "verify_aud": True, "require": ["exp", "iss", "sub"]},
        )
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except pyjwt.InvalidAudienceError:
        raise HTTPException(status_code=401, detail="Invalid audience")
    except pyjwt.InvalidIssuerError:
        raise HTTPException(status_code=401, detail="Invalid issuer")
    except pyjwt.InvalidSignatureError:
        raise HTTPException(status_code=401, detail="Invalid signature")
    except pyjwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

    logger.debug("OIDC auth OK: sub=%s", payload.get("sub"))
    return payload


# ==========================================================================
# FastAPI Security Dependencies
# ==========================================================================

_bearer_scheme = HTTPBearer(auto_error=False)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_VALID_KEYS: set[str] | None = None


def _load_api_keys() -> set[str]:
    global _VALID_KEYS
    if _VALID_KEYS is not None:
        return _VALID_KEYS
    raw = os.getenv("REEFTWIN_API_KEYS", "")
    _VALID_KEYS = {k.strip() for k in raw.split(",") if k.strip()} if raw else set()
    return _VALID_KEYS


async def require_auth(
    request: Request,
    bearer: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
    api_key: str | None = Security(_api_key_header),
) -> dict[str, Any]:
    """Unified auth dependency for all protected endpoints.

    Dispatches to the correct auth backend based on REEFTWIN_AUTH_MODE.
    Returns a dict with at least {"sub": "...", "_roles": [...]}.
    """
    mode = os.getenv("REEFTWIN_AUTH_MODE", "apikey")

    if mode == "none":
        payload = {"sub": "anonymous", "auth_mode": "none"}
        payload["_roles"] = [Role.REEF_ADMIN]
        return payload

    if mode == "oidc":
        if not bearer:
            raise HTTPException(status_code=401, detail="Missing Bearer token")
        payload = validate_jwt(bearer.credentials)
        payload["auth_mode"] = "oidc"
        payload["_roles"] = extract_roles(payload)
        return payload

    # apikey mode
    valid_keys = _load_api_keys()
    if not valid_keys:
        payload = {"sub": "dev-no-keys", "auth_mode": "apikey"}
        payload["_roles"] = [Role.REEF_ADMIN]
        return payload
    if not api_key or api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    payload = {"sub": f"apikey:{api_key[:8]}...", "auth_mode": "apikey"}
    payload["_roles"] = [Role.REEF_ADMIN]
    return payload


# Backward compat
require_api_key = require_auth


def require_permission(*permissions: Permission):
    """FastAPI dependency factory: deny-by-default permission check.

    Usage::

        @app.post("/simulate", dependencies=[Depends(require_permission(Permission.SIMULATE))])
    """
    async def _check(
        request: Request,
        bearer: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
        api_key: str | None = Security(_api_key_header),
    ) -> dict[str, Any]:
        payload = await require_auth(request, bearer, api_key)
        roles: list[Role] = payload.get("_roles", [])

        if not roles:
            raise HTTPException(status_code=403, detail="No recognised role — access denied")

        if not any(user_has_permission(roles, p) for p in permissions):
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions — requires one of: {[p.value for p in permissions]}",
            )

        request.state.auth_payload = payload
        return payload

    return _check


def require_reef_access(permission: Permission):
    """FastAPI dependency for object-level reef_id authorisation.

    Extracts ``reef_id`` from path params, checks role permission,
    then checks object-level ACL.
    """
    async def _check(
        request: Request,
        reef_id: str,
        bearer: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
        api_key: str | None = Security(_api_key_header),
    ) -> dict[str, Any]:
        payload = await require_auth(request, bearer, api_key)
        roles: list[Role] = payload.get("_roles", [])

        if not roles:
            raise HTTPException(status_code=403, detail="No recognised role — access denied")

        if not user_has_permission(roles, permission):
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions — requires: {permission.value}",
            )

        check_reef_access(roles, payload.get("sub", ""), reef_id)
        request.state.auth_payload = payload
        return payload

    return _check


# ==========================================================================
# Input Validation
# ==========================================================================

_REEF_ID_PATTERN = re.compile(r"^[a-z0-9_]{1,64}$")


def validate_reef_id(reef_id: str) -> str:
    if not _REEF_ID_PATTERN.match(reef_id):
        raise HTTPException(status_code=400, detail=f"Invalid reef_id: {reef_id!r}")
    return reef_id


def validate_query_length(query: str, max_length: int = 2000) -> str:
    if len(query) > max_length:
        raise HTTPException(status_code=400, detail=f"Query too long ({len(query)} chars, max {max_length})")
    return query


# ==========================================================================
# Rate Limiting
# ==========================================================================

class RateLimiter:
    def __init__(self, requests_per_minute: int = 30) -> None:
        self._rpm = requests_per_minute
        self._requests: dict[str, list[float]] = defaultdict(list)

    def check(self, client_ip: str) -> None:
        now = time.time()
        self._requests[client_ip] = [t for t in self._requests[client_ip] if t > now - 60]
        if len(self._requests[client_ip]) >= self._rpm:
            raise HTTPException(status_code=429, detail=f"Rate limit exceeded ({self._rpm}/min)")
        self._requests[client_ip].append(now)


llm_rate_limiter = RateLimiter(requests_per_minute=30)


async def check_rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    llm_rate_limiter.check(client_ip)
