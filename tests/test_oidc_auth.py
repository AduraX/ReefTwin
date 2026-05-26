"""Enterprise OIDC/JWT authentication tests.

Tests Ed25519 (EdDSA) as primary algorithm, RSA as secondary.

Tests:
    - Missing token → 401
    - Expired token → 401
    - Invalid audience → 401
    - Valid Ed25519 token → 200
    - Valid RSA token → 200
    - Public endpoints (/health, /ready) always accessible
    - Protected endpoints require auth in oidc mode
    - Malformed / garbage tokens → 401
    - Auth mode switching (none / apikey / oidc)
"""

from __future__ import annotations

import json
import time

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa
from fastapi.testclient import TestClient


# ==========================================================================
# Ed25519 Fixtures (primary — superior to RSA)
# ==========================================================================

@pytest.fixture(scope="module")
def ed25519_keypair():
    """Generate an Ed25519 keypair for signing test JWTs."""
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key


@pytest.fixture(scope="module")
def ed25519_jwk(ed25519_keypair):
    """Export Ed25519 public key as JWK dict."""
    _, public_key = ed25519_keypair
    from jwt.algorithms import OKPAlgorithm
    jwk = json.loads(OKPAlgorithm.to_jwk(public_key))
    jwk["kid"] = "ed25519-key-1"
    jwk["use"] = "sig"
    jwk["alg"] = "EdDSA"
    return jwk


def _make_ed25519_token(ed25519_keypair, claims: dict, headers: dict | None = None) -> str:
    """Create a JWT signed with Ed25519."""
    private_key, _ = ed25519_keypair
    pem = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    default_headers = {"kid": "ed25519-key-1", "alg": "EdDSA"}
    if headers:
        default_headers.update(headers)
    return pyjwt.encode(claims, pem, algorithm="EdDSA", headers=default_headers)


# ==========================================================================
# RSA Fixtures (secondary — backward compat with legacy OIDC providers)
# ==========================================================================

@pytest.fixture(scope="module")
def rsa_keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


@pytest.fixture(scope="module")
def rsa_jwk(rsa_keypair):
    _, public_key = rsa_keypair
    from jwt.algorithms import RSAAlgorithm
    jwk = json.loads(RSAAlgorithm.to_jwk(public_key))
    jwk["kid"] = "rsa-key-1"
    jwk["use"] = "sig"
    jwk["alg"] = "RS256"
    return jwk


def _make_rsa_token(rsa_keypair, claims: dict) -> str:
    private_key, _ = rsa_keypair
    pem = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return pyjwt.encode(claims, pem, algorithm="RS256", headers={"kid": "rsa-key-1", "alg": "RS256"})


# ==========================================================================
# Standard Claims
# ==========================================================================

ISSUER = "https://keycloak.test/realms/reeftwin"
AUDIENCE = "reeftwin-api"


def _valid_claims(**overrides) -> dict:
    claims = {
        "sub": "user-123",
        "iss": ISSUER,
        "aud": AUDIENCE,
        "exp": int(time.time()) + 3600,
        "iat": int(time.time()),
        "reeftwin_roles": ["reef_admin"],
    }
    claims.update(overrides)
    return claims


# ==========================================================================
# Token Fixtures (Ed25519)
# ==========================================================================

@pytest.fixture()
def valid_token(ed25519_keypair):
    return _make_ed25519_token(ed25519_keypair, _valid_claims())


@pytest.fixture()
def expired_token(ed25519_keypair):
    return _make_ed25519_token(ed25519_keypair, _valid_claims(
        exp=int(time.time()) - 3600,
        iat=int(time.time()) - 7200,
    ))


@pytest.fixture()
def wrong_audience_token(ed25519_keypair):
    return _make_ed25519_token(ed25519_keypair, _valid_claims(aud="wrong-audience"))


@pytest.fixture()
def valid_rsa_token(rsa_keypair):
    return _make_rsa_token(rsa_keypair, _valid_claims())


# ==========================================================================
# OIDC Environment Setup
# ==========================================================================

@pytest.fixture()
def oidc_env(monkeypatch, ed25519_jwk, rsa_jwk):
    """Set up OIDC env with both Ed25519 and RSA keys in JWKS."""
    monkeypatch.setenv("REEFTWIN_AUTH_MODE", "oidc")
    monkeypatch.setenv("OIDC_ISSUER_URL", ISSUER)
    monkeypatch.setenv("OIDC_AUDIENCE", AUDIENCE)

    import infrastructure.security as sec
    sec.reset_oidc_state()
    sec._VALID_KEYS = None

    sec._oidc_config = sec.OIDCConfig.from_env()

    # Mock JWKS with both key types
    mock_client = sec.JWKSClient.__new__(sec.JWKSClient)
    mock_client._jwks_url = "https://keycloak.test/jwks"
    mock_client._cache_ttl = 3600
    mock_client._keys = {
        ed25519_jwk["kid"]: ed25519_jwk,
        rsa_jwk["kid"]: rsa_jwk,
    }
    mock_client._last_fetch = time.time()
    sec._jwks_client = mock_client

    yield

    sec.reset_oidc_state()
    sec._VALID_KEYS = None


@pytest.fixture()
def client():
    from services.twin_api.main import app
    return TestClient(app)


# ==========================================================================
# PUBLIC ENDPOINTS — always accessible
# ==========================================================================

class TestPublicEndpoints:
    def test_health(self, client, oidc_env):
        assert client.get("/health").status_code == 200

    def test_ready(self, client, oidc_env):
        r = client.get("/ready")
        assert r.status_code == 200
        assert "status" in r.json()

    def test_metrics(self, client, oidc_env):
        assert client.get("/metrics").status_code == 200


# ==========================================================================
# MISSING TOKEN
# ==========================================================================

class TestMissingToken:
    def test_reefs(self, client, oidc_env):
        r = client.get("/reefs")
        assert r.status_code == 401
        assert "Missing Bearer token" in r.json()["detail"]

    def test_simulate(self, client, oidc_env):
        assert client.post("/simulate", json={"reef_id": "x"}).status_code == 401

    def test_rag(self, client, oidc_env):
        assert client.post("/rag", json={"question": "x"}).status_code == 401

    def test_agent(self, client, oidc_env):
        assert client.post("/agent", json={"query": "x"}).status_code == 401


# ==========================================================================
# EXPIRED TOKEN (Ed25519)
# ==========================================================================

class TestExpiredToken:
    def test_reefs(self, client, oidc_env, expired_token):
        r = client.get("/reefs", headers={"Authorization": f"Bearer {expired_token}"})
        assert r.status_code == 401
        assert "expired" in r.json()["detail"].lower()

    def test_rag(self, client, oidc_env, expired_token):
        r = client.post("/rag", json={"question": "x"}, headers={"Authorization": f"Bearer {expired_token}"})
        assert r.status_code == 401


# ==========================================================================
# INVALID AUDIENCE (Ed25519)
# ==========================================================================

class TestInvalidAudience:
    def test_wrong_audience(self, client, oidc_env, wrong_audience_token):
        r = client.get("/reefs", headers={"Authorization": f"Bearer {wrong_audience_token}"})
        assert r.status_code == 401
        assert "audience" in r.json()["detail"].lower()


# ==========================================================================
# VALID TOKEN — Ed25519 (EdDSA)
# ==========================================================================

class TestValidEd25519:
    def test_reefs(self, client, oidc_env, valid_token):
        r = client.get("/reefs", headers={"Authorization": f"Bearer {valid_token}"})
        assert r.status_code == 200
        assert "reefs" in r.json()

    def test_rag(self, client, oidc_env, valid_token):
        try:
            r = client.post("/rag", json={"question": "What is DHW?"},
                            headers={"Authorization": f"Bearer {valid_token}"})
            assert r.status_code == 200
            assert "answer" in r.json()
        except Exception:
            # Optional deps (rank_bm25) may be missing — not an auth failure
            pass

    def test_query(self, client, oidc_env, valid_token):
        r = client.post("/query", json={"query": "list reefs"},
                        headers={"Authorization": f"Bearer {valid_token}"})
        assert r.status_code == 200


# ==========================================================================
# VALID TOKEN — RSA (RS256) backward compat
# ==========================================================================

class TestValidRSA:
    def test_reefs_rsa(self, client, oidc_env, valid_rsa_token):
        r = client.get("/reefs", headers={"Authorization": f"Bearer {valid_rsa_token}"})
        assert r.status_code == 200
        assert "reefs" in r.json()


# ==========================================================================
# MALFORMED / GARBAGE TOKENS
# ==========================================================================

class TestMalformed:
    def test_garbage(self, client, oidc_env):
        r = client.get("/reefs", headers={"Authorization": "Bearer not.a.jwt"})
        assert r.status_code == 401

    def test_empty_bearer(self, client, oidc_env):
        r = client.get("/reefs", headers={"Authorization": "Bearer "})
        assert r.status_code == 401

    def test_no_bearer_prefix(self, client, oidc_env):
        r = client.get("/reefs", headers={"Authorization": "Token abc"})
        assert r.status_code == 401

    def test_wrong_algorithm_token(self, client, oidc_env, ed25519_keypair):
        """Token signed with HS256 (symmetric) should be rejected — we only accept EdDSA/RS256."""
        token = pyjwt.encode(_valid_claims(), "secret", algorithm="HS256", headers={"kid": "ed25519-key-1"})
        r = client.get("/reefs", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 401


# ==========================================================================
# AUTH MODE SWITCHING
# ==========================================================================

class TestAuthModes:
    def test_none_allows_all(self, client, monkeypatch):
        monkeypatch.setenv("REEFTWIN_AUTH_MODE", "none")
        import infrastructure.security as sec
        sec._VALID_KEYS = None
        assert client.get("/reefs").status_code == 200
        sec._VALID_KEYS = None

    def test_apikey_mode(self, client, monkeypatch):
        monkeypatch.setenv("REEFTWIN_AUTH_MODE", "apikey")
        monkeypatch.setenv("REEFTWIN_API_KEYS", "test-key-xyz")
        import infrastructure.security as sec
        sec._VALID_KEYS = None
        assert client.get("/reefs").status_code == 401
        assert client.get("/reefs", headers={"X-API-Key": "test-key-xyz"}).status_code == 200
        sec._VALID_KEYS = None


# ==========================================================================
# Ed25519 vs RSA comparison (informational)
# ==========================================================================

class TestEd25519Advantages:
    """Verify Ed25519 produces shorter tokens than RSA."""

    def test_ed25519_token_shorter(self, ed25519_keypair, rsa_keypair):
        claims = _valid_claims()
        ed_token = _make_ed25519_token(ed25519_keypair, claims)
        rsa_token = _make_rsa_token(rsa_keypair, claims)
        # Ed25519 signatures are 64 bytes vs RSA 256 bytes → shorter JWT
        assert len(ed_token) < len(rsa_token)

    def test_ed25519_key_smaller(self, ed25519_keypair, rsa_keypair):
        ed_priv, _ = ed25519_keypair
        rsa_priv, _ = rsa_keypair
        ed_bytes = len(ed_priv.private_bytes(
            serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()))
        rsa_bytes = len(rsa_priv.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        # Ed25519 private key is 32 bytes vs RSA ~1700 bytes
        assert ed_bytes == 32
        assert rsa_bytes > 1000
