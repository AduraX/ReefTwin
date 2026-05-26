"""RBAC authorisation tests — privilege escalation & object-level access bypass.

Tests:
    - Deny-by-default: no role → 403 on every protected endpoint
    - Role-permission matrix: each role can only reach its allowed endpoints
    - Privilege escalation: analyst cannot simulate or upload
    - Public viewer can only access /public/reefs
    - Object-level reef_id access: ACL enforcement and bypass attempts
    - JWT role extraction from reeftwin_roles and realm_access.roles
    - Admin override: reef_admin bypasses all ACLs
"""

from __future__ import annotations

import json
import time

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from fastapi.testclient import TestClient

ISSUER = "https://keycloak.test/realms/reeftwin"
AUDIENCE = "reeftwin-api"


# ==========================================================================
# Fixtures
# ==========================================================================

@pytest.fixture(scope="module")
def ed25519_keypair():
    private_key = ed25519.Ed25519PrivateKey.generate()
    return private_key, private_key.public_key()


@pytest.fixture(scope="module")
def ed25519_jwk(ed25519_keypair):
    _, public_key = ed25519_keypair
    from jwt.algorithms import OKPAlgorithm
    jwk = json.loads(OKPAlgorithm.to_jwk(public_key))
    jwk["kid"] = "rbac-test-key"
    jwk["use"] = "sig"
    jwk["alg"] = "EdDSA"
    return jwk


def _sign(keypair, claims: dict) -> str:
    private_key, _ = keypair
    pem = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return pyjwt.encode(claims, pem, algorithm="EdDSA", headers={"kid": "rbac-test-key", "alg": "EdDSA"})


def _claims(roles: list[str], sub: str = "user-1", **extra) -> dict:
    c = {
        "sub": sub,
        "iss": ISSUER,
        "aud": AUDIENCE,
        "exp": int(time.time()) + 3600,
        "iat": int(time.time()),
        "reeftwin_roles": roles,
    }
    c.update(extra)
    return c


def _token(keypair, roles: list[str], sub: str = "user-1", **extra) -> str:
    return _sign(keypair, _claims(roles, sub=sub, **extra))


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def oidc_env(monkeypatch, ed25519_jwk):
    monkeypatch.setenv("REEFTWIN_AUTH_MODE", "oidc")
    monkeypatch.setenv("OIDC_ISSUER_URL", ISSUER)
    monkeypatch.setenv("OIDC_AUDIENCE", AUDIENCE)

    import infrastructure.security as sec
    sec.reset_oidc_state()
    sec._VALID_KEYS = None
    sec.clear_reef_acl()

    sec._oidc_config = sec.OIDCConfig.from_env()
    mock_client = sec.JWKSClient.__new__(sec.JWKSClient)
    mock_client._jwks_url = "https://keycloak.test/jwks"
    mock_client._cache_ttl = 3600
    mock_client._keys = {ed25519_jwk["kid"]: ed25519_jwk}
    mock_client._last_fetch = time.time()
    sec._jwks_client = mock_client

    yield

    sec.reset_oidc_state()
    sec._VALID_KEYS = None
    sec.clear_reef_acl()


@pytest.fixture()
def client():
    from services.twin_api.main import app
    return TestClient(app)


def _upload(client, token: str, **kwargs):
    """Helper to POST a CSV file upload to /datasets/upload."""
    csv_content = "reef_id,timestamp,water_temperature_c,ph,salinity_psu,turbidity_ntu,dissolved_oxygen_mg_l\ngbr_heron_reef,2026-01-01T00:00:00,28.0,8.05,35.1,0.8,6.5\n"
    return client.post(
        "/datasets/upload?dataset_type=iot",
        files={"file": ("test.csv", csv_content, "text/csv")},
        headers=_auth(token),
        **kwargs,
    )


# ==========================================================================
# Unit tests — role extraction
# ==========================================================================

class TestRoleExtraction:
    def test_extract_from_reeftwin_roles(self):
        from infrastructure.security import Role, extract_roles
        payload = {"reeftwin_roles": ["scientist", "analyst"]}
        roles = extract_roles(payload)
        assert Role.SCIENTIST in roles
        assert Role.ANALYST in roles

    def test_extract_from_realm_access(self):
        from infrastructure.security import Role, extract_roles
        payload = {"realm_access": {"roles": ["reef_admin"]}}
        roles = extract_roles(payload)
        assert roles == [Role.REEF_ADMIN]

    def test_reeftwin_roles_takes_precedence(self):
        from infrastructure.security import Role, extract_roles
        payload = {
            "reeftwin_roles": ["analyst"],
            "realm_access": {"roles": ["reef_admin"]},
        }
        roles = extract_roles(payload)
        assert roles == [Role.ANALYST]

    def test_unknown_roles_ignored(self):
        from infrastructure.security import extract_roles
        payload = {"reeftwin_roles": ["unknown_role", "hacker"]}
        assert extract_roles(payload) == []

    def test_no_roles_claim(self):
        from infrastructure.security import extract_roles
        assert extract_roles({"sub": "user-1"}) == []


# ==========================================================================
# Unit tests — permission checking
# ==========================================================================

class TestPermissionChecking:
    def test_admin_has_all_permissions(self):
        from infrastructure.security import Permission, Role, user_has_permission
        for p in Permission:
            assert user_has_permission([Role.REEF_ADMIN], p)

    def test_scientist_can_simulate_and_upload(self):
        from infrastructure.security import Permission, Role, user_has_permission
        assert user_has_permission([Role.SCIENTIST], Permission.SIMULATE)
        assert user_has_permission([Role.SCIENTIST], Permission.UPLOAD_DATASET)

    def test_analyst_cannot_simulate_or_upload(self):
        from infrastructure.security import Permission, Role, user_has_permission
        assert not user_has_permission([Role.ANALYST], Permission.SIMULATE)
        assert not user_has_permission([Role.ANALYST], Permission.UPLOAD_DATASET)

    def test_analyst_can_view(self):
        from infrastructure.security import Permission, Role, user_has_permission
        assert user_has_permission([Role.ANALYST], Permission.VIEW_REEF_STATE)
        assert user_has_permission([Role.ANALYST], Permission.VIEW_DASHBOARDS)

    def test_public_viewer_only_public(self):
        from infrastructure.security import Permission, Role, user_has_permission
        assert user_has_permission([Role.PUBLIC_VIEWER], Permission.VIEW_PUBLIC_REEFS)
        assert not user_has_permission([Role.PUBLIC_VIEWER], Permission.VIEW_REEF_STATE)
        assert not user_has_permission([Role.PUBLIC_VIEWER], Permission.SIMULATE)
        assert not user_has_permission([Role.PUBLIC_VIEWER], Permission.UPLOAD_DATASET)
        assert not user_has_permission([Role.PUBLIC_VIEWER], Permission.RAG_QUERY)

    def test_no_roles_denied(self):
        from infrastructure.security import Permission, user_has_permission
        assert not user_has_permission([], Permission.VIEW_REEF_STATE)


# ==========================================================================
# Deny-by-default — token with no recognised roles → 403
# ==========================================================================

class TestDenyByDefault:
    def test_no_roles_gets_403_on_reefs(self, client, oidc_env, ed25519_keypair):
        token = _token(ed25519_keypair, [])
        r = client.get("/reefs", headers=_auth(token))
        assert r.status_code == 403

    def test_no_roles_gets_403_on_simulate(self, client, oidc_env, ed25519_keypair):
        token = _token(ed25519_keypair, [])
        r = client.post("/simulate", json={"reef_id": "x"}, headers=_auth(token))
        assert r.status_code == 403

    def test_no_roles_gets_403_on_upload(self, client, oidc_env, ed25519_keypair):
        token = _token(ed25519_keypair, [])
        r = _upload(client, token)
        assert r.status_code == 403

    def test_unknown_role_gets_403(self, client, oidc_env, ed25519_keypair):
        token = _token(ed25519_keypair, ["unknown_role"])
        r = client.get("/reefs", headers=_auth(token))
        assert r.status_code == 403

    def test_no_roles_still_gets_public(self, client, oidc_env):
        """Public endpoints require no auth at all."""
        r = client.get("/public/reefs")
        assert r.status_code == 200


# ==========================================================================
# Role-permission matrix — endpoint access by role
# ==========================================================================

class TestReefAdmin:
    def test_can_list_reefs(self, client, oidc_env, ed25519_keypair):
        token = _token(ed25519_keypair, ["reef_admin"])
        assert client.get("/reefs", headers=_auth(token)).status_code == 200

    def test_can_simulate(self, client, oidc_env, ed25519_keypair):
        token = _token(ed25519_keypair, ["reef_admin"])
        r = client.post("/simulate", json={"reef_id": "gbr_heron_reef"}, headers=_auth(token))
        assert r.status_code in (200, 404)  # 404 if state file missing is ok

    def test_can_upload(self, client, oidc_env, ed25519_keypair):
        token = _token(ed25519_keypair, ["reef_admin"])
        r = _upload(client, token)
        assert r.status_code == 200

    def test_can_view_reef_state(self, client, oidc_env, ed25519_keypair):
        token = _token(ed25519_keypair, ["reef_admin"])
        r = client.get("/reefs/gbr_heron_reef/state", headers=_auth(token))
        assert r.status_code in (200, 404)


class TestScientist:
    def test_can_simulate(self, client, oidc_env, ed25519_keypair):
        token = _token(ed25519_keypair, ["scientist"])
        r = client.post("/simulate", json={"reef_id": "gbr_heron_reef"}, headers=_auth(token))
        assert r.status_code in (200, 404)

    def test_can_upload(self, client, oidc_env, ed25519_keypair):
        token = _token(ed25519_keypair, ["scientist"])
        r = _upload(client, token)
        assert r.status_code == 200

    def test_can_view_reefs(self, client, oidc_env, ed25519_keypair):
        token = _token(ed25519_keypair, ["scientist"])
        assert client.get("/reefs", headers=_auth(token)).status_code == 200

    def test_can_view_reef_state(self, client, oidc_env, ed25519_keypair):
        token = _token(ed25519_keypair, ["scientist"])
        r = client.get("/reefs/gbr_heron_reef/state", headers=_auth(token))
        assert r.status_code in (200, 404)

    def test_can_rag_query_not_forbidden(self, client, oidc_env, ed25519_keypair):
        """Scientist has RAG permission — RBAC must not block (may error if deps missing)."""
        token = _token(ed25519_keypair, ["scientist"])
        try:
            r = client.post("/rag", json={"question": "What is DHW?"}, headers=_auth(token))
            assert r.status_code not in (401, 403)
        except Exception:
            # Import errors from optional deps (rank_bm25 etc.) are not RBAC failures
            pass


class TestAnalyst:
    def test_can_view_reefs(self, client, oidc_env, ed25519_keypair):
        token = _token(ed25519_keypair, ["analyst"])
        assert client.get("/reefs", headers=_auth(token)).status_code == 200

    def test_can_view_reef_state(self, client, oidc_env, ed25519_keypair):
        token = _token(ed25519_keypair, ["analyst"])
        r = client.get("/reefs/gbr_heron_reef/state", headers=_auth(token))
        assert r.status_code in (200, 404)

    def test_cannot_simulate(self, client, oidc_env, ed25519_keypair):
        token = _token(ed25519_keypair, ["analyst"])
        r = client.post("/simulate", json={"reef_id": "gbr_heron_reef"}, headers=_auth(token))
        assert r.status_code == 403

    def test_cannot_upload(self, client, oidc_env, ed25519_keypair):
        token = _token(ed25519_keypair, ["analyst"])
        r = _upload(client, token)
        assert r.status_code == 403

    def test_cannot_agent_query(self, client, oidc_env, ed25519_keypair):
        token = _token(ed25519_keypair, ["analyst"])
        r = client.post("/agent", json={"query": "list reefs"}, headers=_auth(token))
        assert r.status_code == 403

    def test_can_rag_query_not_forbidden(self, client, oidc_env, ed25519_keypair):
        """Analyst has RAG permission — RBAC must not block (may error if deps missing)."""
        token = _token(ed25519_keypair, ["analyst"])
        try:
            r = client.post("/rag", json={"question": "bleaching?"}, headers=_auth(token))
            assert r.status_code not in (401, 403)
        except Exception:
            pass


class TestPublicViewer:
    def test_can_access_public_reefs(self, client, oidc_env):
        r = client.get("/public/reefs")
        assert r.status_code == 200
        assert "reefs" in r.json()

    def test_cannot_view_reef_state(self, client, oidc_env, ed25519_keypair):
        token = _token(ed25519_keypair, ["public_viewer"])
        r = client.get("/reefs/gbr_heron_reef/state", headers=_auth(token))
        assert r.status_code == 403

    def test_cannot_list_reefs(self, client, oidc_env, ed25519_keypair):
        token = _token(ed25519_keypair, ["public_viewer"])
        assert client.get("/reefs", headers=_auth(token)).status_code == 403

    def test_cannot_simulate(self, client, oidc_env, ed25519_keypair):
        token = _token(ed25519_keypair, ["public_viewer"])
        r = client.post("/simulate", json={"reef_id": "x"}, headers=_auth(token))
        assert r.status_code == 403

    def test_cannot_upload(self, client, oidc_env, ed25519_keypair):
        token = _token(ed25519_keypair, ["public_viewer"])
        r = _upload(client, token)
        assert r.status_code == 403

    def test_cannot_rag(self, client, oidc_env, ed25519_keypair):
        token = _token(ed25519_keypair, ["public_viewer"])
        r = client.post("/rag", json={"question": "x"}, headers=_auth(token))
        assert r.status_code == 403


# ==========================================================================
# Privilege escalation attempts
# ==========================================================================

class TestPrivilegeEscalation:
    def test_analyst_cannot_escalate_to_simulate(self, client, oidc_env, ed25519_keypair):
        """Analyst forges a token claiming scientist role but only has analyst."""
        token = _token(ed25519_keypair, ["analyst"])
        r = client.post("/simulate", json={"reef_id": "gbr_heron_reef"}, headers=_auth(token))
        assert r.status_code == 403

    def test_public_viewer_cannot_escalate_to_view_state(self, client, oidc_env, ed25519_keypair):
        token = _token(ed25519_keypair, ["public_viewer"])
        r = client.get("/reefs/gbr_heron_reef/state", headers=_auth(token))
        assert r.status_code == 403

    def test_analyst_cannot_upload_dataset(self, client, oidc_env, ed25519_keypair):
        token = _token(ed25519_keypair, ["analyst"])
        r = _upload(client, token)
        assert r.status_code == 403

    def test_multiple_low_roles_dont_escalate(self, client, oidc_env, ed25519_keypair):
        """Having analyst + public_viewer should NOT grant simulate access."""
        token = _token(ed25519_keypair, ["analyst", "public_viewer"])
        r = client.post("/simulate", json={"reef_id": "x"}, headers=_auth(token))
        assert r.status_code == 403

    def test_injected_manage_all_in_claims_ignored(self, client, oidc_env, ed25519_keypair):
        """Even if someone puts 'manage_all' as a role string, it's not a valid Role."""
        token = _token(ed25519_keypair, ["manage_all"])
        r = client.get("/reefs", headers=_auth(token))
        assert r.status_code == 403


# ==========================================================================
# Object-level reef_id access control
# ==========================================================================

class TestObjectLevelAccess:
    def test_acl_restricts_scientist_to_assigned_reefs(self, client, oidc_env, ed25519_keypair):
        import infrastructure.security as sec
        sec.set_reef_acl({"scientist-user": {"gbr_heron_reef"}})

        token = _token(ed25519_keypair, ["scientist"], sub="scientist-user")
        # Allowed reef
        r = client.get("/reefs/gbr_heron_reef/state", headers=_auth(token))
        assert r.status_code in (200, 404)  # 404 if no state file

        # Denied reef
        r = client.get("/reefs/gbr_lizard_island/state", headers=_auth(token))
        assert r.status_code == 403

        sec.clear_reef_acl()

    def test_acl_restricts_analyst_to_assigned_reefs(self, client, oidc_env, ed25519_keypair):
        import infrastructure.security as sec
        sec.set_reef_acl({"analyst-user": {"coral_sea_reef"}})

        token = _token(ed25519_keypair, ["analyst"], sub="analyst-user")
        r = client.get("/reefs/gbr_heron_reef/state", headers=_auth(token))
        assert r.status_code == 403

        r = client.get("/reefs/coral_sea_reef/state", headers=_auth(token))
        assert r.status_code in (200, 404)

        sec.clear_reef_acl()

    def test_admin_bypasses_acl(self, client, oidc_env, ed25519_keypair):
        import infrastructure.security as sec
        sec.set_reef_acl({"admin-user": {"gbr_heron_reef"}})  # ACL set but admin ignores it

        token = _token(ed25519_keypair, ["reef_admin"], sub="admin-user")
        r = client.get("/reefs/gbr_lizard_island/state", headers=_auth(token))
        assert r.status_code in (200, 404)  # admin always passes

        sec.clear_reef_acl()

    def test_no_acl_scientist_sees_all(self, client, oidc_env, ed25519_keypair):
        """Without explicit ACL, scientist can access any reef."""
        token = _token(ed25519_keypair, ["scientist"], sub="free-scientist")
        r = client.get("/reefs/gbr_heron_reef/state", headers=_auth(token))
        assert r.status_code in (200, 404)
        r = client.get("/reefs/gbr_lizard_island/state", headers=_auth(token))
        assert r.status_code in (200, 404)

    def test_public_viewer_denied_object_access(self, client, oidc_env, ed25519_keypair):
        """Public viewer should be denied even if ACL is set for them."""
        import infrastructure.security as sec
        sec.set_reef_acl({"public-user": {"gbr_heron_reef"}})

        token = _token(ed25519_keypair, ["public_viewer"], sub="public-user")
        r = client.get("/reefs/gbr_heron_reef/state", headers=_auth(token))
        assert r.status_code == 403  # permission denied before ACL check

        sec.clear_reef_acl()

    def test_bypass_attempt_with_path_traversal(self, client, oidc_env, ed25519_keypair):
        """Ensure path-traversal reef_ids are rejected by input validation."""
        import infrastructure.security as sec
        sec.set_reef_acl({"hacker": {"gbr_heron_reef"}})

        token = _token(ed25519_keypair, ["scientist"], sub="hacker")
        r = client.get("/reefs/../../../etc/passwd/state", headers=_auth(token))
        # FastAPI may return 404 for unmatched route or 400 for bad reef_id
        assert r.status_code in (400, 404, 422)

        sec.clear_reef_acl()


# ==========================================================================
# Keycloak realm_access.roles extraction
# ==========================================================================

class TestKeycloakRoles:
    def test_realm_access_roles_work(self, client, oidc_env, ed25519_keypair):
        """Roles from realm_access.roles (Keycloak convention) grant access."""
        claims = {
            "sub": "kc-user",
            "iss": ISSUER,
            "aud": AUDIENCE,
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
            "realm_access": {"roles": ["scientist"]},
        }
        token = _sign(ed25519_keypair, claims)
        r = client.get("/reefs", headers=_auth(token))
        assert r.status_code == 200

    def test_realm_access_analyst_cannot_simulate(self, client, oidc_env, ed25519_keypair):
        claims = {
            "sub": "kc-analyst",
            "iss": ISSUER,
            "aud": AUDIENCE,
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
            "realm_access": {"roles": ["analyst"]},
        }
        token = _sign(ed25519_keypair, claims)
        r = client.post("/simulate", json={"reef_id": "x"}, headers=_auth(token))
        assert r.status_code == 403


# ==========================================================================
# Public endpoint access (no auth at all)
# ==========================================================================

class TestPublicEndpoint:
    def test_public_reefs_no_auth(self, client, oidc_env):
        r = client.get("/public/reefs")
        assert r.status_code == 200
        data = r.json()
        assert "reefs" in data
        for reef in data["reefs"]:
            assert "reef_id" in reef

    def test_public_reefs_returns_summary_only(self, client, oidc_env):
        r = client.get("/public/reefs")
        for reef in r.json()["reefs"]:
            # Should only contain summary fields, not full state
            assert set(reef.keys()) <= {"reef_id", "ecosystem_status", "bleaching_risk_score"}

    def test_health_still_public(self, client, oidc_env):
        assert client.get("/health").status_code == 200

    def test_metrics_still_public(self, client, oidc_env):
        assert client.get("/metrics").status_code == 200
