from __future__ import annotations

from echo_agent.config.schema import GatewayAuthConfig
from echo_agent.gateway.auth import GatewayAuth


def test_gateway_auth_token_and_allowlist(tmp_path) -> None:
    auth = GatewayAuth(
        GatewayAuthConfig(
            mode="allowlist",
            allowed_users=["slack:u1"],
            admin_users=["admin"],
            api_tokens=["secret"],
        ),
        tmp_path,
    )

    assert auth.authenticate_token("secret")
    assert not auth.authenticate_token("wrong")
    assert auth.is_authorized("slack", "u1")
    assert not auth.is_authorized("slack", "u2")
    assert auth.is_admin("slack", "admin")
    assert auth.is_admin("slack", "u2", token="secret")
