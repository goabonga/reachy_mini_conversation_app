"""Tests for OAuth2 authentication tools."""

import os
import asyncio
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.authenticate import (
    Logout,
    Whoami,
    CheckToken,
    Authenticate,
    CancelAuthentication,
    cancel_polling,
    get_token_info,
    is_polling_active,
    decode_jwt_payload,
    format_code_for_speech,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_deps() -> ToolDependencies:
    """Create mock ToolDependencies."""
    return ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
    )


@pytest.fixture
def sample_jwt() -> str:
    """Create a sample JWT token for testing (not cryptographically valid)."""
    # Header: {"alg": "RS256", "typ": "JWT"}
    # Payload: {"sub": "user123", "name": "John Doe", "email": "john@example.com", "exp": 9999999999}
    # This is a base64url encoded payload
    import json
    import base64

    header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256", "typ": "JWT"}).encode()).decode().rstrip("=")
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "sub": "user123",
                "name": "John Doe",
                "email": "john@example.com",
                "preferred_username": "johndoe",
                "exp": 9999999999,
            }
        ).encode()
    ).decode().rstrip("=")
    signature = "fake_signature"
    return f"{header}.{payload}.{signature}"


@pytest.fixture
def expired_jwt() -> str:
    """Create a JWT token with expired timestamp."""
    import json
    import base64

    header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256", "typ": "JWT"}).encode()).decode().rstrip("=")
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "sub": "user123",
                "name": "John Doe",
                "exp": 1000000000,  # Expired in 2001
            }
        ).encode()
    ).decode().rstrip("=")
    signature = "fake_signature"
    return f"{header}.{payload}.{signature}"


@pytest.fixture(autouse=True)
def clean_env() -> Generator[None, None, None]:
    """Clean environment variables before and after each test."""
    # Save original
    original_token = os.environ.get("ACCESS_TOKEN")
    original_client_id = os.environ.get("OAUTH2_CLIENT_ID")

    # Clear before test
    os.environ.pop("ACCESS_TOKEN", None)

    yield

    # Restore after test
    os.environ.pop("ACCESS_TOKEN", None)
    if original_token:
        os.environ["ACCESS_TOKEN"] = original_token
    if original_client_id:
        os.environ["OAUTH2_CLIENT_ID"] = original_client_id


@pytest.fixture(autouse=True)
def reset_polling_state() -> Generator[None, None, None]:
    """Reset global polling state before each test."""
    import reachy_mini_conversation_app.tools.authenticate as auth_module

    auth_module._polling_task = None
    auth_module._polling_cancelled = False
    yield
    auth_module._polling_task = None
    auth_module._polling_cancelled = False


# ============================================================================
# Unit Tests: Helper Functions
# ============================================================================


class TestFormatCodeForSpeech:
    """Tests for format_code_for_speech function."""

    def test_simple_code(self) -> None:
        """Test formatting a simple code."""
        assert format_code_for_speech("ABCD1234") == "A B C D 1 2 3 4"

    def test_code_with_dash(self) -> None:
        """Test formatting a code with dashes."""
        assert format_code_for_speech("ABCD-1234") == "A B C D 1 2 3 4"

    def test_code_with_spaces(self) -> None:
        """Test formatting a code with spaces."""
        assert format_code_for_speech("AB CD 12 34") == "A B C D 1 2 3 4"

    def test_lowercase_code(self) -> None:
        """Test formatting a lowercase code (should uppercase)."""
        assert format_code_for_speech("abcd1234") == "A B C D 1 2 3 4"


class TestDecodeJwtPayload:
    """Tests for decode_jwt_payload function."""

    def test_valid_jwt(self, sample_jwt: str) -> None:
        """Test decoding a valid JWT."""
        payload = decode_jwt_payload(sample_jwt)
        assert payload is not None
        assert payload["sub"] == "user123"
        assert payload["name"] == "John Doe"
        assert payload["email"] == "john@example.com"

    def test_invalid_jwt_format(self) -> None:
        """Test decoding an invalid JWT (not 3 parts)."""
        assert decode_jwt_payload("not.a.valid.jwt.token") is None
        assert decode_jwt_payload("onlyonepart") is None
        assert decode_jwt_payload("two.parts") is None

    def test_invalid_base64(self) -> None:
        """Test decoding a JWT with invalid base64."""
        assert decode_jwt_payload("header.!!!invalid!!!.signature") is None


class TestGetTokenInfo:
    """Tests for get_token_info function."""

    def test_no_token(self) -> None:
        """Test when no token is set."""
        assert get_token_info() is None

    def test_with_valid_token(self, sample_jwt: str) -> None:
        """Test when a valid token is set."""
        os.environ["ACCESS_TOKEN"] = sample_jwt
        info = get_token_info()
        assert info is not None
        assert info["sub"] == "user123"


class TestPollingState:
    """Tests for polling state management."""

    def test_is_polling_active_no_task(self) -> None:
        """Test is_polling_active when no task exists."""
        assert is_polling_active() is False

    def test_is_polling_active_with_done_task(self) -> None:
        """Test is_polling_active when task is done."""
        import reachy_mini_conversation_app.tools.authenticate as auth_module

        mock_task = MagicMock()
        mock_task.done.return_value = True
        auth_module._polling_task = mock_task
        assert is_polling_active() is False

    def test_is_polling_active_with_running_task(self) -> None:
        """Test is_polling_active when task is running."""
        import reachy_mini_conversation_app.tools.authenticate as auth_module

        mock_task = MagicMock()
        mock_task.done.return_value = False
        auth_module._polling_task = mock_task
        assert is_polling_active() is True

    def test_cancel_polling_no_task(self) -> None:
        """Test cancel_polling when no task exists."""
        assert cancel_polling() is False

    def test_cancel_polling_with_running_task(self) -> None:
        """Test cancel_polling when task is running."""
        import reachy_mini_conversation_app.tools.authenticate as auth_module

        mock_task = MagicMock()
        mock_task.done.return_value = False
        auth_module._polling_task = mock_task

        assert cancel_polling() is True
        assert auth_module._polling_cancelled is True
        mock_task.cancel.assert_called_once()


# ============================================================================
# Integration Tests: Authenticate Tool
# ============================================================================


class TestAuthenticateTool:
    """Tests for the Authenticate tool."""

    @pytest.mark.asyncio
    async def test_no_client_id(self, mock_deps: ToolDependencies) -> None:
        """Test authenticate fails when client ID is not configured."""
        tool = Authenticate()

        with patch(
            "reachy_mini_conversation_app.tools.authenticate.config"
        ) as mock_config:
            mock_config.OAUTH2_CLIENT_ID = None

            result = await tool(mock_deps)

            assert result["status"] == "error"
            assert "client id" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_already_authenticated(self, mock_deps: ToolDependencies, sample_jwt: str) -> None:
        """Test authenticate returns early when already authenticated."""
        os.environ["ACCESS_TOKEN"] = sample_jwt
        tool = Authenticate()

        with patch(
            "reachy_mini_conversation_app.tools.authenticate.config"
        ) as mock_config:
            mock_config.OAUTH2_CLIENT_ID = "test-client-id"

            result = await tool(mock_deps)

            assert result["status"] == "already_authenticated"
            assert "already connected" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_successful_device_code_request(self, mock_deps: ToolDependencies) -> None:
        """Test successful device code request."""
        tool = Authenticate()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"device_code": "dev123", "user_code": "ABCD-1234", "verification_uri": "https://auth.example.com/device", "expires_in": 600, "interval": 5}'
        mock_response.json.return_value = {
            "device_code": "dev123",
            "user_code": "ABCD-1234",
            "verification_uri": "https://auth.example.com/device",
            "expires_in": 600,
            "interval": 5,
        }
        mock_response.raise_for_status = MagicMock()

        with patch(
            "reachy_mini_conversation_app.tools.authenticate.config"
        ) as mock_config, patch(
            "reachy_mini_conversation_app.tools.authenticate.httpx.AsyncClient"
        ) as mock_client_class:
            mock_config.OAUTH2_CLIENT_ID = "test-client-id"
            mock_config.OAUTH2_ISSUER = "https://auth.example.com"

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await tool(mock_deps)

            assert result["status"] == "pending"
            assert result["user_code"] == "ABCD-1234"
            assert result["verification_uri"] == "https://auth.example.com/device"
            assert "A B C D 1 2 3 4" in result["message"]

    @pytest.mark.asyncio
    async def test_device_code_request_failure(self, mock_deps: ToolDependencies) -> None:
        """Test handling of device code request failure."""
        import httpx

        tool = Authenticate()

        with patch(
            "reachy_mini_conversation_app.tools.authenticate.config"
        ) as mock_config, patch(
            "reachy_mini_conversation_app.tools.authenticate.httpx.AsyncClient"
        ) as mock_client_class:
            mock_config.OAUTH2_CLIENT_ID = "test-client-id"
            mock_config.OAUTH2_ISSUER = "https://auth.example.com"

            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.HTTPError("Connection failed")
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await tool(mock_deps)

            assert result["status"] == "error"
            assert "couldn't start" in result["message"].lower()


# ============================================================================
# Integration Tests: Whoami Tool
# ============================================================================


class TestWhoamiTool:
    """Tests for the Whoami tool."""

    @pytest.mark.asyncio
    async def test_not_authenticated(self, mock_deps: ToolDependencies) -> None:
        """Test whoami when not authenticated."""
        tool = Whoami()
        result = await tool(mock_deps)

        assert result["status"] == "not_authenticated"
        assert "not connected" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_successful_userinfo(self, mock_deps: ToolDependencies, sample_jwt: str) -> None:
        """Test successful userinfo request."""
        os.environ["ACCESS_TOKEN"] = sample_jwt
        tool = Whoami()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"sub": "user123", "name": "John Doe", "email": "john@example.com"}'
        mock_response.json.return_value = {
            "sub": "user123",
            "name": "John Doe",
            "email": "john@example.com",
        }
        mock_response.raise_for_status = MagicMock()

        with patch(
            "reachy_mini_conversation_app.tools.authenticate.config"
        ) as mock_config, patch(
            "reachy_mini_conversation_app.tools.authenticate.httpx.AsyncClient"
        ) as mock_client_class:
            mock_config.OAUTH2_ISSUER = "https://auth.example.com"

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await tool(mock_deps)

            assert result["status"] == "authenticated"
            assert result["user"]["name"] == "John Doe"
            assert result["user"]["email"] == "john@example.com"
            assert "John Doe" in result["message"]

    @pytest.mark.asyncio
    async def test_token_expired(self, mock_deps: ToolDependencies, sample_jwt: str) -> None:
        """Test whoami when token is expired (401 response)."""
        os.environ["ACCESS_TOKEN"] = sample_jwt
        tool = Whoami()

        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch(
            "reachy_mini_conversation_app.tools.authenticate.config"
        ) as mock_config, patch(
            "reachy_mini_conversation_app.tools.authenticate.httpx.AsyncClient"
        ) as mock_client_class:
            mock_config.OAUTH2_ISSUER = "https://auth.example.com"

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await tool(mock_deps)

            assert result["status"] == "token_expired"
            assert "expired" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_fallback_to_jwt(self, mock_deps: ToolDependencies, sample_jwt: str) -> None:
        """Test fallback to JWT decoding when userinfo fails."""
        import httpx

        os.environ["ACCESS_TOKEN"] = sample_jwt
        tool = Whoami()

        with patch(
            "reachy_mini_conversation_app.tools.authenticate.config"
        ) as mock_config, patch(
            "reachy_mini_conversation_app.tools.authenticate.httpx.AsyncClient"
        ) as mock_client_class:
            mock_config.OAUTH2_ISSUER = "https://auth.example.com"

            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.HTTPError("Connection failed")
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await tool(mock_deps)

            # Should fallback to JWT and still return authenticated
            assert result["status"] == "authenticated"
            assert result["user"]["name"] == "John Doe"


# ============================================================================
# Integration Tests: CancelAuthentication Tool
# ============================================================================


class TestCancelAuthenticationTool:
    """Tests for the CancelAuthentication tool."""

    @pytest.mark.asyncio
    async def test_no_active_auth(self, mock_deps: ToolDependencies) -> None:
        """Test cancel when no authentication is in progress."""
        tool = CancelAuthentication()
        result = await tool(mock_deps)

        assert result["status"] == "no_active_auth"
        assert "no authentication" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_cancel_active_auth(self, mock_deps: ToolDependencies) -> None:
        """Test cancelling an active authentication."""
        import reachy_mini_conversation_app.tools.authenticate as auth_module

        mock_task = MagicMock()
        mock_task.done.return_value = False
        auth_module._polling_task = mock_task

        tool = CancelAuthentication()
        result = await tool(mock_deps)

        assert result["status"] == "cancelled"
        assert "cancelled" in result["message"].lower()
        mock_task.cancel.assert_called_once()


# ============================================================================
# Integration Tests: Logout Tool
# ============================================================================


class TestLogoutTool:
    """Tests for the Logout tool."""

    @pytest.mark.asyncio
    async def test_not_authenticated(self, mock_deps: ToolDependencies) -> None:
        """Test logout when not authenticated."""
        tool = Logout()
        result = await tool(mock_deps)

        assert result["status"] == "not_authenticated"
        assert "not connected" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_successful_logout(self, mock_deps: ToolDependencies, sample_jwt: str) -> None:
        """Test successful logout."""
        os.environ["ACCESS_TOKEN"] = sample_jwt
        tool = Logout()

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch(
            "reachy_mini_conversation_app.tools.authenticate.config"
        ) as mock_config, patch(
            "reachy_mini_conversation_app.tools.authenticate.httpx.AsyncClient"
        ) as mock_client_class:
            mock_config.OAUTH2_CLIENT_ID = "test-client-id"
            mock_config.OAUTH2_ISSUER = "https://auth.example.com"

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await tool(mock_deps)

            assert result["status"] == "logged_out"
            assert "logged out" in result["message"].lower()
            assert os.environ.get("ACCESS_TOKEN") is None

    @pytest.mark.asyncio
    async def test_logout_clears_token_on_failure(self, mock_deps: ToolDependencies, sample_jwt: str) -> None:
        """Test that logout clears local token even if revocation fails."""
        import httpx

        os.environ["ACCESS_TOKEN"] = sample_jwt
        tool = Logout()

        with patch(
            "reachy_mini_conversation_app.tools.authenticate.config"
        ) as mock_config, patch(
            "reachy_mini_conversation_app.tools.authenticate.httpx.AsyncClient"
        ) as mock_client_class:
            mock_config.OAUTH2_CLIENT_ID = "test-client-id"
            mock_config.OAUTH2_ISSUER = "https://auth.example.com"

            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.HTTPError("Connection failed")
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await tool(mock_deps)

            # Should still log out locally
            assert result["status"] == "logged_out"
            assert os.environ.get("ACCESS_TOKEN") is None


# ============================================================================
# Integration Tests: CheckToken Tool
# ============================================================================


class TestCheckTokenTool:
    """Tests for the CheckToken tool."""

    @pytest.mark.asyncio
    async def test_not_authenticated(self, mock_deps: ToolDependencies) -> None:
        """Test check_token when not authenticated."""
        tool = CheckToken()
        result = await tool(mock_deps)

        assert result["status"] == "not_authenticated"
        assert "not connected" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_token_valid(self, mock_deps: ToolDependencies, sample_jwt: str) -> None:
        """Test check_token with a valid token."""
        os.environ["ACCESS_TOKEN"] = sample_jwt
        tool = CheckToken()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"active": true, "username": "johndoe", "exp": 9999999999}'
        mock_response.json.return_value = {
            "active": True,
            "username": "johndoe",
            "exp": 9999999999,
        }
        mock_response.raise_for_status = MagicMock()

        with patch(
            "reachy_mini_conversation_app.tools.authenticate.config"
        ) as mock_config, patch(
            "reachy_mini_conversation_app.tools.authenticate.httpx.AsyncClient"
        ) as mock_client_class:
            mock_config.OAUTH2_CLIENT_ID = "test-client-id"
            mock_config.OAUTH2_ISSUER = "https://auth.example.com"

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await tool(mock_deps)

            assert result["status"] == "token_valid"
            assert result["active"] is True
            assert "valid" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_token_invalid(self, mock_deps: ToolDependencies, sample_jwt: str) -> None:
        """Test check_token with an invalid/revoked token."""
        os.environ["ACCESS_TOKEN"] = sample_jwt
        tool = CheckToken()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"active": false}'
        mock_response.json.return_value = {"active": False}
        mock_response.raise_for_status = MagicMock()

        with patch(
            "reachy_mini_conversation_app.tools.authenticate.config"
        ) as mock_config, patch(
            "reachy_mini_conversation_app.tools.authenticate.httpx.AsyncClient"
        ) as mock_client_class:
            mock_config.OAUTH2_CLIENT_ID = "test-client-id"
            mock_config.OAUTH2_ISSUER = "https://auth.example.com"

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await tool(mock_deps)

            assert result["status"] == "token_invalid"
            assert result["active"] is False
            # Token should be cleared
            assert os.environ.get("ACCESS_TOKEN") is None

    @pytest.mark.asyncio
    async def test_introspection_failure(self, mock_deps: ToolDependencies, sample_jwt: str) -> None:
        """Test check_token when introspection request fails."""
        import httpx

        os.environ["ACCESS_TOKEN"] = sample_jwt
        tool = CheckToken()

        with patch(
            "reachy_mini_conversation_app.tools.authenticate.config"
        ) as mock_config, patch(
            "reachy_mini_conversation_app.tools.authenticate.httpx.AsyncClient"
        ) as mock_client_class:
            mock_config.OAUTH2_CLIENT_ID = "test-client-id"
            mock_config.OAUTH2_ISSUER = "https://auth.example.com"

            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.HTTPError("Connection failed")
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await tool(mock_deps)

            assert result["status"] == "error"
            assert "couldn't verify" in result["message"].lower()


# ============================================================================
# Integration Tests: Polling Behavior
# ============================================================================


class TestPollingBehavior:
    """Tests for the background polling behavior."""

    @pytest.mark.asyncio
    async def test_polling_cancellation(self) -> None:
        """Test that polling can be cancelled."""
        import reachy_mini_conversation_app.tools.authenticate as auth_module

        auth_module._polling_cancelled = False

        async def fake_poll() -> str:
            while not auth_module._polling_cancelled:
                await asyncio.sleep(0.1)
            return "cancelled"

        task = asyncio.create_task(fake_poll())
        auth_module._polling_task = task  # type: ignore[assignment]

        await asyncio.sleep(0.05)
        assert is_polling_active() is True

        cancel_polling()
        await asyncio.sleep(0.15)

        assert auth_module._polling_cancelled is True
