"""OAuth2 Device Code authentication tool for Goabonga Cloud."""

import os
import json
import base64
import asyncio
import logging
from typing import Any, Dict, Optional

import httpx

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Global state for managing authentication polling
_polling_task: Optional[asyncio.Task[None]] = None
_polling_cancelled = False


def cancel_polling() -> bool:
    """Cancel any active polling task. Returns True if a task was cancelled."""
    global _polling_task, _polling_cancelled
    if _polling_task and not _polling_task.done():
        _polling_cancelled = True
        _polling_task.cancel()
        logger.info("Authentication polling cancelled")
        return True
    return False


def is_polling_active() -> bool:
    """Check if authentication polling is currently active."""
    return _polling_task is not None and not _polling_task.done()


def decode_jwt_payload(token: str) -> Optional[Dict[str, Any]]:
    """Decode JWT payload without verification (for reading claims only)."""
    try:
        # JWT format: header.payload.signature
        parts = token.split(".")
        if len(parts) != 3:
            logger.warning("Invalid JWT format")
            return None

        # Decode payload (second part)
        payload_b64 = parts[1]
        # Add padding if needed
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding

        payload_json = base64.urlsafe_b64decode(payload_b64)
        result: Dict[str, Any] = json.loads(payload_json)
        return result
    except Exception as e:
        logger.error(f"Failed to decode JWT: {e}")
        return None


def get_token_info() -> Optional[Dict[str, Any]]:
    """Get decoded token information from ACCESS_TOKEN env var."""
    token = os.environ.get("ACCESS_TOKEN")
    if not token:
        return None
    return decode_jwt_payload(token)


def format_code_for_speech(code: str) -> str:
    """Format user code for clear speech (e.g., 'ABCD-1234' -> 'A B C D 1 2 3 4')."""
    clean = code.replace("-", "").replace(" ", "")
    return " ".join(clean.upper())


class Authenticate(Tool):
    """OAuth2 Device Code authentication for Goabonga Cloud."""

    name = "authenticate"
    description = (
        "Authenticate the robot with Goabonga Cloud using OAuth2 Device Code. "
        "Call this when the user wants to connect to their Goabonga account. "
        "The robot will announce a code that the user must enter on their phone or computer."
    )
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Start OAuth2 Device Code flow and return message for Reachy to speak."""
        logger.info("=== AUTHENTICATE TOOL CALLED ===")
        logger.info(f"kwargs received: {kwargs}")

        client_id = config.OAUTH2_CLIENT_ID
        logger.info(f"OAUTH2_CLIENT_ID configured: {bool(client_id)}")
        if not client_id:
            return {
                "status": "error",
                "message": "OAuth2 client ID is not configured. Please set OAUTH2_CLIENT_ID.",
            }

        # Check if already authenticated
        existing_token = os.environ.get("ACCESS_TOKEN")
        logger.info(f"Existing ACCESS_TOKEN: {bool(existing_token)}")
        if existing_token:
            logger.info("Already authenticated, returning early")
            return {
                "status": "already_authenticated",
                "message": "I'm already connected to your Goabonga account.",
            }

        issuer = getattr(config, "OAUTH2_ISSUER", "https://auth.goabonga.com")
        device_url = f"{issuer}/oauth2/device"
        token_url = f"{issuer}/oauth2/token"
        logger.info(f"OAuth2 issuer: {issuer}")
        logger.info(f"Device URL: {device_url}")
        logger.info(f"Token URL: {token_url}")

        try:
            # Step 1: Request device code
            logger.info("Requesting device code...")
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    device_url,
                    data={
                        "client_id": client_id,
                        "scope": "openid profile email",
                    },
                )
                logger.info(f"Device code response status: {resp.status_code}")
                logger.info(f"Device code response body: {resp.text}")
                resp.raise_for_status()
                data = resp.json()

            device_code = data["device_code"]
            user_code = data["user_code"]
            verification_uri = data.get("verification_uri", f"{issuer}/device")
            expires_in = data.get("expires_in", 600)
            interval = data.get("interval", 5)

            logger.info(f"Got device_code: {device_code[:10]}...")
            logger.info(f"Got user_code: {user_code}")
            logger.info(f"Verification URI: {verification_uri}")
            logger.info(f"Expires in: {expires_in}s, interval: {interval}s")

            # Format for speech
            spoken_code = format_code_for_speech(user_code)
            spoken_url = verification_uri.replace("https://", "").replace("/", " slash ")
            logger.info(f"Spoken code: {spoken_code}")
            logger.info(f"Spoken URL: {spoken_url}")

            # Return message for Reachy to speak
            speech_message = (
                f"To authenticate me, go to {spoken_url} "
                f"and enter the code: {spoken_code}. "
                f"I'll wait for you to complete the authorization."
            )
            logger.info(f"Speech message: {speech_message}")

            # Start background polling task
            logger.info("Starting background polling task...")
            global _polling_task, _polling_cancelled
            _polling_cancelled = False
            _polling_task = asyncio.create_task(
                self._poll_for_token(token_url, client_id, device_code, expires_in, interval)
            )

            result = {
                "status": "pending",
                "message": speech_message,
                "user_code": user_code,
                "verification_uri": verification_uri,
            }
            logger.info(f"Returning result: {result}")
            return result

        except httpx.HTTPError as e:
            logger.error(f"Device code request failed: {e}")
            response = getattr(e, "response", None)
            if response is not None:
                logger.error(f"Response status: {response.status_code}")
                logger.error(f"Response body: {response.text}")
            return {
                "status": "error",
                "message": "I couldn't start the authentication process. Please try again later.",
            }
        except Exception as e:
            logger.exception(f"Unexpected error in authenticate tool: {e}")
            return {
                "status": "error",
                "message": f"An unexpected error occurred: {e}",
            }

    async def _poll_for_token(
        self,
        token_url: str,
        client_id: str,
        device_code: str,
        expires_in: int,
        interval: int,
    ) -> None:
        """Poll for token in background and store when complete."""
        global _polling_cancelled
        deadline = asyncio.get_event_loop().time() + expires_in

        while asyncio.get_event_loop().time() < deadline:
            # Check if cancelled
            if _polling_cancelled:
                logger.info("Polling cancelled by user")
                return

            await asyncio.sleep(interval)

            # Check again after sleep
            if _polling_cancelled:
                logger.info("Polling cancelled by user")
                return

            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        token_url,
                        data={
                            "client_id": client_id,
                            "device_code": device_code,
                            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                        },
                    )

                # Success - token received
                if resp.status_code == 200:
                    data = resp.json()
                    access_token = data.get("access_token")
                    if access_token:
                        os.environ["ACCESS_TOKEN"] = access_token
                        logger.info("Successfully authenticated with Goabonga Cloud")
                        return

                # Handle OAuth2 error responses
                try:
                    error_data = resp.json()
                    error = error_data.get("error", "")
                except Exception:
                    logger.warning(f"Token polling failed with status {resp.status_code}")
                    continue

                if error == "authorization_pending":
                    logger.debug("Authorization pending, continuing to poll...")
                    continue  # Keep polling
                elif error == "slow_down":
                    interval += 5  # Back off
                    logger.debug(f"Slowing down, new interval: {interval}s")
                    continue
                elif error in ("expired_token", "access_denied"):
                    logger.warning(f"Device code auth failed: {error}")
                    return
                else:
                    continue  # Keep trying instead of giving up

            except Exception as e:
                logger.error(f"Error during token polling: {e}")
                continue

        logger.warning("Device code expired before user completed authorization")


class Whoami(Tool):
    """Get information about the currently authenticated user."""

    name = "whoami"
    description = (
        "Get information about the currently authenticated Goabonga Cloud user. "
        "Call this when the user asks who is logged in, or wants to know their account details. "
        "Returns the user's name, email, and other profile information."
    )
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Return information about the authenticated user from userinfo endpoint."""
        logger.info("=== WHOAMI TOOL CALLED ===")

        access_token = os.environ.get("ACCESS_TOKEN")
        if not access_token:
            logger.info("No access token found")
            return {
                "status": "not_authenticated",
                "message": "I'm not connected to any Goabonga account yet. Would you like me to authenticate?",
            }

        # Call userinfo endpoint
        issuer = getattr(config, "OAUTH2_ISSUER", "https://auth.goabonga.com")
        userinfo_url = f"{issuer}/userinfo"
        logger.info(f"Calling userinfo endpoint: {userinfo_url}")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    userinfo_url,
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                logger.info(f"Userinfo response status: {resp.status_code}")
                logger.info(f"Userinfo response body: {resp.text}")

                if resp.status_code == 401:
                    # Token expired or invalid
                    logger.warning("Access token is invalid or expired")
                    return {
                        "status": "token_expired",
                        "message": "Your session has expired. Would you like me to re-authenticate?",
                    }

                resp.raise_for_status()
                user_info = resp.json()

        except httpx.HTTPError as e:
            logger.error(f"Userinfo request failed: {e}")
            # Fallback to JWT decoding if userinfo fails
            return await self._fallback_to_jwt()

        logger.info(f"User info: {user_info}")

        # Extract user information
        name = user_info.get("name") or user_info.get("preferred_username") or "Unknown"
        email = user_info.get("email", "")
        subject = user_info.get("sub", "")
        picture = user_info.get("picture", "")

        # Build a human-readable message
        parts = []
        if name and name != "Unknown":
            parts.append(f"You are logged in as {name}")
        if email:
            parts.append(f"your email is {email}")

        if parts:
            message = ", ".join(parts) + "."
        else:
            message = f"You are authenticated with user ID {subject}."

        # Check token expiration from JWT
        token_info = get_token_info()
        if token_info:
            exp = token_info.get("exp")
            if exp:
                import time
                remaining = exp - time.time()
                if remaining > 0:
                    minutes = int(remaining / 60)
                    if minutes > 60:
                        hours = minutes // 60
                        message += f" Your session is valid for about {hours} more hours."
                    elif minutes > 0:
                        message += f" Your session is valid for about {minutes} more minutes."
                    else:
                        message += " Your session is about to expire."
                else:
                    message += " Your session has expired, you may need to re-authenticate."

        return {
            "status": "authenticated",
            "message": message,
            "user": {
                "name": name,
                "email": email,
                "subject": subject,
                "picture": picture,
            },
            "userinfo": user_info,
        }

    async def _fallback_to_jwt(self) -> Dict[str, Any]:
        """Fallback to JWT decoding if userinfo endpoint fails."""
        logger.info("Falling back to JWT decoding")
        token_info = get_token_info()

        if not token_info:
            return {
                "status": "error",
                "message": "I couldn't retrieve your account information. Please try again.",
            }

        name = token_info.get("name") or token_info.get("preferred_username") or "Unknown"
        email = token_info.get("email", "")
        subject = token_info.get("sub", "")

        parts = []
        if name and name != "Unknown":
            parts.append(f"You are logged in as {name}")
        if email:
            parts.append(f"your email is {email}")

        if parts:
            message = ", ".join(parts) + "."
        else:
            message = f"You are authenticated with user ID {subject}."

        return {
            "status": "authenticated",
            "message": message,
            "user": {
                "name": name,
                "email": email,
                "subject": subject,
            },
            "token_claims": token_info,
        }


class CancelAuthentication(Tool):
    """Cancel an ongoing authentication process."""

    name = "cancel_authentication"
    description = (
        "Cancel an ongoing OAuth2 Device Code authentication process. "
        "Call this when the user wants to stop or cancel the authentication, "
        "or says things like 'never mind', 'cancel', 'stop the authentication'."
    )
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Cancel any active authentication polling."""
        logger.info("=== CANCEL_AUTHENTICATION TOOL CALLED ===")

        if is_polling_active():
            cancel_polling()
            return {
                "status": "cancelled",
                "message": "OK, I've cancelled the authentication process.",
            }
        else:
            return {
                "status": "no_active_auth",
                "message": "There's no authentication in progress to cancel.",
            }


class Logout(Tool):
    """Revoke the current access token and log out."""

    name = "logout"
    description = (
        "Log out from Goabonga Cloud by revoking the current access token. "
        "Call this when the user wants to disconnect, log out, or sign out. "
        "After logout, the robot will need to re-authenticate to access cloud services."
    )
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Revoke the access token and clear local state."""
        logger.info("=== LOGOUT TOOL CALLED ===")

        access_token = os.environ.get("ACCESS_TOKEN")
        if not access_token:
            logger.info("No access token to revoke")
            return {
                "status": "not_authenticated",
                "message": "I'm not connected to any account, so there's nothing to log out from.",
            }

        # Call revocation endpoint
        issuer = getattr(config, "OAUTH2_ISSUER", "https://auth.goabonga.com")
        revoke_url = f"{issuer}/oauth2/revoke"
        client_id = config.OAUTH2_CLIENT_ID

        logger.info(f"Revoking token at: {revoke_url}")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    revoke_url,
                    data={
                        "token": access_token,
                        "token_type_hint": "access_token",
                        "client_id": client_id,
                    },
                )
                logger.info(f"Revoke response status: {resp.status_code}")

                # RFC 7009: revocation endpoint returns 200 even if token was already invalid
                # So we just clear local state regardless
                if resp.status_code in (200, 204):
                    logger.info("Token revoked successfully")
                else:
                    logger.warning(f"Revoke returned unexpected status: {resp.status_code}")

        except httpx.HTTPError as e:
            logger.error(f"Revoke request failed: {e}")
            # Still clear local state even if revocation fails

        # Clear local token
        os.environ.pop("ACCESS_TOKEN", None)
        logger.info("Local token cleared")

        return {
            "status": "logged_out",
            "message": "OK, I've logged out from your Goabonga account. You'll need to re-authenticate to use cloud services.",
        }


class CheckToken(Tool):
    """Introspect the current access token to check its validity."""

    name = "check_token"
    description = (
        "Check if the current access token is still valid. "
        "Call this when the user asks about the token status, validity, or wants to verify their session. "
        "Returns detailed information about the token's active status and remaining validity."
    )
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Introspect the access token to check validity."""
        logger.info("=== CHECK_TOKEN TOOL CALLED ===")

        access_token = os.environ.get("ACCESS_TOKEN")
        if not access_token:
            logger.info("No access token to check")
            return {
                "status": "not_authenticated",
                "message": "I'm not connected to any account. There's no token to check.",
            }

        # Call introspection endpoint
        issuer = getattr(config, "OAUTH2_ISSUER", "https://auth.goabonga.com")
        introspect_url = f"{issuer}/oauth2/introspect"
        client_id = config.OAUTH2_CLIENT_ID

        logger.info(f"Introspecting token at: {introspect_url}")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    introspect_url,
                    data={
                        "token": access_token,
                        "token_type_hint": "access_token",
                        "client_id": client_id,
                    },
                )
                logger.info(f"Introspect response status: {resp.status_code}")
                logger.info(f"Introspect response body: {resp.text}")

                resp.raise_for_status()
                introspection = resp.json()

        except httpx.HTTPError as e:
            logger.error(f"Introspect request failed: {e}")
            return {
                "status": "error",
                "message": "I couldn't verify the token status. Please try again.",
            }

        logger.info(f"Introspection result: {introspection}")

        is_active = introspection.get("active", False)

        if not is_active:
            # Token is not valid - clear local state
            os.environ.pop("ACCESS_TOKEN", None)
            logger.info("Token is inactive, cleared local state")
            return {
                "status": "token_invalid",
                "message": "Your session is no longer valid. Would you like me to re-authenticate?",
                "active": False,
            }

        # Token is active - extract info
        username = introspection.get("username", "")
        scope = introspection.get("scope", "")
        exp = introspection.get("exp")
        client_id_from_token = introspection.get("client_id", "")

        # Build message
        message = "Your session is active and valid."

        if exp:
            import time
            remaining = exp - time.time()
            if remaining > 0:
                minutes = int(remaining / 60)
                if minutes > 60:
                    hours = minutes // 60
                    message += f" It will expire in about {hours} hours."
                elif minutes > 0:
                    message += f" It will expire in about {minutes} minutes."
                else:
                    message += " It will expire very soon."

        return {
            "status": "token_valid",
            "message": message,
            "active": True,
            "introspection": {
                "username": username,
                "scope": scope,
                "exp": exp,
                "client_id": client_id_from_token,
            },
        }
