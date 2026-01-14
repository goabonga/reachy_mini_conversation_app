"""OAuth2 Device Code authentication tool for Goabonga Cloud."""

import asyncio
import logging
import os
from typing import Any, Dict

import httpx

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)


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
        client_id = config.OAUTH2_CLIENT_ID
        if not client_id:
            return {
                "status": "error",
                "message": "OAuth2 client ID is not configured. Please set OAUTH2_CLIENT_ID.",
            }

        # Check if already authenticated
        if os.environ.get("ACCESS_TOKEN"):
            return {
                "status": "already_authenticated",
                "message": "I'm already connected to your Goabonga account.",
            }

        issuer = getattr(config, "OAUTH2_ISSUER", "https://auth.goabonga.com")
        device_url = f"{issuer}/oauth2/device"
        token_url = f"{issuer}/oauth2/token"

        try:
            # Step 1: Request device code
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    device_url,
                    data={
                        "client_id": client_id,
                        "scope": "openid profile email",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            device_code = data["device_code"]
            user_code = data["user_code"]
            verification_uri = data.get("verification_uri", f"{issuer}/device")
            expires_in = data.get("expires_in", 600)
            interval = data.get("interval", 5)

            # Format for speech
            spoken_code = format_code_for_speech(user_code)
            spoken_url = verification_uri.replace("https://", "").replace("/", " slash ")

            # Return message for Reachy to speak
            speech_message = (
                f"To authenticate me, go to {spoken_url} "
                f"and enter the code: {spoken_code}. "
                f"I'll wait for you to complete the authorization."
            )

            # Start background polling task
            asyncio.create_task(
                self._poll_for_token(token_url, client_id, device_code, expires_in, interval)
            )

            return {
                "status": "pending",
                "message": speech_message,
                "user_code": user_code,
                "verification_uri": verification_uri,
            }

        except httpx.HTTPError as e:
            logger.error(f"Device code request failed: {e}")
            return {
                "status": "error",
                "message": "I couldn't start the authentication process. Please try again later.",
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
        deadline = asyncio.get_event_loop().time() + expires_in

        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(interval)

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

                if resp.status_code == 200:
                    data = resp.json()
                    access_token = data.get("access_token")
                    if access_token:
                        os.environ["ACCESS_TOKEN"] = access_token
                        logger.info("Successfully authenticated with Goabonga Cloud")
                        return

                # Handle pending/errors
                error_data = resp.json() if resp.status_code == 400 else {}
                error = error_data.get("error", "")

                if error == "authorization_pending":
                    continue  # Keep polling
                elif error == "slow_down":
                    interval += 5  # Back off
                    continue
                elif error in ("expired_token", "access_denied"):
                    logger.warning(f"Device code auth failed: {error}")
                    return
                else:
                    logger.warning(f"Unexpected error during polling: {error}")
                    return

            except Exception as e:
                logger.error(f"Error during token polling: {e}")
                continue

        logger.warning("Device code expired before user completed authorization")
