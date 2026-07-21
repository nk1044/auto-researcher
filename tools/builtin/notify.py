"""Built-in Discord notification tool.

Setup (takes ~2 minutes):
  1. Open any Discord channel you want notifications in.
  2. Click "Edit Channel" (gear icon) → Integrations → Webhooks → New Webhook.
  3. Give it a name (e.g. "Auto-Researcher"), click "Copy Webhook URL".
  4. Paste the URL into config.yaml:

       notifications:
         discord_webhook_url: "https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN"

That's all — no bot token, no developer portal, no OAuth.
"""

from __future__ import annotations

import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

from tools.decorator import tool


def _load_webhook_url() -> str:
    """Read the Discord webhook URL from config.yaml (used inside subprocess)."""
    # Walk up from this file to find config.yaml at project root
    root = Path(__file__).resolve().parent.parent.parent
    cfg_path = root / "config.yaml"
    if not cfg_path.exists():
        return ""
    try:
        # Avoid importing yaml in the subprocess — parse the simple line ourselves
        for line in cfg_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("discord_webhook_url:"):
                value = stripped.split(":", 1)[1].strip().strip('"').strip("'")
                if value.startswith("https://discord.com/api/webhooks/"):
                    return value
    except OSError:
        pass
    return ""


def _post_webhook(webhook_url: str, payload: dict) -> tuple[bool, str]:
    """POST JSON payload to a Discord webhook URL."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "auto-researcher/1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return True, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}: {exc.read().decode()[:200]}"
    except Exception as exc:
        return False, str(exc)


@tool(name="send_discord", description="Send a Discord notification via the configured webhook", kind="action")
def send_discord(
    workspace: str,
    message: str,
    title: str = "",
    color: int = 5793266,   # default: blue
) -> dict:
    """Send a message to Discord via the webhook URL in config.yaml.

    Args:
        workspace: auto-injected — do NOT pass.
        message: Main message body (supports Discord markdown).
        title: Optional bold title shown above the message.
        color: Embed sidebar color as decimal RGB (default: blue 0x587FF2).

    Common colors:
        Blue (info):    5793266   (0x587FF2)
        Green (ok):     3978097   (0x3CB371)
        Yellow (warn):  16753920  (0xFF8C00)
        Red (error):    16007990  (0xF44336)

    Returns:
        {"sent": bool, "detail": str}
    """
    webhook_url = _load_webhook_url()
    if not webhook_url:
        return {
            "sent": False,
            "detail": (
                "No Discord webhook URL configured. Add to config.yaml:\n"
                "notifications:\n"
                "  discord_webhook_url: \"https://discord.com/api/webhooks/YOUR_ID/TOKEN\""
            ),
        }

    embed: dict = {"description": message, "color": color}
    if title:
        embed["title"] = title

    ok, detail = _post_webhook(webhook_url, {"embeds": [embed]})
    return {"sent": ok, "detail": detail}
