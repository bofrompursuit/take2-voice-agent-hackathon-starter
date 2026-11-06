"""Text the caller a summary after the call, via Twilio.

LiveKit phone numbers are voice-only, so texts go out from a Twilio number.
Set these in .env.local (and as secrets on the deployed agent):

  TWILIO_ACCOUNT_SID   = ACxxxxxxxx...
  TWILIO_AUTH_TOKEN    = your auth token
  TWILIO_FROM_NUMBER   = +1XXXXXXXXXX   (a Twilio number that can send SMS)

If any are missing, texting is skipped and the summary is only logged.
"""

import logging
import os
import re

import aiohttp
from livekit.agents import llm

logger = logging.getLogger("sms")

# Delivery tags like <expr type="expression" label="happy"/> that expressive TTS reads.
_MARKUP = re.compile(r"<[^>]+>")

SUMMARY_PROMPT = """
You write the text message a career coach sends a caller right after a phone
session. Using the transcript, write under 400 characters of plain text: one
sentence on what you talked about, then the next steps the caller mentioned or
agreed to, each with its timeframe if one came up. Start with "PAT here."
No markdown or emojis. If nothing useful was discussed, reply with only: NONE
"""


async def send_sms(to: str, body: str) -> bool:
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    sender = os.getenv("TWILIO_FROM_NUMBER")
    if not (sid and token and sender):
        logger.info("Twilio not configured; summary not texted: %s", body)
        return False

    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    data = {"To": to, "From": sender, "Body": body}
    async with (
        aiohttp.ClientSession(auth=aiohttp.BasicAuth(sid, token)) as http,
        http.post(url, data=data) as resp,
    ):
        if resp.status >= 300:
            logger.error("SMS failed (%s): %s", resp.status, await resp.text())
            return False
    logger.info("summary texted to caller")
    return True


async def summarize_call(model: llm.LLM, history: llm.ChatContext) -> str | None:
    """Fallback for calls that ended before an action plan was saved."""
    lines = []
    for msg in history.messages():
        if msg.role in ("user", "assistant") and msg.text_content:
            speaker = "Caller" if msg.role == "user" else "Coach"
            lines.append(f"{speaker}: {_MARKUP.sub('', msg.text_content).strip()}")
    if sum(line.startswith("Caller:") for line in lines) < 2:
        return None

    ctx = llm.ChatContext.empty()
    ctx.add_message(role="system", content=SUMMARY_PROMPT.strip())
    ctx.add_message(role="user", content="\n".join(lines))
    text = ""
    async with model.chat(chat_ctx=ctx) as stream:
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                text += chunk.delta.content
    text = _MARKUP.sub("", text).strip()
    return None if not text or text == "NONE" else text
