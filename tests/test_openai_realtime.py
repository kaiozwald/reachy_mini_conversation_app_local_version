"""Tests for the local realtime handler."""

import asyncio
from typing import Any
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

import reachy_mini_conversation_app.openai_realtime as rt_mod
from reachy_mini_conversation_app.openai_realtime import LocalRealtimeHandler, OpenaiRealtimeHandler
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _build_handler(loop: asyncio.AbstractEventLoop) -> LocalRealtimeHandler:
    asyncio.set_event_loop(loop)
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    return LocalRealtimeHandler(deps)


def test_format_timestamp_uses_wall_clock() -> None:
    """Test that format_timestamp uses wall clock time."""
    loop = asyncio.new_event_loop()
    try:
        print("Testing format_timestamp...")
        handler = _build_handler(loop)
        formatted = handler.format_timestamp()
        print(f"Formatted timestamp: {formatted}")
    finally:
        asyncio.set_event_loop(None)
        loop.close()

    # Extract year from "[YYYY-MM-DD ...]"
    year = int(formatted[1:5])
    assert year == datetime.now(timezone.utc).year


def test_backwards_compatibility_alias() -> None:
    """Test that OpenaiRealtimeHandler is an alias for LocalRealtimeHandler."""
    assert OpenaiRealtimeHandler is LocalRealtimeHandler


def test_is_ready_property() -> None:
    """Test _is_ready property returns False when endpoints not configured."""
    loop = asyncio.new_event_loop()
    try:
        handler = _build_handler(loop)
        # Without any endpoints configured, _is_ready should be False
        assert handler._is_ready is False
    finally:
        asyncio.set_event_loop(None)
        loop.close()


@pytest.mark.asyncio
async def test_get_available_voices_returns_empty() -> None:
    """Test get_available_voices returns empty list for local TTS."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = LocalRealtimeHandler(deps)
    voices = await handler.get_available_voices()
    assert voices == []


@pytest.mark.asyncio
async def test_apply_personality() -> None:
    """Test apply_personality updates config and clears history."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = LocalRealtimeHandler(deps)

    # Add some fake conversation history
    handler._conversation_history = [{"role": "user", "content": "test"}]

    # Apply a personality (None = built-in default)
    result = await handler.apply_personality(None)

    # Conversation history should be cleared
    assert handler._conversation_history == []
    assert "built-in default" in result
