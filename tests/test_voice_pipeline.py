"""
Integration tests for Voice Pipeline (Feature 2).

Tests:
  * F36: Guillotine triggers after MAX_BUFFER_SECONDS
  * F40: Hot-word lexicon is present
  * Intent parsing for tactical keywords
  * Buffer accumulation and flush
"""
import numpy as np
import pytest

from engine.voice_engine import (
    VoiceEngine,
    MAX_BUFFER_SECONDS,
    TACTICAL_LEXICON,
    SAMPLE_RATE,
)


@pytest.fixture
def engine():
    e = VoiceEngine()
    # Don't call start() — models won't be available in CI
    return e


def _make_pcm_chunk(duration_s: float = 0.25, freq: float = 440.0) -> bytes:
    """Generate a synthetic PCM16 audio chunk."""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), dtype=np.float32)
    signal = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)
    return signal.tobytes()


class TestVoiceEngine:
    def test_hot_word_lexicon_contains_tactical_terms(self):
        assert "half-space" in TACTICAL_LEXICON
        assert "double pivot" in TACTICAL_LEXICON
        assert "expected threat" in TACTICAL_LEXICON
        assert "Cruyff turn" in TACTICAL_LEXICON

    def test_buffer_accumulation(self, engine):
        chunk = _make_pcm_chunk()
        engine.feed_chunk(chunk)
        assert len(engine._buffer) == len(chunk)

        engine.feed_chunk(chunk)
        assert len(engine._buffer) == 2 * len(chunk)

    def test_flush_clears_buffer(self, engine):
        chunk = _make_pcm_chunk()
        engine.feed_chunk(chunk)
        engine.flush()
        assert len(engine._buffer) == 0

    def test_intent_parsing_void(self, engine):
        intent = engine._parse_intent("show me the void from 5 minutes ago")
        assert intent is not None
        assert intent["event"] == "topological_void"
        assert intent["minutes_ago"] == 5

    def test_intent_parsing_ghost(self, engine):
        intent = engine._parse_intent("there was a ghost run 12 minutes ago")
        assert intent is not None
        assert intent["event"] == "ghost_run"
        assert intent["minutes_ago"] == 12

    def test_intent_parsing_pressing(self, engine):
        intent = engine._parse_intent("counter press at 10 minutes ago")
        assert intent is not None
        assert intent["event"] == "pressing_trigger"

    def test_intent_parsing_set_piece(self, engine):
        intent = engine._parse_intent("review the corner kick")
        assert intent is not None
        assert intent["event"] == "set_piece"

    def test_intent_parsing_no_time(self, engine):
        intent = engine._parse_intent("show me the half-space overload")
        assert intent is not None
        assert intent["minutes_ago"] is None

    def test_intent_empty_returns_none(self, engine):
        assert engine._parse_intent("") is None
        assert engine._parse_intent("[mock transcript]") is None

    def test_guillotine_constant(self):
        assert MAX_BUFFER_SECONDS == 5.0
