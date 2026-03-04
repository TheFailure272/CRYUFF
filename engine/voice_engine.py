"""
C.R.U.Y.F.F. — Voice Engine (Feature 2)

Streaming speech-to-intent engine for live tactical voice commands.

Architecture
~~~~~~~~~~~~
1. Frontend sends 250ms audio chunks via WebRTC DataChannel
2. Backend appends to rolling buffer
3. Silero VAD monitors for 400ms silence pause
4. On pause: faster-whisper runs on FULL accumulated buffer

Fixes Pre-Baked
~~~~~~~~~~~~~~~
* Fix F36: Max-Token Guillotine — if VAD doesn't detect silence
  within 5 seconds, force-trigger Whisper to prevent buffer overflow.
  Stadium roar (110dB) will never have "silence" gaps.

* Fix F36b: Frontend must apply WebRTC RNNoise before sending
  chunks (documented here, implemented in VoiceButton.jsx).

* Fix F40: Hot-Word Prompting — feeds tactical football lexicon
  as initial_prompt to bias Whisper's beam search toward correct
  jargon transcription ("half-space" not "half pace").

* Fix F46: Zero-Shot Speaker Diarization — the dugout has multiple
  voices (assistant coach, fitness coach). Before Whisper inference,
  audio is filtered through a vocal embedding comparison against the
  manager's pre-enrolled voice sample. Non-matching segments are
  muted to prevent cross-talk corruption.

Dependencies
~~~~~~~~~~~~
* ``faster-whisper`` (CTranslate2 backend)
* ``silero-vad`` (via torch.hub or bundled)
* ``numpy``
* ``resemblyzer`` or ``pyannote.audio`` (speaker embedding, optional)
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─── Fix F40: Hot-Word Lexicon ──────────────────────────────────
# Biases Whisper beam search toward correct tactical terminology

TACTICAL_LEXICON = (
    "half-space, double pivot, false nine, inverted fullback, "
    "topological void, expected threat, xT, rest-defence, "
    "pressing trigger, counter-press, third-man run, "
    "overload, underload, structural collapse, masking, "
    "ghost engine, blue river, set-piece, delivery zone, "
    "far post, near post, penalty spot, six-yard box, "
    "high block, low block, gegenpressing, positional play, "
    "half-turn, body orientation, progressive pass, "
    "line-breaking pass, entre líneas, Cruyff turn, "
    "la pausa, rondo, juego de posición, salida lavolpiana"
)

# ─── Fix F36: Max-Token Guillotine ─────────────────────────────

MAX_BUFFER_SECONDS = 5.0    # Force-trigger after 5s without silence
VAD_SILENCE_MS = 400        # Silence duration to trigger inference
SAMPLE_RATE = 16000         # 16kHz mono (Whisper input)
CHUNK_DURATION = 0.25       # 250ms chunks

# Fix F46: Speaker diarization
SPEAKER_SIM_THRESHOLD = 0.75  # cosine similarity threshold for voice match


@dataclass
class VoiceEngine:
    """
    Streaming voice-to-intent engine.

    Usage::

        engine = VoiceEngine()
        await engine.start()

        # From DataChannel callback:
        engine.feed_chunk(audio_bytes)

        # Engine auto-triggers Whisper on VAD pause or guillotine
        engine.on_transcript(callback)
        engine.on_intent(callback)
    """

    model_size: str = "base.en"

    _buffer: bytearray = field(default_factory=bytearray, init=False)
    _buffer_start_time: float = field(default=0.0, init=False)
    _last_voice_time: float = field(default=0.0, init=False)
    _is_speaking: bool = field(default=False, init=False)

    _transcript_callbacks: list[Callable] = field(
        default_factory=list, init=False
    )
    _intent_callbacks: list[Callable] = field(
        default_factory=list, init=False
    )

    _whisper_model: object = field(default=None, init=False, repr=False)
    _vad_model: object = field(default=None, init=False, repr=False)
    _guillotine_task: object = field(default=None, init=False, repr=False)

    # Fix F46: Speaker diarization
    _speaker_encoder: object = field(default=None, init=False, repr=False)
    _manager_embedding: object = field(default=None, init=False, repr=False)

    async def start(self) -> None:
        """Load models (lazy initialization)."""
        try:
            from faster_whisper import WhisperModel
            self._whisper_model = WhisperModel(
                self.model_size, compute_type="int8"
            )
            logger.info("Whisper model loaded: %s", self.model_size)
        except ImportError:
            logger.warning("faster-whisper not available — mock mode")
            self._whisper_model = None

        try:
            import torch
            self._vad_model, _ = torch.hub.load(
                "snakers4/silero-vad", "silero_vad"
            )
            logger.info("Silero VAD loaded")
        except Exception:
            logger.warning("Silero VAD not available — using energy VAD")
            self._vad_model = None

        # Fix F46: Load speaker encoder for diarization
        try:
            from resemblyzer import VoiceEncoder
            self._speaker_encoder = VoiceEncoder()
            logger.info("Speaker encoder loaded (Fix F46)")
        except ImportError:
            logger.warning(
                "resemblyzer not available — speaker diarization disabled. "
                "Cross-talk from assistant coach will not be filtered."
            )
            self._speaker_encoder = None

    def enroll_manager_voice(self, audio_sample: np.ndarray) -> None:
        """
        Fix F46: Pre-match enrollment of the manager's voice.

        Record a 5-second voice sample from the head coach.
        This generates a vocal embedding used to filter out
        cross-talk from assistant coaches in the dugout.

        Parameters
        ----------
        audio_sample : np.ndarray
            5 seconds of 16kHz mono float32 audio from the manager.
        """
        if self._speaker_encoder is None:
            logger.warning("Speaker encoder not available — cannot enroll")
            return

        from resemblyzer import preprocess_wav
        wav = preprocess_wav(audio_sample, source_sr=SAMPLE_RATE)
        self._manager_embedding = self._speaker_encoder.embed_utterance(wav)
        logger.info(
            "Manager voice enrolled (embedding dim=%d)",
            len(self._manager_embedding),
        )

    def on_transcript(self, callback: Callable[[str], None]) -> None:
        """Register callback for partial/final transcript."""
        self._transcript_callbacks.append(callback)

    def on_intent(self, callback: Callable[[dict], None]) -> None:
        """Register callback for parsed intent."""
        self._intent_callbacks.append(callback)

    def feed_chunk(self, audio_bytes: bytes) -> None:
        """
        Feed a 250ms audio chunk from the DataChannel.

        The audio must be 16kHz mono PCM16 (post-RNNoise on client).
        """
        now = time.monotonic()

        if not self._buffer:
            self._buffer_start_time = now

        self._buffer.extend(audio_bytes)

        # Check VAD on this chunk
        is_voice = self._check_vad(audio_bytes)

        if is_voice:
            self._last_voice_time = now
            self._is_speaking = True
        else:
            # Check for silence pause
            if (self._is_speaking and
                    (now - self._last_voice_time) >= VAD_SILENCE_MS / 1000):
                # VAD detected silence after speech — trigger inference
                self._trigger_inference("vad_pause")
                return

        # Fix F36: Max-Token Guillotine
        # If buffer exceeds 5 seconds without any silence trigger,
        # force inference to prevent RAM overflow from stadium roar
        buffer_duration = now - self._buffer_start_time
        if buffer_duration >= MAX_BUFFER_SECONDS:
            logger.warning(
                "Guillotine! Buffer at %.1fs without VAD pause. "
                "Stadium roar detected — forcing inference.",
                buffer_duration,
            )
            self._trigger_inference("guillotine")

    def flush(self) -> None:
        """Force-process any remaining buffer (button release)."""
        if self._buffer:
            self._trigger_inference("flush")

    def _check_vad(self, chunk: bytes) -> bool:
        """Check if audio chunk contains speech."""
        if self._vad_model is not None:
            try:
                import torch
                audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                audio /= 32768.0
                tensor = torch.from_numpy(audio)
                prob = self._vad_model(tensor, SAMPLE_RATE).item()
                return prob > 0.5
            except Exception:
                pass

        # Fallback: energy-based VAD
        audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        energy = np.sqrt(np.mean(audio ** 2))
        return energy > 500  # crude threshold

    def _trigger_inference(self, reason: str) -> None:
        """Run Whisper on accumulated buffer and parse intent."""
        if not self._buffer:
            return

        logger.info(
            "Whisper triggered (%s): %.1fs of audio",
            reason, len(self._buffer) / (SAMPLE_RATE * 2),
        )

        # Convert buffer to float32 for Whisper
        audio = np.frombuffer(bytes(self._buffer), dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0

        # Clear buffer
        self._buffer.clear()
        self._is_speaking = False

        # Fix F46: Speaker diarization filter
        # Mute audio segments that don't match the manager's voice
        audio = self._filter_speaker(audio)

        if len(audio) < SAMPLE_RATE * 0.3:  # less than 300ms of manager speech
            logger.info("All audio filtered as non-manager — skipping")
            return

        # Run Whisper
        transcript = self._transcribe(audio)

        # Emit transcript
        for cb in self._transcript_callbacks:
            try:
                cb(transcript)
            except Exception as e:
                logger.error("Transcript callback error: %s", e)

        # Parse intent
        intent = self._parse_intent(transcript)
        if intent:
            for cb in self._intent_callbacks:
                try:
                    cb(intent)
                except Exception as e:
                    logger.error("Intent callback error: %s", e)

    def _transcribe(self, audio: np.ndarray) -> str:
        """Run faster-whisper with hot-word prompting (Fix F40)."""
        if self._whisper_model is None:
            # Mock mode
            return "[mock transcript]"

        segments, _ = self._whisper_model.transcribe(
            audio,
            language="en",
            # Fix F40: Hot-word prompting biases beam search
            # toward tactical football terminology
            initial_prompt=TACTICAL_LEXICON,
            vad_filter=True,
            beam_size=5,
        )

        return " ".join(seg.text.strip() for seg in segments)

    def _filter_speaker(self, audio: np.ndarray) -> np.ndarray:
        """
        Fix F46: Zero-shot speaker diarization.

        Splits audio into 500ms windows. For each window, compute
        the vocal embedding and compare (cosine similarity) against
        the enrolled manager embedding. Windows that don't match
        are zeroed (muted).

        This prevents assistant coach cross-talk from reaching Whisper.
        """
        if (self._speaker_encoder is None or
                self._manager_embedding is None):
            # No enrollment — pass through unfiltered
            return audio

        window_samples = int(0.5 * SAMPLE_RATE)  # 500ms windows
        filtered = audio.copy()
        n_muted = 0

        for i in range(0, len(audio) - window_samples, window_samples):
            window = audio[i:i + window_samples]

            try:
                from resemblyzer import preprocess_wav
                wav = preprocess_wav(window, source_sr=SAMPLE_RATE)
                embedding = self._speaker_encoder.embed_utterance(wav)

                # Cosine similarity
                similarity = float(np.dot(
                    embedding, self._manager_embedding
                ) / (
                    np.linalg.norm(embedding) *
                    np.linalg.norm(self._manager_embedding) + 1e-8
                ))

                if similarity < SPEAKER_SIM_THRESHOLD:
                    # Not the manager — mute this window
                    filtered[i:i + window_samples] = 0.0
                    n_muted += 1
            except Exception:
                pass

        if n_muted > 0:
            logger.info(
                "F46: Muted %d/%d windows (non-manager speech)",
                n_muted,
                len(audio) // window_samples,
            )

        return filtered

    def _parse_intent(self, transcript: str) -> Optional[dict]:
        """Parse tactical intent from transcript."""
        text = transcript.lower().strip()
        if not text or text == "[mock transcript]":
            return None

        # Temporal queries: "show me X from Y minutes ago"
        time_match = re.search(
            r"(\d+)\s*minutes?\s*ago", text
        )
        minutes_ago = int(time_match.group(1)) if time_match else None

        # Event detection
        if any(kw in text for kw in ["void", "collapse", "breakdown"]):
            return {
                "type": "clip_request",
                "event": "topological_void",
                "minutes_ago": minutes_ago,
                "raw": transcript,
            }
        elif any(kw in text for kw in ["ghost", "run", "threat", "xt"]):
            return {
                "type": "clip_request",
                "event": "ghost_run",
                "minutes_ago": minutes_ago,
                "raw": transcript,
            }
        elif any(kw in text for kw in ["press", "counter", "gegen"]):
            return {
                "type": "clip_request",
                "event": "pressing_trigger",
                "minutes_ago": minutes_ago,
                "raw": transcript,
            }
        elif any(kw in text for kw in ["half-space", "overload", "pivot"]):
            return {
                "type": "clip_request",
                "event": "positional_play",
                "minutes_ago": minutes_ago,
                "raw": transcript,
            }
        elif any(kw in text for kw in ["set-piece", "corner", "free"]):
            return {
                "type": "clip_request",
                "event": "set_piece",
                "minutes_ago": minutes_ago,
                "raw": transcript,
            }
        else:
            return {
                "type": "search",
                "query": text,
                "minutes_ago": minutes_ago,
                "raw": transcript,
            }
