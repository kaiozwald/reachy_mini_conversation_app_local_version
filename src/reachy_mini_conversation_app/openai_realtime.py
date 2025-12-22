"""Local realtime audio handler for Reachy Mini conversation app.

This module provides a fully local audio pipeline using:
- Local VAD (Voice Activity Detection)
- Local ASR (Automatic Speech Recognition)
- Local LLM (Language Model)
- Local TTS (Text-to-Speech via Chatterbox)
"""

import base64
import asyncio
import logging
from typing import Any, Final, Tuple, Literal, Optional
from pathlib import Path
from datetime import datetime

import aiohttp
import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample
from gradio_client import Client as GradioClient

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import get_session_instructions
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies

# Use OpenAI-compatible client for local LLM (vLLM)
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore


logger = logging.getLogger(__name__)

INPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000
OUTPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000


class LocalRealtimeHandler(AsyncStreamHandler):
    """A fully local realtime handler for fastrtc Stream.

    Uses local endpoints for VAD, ASR, LLM, and TTS instead of cloud services.
    """

    def __init__(self, deps: "ToolDependencies", gradio_mode: bool = False, instance_path: Optional[str] = None):
        """Initialize the handler."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=OUTPUT_SAMPLE_RATE,
            input_sample_rate=INPUT_SAMPLE_RATE,
        )

        # Override typing of the sample rates
        self.output_sample_rate: Literal[24000] = self.output_sample_rate
        self.input_sample_rate: Literal[24000] = self.input_sample_rate

        self.deps = deps

        self.output_sample_rate = OUTPUT_SAMPLE_RATE
        self.input_sample_rate = INPUT_SAMPLE_RATE

        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()

        self.start_time = asyncio.get_event_loop().time()
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        # Internal lifecycle flags
        self._shutdown_requested: bool = False
        self._connected_event: asyncio.Event = asyncio.Event()

        # Chatterbox TTS (Gradio endpoint)
        self._chatterbox_client: GradioClient | None = None
        self._chatterbox_ref_audio: str | None = None
        if config.CHATTERBOX_ENDPOINT:
            # Reference audio for voice cloning
            if config.CHATTERBOX_REF_AUDIO:
                self._chatterbox_ref_audio = config.CHATTERBOX_REF_AUDIO
            else:
                # Use alfred voice file from project root
                project_dir = Path(__file__).parent.parent.parent
                alfred_path = project_dir / "alfred_1_isolated.wav"
                if alfred_path.exists():
                    self._chatterbox_ref_audio = str(alfred_path)
                    logger.info("Using alfred voice for TTS: %s", self._chatterbox_ref_audio)
            try:
                self._chatterbox_client = GradioClient(config.CHATTERBOX_ENDPOINT)
                logger.info("Chatterbox TTS client initialized at %s", config.CHATTERBOX_ENDPOINT)
            except Exception as e:
                logger.error("Failed to initialize Chatterbox TTS client: %s", e)
                self._chatterbox_client = None

        # Local LLM client (if configured) - uses OpenAI-compatible API (vLLM)
        self._local_llm_client: AsyncOpenAI | None = None
        self._local_llm_model: str = config.LOCAL_LLM_MODEL or "Qwen3-30B"
        self._conversation_history: list[dict[str, Any]] = []
        self._pending_response_id: str | None = None  # Track OpenAI response to cancel
        if config.LOCAL_LLM_ENDPOINT:
            try:
                self._local_llm_client = AsyncOpenAI(
                    base_url=config.LOCAL_LLM_ENDPOINT,
                    api_key="not-needed",  # vLLM doesn't require API key
                )
                logger.info("Local LLM client initialized at %s with model %s",
                           config.LOCAL_LLM_ENDPOINT, self._local_llm_model)
            except Exception as e:
                logger.error("Failed to initialize local LLM client: %s", e)
                self._local_llm_client = None

        # Local ASR client (if configured) - uses Gradio API (GLM-ASR-Nano)
        self._local_asr_client: GradioClient | None = None
        self._audio_buffer: list[bytes] = []  # Buffer for audio during speech
        self._is_speech_active: bool = False
        if config.LOCAL_ASR_ENDPOINT:
            try:
                self._local_asr_client = GradioClient(config.LOCAL_ASR_ENDPOINT)
                logger.info("Local ASR client initialized at %s", config.LOCAL_ASR_ENDPOINT)
            except Exception as e:
                logger.error("Failed to initialize local ASR client: %s", e)
                self._local_asr_client = None

        # Local VAD endpoint (if configured) - Flask API for smart turn detection
        self._local_vad_endpoint: str | None = config.LOCAL_VAD_ENDPOINT
        # Local VAD state for energy-based speech detection
        self._vad_energy_threshold: float = 0.01  # RMS threshold for speech detection
        self._vad_silence_frames: int = 0  # Count of consecutive silent frames
        self._vad_silence_threshold: int = 15  # Frames of silence before checking turn completion (~0.5s at 30fps)
        self._vad_min_speech_frames: int = 5  # Minimum frames of speech before considering it valid
        self._vad_speech_frames: int = 0  # Count of speech frames
        self._vad_processing: bool = False  # Prevent concurrent processing
        if self._local_vad_endpoint:
            logger.info("Local VAD enabled at %s (with energy-based speech detection)", self._local_vad_endpoint)

        # Validate that all required local endpoints are configured
        if not self._is_ready:
            logger.warning(
                "Not all local endpoints configured. Required: LOCAL_VAD_ENDPOINT, "
                "LOCAL_ASR_ENDPOINT, LOCAL_LLM_ENDPOINT, CHATTERBOX_ENDPOINT"
            )
        else:
            logger.info("All local endpoints configured - ready for local-only operation")

    @property
    def _is_ready(self) -> bool:
        """Check if all required local endpoints are configured."""
        return bool(
            self._local_vad_endpoint
            and self._local_asr_client
            and self._local_llm_client
            and self._chatterbox_client
        )

    def copy(self) -> "LocalRealtimeHandler":
        """Create a copy of the handler."""
        return LocalRealtimeHandler(self.deps, self.gradio_mode, self.instance_path)

    def _split_into_chunks(self, text: str, max_chars: int = 150) -> list[str]:
        """Split text into optimal chunks for TTS streaming.

        Uses a waterfall approach inspired by mlx-audio:
        1. First try to split at sentence boundaries (.!?…)
        2. Then try clause boundaries (:;)
        3. Then try phrase boundaries (,—)
        4. Finally fall back to space boundaries

        Args:
            text: The text to split.
            max_chars: Maximum characters per chunk (default 250 for fast response).

        Returns:
            List of text chunks to synthesize separately.
        """
        import re

        text = text.strip()
        if not text:
            return []

        # If text is short enough, return as-is
        if len(text) <= max_chars:
            return [text]

        chunks = []
        remaining = text

        # Waterfall punctuation priorities (strongest to weakest break points)
        waterfall = [
            r'([.!?…]+[\"\'\)]?\s+)',  # Sentence endings (with optional quotes/parens)
            r'([:;]\s+)',               # Clause separators
            r'([,—]\s+)',               # Phrase separators
            r'(\s+)',                   # Any whitespace (last resort)
        ]

        while remaining:
            if len(remaining) <= max_chars:
                chunks.append(remaining.strip())
                break

            # Try each punctuation level to find a good break point
            best_break = None
            for pattern in waterfall:
                # Find all matches within the max_chars window
                matches = list(re.finditer(pattern, remaining[:max_chars + 50]))
                if matches:
                    # Take the last match that's within or close to max_chars
                    for match in reversed(matches):
                        if match.end() <= max_chars + 20:  # Allow slight overflow for natural breaks
                            best_break = match.end()
                            break
                    if best_break:
                        break

            if best_break and best_break > 20:  # Don't create tiny chunks
                chunk = remaining[:best_break].strip()
                remaining = remaining[best_break:].strip()
            else:
                # No good break point found, force break at max_chars
                # Try to at least break at a space
                space_idx = remaining[:max_chars].rfind(' ')
                if space_idx > 20:
                    chunk = remaining[:space_idx].strip()
                    remaining = remaining[space_idx:].strip()
                else:
                    chunk = remaining[:max_chars].strip()
                    remaining = remaining[max_chars:].strip()

            if chunk:
                chunks.append(chunk)

        return chunks

    async def _synthesize_with_chatterbox(self, text: str) -> None:
        """Synthesize text using Chatterbox TTS and stream audio for immediate playback.

        Uses waterfall chunking to split text at natural boundaries for faster
        time-to-first-audio while maintaining natural speech flow.

        Args:
            text: The text to synthesize.

        """
        if not self._chatterbox_client:
            logger.warning("Chatterbox client not configured, skipping TTS")
            return

        # Split into optimal chunks using waterfall approach
        chunks = self._split_into_chunks(text)
        logger.info("TTS: input text (%d chars): %s", len(text), text[:100])
        logger.info("TTS: split into %d chunks: %s", len(chunks), [c[:30] + "..." if len(c) > 30 else c for c in chunks])

        for i, chunk in enumerate(chunks):
            logger.info("TTS: synthesizing chunk %d/%d: %s", i + 1, len(chunks), chunk[:50])
            await self._synthesize_sentence(chunk)

    async def _synthesize_sentence(self, text: str) -> None:
        """Synthesize a single sentence using Chatterbox Gradio endpoint.

        Args:
            text: The sentence to synthesize.
        """
        if not self._chatterbox_client:
            return

        try:
            logger.debug("TTS for sentence: %s", text[:50])

            # Run the blocking Gradio client call in a thread pool
            from gradio_client import handle_file

            # Wrap the reference audio file for Gradio
            ref_audio = handle_file(self._chatterbox_ref_audio) if self._chatterbox_ref_audio else None

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._chatterbox_client.predict(
                    text,                    # text
                    ref_audio,               # audio_prompt_path (wrapped)
                    0.8,                     # temperature
                    0,                       # seed_num
                    0.0,                     # min_p
                    0.95,                    # top_p
                    1000,                    # top_k
                    1.2,                     # repetition_penalty
                    True,                    # norm_loudness
                    fn_index=9,
                ),
            )

            # Handle different Gradio return formats
            if isinstance(result, str):
                # It's a file path - read the audio file
                import scipy.io.wavfile as wavfile
                sample_rate, audio_data = wavfile.read(result)
            elif isinstance(result, tuple) and len(result) >= 2:
                sample_rate, audio_data = result[0], result[1]
            elif isinstance(result, np.ndarray):
                sample_rate = 24000
                audio_data = result
            else:
                logger.error("Unexpected Chatterbox result format: %s", type(result))
                return

            # Convert to numpy array if needed
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data)

            # Resample to output sample rate if different
            if sample_rate != self.output_sample_rate:
                num_samples = int(len(audio_data) * self.output_sample_rate / sample_rate)
                audio_data = resample(audio_data, num_samples)

            # Convert to int16
            audio_data = audio_to_int16(audio_data)

            # Feed to head wobbler if available
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.feed(base64.b64encode(audio_data.tobytes()).decode("utf-8"))

            # Queue audio in chunks for smoother playback
            chunk_size = 4800  # 200ms at 24kHz
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                await self.output_queue.put(
                    (self.output_sample_rate, chunk.reshape(1, -1)),
                )

            logger.debug("TTS sentence complete: %s", text[:30])

        except Exception as e:
            logger.error("Chatterbox TTS synthesis failed: %s", e)

    async def _check_turn_complete(self, audio_data: bytes) -> bool:
        """Check if the user's turn is complete using local VAD.

        Args:
            audio_data: Raw PCM audio bytes (16-bit, 24kHz, mono)

        Returns:
            True if turn is complete, False if user might still be speaking
        """
        if not self._local_vad_endpoint:
            return True  # No VAD configured, assume complete

        try:
            import aiohttp
            import tempfile
            import wave

            # Save audio to temp WAV file for the VAD server
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                with wave.open(f, 'wb') as wav:
                    wav.setnchannels(1)  # mono
                    wav.setsampwidth(2)  # 16-bit
                    wav.setframerate(self.input_sample_rate)  # 24kHz
                    wav.writeframes(audio_data)

            # Read the WAV file and encode as base64
            with open(temp_path, 'rb') as f:
                audio_bytes = f.read()

            import base64
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

            # Clean up temp file
            try:
                import os
                os.unlink(temp_path)
            except Exception:
                pass

            # Call VAD endpoint
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._local_vad_endpoint}/predict",
                    json={"audio_base64": audio_b64},
                    timeout=aiohttp.ClientTimeout(total=5.0)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        prediction = result.get("prediction", 1)
                        probability = result.get("probability", 1.0)
                        status = result.get("status", "complete")
                        logger.info("VAD result: %s (probability=%.2f)", status, probability)
                        return prediction == 1  # 1 = complete, 0 = incomplete
                    else:
                        logger.warning("VAD request failed with status %d", resp.status)
                        return True  # Assume complete on error

        except Exception as e:
            logger.error("VAD check failed: %s", e)
            return True  # Assume complete on error

    async def _transcribe_with_local_asr(self, audio_data: bytes) -> str | None:
        """Transcribe audio using local ASR (GLM-ASR-Nano).

        Args:
            audio_data: Raw PCM audio bytes (16-bit, 24kHz, mono)

        Returns:
            Transcribed text or None if failed
        """
        if not self._local_asr_client:
            logger.warning("Local ASR client not available")
            return None

        try:
            import tempfile
            import wave

            # Save audio buffer to temp WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                with wave.open(f, 'wb') as wav:
                    wav.setnchannels(1)  # mono
                    wav.setsampwidth(2)  # 16-bit
                    wav.setframerate(self.input_sample_rate)  # 24kHz
                    wav.writeframes(audio_data)

            logger.debug("Saved audio buffer to %s (%d bytes)", temp_path, len(audio_data))

            # Call local ASR via Gradio
            from gradio_client import handle_file
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._local_asr_client.predict(
                    handle_file(temp_path),  # Wrap file path for Gradio
                    fn_index=0,  # First (and only) function in the Interface
                )
            )

            # Clean up temp file
            try:
                import os
                os.unlink(temp_path)
            except Exception:
                pass

            if isinstance(result, str) and result.strip():
                transcript = result.strip()
                # Filter out error messages
                if transcript.startswith("[") and transcript.endswith("]"):
                    logger.warning("ASR returned placeholder: %s", transcript)
                    return None
                logger.info("Local ASR transcription: %s", transcript)
                return transcript

            logger.warning("Local ASR returned empty result")
            return None

        except Exception as e:
            logger.error("Local ASR transcription failed: %s", e)
            return None

    async def _process_local_asr(self, audio_data: bytes, check_vad: bool = True) -> None:
        """Process audio with local ASR and generate response with local LLM.

        Args:
            audio_data: Raw PCM audio bytes from the speech buffer.
            check_vad: Whether to check VAD first (default True).
        """
        # Check if turn is complete using local VAD
        if check_vad and self._local_vad_endpoint:
            is_complete = await self._check_turn_complete(audio_data)
            if not is_complete:
                logger.info("VAD says turn incomplete - waiting for more speech")
                # Re-enable speech buffering to capture more audio
                self._is_speech_active = True
                # Put the audio back in the buffer
                self._audio_buffer.append(audio_data)
                return

        # Transcribe with local ASR
        transcript = await self._transcribe_with_local_asr(audio_data)
        if not transcript:
            logger.warning("Local ASR returned no transcription")
            return

        # Show transcription in UI
        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))

        # Generate response with local LLM
        if self._local_llm_client:
            await self._generate_local_response(transcript)
        else:
            logger.warning("Local LLM not available, cannot generate response")

    async def _generate_local_response(self, user_message: str) -> None:
        """Generate a response using the local LLM and send to Chatterbox.

        Args:
            user_message: The user's transcribed message.

        """
        if not self._local_llm_client:
            logger.warning("Local LLM client not available")
            return

        try:
            # Add user message to conversation history
            self._conversation_history.append({"role": "user", "content": user_message})

            # Build messages with system prompt
            messages = [
                {"role": "system", "content": get_session_instructions()},
                *self._conversation_history[-20:]  # Keep last 20 messages for context
            ]

            logger.debug("Calling local LLM with %d messages", len(messages))

            # Call local LLM (no tool support - using base instruct model)
            response = await self._local_llm_client.chat.completions.create(
                model=self._local_llm_model,
                messages=messages,
                max_tokens=512,
                temperature=0.7,
            )

            choice = response.choices[0]
            assistant_message = choice.message

            # Get the text content
            text_response = assistant_message.content
            if text_response:
                # Clean up Qwen thinking tags if present
                if "<think>" in text_response:
                    # Remove thinking section
                    import re
                    text_response = re.sub(r'<think>.*?</think>', '', text_response, flags=re.DOTALL).strip()

                logger.info("Local LLM response: %s", text_response[:100])

                # Add to conversation history
                self._conversation_history.append({"role": "assistant", "content": text_response})

                # Show in UI
                await self.output_queue.put(
                    AdditionalOutputs({"role": "assistant", "content": text_response})
                )

                # Synthesize with Chatterbox
                if self._chatterbox_client:
                    await self._synthesize_with_chatterbox(text_response)

        except Exception as e:
            logger.error("Local LLM generation failed: %s", e)
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": f"[error] LLM failed: {e}"})
            )

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a new personality (profile) at runtime.

        Updates the global config's selected profile for subsequent calls.
        The new personality will be used for the next LLM request.

        Returns a short status message for UI feedback.
        """
        try:
            # Update the in-process config value and env
            from reachy_mini_conversation_app.config import config as _config
            from reachy_mini_conversation_app.config import set_custom_profile

            set_custom_profile(profile)
            logger.info(
                "Set custom profile to %r (config=%r)", profile, getattr(_config, "REACHY_MINI_CUSTOM_PROFILE", None)
            )

            try:
                _ = get_session_instructions()  # Validate the profile loads correctly
            except BaseException as e:  # catch SystemExit from prompt loader without crashing
                logger.error("Failed to resolve personality content: %s", e)
                return f"Failed to apply personality: {e}"

            # Clear conversation history to start fresh with new personality
            self._conversation_history.clear()
            logger.info("Applied personality: %s", profile or "built-in default")
            return f"Applied personality: {profile or 'built-in default'}"
        except Exception as e:
            logger.error("Error applying personality '%s': %s", profile, e)
            return f"Failed to apply personality: {e}"

    async def start_up(self) -> None:
        """Start the local audio processing session."""
        if not self._is_ready:
            logger.error(
                "Cannot start: missing required local endpoints. "
                "Configure LOCAL_VAD_ENDPOINT, LOCAL_ASR_ENDPOINT, LOCAL_LLM_ENDPOINT, CHATTERBOX_ENDPOINT"
            )
            # Still run the session to keep the handler alive, but it won't process audio
        await self._run_local_session()

    async def _run_local_session(self) -> None:
        """Run the local audio processing session.

        This handles the entire pipeline locally:
        - Energy-based VAD for speech start detection
        - Smart-turn VAD for turn completion
        - Local ASR (via Gradio endpoint)
        - Local LLM (via vLLM-compatible endpoint)
        - Local TTS (Chatterbox via Gradio)
        """
        logger.info("Local session started - VAD, ASR, LLM, and TTS all running locally")

        # Signal that we're ready to receive audio
        self._connected_event.set()

        # The audio processing happens in receive() which is called by the audio input stream
        # We just need to keep this session alive
        while not self._shutdown_requested:
            await asyncio.sleep(0.1)

        logger.info("Local session ended")

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from the microphone and process it locally.

        Uses energy-based VAD for speech detection and smart-turn VAD for
        turn completion detection.

        Handles both mono and stereo audio formats, converting to the expected
        mono format. Resamples if the input sample rate differs from the expected rate.

        Args:
            frame: A tuple containing (sample_rate, audio_data).
        """
        if not self._is_ready:
            return

        input_sample_rate, audio_frame = frame

        # Reshape if needed
        if audio_frame.ndim == 2:
            # Scipy channels last convention
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            # Multiple channels -> Mono channel
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        # Resample if needed
        if self.input_sample_rate != input_sample_rate:
            audio_frame = resample(audio_frame, int(len(audio_frame) * self.input_sample_rate / input_sample_rate))

        # Cast if needed
        audio_frame = audio_to_int16(audio_frame)

        # Local VAD: energy-based speech detection + smart-turn for completion
        # Calculate RMS energy
        audio_float = audio_frame.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_float ** 2))

        if rms > self._vad_energy_threshold:
            # Speech detected
            self._vad_silence_frames = 0
            self._vad_speech_frames += 1

            if not self._is_speech_active and self._vad_speech_frames >= self._vad_min_speech_frames:
                # Speech started
                self._is_speech_active = True
                self._audio_buffer.clear()
                self.deps.movement_manager.set_listening(True)
                logger.info("Local VAD: speech started")

            if self._is_speech_active:
                self._audio_buffer.append(audio_frame.tobytes())
        else:
            # Silence detected
            if self._is_speech_active:
                self._audio_buffer.append(audio_frame.tobytes())  # Include trailing silence
                self._vad_silence_frames += 1

                if self._vad_silence_frames >= self._vad_silence_threshold and not self._vad_processing:
                    # Enough silence - check with smart-turn VAD
                    self._vad_processing = True
                    audio_data = b''.join(self._audio_buffer)
                    logger.info("Local VAD: silence detected, checking turn completion (%d bytes)", len(audio_data))

                    # Process in background
                    asyncio.create_task(self._process_with_local_vad(audio_data))
            else:
                self._vad_speech_frames = 0  # Reset speech counter during silence

    async def _process_with_local_vad(self, audio_data: bytes) -> None:
        """Process audio with local VAD check then ASR/LLM if turn complete."""
        try:
            is_complete = await self._check_turn_complete(audio_data)

            if is_complete:
                logger.info("Local VAD: turn complete, proceeding to ASR")
                self._is_speech_active = False
                self._vad_speech_frames = 0
                self._audio_buffer.clear()
                self.deps.movement_manager.set_listening(False)

                # Process with ASR and LLM (skip VAD check since we just did it)
                await self._process_local_asr(audio_data, check_vad=False)
            else:
                logger.info("Local VAD: turn incomplete, continuing to listen")
                # Keep listening - don't clear buffer, just reset silence counter
                self._vad_silence_frames = 0
        finally:
            self._vad_processing = False

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio frame to be played by the speaker."""
        # sends to the stream the stuff put in the output queue by the openai event handler
        # This is called periodically by the fastrtc Stream
        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._shutdown_requested = True
        logger.info("Shutting down local realtime handler")

        # Clear any remaining items in the output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def format_timestamp(self) -> str:
        """Format current timestamp with date, time, and elapsed seconds."""
        loop_time = asyncio.get_event_loop().time()  # monotonic
        elapsed_seconds = loop_time - self.start_time
        dt = datetime.now()  # wall-clock
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed_seconds:.1f}s]"

    async def get_available_voices(self) -> list[str]:
        """Return available voices.

        With local TTS (Chatterbox), voice selection is handled via reference
        audio files rather than voice names, so this returns an empty list.
        """
        # Chatterbox uses reference audio instead of voice names
        return []


# Backwards compatibility alias
OpenaiRealtimeHandler = LocalRealtimeHandler
