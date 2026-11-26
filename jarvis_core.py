"""
JARVIS Core Controller.
Coordinates ActivationManager, VoiceEngine, SpeechEngine, AIBrain,
DesktopController, and MemoryManager into a cohesive system.
"""
import logging
import re
import threading
from typing import Callable, Optional

from activation_manager import ActivationManager, AssistantState
from ai_brain import AIBrain
from config.config import get_config
from desktop_control import DesktopController
from memory_manager import MemoryManager
from speech_engine import SpeechEngine
from voice_engine import VoiceEngine

logger = logging.getLogger("Jarvis.Core")


class JarvisCore:
    """Central orchestrator for the JARVIS AI Desktop Assistant."""

    def __init__(
        self,
        on_state_change: Optional[Callable[[AssistantState], None]] = None,
        on_transcript_update: Optional[Callable[[str, str], None]] = None,
        on_audio_level: Optional[Callable[[float], None]] = None,
    ):
        self.config = get_config()
        self.user_title = self.config.get("user_title", "Sir")

        # UI Callbacks
        self.on_state_change_cb = on_state_change
        self.on_transcript_cb = on_transcript_update
        self.on_audio_level_cb = on_audio_level

        # Subsystems
        self.memory = MemoryManager()
        self.desktop = DesktopController(user_title=self.user_title)
        self.brain = AIBrain(memory_manager=self.memory)

        # Speech Engine (TTS)
        self.speech = SpeechEngine(
            on_start=self._handle_speech_start,
            on_finish=self._handle_speech_finish,
        )

        # Voice Engine (STT & Mic)
        self.voice = VoiceEngine(
            on_text_recognized=self.process_command,
            on_audio_level=self._handle_audio_level,
            on_speech_start=self._handle_user_speaking_start,
            on_speech_end=self._handle_user_speaking_end,
        )

        # Activation Manager
        self.activation = ActivationManager(
            on_state_changed=self._handle_state_changed,
            on_activated=self._on_activated_callback,
            on_deactivated=self._on_deactivated_callback,
        )

        self._processing_lock = threading.Lock()

    # --- Properties ---
    @property
    def state(self) -> AssistantState:
        return self.activation.state

    @property
    def is_active(self) -> bool:
        return self.activation.is_active

    # --- Activation / Deactivation Methods ---
    def activate(self) -> None:
        """User clicked ACTIVATE JARVIS or activated via Tray."""
        if self.is_active:
            return
        logger.info("Core: Initiating activation.")
        self.activation.activate()

    def deactivate(self) -> None:
        """User clicked DEACTIVATE JARVIS or deactivated via Tray."""
        if not self.is_active:
            return
        logger.info("Core: Initiating deactivation.")
        self.activation.deactivate()

    def _on_activated_callback(self) -> None:
        """Triggered by ActivationManager during activate()."""
        # Start microphone continuous listening
        self.voice.start_listening()
        # Speak startup greeting
        greeting = f"JARVIS is online, {self.user_title}. I'm listening."
        self._post_transcript("JARVIS", greeting)
        self.speech.speak(greeting)

    def _on_deactivated_callback(self) -> None:
        """Triggered by ActivationManager during deactivate()."""
        # 1. Stop speech playback immediately
        self.speech.stop()
        # 2. Stop microphone capture and release hardware
        self.voice.stop_listening()
        self._post_transcript("SYSTEM", "JARVIS deactivated. Microphone released.")

    # --- State & Audio Routing ---
    def _handle_state_changed(self, new_state: AssistantState) -> None:
        if self.on_state_change_cb:
            try:
                self.on_state_change_cb(new_state)
            except Exception as e:
                logger.error(f"Error notifying UI of state change: {e}")

    def _handle_audio_level(self, level: float) -> None:
        if self.on_audio_level_cb:
            try:
                self.on_audio_level_cb(level)
            except Exception:
                pass

    def _handle_user_speaking_start(self) -> None:
        if self.is_active and self.state != AssistantState.SPEAKING:
            self.activation.set_state(AssistantState.LISTENING)

    def _handle_user_speaking_end(self) -> None:
        if self.is_active and self.state == AssistantState.LISTENING:
            self.activation.set_state(AssistantState.THINKING)

    def _handle_speech_start(self) -> None:
        if self.is_active:
            self.activation.set_state(AssistantState.SPEAKING)

    def _handle_speech_finish(self) -> None:
        """
        Crucial requirement: When speech response finishes,
        automatically return to ACTIVE listening mode!
        """
        if self.is_active:
            self.voice.resume_listening()
            self.activation.on_command_completed()

    # --- Command Processing Pipeline ---
    def process_command(self, text: str) -> None:
        """
        Processes recognized speech or text input.
        Runs asynchronously to prevent blocking the audio/UI loop.
        """
        if not self.is_active or not text.strip():
            return

        worker = threading.Thread(
            target=self._process_command_worker,
            args=(text.strip(),),
            name="JarvisCommandWorker",
            daemon=True,
        )
        worker.start()

    def _process_command_worker(self, text: str) -> None:
        with self._processing_lock:
            if not self.is_active:
                return

            self._post_transcript("USER", text)
            self.activation.set_state(AssistantState.THINKING)
            # Temporarily pause mic so Jarvis doesn't capture room echo
            self.voice.pause_listening()

            clean_lower = text.lower()

            # Check for API key registration in chat
            key_match = re.search(r"\b(?:gemini|api)?\s*key\s*(?:is|to|=)?\s*([A-Za-z0-9_-]{35,60})\b", text, re.IGNORECASE)
            if not key_match:
                key_match = re.search(r"\b(AIzaSy[A-Za-z0-9_-]{33})\b", text)

            if key_match:
                new_key = key_match.group(1).strip()
                self.config.set("gemini_api_key", new_key)
                self.config.set("llm_provider", "gemini")
                ack = f"Google Gemini API key registered successfully, {self.user_title}. Full Gemini neural reasoning and world knowledge are now active."
                self._post_transcript("JARVIS", ack)
                self.speech.speak(ack)
                return

            # Check explicit deactivation voice command
            if re.search(r"\b(deactivate jarvis|deactivate|go to sleep|shut down jarvis|power down|goodbye jarvis)\b", clean_lower):
                farewell = f"Deactivating systems now, {self.user_title}. Have a pleasant day."
                self._post_transcript("JARVIS", farewell)
                self.speech.speak(farewell, block=True)
                self.deactivate()
                return

            # Check Desktop Automation Intents first
            response_text = self.desktop.handle_command(text)

            # If not an automation command, pass to AI Brain
            if response_text is None:
                try:
                    response_text = self.brain.generate_response(text)
                except Exception as e:
                    logger.error(f"AI Brain error: {e}")
                    response_text = f"My apologies {self.user_title}, I encountered an issue processing that query."

            self._post_transcript("JARVIS", response_text)

            if not self.is_active:
                return

            # Speak the response using neural TTS
            self.speech.speak(response_text)

    def _post_transcript(self, sender: str, message: str) -> None:
        if self.on_transcript_cb:
            try:
                self.on_transcript_cb(sender, message)
            except Exception as e:
                logger.error(f"Error updating transcript UI: {e}")
