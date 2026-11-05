"""Turn text into speech and play it on the robot's speaker.

Two voices are available:
    huggingface  Chatterbox on Hugging Face: natural voice, needs internet,
                 and has a free usage limit (set HF_TOKEN for more quota)
    local        the offline Windows voice (pyttsx3): sounds robotic,
                 but has no limits and works without internet

Run this file directly to test speech:
    python text_to_speech.py --text "Hello" --provider local
"""

import argparse
import os
import tempfile
import time
import wave
from contextlib import nullcontext
from functools import lru_cache

from gradio_client import Client, handle_file
from reachy_mini import ReachyMini

from robot import connect

HF_SPACE = "ResembleAI/Chatterbox-Multilingual-TTS"
PROVIDERS = ("huggingface", "local")
LANGUAGES = [
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it",
    "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
]
# Sample voice that Chatterbox imitates (zero-shot voice cloning).
DEFAULT_REF_AUDIO = (
    "https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav"
)
LOCAL_VOICE_RATE = 175  # words per minute for the offline voice


@lru_cache(maxsize=1)
def _hf_client() -> Client:
    """Connect to the Hugging Face voice service once and reuse it."""
    return Client(HF_SPACE, hf_token=os.getenv("HF_TOKEN"))


def synthesize(text: str, lang: str, ref_audio: str) -> str:
    """Create speech with Chatterbox and return the path of the audio file."""
    if not ref_audio.startswith(("http://", "https://")):
        ref_audio = os.path.expanduser(ref_audio)
    audio_path = _hf_client().predict(
        text_input=text,
        language_id=lang,
        audio_prompt_path_input=handle_file(ref_audio),
        api_name="/generate_tts_audio",
    )
    return str(audio_path)


def synthesize_local(text: str) -> str:
    """Create speech with the offline Windows voice and return the WAV path."""
    try:
        import pyttsx3
    except ImportError as exc:
        raise RuntimeError(
            "Offline speech requires pyttsx3. Install it in the active "
            "environment with: python -m pip install pyttsx3"
        ) from exc

    # One file per process: each reply replaces the previous one. Delete the
    # old file first, so the check below cannot pass on stale audio.
    output_path = os.path.join(tempfile.gettempdir(), f"reachy_tts_{os.getpid()}.wav")
    if os.path.exists(output_path):
        os.remove(output_path)

    engine = pyttsx3.init()
    engine.setProperty("rate", LOCAL_VOICE_RATE)
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    if not os.path.isfile(output_path):
        raise RuntimeError("pyttsx3 did not create the requested WAV file.")
    return output_path


def audio_duration_s(audio_path: str, text: str) -> float:
    """Return how long the audio lasts, or estimate it from the word count."""
    try:
        with wave.open(audio_path, "rb") as wav:
            return wav.getnframes() / wav.getframerate()
    except (wave.Error, EOFError, OSError):
        # Not a plain WAV file: assume about 2.5 spoken words per second.
        return max(1.0, len(text.split()) / 2.5 + 0.5)


def speak(
    text: str,
    lang: str,
    ref_audio: str,
    mini: ReachyMini | None = None,
    provider: str = "huggingface",
) -> None:
    """Say *text* on the robot; connects to the robot itself if *mini* is None."""
    if provider == "local":
        print(f"Synthesizing {len(text)} chars with the offline voice...")
        audio_path = synthesize_local(text)
    elif provider == "huggingface":
        print(f"Synthesizing {len(text)} chars ({lang}) with Chatterbox...")
        audio_path = synthesize(text, lang, ref_audio)
    else:
        raise ValueError(f"Unknown provider {provider!r}; use one of {PROVIDERS}.")

    duration = audio_duration_s(audio_path, text)
    print(f"Playing {duration:.1f} s of audio.")

    # Reuse the caller's connection if given; otherwise open (and close) our own.
    with (nullcontext(mini) if mini is not None else connect()) as robot:
        robot.media.play_sound(audio_path)
        time.sleep(duration + 0.5)  # play_sound returns immediately, so wait


def main() -> None:
    """Speak one sentence from the command line, to test the voice."""
    parser = argparse.ArgumentParser(description="Make Reachy Mini say a sentence.")
    parser.add_argument(
        "--text",
        default="Hello, I am Reachy Mini.",
        help="Text to say (Chatterbox accepts up to 300 characters).",
    )
    parser.add_argument(
        "--lang", default="en", choices=LANGUAGES, help="Language code (ISO 639-1)."
    )
    parser.add_argument(
        "--ref-audio",
        default=DEFAULT_REF_AUDIO,
        help="Voice sample for Chatterbox to imitate (URL or local path).",
    )
    parser.add_argument(
        "--provider", default="huggingface", choices=PROVIDERS, help="Which voice to use."
    )
    args = parser.parse_args()
    speak(args.text, args.lang, args.ref_audio, provider=args.provider)


if __name__ == "__main__":
    main()