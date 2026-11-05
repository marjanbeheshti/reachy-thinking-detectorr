"""Record speech from the robot's microphone and turn it into text.

record_response() listens until someone starts talking, keeps recording,
and stops after a short silence. transcribe() turns that recording into
text with faster-whisper.

Run this file directly to test the microphone: it records for 5 seconds
and plays the recording back through the robot's speaker.
"""

import time
from functools import lru_cache

import numpy as np
from reachy_mini import ReachyMini

from robot import connect

TEST_RECORD_SECONDS = 5.0     # length of the microphone test
LISTEN_TIMEOUT_SECONDS = 15.0  # give up if nobody starts talking by then
END_SPEECH_SECONDS = 1.0      # this much silence means the speaker is done
VAD_THRESHOLD = 0.02          # volume (RMS) above this counts as speech
WHISPER_SAMPLE_RATE = 16_000  # faster-whisper expects 16 kHz mono audio
WHISPER_MODEL_SIZE = "base"


def _as_mono(audio: np.ndarray) -> np.ndarray:
    """Convert mono or multi-channel audio to a one-dimensional float array."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        return audio.mean(axis=1).astype(np.float32)
    if audio.ndim == 1:
        return audio
    raise ValueError(f"Unexpected audio shape: {audio.shape}")


def speech_rms(audio: np.ndarray) -> float:
    """Return the root-mean-square volume of one audio chunk."""
    mono = _as_mono(audio)
    if mono.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(mono))))


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Change the sample rate with linear interpolation (no scipy needed)."""
    if source_rate == target_rate:
        return audio
    target_size = round(audio.size * target_rate / source_rate)
    source_positions = np.arange(audio.size, dtype=np.float32)
    target_positions = np.linspace(0, audio.size - 1, target_size, dtype=np.float32)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def record(mini: ReachyMini, seconds: float) -> np.ndarray:
    """Record from the robot's microphone for a fixed number of seconds."""
    mini.media.start_recording()
    print(f"Recording for up to {seconds:.1f} s. Speak into the robot...")

    chunks: list[np.ndarray] = []
    started_at = time.monotonic()
    try:
        while time.monotonic() - started_at < seconds:
            chunk = mini.media.get_audio_sample()
            if chunk is None:
                time.sleep(0.01)  # no new audio yet
                continue
            chunks.append(chunk)
    except KeyboardInterrupt:
        print("\nRecording stopped early.")
    finally:
        mini.media.stop_recording()

    if not chunks:
        raise RuntimeError(
            "No microphone samples received. Is the robot daemon running, and "
            "is the microphone free (close the conversation app if it is open)?"
        )

    audio = np.concatenate(chunks, axis=0).astype(np.float32)
    sample_rate = mini.media.get_input_audio_samplerate()
    print(f"Captured {audio.shape[0] / sample_rate:.2f} s, shape={audio.shape}")
    return audio


def record_response(
    mini: ReachyMini,
    vad_threshold: float = VAD_THRESHOLD,
    timeout_seconds: float = LISTEN_TIMEOUT_SECONDS,
    end_speech_seconds: float = END_SPEECH_SECONDS,
) -> np.ndarray:
    """Wait for speech, record it, and stop after a period of silence."""
    mini.media.start_recording()
    chunks: list[np.ndarray] = []
    speech_started = False
    silent_seconds = 0.0
    started_at = last_time = time.monotonic()

    print("Listening for a response...")
    try:
        while time.monotonic() - started_at < timeout_seconds:
            now = time.monotonic()
            elapsed, last_time = now - last_time, now

            chunk = mini.media.get_audio_sample()
            if chunk is None:
                time.sleep(0.01)  # no new audio yet
                continue

            chunk = _as_mono(chunk)
            chunks.append(chunk)

            if speech_rms(chunk) > vad_threshold:
                if not speech_started:
                    print("Speech detected.")
                speech_started = True
                silent_seconds = 0.0
            elif speech_started:
                silent_seconds += elapsed
                if silent_seconds >= end_speech_seconds:
                    print("End of speech detected.")
                    break
    except KeyboardInterrupt:
        print("\nRecording stopped by the user.")
    finally:
        mini.media.stop_recording()

    if not speech_started or not chunks:
        raise RuntimeError(
            "No speech detected before the timeout. Check the robot microphone."
        )
    return np.concatenate(chunks, axis=0).astype(np.float32)


@lru_cache(maxsize=1)
def _whisper_model():
    """Load the speech-recognition model once and reuse it for every call."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Speech recognition requires faster-whisper. Install it in the "
            "active environment with: python -m pip install faster-whisper"
        ) from exc
    return WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")


def transcribe(
    audio: np.ndarray,
    sample_rate: int = WHISPER_SAMPLE_RATE,
    language: str = "en",
) -> str:
    """Convert robot microphone samples to text with faster-whisper."""
    audio = _resample(audio, sample_rate, WHISPER_SAMPLE_RATE)
    segments, _ = _whisper_model().transcribe(audio, language=language)
    return " ".join(segment.text.strip() for segment in segments).strip()


def play(mini: ReachyMini, audio: np.ndarray) -> None:
    """Play raw audio on the robot's speaker and wait until it finishes."""
    sample_rate = mini.media.get_output_audio_samplerate()
    mini.media.start_playing()
    print("Playing back on the robot speaker...")
    mini.media.push_audio_sample(audio)
    time.sleep(len(audio) / sample_rate)  # pushing is non-blocking, so wait
    mini.media.stop_playing()


def main() -> None:
    """Microphone test: record a few seconds, then play it back."""
    with connect(media_backend="default") as mini:
        audio = record(mini, TEST_RECORD_SECONDS)
        play(mini, audio)
    print("Done.")


if __name__ == "__main__":
    main()