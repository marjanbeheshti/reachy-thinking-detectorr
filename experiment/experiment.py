"""Robot side of the conversation: listen, send, receive, and speak.

Three tasks run at the same time and share one robot connection:
    listen_and_send   record speech from the robot's microphone, turn it
                      into text, and send it to the server
    receive_messages  collect replies from the server into a queue
    speak_messages    take replies from the queue and say them out loud

While the robot is speaking, the microphone input is ignored, so the
robot does not hear its own voice and send it back as a new question.

Settings (optional, set in the terminal before running):
    REACHY_NODE_WS_URL   server address (default ws://localhost:8080)
    REACHY_TTS_PROVIDER  "huggingface" (default) or "local" (offline voice)
"""

import asyncio
import os

import websockets

from record import record_response, transcribe
from robot import connect
from text_to_speech import DEFAULT_REF_AUDIO, speak

SERVER_URL = os.getenv("REACHY_NODE_WS_URL", "ws://localhost:8080")
TTS_PROVIDER = os.getenv("REACHY_TTS_PROVIDER", "huggingface")


def websocket_text(message: str | bytes) -> str:
    """Return a WebSocket message as text; some libraries deliver bytes."""
    if isinstance(message, bytes):
        return message.decode("utf-8")
    return message


async def main() -> None:
    # Keep the server and robot connections open for the whole conversation.
    async with websockets.connect(SERVER_URL) as ws:
        with connect(media_backend="webrtc") as mini:
            replies: asyncio.Queue[str | None] = asyncio.Queue()
            robot_quiet = asyncio.Event()  # set whenever the robot is not speaking
            robot_quiet.set()
            replies_spoken = 0  # counts finished replies, to detect echo
            input_rate = mini.media.get_input_audio_samplerate()

            async def receive_messages() -> None:
                """Queue every server reply; None tells the speaker to stop."""
                try:
                    async for message in ws:
                        await replies.put(websocket_text(message))
                finally:
                    await replies.put(None)

            async def speak_messages() -> None:
                """Say each queued reply out loud, one at a time."""
                nonlocal replies_spoken
                while (reply := await replies.get()) is not None:
                    robot_quiet.clear()
                    try:
                        # Speech synthesis blocks, so run it in a worker thread.
                        await asyncio.to_thread(
                            speak,
                            text=reply,
                            lang="en",
                            ref_audio=DEFAULT_REF_AUDIO,
                            mini=mini,
                            provider=TTS_PROVIDER,
                        )
                    finally:
                        replies_spoken += 1
                        robot_quiet.set()

            async def listen_and_send() -> None:
                """Record each utterance, transcribe it, and send the text."""
                while True:
                    await robot_quiet.wait()  # never record over the robot's voice
                    spoken_before = replies_spoken

                    try:
                        audio = await asyncio.to_thread(record_response, mini)
                    except RuntimeError as exc:
                        print(exc)
                        await asyncio.sleep(0.1)
                        continue

                    text = await asyncio.to_thread(transcribe, audio, input_rate, "en")

                    # If the robot spoke during this recording, part of it may
                    # be the robot's own voice, so throw the recording away.
                    if not robot_quiet.is_set() or replies_spoken != spoken_before:
                        continue

                    text = text.strip()
                    if not text:
                        print("No words were recognized; listening again.")
                        continue
                    print(f"You said: {text}")
                    await ws.send(text)

            tasks = [
                asyncio.create_task(receive_messages()),
                asyncio.create_task(speak_messages()),
                asyncio.create_task(listen_and_send()),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                # If one task fails or the server closes, stop the others too.
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())