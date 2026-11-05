"""Print every message that passes through the Wizard-of-Oz server.

A small debugging tool: it connects to the server as a dashboard client
and shows each message as it arrives, so you can check that the robot's
transcriptions reach the server and that the replies come back.

Run it while the server (node wizardOfOz/index.js) is running:
    python watch_messages.py
Press Ctrl+C to stop.
"""

import asyncio
import json

import websockets

SERVER_URL = "ws://localhost:8080/?dashboard=true"


async def main() -> None:
    print("Connecting...")
    async with websockets.connect(SERVER_URL) as ws:
        print("Connected. Speak to the robot (or type in the dashboard).\n")
        async for raw in ws:
            print("RAW:", raw)
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                continue  # not JSON; the raw line above is enough
            print(f"   direction = {message.get('direction')}"
                  f" | text = {message.get('text')}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")