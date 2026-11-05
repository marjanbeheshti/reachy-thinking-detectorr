"""Connect to the Reachy Mini robot over the local network.

The other experiment scripts call connect() instead of creating
ReachyMini themselves, so the robot's address is set in one place.
If your robot has a different IP address, set it before running:
    set REACHY_HOST=192.168.1.42
"""

import os

from reachy_mini import ReachyMini

ROBOT_HOST = os.getenv("REACHY_HOST", "192.168.50.186")
ROBOT_PORT = int(os.getenv("REACHY_PORT", "8000"))


def connect(
    media_backend: str = "default",
    host: str = ROBOT_HOST,
    port: int = ROBOT_PORT,
) -> ReachyMini:
    """Return a connection to the robot; use it in a `with` block."""
    return ReachyMini(
        host=host,
        port=port,
        connection_mode="network",
        media_backend=media_backend,
    )