"""Backward-compatible launcher for the headless Q-Mobile gateway.

The old development viewer decoded video with OpenCV. Production Arduino
behavior lives in main.py and relays encoded frames without decoding them.
"""

import asyncio
import sys

import main as gateway


if __name__ == "__main__":
    if len(sys.argv) > 1:
        gateway.ARM_PC_IP = sys.argv[1]
    try:
        asyncio.run(gateway.main())
    except KeyboardInterrupt:
        print("\nShutting down.")
