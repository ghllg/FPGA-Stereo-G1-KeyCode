"""Keep a slow display from blocking physics or silently losing its subscription."""
import asyncio
from collections import deque


class StreamClient:
    def __init__(self, ws):
        self.ws = ws
        self.messages = deque()
        self.pose = None
        self.wake = asyncio.Event()
        self.task = asyncio.create_task(self.write())

    def offer(self, packet, payload):
        if packet['type'] == 'pose' and not packet.get('finished'):
            self.pose = payload
        else:
            # Session changes and terminal states must never leave an older
            # unsent display pose behind them.
            if packet['type'] in ('ready', 'reset', 'idle', 'error', 'pose'):
                self.pose = None
            self.messages.append(payload)
        self.wake.set()

    async def write(self):
        try:
            while True:
                await self.wake.wait()
                while self.messages or self.pose is not None:
                    if self.messages:
                        payload = self.messages.popleft()
                    else:
                        payload, self.pose = self.pose, None
                    await asyncio.wait_for(self.ws.send_str(payload), 5)
                self.wake.clear()
        except (ConnectionError, asyncio.TimeoutError):
            # A disconnected subscriber must reconnect, rather than continue
            # sending input while receiving no simulation state.
            await self.ws.close(code=1013, message=b'Display connection stalled; reconnect')

    async def close(self):
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
