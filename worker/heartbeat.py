import asyncio
import logging
import httpx
from shared.config import COORDINATOR_URL, HEARTBEAT_INTERVAL

log = logging.getLogger(__name__)


class Heartbeat:
    def __init__(self, worker_id: str, status_fn):
        self.worker_id = worker_id
        self.status_fn = status_fn  # callable returning current status string
        self._running = False

    async def run(self):
        self._running = True
        while self._running:
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"{COORDINATOR_URL}/heartbeat",
                        json={"worker_id": self.worker_id, "status": self.status_fn()},
                        timeout=5.0,
                    )
            except Exception as e:
                log.warning(f"Heartbeat failed: {e}")
            await asyncio.sleep(HEARTBEAT_INTERVAL)

    def stop(self):
        self._running = False