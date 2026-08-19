import asyncio
import logging
import time
from coordinator.registry import WorkerRegistry
from shared.config import HEARTBEAT_TIMEOUT

log = logging.getLogger(__name__)


class HealthMonitor:
    def __init__(self, registry: WorkerRegistry):
        self.registry = registry
        self._running = False

    async def run(self):
        self._running = True
        while self._running:
            now = time.time()
            for w in self.registry.all():
                if w.status in ("active", "loading"):
                    elapsed = now - w.last_heartbeat
                    if elapsed > HEARTBEAT_TIMEOUT:
                        log.warning(f"Worker {w.hostname} ({w.layer_start}-{w.layer_end}) timed out — marking DEAD")
                        self.registry.mark_dead(w.worker_id)
            await asyncio.sleep(5)

    def stop(self):
        self._running = False