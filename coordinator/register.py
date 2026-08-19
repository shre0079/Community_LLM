import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class WorkerStatus(str, Enum):
    REGISTERING = "registering"
    DOWNLOADING = "downloading"
    LOADING = "loading"
    ACTIVE = "active"
    DEAD = "dead"


@dataclass
class WorkerInfo:
    worker_id: str
    hostname: str
    ip: str
    http_port: int
    zmq_in_port: int
    gpu_name: str
    vram_gb: float
    layer_start: Optional[int] = None
    layer_end: Optional[int] = None
    is_first: bool = False
    is_last: bool = False
    status: WorkerStatus = WorkerStatus.REGISTERING
    last_heartbeat: float = field(default_factory=time.time)


class WorkerRegistry:
    def __init__(self):
        self._workers: Dict[str, WorkerInfo] = {}

    def add(self, **kwargs) -> WorkerInfo:
        wid = str(uuid.uuid4())
        w = WorkerInfo(worker_id=wid, **kwargs)
        self._workers[wid] = w
        return w

    def get(self, wid: str) -> Optional[WorkerInfo]:
        return self._workers.get(wid)

    def all(self) -> List[WorkerInfo]:
        return list(self._workers.values())

    def active(self) -> List[WorkerInfo]:
        return [w for w in self._workers.values() if w.status == WorkerStatus.ACTIVE]

    def heartbeat(self, wid: str, status: WorkerStatus):
        if wid in self._workers:
            self._workers[wid].last_heartbeat = time.time()
            self._workers[wid].status = status

    def mark_dead(self, wid: str):
        if wid in self._workers:
            self._workers[wid].status = WorkerStatus.DEAD

    def remove(self, wid: str):
        self._workers.pop(wid, None)

    def pipeline(self) -> List[WorkerInfo]:
        """Active workers sorted by layer_start."""
        workers = [w for w in self.active() if w.layer_start is not None]
        return sorted(workers, key=lambda w: w.layer_start)

    def pipeline_ready(self) -> bool:
        from shared.config import TOTAL_LAYERS
        covered = set()
        for w in self.pipeline():
            for i in range(w.layer_start, w.layer_end + 1):
                covered.add(i)
        return len(covered) == TOTAL_LAYERS