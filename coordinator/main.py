import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from coordinator.registry import WorkerRegistry, WorkerStatus
from coordinator.scheduler import Scheduler
from coordinator.health_monitor import HealthMonitor

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

registry = WorkerRegistry()
scheduler = Scheduler(registry)
monitor = HealthMonitor(registry)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(monitor.run())
    yield
    monitor.stop()
    task.cancel()


app = FastAPI(title="OLMoE Coordinator", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="coordinator/static"), name="static")


# ── Schemas ──────────────────────────────────────────────────────────────────

class RegisterReq(BaseModel):
    hostname: str
    ip: str
    http_port: int
    zmq_in_port: int
    gpu_name: str
    vram_gb: float

class RegisterResp(BaseModel):
    worker_id: str
    layer_start: int
    layer_end: int
    is_first: bool
    is_last: bool
    model_id: str

class HeartbeatReq(BaseModel):
    worker_id: str
    status: str
    vram_used_gb: Optional[float] = None

class InferReq(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.8


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/register", response_model=RegisterResp)
async def register(req: RegisterReq):
    worker = registry.add(
        hostname=req.hostname,
        ip=req.ip,
        http_port=req.http_port,
        zmq_in_port=req.zmq_in_port,
        gpu_name=req.gpu_name,
        vram_gb=req.vram_gb,
    )
    try:
        start, end, is_first, is_last = scheduler.assign(worker.worker_id)
    except ValueError as e:
        registry.remove(worker.worker_id)
        raise HTTPException(400, str(e))

    log.info(f"Registered {req.hostname} → layers {start}–{end} ({'first' if is_first else ''} {'last' if is_last else ''})")
    return RegisterResp(
        worker_id=worker.worker_id,
        layer_start=start,
        layer_end=end,
        is_first=is_first,
        is_last=is_last,
        model_id="allenai/OLMoE-1B-7B",
    )


@app.post("/heartbeat")
async def heartbeat(req: HeartbeatReq):
    w = registry.get(req.worker_id)
    if not w:
        raise HTTPException(404, "Unknown worker")
    registry.heartbeat(req.worker_id, WorkerStatus(req.status))
    return {"ok": True}


@app.get("/topology")
async def topology():
    """
    Returns full pipeline topology once all 3 workers are active.
    Workers poll this until ready=True.
    """
    pipeline = registry.pipeline()
    ready = registry.pipeline_ready()
    return {
        "ready": ready,
        "workers": [
            {
                "worker_id": w.worker_id,
                "ip": w.ip,
                "http_port": w.http_port,
                "zmq_in_port": w.zmq_in_port,
                "layer_start": w.layer_start,
                "layer_end": w.layer_end,
                "is_first": w.is_first,
                "is_last": w.is_last,
            }
            for w in pipeline
        ],
    }


@app.get("/workers")
async def list_workers():
    now = time.time()
    return {
        "pipeline_ready": registry.pipeline_ready(),
        "workers": [
            {
                "hostname": w.hostname,
                "ip": w.ip,
                "gpu": w.gpu_name,
                "vram_gb": w.vram_gb,
                "layers": f"{w.layer_start}–{w.layer_end}" if w.layer_start is not None else "—",
                "status": w.status,
                "heartbeat_ago_s": round(now - w.last_heartbeat, 1),
            }
            for w in registry.all()
        ],
    }


@app.post("/infer")
async def infer(req: InferReq):
    if not registry.pipeline_ready():
        raise HTTPException(503, "Pipeline not ready — waiting for all workers")

    # Route to first worker
    first = registry.pipeline()[0]
    url = f"http://{first.ip}:{first.http_port}/infer"

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            resp = await client.post(url, json=req.model_dump())
            return resp.json()
        except Exception as e:
            raise HTTPException(502, f"Worker error: {e}")


@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_ready": registry.pipeline_ready()}


@app.get("/", response_class=HTMLResponse)
async def ui():
    with open("coordinator/static/index.html") as f:
        return f.read()