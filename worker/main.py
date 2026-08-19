import asyncio
import logging
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import httpx
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.config import (
    COORDINATOR_URL, WORKER_HTTP_PORT,
    ZMQ_IN_PORT, ZMQ_RESULT_PORT
)
from worker.model import ShardRunner
from worker.pipeline import Pipeline
from worker.heartbeat import Heartbeat

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Global state ──────────────────────────────────────────────────────────────
runner: ShardRunner = None
pipeline: Pipeline = None
heartbeat: Heartbeat = None
worker_id: str = None
current_status: str = "loading"
executor = ThreadPoolExecutor(max_workers=1)


# ── Registration + Topology ───────────────────────────────────────────────────

async def register() -> dict:
    my_ip = os.getenv("MY_IP", socket.gethostbyname(socket.gethostname()))
    zmq_port = ZMQ_RESULT_PORT  # Worker 1 listens on 5556 for return tokens

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{COORDINATOR_URL}/register",
            json={
                "hostname": socket.gethostname(),
                "ip": my_ip,
                "http_port": WORKER_HTTP_PORT,
                "zmq_in_port": zmq_port,  # what we tell coordinator we listen on
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()


async def wait_for_topology() -> dict:
    """Poll coordinator until all workers are registered and active."""
    log.info("Waiting for full pipeline topology...")
    async with httpx.AsyncClient() as client:
        while True:
            try:
                resp = await client.get(f"{COORDINATOR_URL}/topology", timeout=5.0)
                data = resp.json()
                if data["ready"]:
                    log.info("Pipeline topology ready.")
                    return data
            except Exception as e:
                log.warning(f"Topology not ready: {e}")
            await asyncio.sleep(5)


def setup_zmq(my_assignment: dict, topology: dict):
    """Wire up ZMQ sockets based on pipeline topology."""
    global pipeline
    is_first = my_assignment["is_first"]
    is_last = my_assignment["is_last"]

    # Determine which port WE listen on
    if is_first:
        my_in_port = ZMQ_RESULT_PORT  # Worker 1 receives tokens from Worker 3
    else:
        my_in_port = ZMQ_IN_PORT      # Workers 2,3 receive hidden states

    pipeline = Pipeline(is_first=is_first, is_last=is_last, zmq_in_port=my_in_port)
    pipeline.bind_pull()

    # Find the next worker to push to
    workers = topology["workers"]
    my_start = my_assignment["layer_start"]

    if is_last:
        # Last worker pushes tokens back to First worker
        first = next(w for w in workers if w["is_first"])
        pipeline.connect_push(first["ip"], ZMQ_RESULT_PORT)
    else:
        # Push hidden states to next worker in pipeline
        next_w = next(w for w in workers if w["layer_start"] == my_assignment["layer_end"] + 1)
        pipeline.connect_push(next_w["ip"], ZMQ_IN_PORT)

    log.info(f"ZMQ pipeline wired up. is_first={is_first}, is_last={is_last}")


# ── Generation Loop ───────────────────────────────────────────────────────────

def run_middle_worker_loop():
    """Workers 2 and 3: continuously receive, process, forward."""
    log.info("Middle/last worker loop started.")
    while True:
        try:
            if runner.is_last:
                hidden = pipeline.recv_hidden()
                next_token = runner.forward(hidden)
                token_id = next_token.item()
                is_done = (token_id == runner.eos_token_id)
                pipeline.send_token(token_id, is_done)
            else:
                hidden = pipeline.recv_hidden()
                out = runner.forward(hidden)
                pipeline.send_hidden(out)
        except Exception as e:
            log.error(f"Worker loop error: {e}")


def generate(prompt: str, max_new_tokens: int, temperature: float) -> str:
    """
    Full autoregressive generation loop — runs on Worker 1.
    Each step: embed + own layers → send → receive token → repeat.
    """
    input_ids = runner.encode(prompt)  # [1, prompt_len]
    generated_ids = input_ids.clone()
    generated_tokens = []
    
    start = time.time()

    for step in range(max_new_tokens):
        # Worker 1 forward pass on its layers
        hidden = runner.forward(generated_ids)  # [1, seq_len, 2048]

        # Send hidden states down the pipeline
        pipeline.send_hidden(hidden)

        # Wait for generated token from Worker 3
        token_id, is_done = pipeline.recv_token()

        generated_tokens.append(token_id)
        new_tok = torch.tensor([[token_id]])
        generated_ids = torch.cat([generated_ids, new_tok], dim=1)

        if is_done or token_id == runner.eos_token_id:
            break

    elapsed = time.time() - start
    tokens_per_sec = len(generated_tokens) / elapsed
    log.info(f"Generated {len(generated_tokens)} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")

    return runner.decode(torch.tensor(generated_tokens))


# ── FastAPI App ───────────────────────────────────────────────────────────────

class InferReq(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.8


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runner, worker_id, current_status

    # 1. Register with coordinator
    current_status = "registering"
    assignment = await register()
    worker_id = assignment["worker_id"]
    log.info(f"Assigned layers {assignment['layer_start']}–{assignment['layer_end']}")

    # 2. Load model shard
    current_status = "loading"
    runner = ShardRunner(
        layer_start=assignment["layer_start"],
        layer_end=assignment["layer_end"],
        is_first=assignment["is_first"],
        is_last=assignment["is_last"],
    )
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, runner.load)

    # 3. Wait for full topology, set up ZMQ
    topology = await wait_for_topology()
    setup_zmq(assignment, topology)

    # 4. Start heartbeat
    hb = Heartbeat(worker_id, lambda: current_status)
    asyncio.create_task(hb.run())

    # 5. Start processing loop for middle/last workers
    if not assignment["is_first"]:
        loop.run_in_executor(executor, run_middle_worker_loop)

    current_status = "active"
    log.info("Worker is ACTIVE and ready.")
    yield

    pipeline.close()


app = FastAPI(title="OLMoE Worker", lifespan=lifespan)


@app.post("/infer")
async def infer(req: InferReq):
    if not runner.is_first:
        raise HTTPException(400, "Only the first worker accepts inference requests")
    global current_status
    current_status = "active"

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            executor,
            generate,
            req.prompt,
            req.max_new_tokens,
            req.temperature,
        )
    except Exception as e:
        log.error(f"Generation error: {e}")
        raise HTTPException(500, str(e))

    return {
        "response": result,
        "prompt": req.prompt,
        "vram_used_gb": runner.vram_used_gb(),
    }


@app.get("/health")
async def health():
    return {
        "status": current_status,
        "worker_id": worker_id,
        "layers": f"{runner.layer_start}–{runner.layer_end}" if runner else "loading",
        "vram_used_gb": runner.vram_used_gb() if runner else 0,
    }


if __name__ == "__main__":
    coordinator = os.getenv("COORDINATOR_HOST", "localhost")
    os.environ["COORDINATOR_HOST"] = coordinator
    uvicorn.run("worker.main:app", host="0.0.0.0", port=WORKER_HTTP_PORT, log_level="info")