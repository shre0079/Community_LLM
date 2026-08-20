import asyncio
import logging
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import httpx
import torch
import uvicorn
import zmq.error
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.config import (
    COORDINATOR_URL,
    WORKER_HTTP_PORT,
    ZMQ_RECV_TIMEOUT_MS,
)
from worker.heartbeat import Heartbeat
from worker.model import ShardRunner
from worker.pipeline import Pipeline

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ── Global state ──────────────────────────────────────────────────────────────

runner:         ShardRunner | None = None
pipeline_conn:  Pipeline    | None = None
worker_id:      str | None         = None
current_status: str                = "registering"
assignment:     dict               = {}

# Dedicated executors — ZMQ loop and inference never compete for the same thread.
_zmq_executor   = ThreadPoolExecutor(max_workers=1, thread_name_prefix="zmq")
_infer_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
_stop_event     = threading.Event()


# ── Registration ──────────────────────────────────────────────────────────────

async def _register() -> dict:
    my_ip = os.getenv("MY_IP") or _lan_ip()
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{COORDINATOR_URL}/register",
            json={
                "hostname":  socket.gethostname(),
                "ip":        my_ip,
                "http_port": WORKER_HTTP_PORT,
                "gpu_name":  (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else "CPU"
                ),
                "vram_gb": (
                    torch.cuda.get_device_properties(0).total_memory / 1e9
                    if torch.cuda.is_available() else 0.0
                ),
            },
        )
        resp.raise_for_status()
        return resp.json()


def _lan_ip() -> str:
    """Return the machine's outbound LAN IP (avoids 127.0.0.1)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return socket.gethostbyname(socket.gethostname())


# ── Topology polling ──────────────────────────────────────────────────────────

async def _wait_for_topology() -> dict:
    """
    Poll /topology until all workers are in 'loaded' or 'active' state.
    Workers reach this point only after finishing model load, so
    pipeline_ready=True means ZMQ wiring can safely proceed.
    """
    log.info("Waiting for full pipeline topology…")
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            try:
                resp = await client.get(f"{COORDINATOR_URL}/topology")
                data = resp.json()
                # Fix: count the actual workers returned, not a tautology
                n_loaded = len(data["workers"])
                log.info(f"Topology: {n_loaded}/3 workers loaded, ready={data['ready']}")
                if data["ready"]:
                    return data
            except Exception as e:
                log.debug(f"Topology poll error: {e}")
            await asyncio.sleep(5)


# ── ZMQ wiring ────────────────────────────────────────────────────────────────

def _wire_zmq(my_assignment: dict, topology: dict):
    global pipeline_conn

    is_first     = my_assignment["is_first"]
    is_last      = my_assignment["is_last"]
    zmq_in_port  = my_assignment["zmq_in_port"]

    pipeline_conn = Pipeline(
        is_first=is_first,
        is_last=is_last,
        zmq_in_port=zmq_in_port,
        recv_timeout_ms=ZMQ_RECV_TIMEOUT_MS,
    )
    pipeline_conn.bind_pull()

    workers   = topology["workers"]
    layer_end = my_assignment["layer_end"]

    if is_last:
        # Last worker pushes tokens back to first worker.
        first = next(w for w in workers if w["is_first"])
        pipeline_conn.connect_push(first["ip"], first["zmq_in_port"])
    else:
        # Push hidden states to the next worker in the pipeline.
        next_w = next(w for w in workers if w["layer_start"] == layer_end + 1)
        pipeline_conn.connect_push(next_w["ip"], next_w["zmq_in_port"])

    log.info(f"ZMQ wired. is_first={is_first} is_last={is_last}")


# ── ZMQ processing loop (middle & last workers) ───────────────────────────────

def _zmq_loop():
    """
    Runs in _zmq_executor (dedicated thread, never blocks inference).

    Middle worker: recv hidden → forward → send hidden (+ metadata)
    Last worker:   recv hidden → forward with temperature → send token
    """
    log.info("ZMQ processing loop started.")
    while not _stop_event.is_set():
        try:
            hidden, meta = pipeline_conn.recv_hidden()   # raises Again on timeout
            temperature  = float(meta.get("temperature", 1.0))
            step         = meta.get("step", -1)

            out = runner.forward(hidden, temperature=temperature)

            if runner.is_last:
                token_id = out.item()
                is_done  = (token_id == runner.eos_token_id)
                pipeline_conn.send_token(token_id, is_done)
                log.debug(f"Step {step}: token={token_id} done={is_done}")
            else:
                # Forward metadata so the last worker receives temperature.
                pipeline_conn.send_hidden(out, meta)
                log.debug(f"Step {step}: forwarded hidden {out.shape}")

        except zmq.error.Again:
            # Timeout — normal during idle periods. Check stop_event and loop.
            continue
        except Exception as e:
            log.error(f"ZMQ loop error: {e}", exc_info=True)
            time.sleep(0.1)

    log.info("ZMQ processing loop stopped.")


# ── Autoregressive generation (first worker only) ─────────────────────────────

def _generate(prompt: str, max_new_tokens: int, temperature: float) -> dict:
    input_ids       = runner.encode(prompt)      # [1, prompt_len]
    generated_ids   = input_ids.clone()
    generated_tokens: list[int] = []
    t0 = time.time()

    for step in range(max_new_tokens):
        # Our layers: embed + layers 0-5
        hidden = runner.forward(generated_ids, temperature=temperature)

        # Log transfer cost — grows linearly with sequence length
        nbytes = hidden.element_size() * hidden.nelement()
        log.info(
            f"Step {step + 1}/{max_new_tokens} "
            f"seq_len={generated_ids.shape[1]} "
            f"payload={nbytes / 1024:.1f} KB"
        )

        # Send downstream with temperature embedded in metadata
        pipeline_conn.send_hidden(hidden, {"temperature": temperature, "step": step})

        # Wait for next-token from last worker
        try:
            token_id, is_done = pipeline_conn.recv_token()
        except zmq.error.Again:
            raise TimeoutError(
                "No token received within timeout window. "
                "A downstream worker may be dead."
            )

        generated_tokens.append(token_id)
        generated_ids = torch.cat(
            [generated_ids, torch.tensor([[token_id]], dtype=torch.long)], dim=1
        )

        if is_done:
            break

    elapsed   = time.time() - t0
    tok_per_s = len(generated_tokens) / elapsed if elapsed > 0 else 0.0

    log.info(
        f"Done. {len(generated_tokens)} tokens in {elapsed:.2f}s "
        f"({tok_per_s:.2f} tok/s)"
    )

    return {
        "response":         runner.decode(generated_tokens),
        "tokens_generated": len(generated_tokens),
        "elapsed_s":        round(elapsed, 3),
        "tok_per_s":        round(tok_per_s, 2),
        "vram_used_gb":     round(runner.vram_used_gb(), 3),
    }


# ── FastAPI ───────────────────────────────────────────────────────────────────

class InferReq(BaseModel):
    prompt:         str
    max_new_tokens: int   = 200
    temperature:    float = 0.8

async def _notify_status(status: str):
    """
    Fire a single immediate heartbeat to the coordinator.
    Used at status transitions so the coordinator learns without
    waiting for the next heartbeat interval.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"{COORDINATOR_URL}/heartbeat",
                json={"worker_id": worker_id, "status": status},
            )
        log.info(f"Notified coordinator: status={status!r}")
    except Exception as e:
        log.warning(f"Immediate status notify failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global runner, worker_id, current_status, assignment

    # ── Step 1: Register ──────────────────────────────────────────────────
    current_status = "registering"
    result         = await _register()
    worker_id      = result["worker_id"]
    assignment     = result
    log.info(
        f"Registered. ID={worker_id} "
        f"layers={result['layer_start']}–{result['layer_end']} "
        f"zmq_in_port={result['zmq_in_port']}"
    )

    # ── Step 2: Load model ────────────────────────────────────────────────
    current_status = "loading"
    runner = ShardRunner(
        layer_start=result["layer_start"],
        layer_end=result["layer_end"],
        is_first=result["is_first"],
        is_last=result["is_last"],
    )
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as load_pool:
        await loop.run_in_executor(load_pool, runner.load)

    # ── Step 3: Set loaded + immediately notify coordinator ───────────────
    # This MUST happen before topology polling.
    # The heartbeat fires every 5s; waiting for it would stall all 3 workers.
    current_status = "loaded"
    await _notify_status("loaded")
    log.info("Status → loaded. Notified coordinator. Waiting for other workers…")

    # ── Step 4: Start heartbeat NOW (before topology poll) ────────────────
    # Heartbeat keeps the coordinator updated while we wait for peers.
    hb      = Heartbeat(worker_id, lambda: current_status)
    hb_task = asyncio.create_task(hb.run())

    # ── Step 5: Poll topology until all workers are loaded ────────────────
    topology = await _wait_for_topology()

    # ── Step 6: Wire ZMQ ─────────────────────────────────────────────────
    _wire_zmq(assignment, topology)

    # ── Step 7: Go active ─────────────────────────────────────────────────
    current_status = "active"
    await _notify_status("active")

    # ── Step 8: Start ZMQ loop for middle / last workers ─────────────────
    zmq_future = None
    if not result["is_first"]:
        _stop_event.clear()
        zmq_future = loop.run_in_executor(_zmq_executor, _zmq_loop)

    log.info(
        f"Worker ACTIVE — layers {result['layer_start']}–{result['layer_end']} "
        f"is_first={result['is_first']} is_last={result['is_last']}"
    )

    yield   # ─── application runs ───────────────────────────────────────────

    # ── Shutdown ──────────────────────────────────────────────────────────
    log.info("Shutting down worker…")
    _stop_event.set()
    hb.stop()
    hb_task.cancel()
    if pipeline_conn:
        pipeline_conn.close()
    log.info("Worker shut down cleanly.")


app = FastAPI(title="OLMoE Worker", lifespan=lifespan)


@app.post("/infer")
async def infer(req: InferReq):
    if not runner or not runner.is_first:
        raise HTTPException(
            status_code=400,
            detail="Only the first worker accepts inference requests.",
        )
    if current_status != "active":
        raise HTTPException(
            status_code=503,
            detail=f"Worker not ready (status={current_status!r})",
        )
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _infer_executor, _generate,
            req.prompt, req.max_new_tokens, req.temperature,
        )
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        log.exception("Inference error")
        raise HTTPException(status_code=500, detail=str(e))
    return result


@app.get("/health")
async def health():
    return {
        "status":       current_status,
        "worker_id":    worker_id,
        "layers":       f"{runner.layer_start}–{runner.layer_end}" if runner else None,
        "is_first":     assignment.get("is_first"),
        "is_last":      assignment.get("is_last"),
        "vram_used_gb": runner.vram_used_gb() if runner else 0.0,
    }


if __name__ == "__main__":
    uvicorn.run("worker.main:app", host="0.0.0.0", port=WORKER_HTTP_PORT, log_level="info")