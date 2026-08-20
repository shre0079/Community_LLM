import logging

import torch
import zmq

from shared.tensor_io import deserialize, deserialize_token, serialize, serialize_token

log = logging.getLogger(__name__)


class Pipeline:
    """
    ZMQ wiring for the 3-worker pipeline.

    Ports (set by coordinator, returned in registration response):
      Worker 1  PULL :5556  ← tokens from Worker 3
      Worker 1  PUSH → Worker 2 :5555
      Worker 2  PULL :5555  ← hidden states from Worker 1
      Worker 2  PUSH → Worker 3 :5555
      Worker 3  PULL :5555  ← hidden states from Worker 2
      Worker 3  PUSH → Worker 1 :5556
    """

    def __init__(
        self,
        is_first: bool,
        is_last: bool,
        zmq_in_port: int,
        recv_timeout_ms: int = 30_000,
    ):
        self.is_first        = is_first
        self.is_last         = is_last
        self.zmq_in_port     = zmq_in_port
        self.recv_timeout_ms = recv_timeout_ms

        self.ctx  = zmq.Context()
        self.pull: zmq.Socket | None = None
        self.push: zmq.Socket | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def bind_pull(self):
        self.pull = self.ctx.socket(zmq.PULL)
        self.pull.setsockopt(zmq.RCVTIMEO, self.recv_timeout_ms)
        self.pull.bind(f"tcp://*:{self.zmq_in_port}")
        log.info(f"PULL bound on :{self.zmq_in_port}  timeout={self.recv_timeout_ms}ms")

    def connect_push(self, target_ip: str, target_port: int):
        self.push = self.ctx.socket(zmq.PUSH)
        self.push.connect(f"tcp://{target_ip}:{target_port}")
        log.info(f"PUSH connected → {target_ip}:{target_port}")

    def close(self):
        for s in (self.pull, self.push):
            if s:
                s.close()
        self.ctx.term()

    # ── Hidden-state transfer ─────────────────────────────────────────────────

    def send_hidden(self, tensor: torch.Tensor, meta: dict = None):
        """Send hidden states + metadata to next worker."""
        parts  = serialize(tensor, meta)
        nbytes = sum(len(p) for p in parts)
        log.debug(f"send_hidden {tensor.shape}  {nbytes / 1024:.1f} KB")
        self.push.send_multipart(parts)

    def recv_hidden(self) -> tuple[torch.Tensor, dict]:
        """
        Receive hidden states + metadata from previous worker.
        Raises zmq.error.Again on timeout.
        """
        parts          = self.pull.recv_multipart()   # Again on timeout
        tensor, meta   = deserialize(parts)
        log.debug(f"recv_hidden {tensor.shape}")
        return tensor, meta

    # ── Token transfer ────────────────────────────────────────────────────────

    def send_token(self, token_id: int, is_done: bool):
        self.push.send(serialize_token(token_id, is_done))

    def recv_token(self) -> tuple[int, bool]:
        """
        Receive next-token result from last worker.
        Raises zmq.error.Again on timeout.
        """
        data = self.pull.recv()                       # Again on timeout
        return deserialize_token(data)