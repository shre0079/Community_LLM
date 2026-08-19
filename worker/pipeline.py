import logging
import zmq
import torch
from shared.tensor_io import serialize, deserialize, serialize_token, deserialize_token

log = logging.getLogger(__name__)


class Pipeline:
    """
    ZMQ socket management for the worker pipeline.

    Topology:
      Worker1.PUSH → Worker2.PULL  (hidden states)
      Worker2.PUSH → Worker3.PULL  (hidden states)
      Worker3.PUSH → Worker1.PULL  (generated token + is_done)

    Worker1 uses ZMQ_IN_PORT=5556 (result return from Worker3)
    Worker2,3 use ZMQ_IN_PORT=5555 (hidden states from previous)
    """

    def __init__(self, is_first: bool, is_last: bool, zmq_in_port: int):
        self.is_first = is_first
        self.is_last = is_last
        self.zmq_in_port = zmq_in_port
        self.ctx = zmq.Context()
        self.pull = None
        self.push = None

    def bind_pull(self):
        """Bind our incoming PULL socket."""
        self.pull = self.ctx.socket(zmq.PULL)
        self.pull.bind(f"tcp://*:{self.zmq_in_port}")
        log.info(f"PULL socket bound on port {self.zmq_in_port}")

    def connect_push(self, next_ip: str, next_port: int):
        """Connect our PUSH socket to the next worker's PULL socket."""
        self.push = self.ctx.socket(zmq.PUSH)
        self.push.connect(f"tcp://{next_ip}:{next_port}")
        log.info(f"PUSH socket connected to {next_ip}:{next_port}")

    def send_hidden(self, tensor: torch.Tensor):
        parts = serialize(tensor)
        self.push.send_multipart(parts)

    def recv_hidden(self) -> torch.Tensor:
        parts = self.pull.recv_multipart()
        tensor, _ = deserialize(parts)
        return tensor

    def send_token(self, token_id: int, is_done: bool):
        self.push.send(serialize_token(token_id, is_done))

    def recv_token(self) -> tuple[int, bool]:
        data = self.pull.recv()
        return deserialize_token(data)

    def close(self):
        if self.pull:
            self.pull.close()
        if self.push:
            self.push.close()
        self.ctx.term()