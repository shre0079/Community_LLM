import json

import numpy as np
import torch


def serialize(tensor: torch.Tensor, meta: dict = None) -> list[bytes]:
    """Pack a tensor + metadata dict into a 2-part ZMQ multipart message."""
    arr = tensor.detach().cpu().to(torch.float16).numpy()
    metadata = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        **(meta or {}),
    }
    return [json.dumps(metadata).encode(), arr.tobytes()]


def deserialize(parts: list[bytes]) -> tuple[torch.Tensor, dict]:
    """Unpack a 2-part ZMQ multipart message into (tensor, metadata)."""
    meta  = json.loads(parts[0].decode())
    shape = meta.pop("shape")
    dtype = meta.pop("dtype")
    arr   = np.frombuffer(parts[1], dtype=np.dtype(dtype)).reshape(shape)
    return torch.from_numpy(arr.copy()), meta


def serialize_token(token_id: int, is_done: bool) -> bytes:
    return json.dumps({"token_id": token_id, "is_done": is_done}).encode()


def deserialize_token(data: bytes) -> tuple[int, bool]:
    obj = json.loads(data.decode())
    return obj["token_id"], obj["is_done"]