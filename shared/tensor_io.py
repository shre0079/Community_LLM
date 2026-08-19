import io
import json
import numpy as np
import torch


def serialize(tensor: torch.Tensor, meta: dict = None) -> list[bytes]:
    """Returns a ZMQ multipart message: [metadata_bytes, tensor_bytes]"""
    arr = tensor.detach().cpu().to(torch.float16).numpy()
    metadata = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        **(meta or {})
    }
    return [json.dumps(metadata).encode(), arr.tobytes()]


def deserialize(parts: list[bytes]) -> tuple[torch.Tensor, dict]:
    """Reconstruct tensor from ZMQ multipart message."""
    meta = json.loads(parts[0])
    shape = meta.pop("shape")
    dtype = meta.pop("dtype")
    arr = np.frombuffer(parts[1], dtype=np.dtype(dtype)).reshape(shape)
    return torch.from_numpy(arr.copy()), meta


def serialize_token(token_id: int, is_done: bool) -> bytes:
    return json.dumps({"token_id": token_id, "is_done": is_done}).encode()


def deserialize_token(data: bytes) -> tuple[int, bool]:
    obj = json.loads(data)
    return obj["token_id"], obj["is_done"]