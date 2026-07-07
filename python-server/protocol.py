"""
Length-prefixed JSON protocol over stdin/stdout.

Each message is encoded as:
  [4 bytes: big-endian uint32 length][JSON body]

This avoids newline-delimited issues with large code contexts
and provides clean message boundaries.
"""

import json
import struct
import sys
from typing import AsyncIterator, Optional


# --- Encoding ---

def encode_message(data: dict) -> bytes:
    """Serialize a dict to a length-prefixed message."""
    body = json.dumps(data).encode("utf-8")
    return struct.pack("!I", len(body)) + body


def encode_token(token: str, msg_id: int) -> bytes:
    """Encode a streaming token message."""
    return encode_message({
        "type": "stream",
        "id": msg_id,
        "token": token,
    })


def encode_complete(completion: str, msg_id: int) -> bytes:
    """Encode a completed response message."""
    return encode_message({
        "type": "complete",
        "id": msg_id,
        "completion": completion,
    })


def encode_error(message: str, msg_id: Optional[int] = None) -> bytes:
    """Encode an error message."""
    return encode_message({
        "type": "error",
        "id": msg_id,
        "error": message,
    })


# --- Decoding ---

def _read_exact(n: int) -> bytes:
    """Read exactly n bytes from stdin."""
    result = bytearray()
    while len(result) < n:
        chunk = sys.stdin.buffer.read(n - len(result))
        if not chunk:
            raise EOFError("Unexpected EOF while reading message")
        result.extend(chunk)
    return bytes(result)


def decode_next_message() -> dict:
    """Read and decode the next message from stdin.

    Returns a dict parsed from the JSON body.
    Raises EOFError if the stream is closed.
    """
    # Read 4-byte length prefix
    length_bytes = _read_exact(4)
    length = struct.unpack("!I", length_bytes)[0]

    # Read JSON body
    body_bytes = _read_exact(length)
    return json.loads(body_bytes.decode("utf-8"))


def message_writer():
    """Yield a write function for sending messages to stdout.

    Usage:
        write = message_writer()
        write(encode_message({...}))
    """
    def write(data: bytes) -> None:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    return write
