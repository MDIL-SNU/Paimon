import os
import os.path as osp
from pathlib import Path
import sys
import asyncio
from asyncio import StreamReader, StreamWriter
import json
import random
import string
import codecs


yaml_path = os.getenv("PAIMON_YAML", os.path.expanduser("~/.config/paimon.yaml"))

_alphabet = string.ascii_lowercase + string.digits


def _id_gen():
    return "".join(random.choices(_alphabet, k=8))


ID = _id_gen()
SOCKET_PATH = f"/tmp/socket_test_{ID}"
CLIENT_SCRIPT = Path(__file__).parent / "runner.py"


async def handle_turn_connection(reader: StreamReader, writer: StreamWriter):
    message = input(">")
    if message == "close":
        request = {
            "type": "terminate",
        }
    else:
        request = {
            "type": "user_msg",
            "user_msg": message,
        }

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    payload = json.dumps(request).encode()
    writer.write(payload)
    await writer.drain()
    writer.write_eof()

    while True:
        b = await reader.read(4096)
        if not b:
            break
        print(decoder.decode(b), end="", flush=True)
    print(decoder.decode(b"", final=True), flush=True)  # end with newline

    writer.close()
    await writer.wait_closed()


async def main():
    # create unix domain socket
    # the callback is called whenever a new client connection is est.
    server = await asyncio.start_unix_server(handle_turn_connection, path=SOCKET_PATH)

    stdout = open(f"/tmp/server_debug_{ID}.stdout", "w")
    stderr = open(f"/tmp/server_debug_{ID}.stderr", "w")

    # start a runner process, with created path to the socket
    proc = await asyncio.create_subprocess_exec(
        sys.executable, CLIENT_SCRIPT, 
        "--id", ID,
        "--config", yaml_path,
        "--socket", SOCKET_PATH,
        stdout=stdout,
        stderr=stderr,
    )

    async with server:
        try:
            await proc.wait()
        finally:
            if proc.returncode is None:
                proc.terminate()
                await proc.wait()
            if os.path.exists(SOCKET_PATH):
                os.remove(SOCKET_PATH)


if __name__ == "__main__":
    asyncio.run(main())
