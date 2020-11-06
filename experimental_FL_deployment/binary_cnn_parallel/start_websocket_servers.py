import subprocess
from pathlib import Path
import signal
import sys

import warnings
warnings.filterwarnings("ignore")


python = Path(sys.executable).name

FILE_PATH = Path(__file__).resolve().parents[1].joinpath("run_websocket_server_3_workers_a_BINARY.py")

call_alice = [
    python,
    FILE_PATH,
    "--port",
    "8777",
    "--id",
    "alice",
    "--host",
    "127.0.0.1",
    "--notebook",
    "cnn-parallel",
]

call_bob = [
    python,
    FILE_PATH,
    "--port",
    "8778",
    "--id",
    "bob",
    "--host",
    "127.0.0.1",
    "--notebook",
    "cnn-parallel",
]

call_charlie = [
    python,
    FILE_PATH,
    "--port",
    "8779",
    "--id",
    "charlie",
    "--host",
    "127.0.0.1",
    "--notebook",
    "cnn-parallel",
]

call_testing = [
    python,
    FILE_PATH,
    "--port",
    "8780",
    "--id",
    "testing",
    "--testing",
    "--host",
    "127.0.0.1",
    "--notebook",
    "cnn-parallel",
]

print("Starting server for Alice")
process_alice = subprocess.Popen(call_alice)

print("Starting server for Bob")
process_bob = subprocess.Popen(call_bob)

print("Starting server for Charlie")
process_charlie = subprocess.Popen(call_charlie)

print("Starting server for Testing")
process_testing = subprocess.Popen(call_testing)


def signal_handler(sig, frame):
    print("You pressed Ctrl+C!")
    for p in [process_alice, process_bob, process_charlie, process_testing]:
        p.terminate()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

signal.pause()
