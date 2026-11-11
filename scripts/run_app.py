"""Run the local app on 127.0.0.1 (port from config/app.yaml unless --port is given)."""

import argparse

import uvicorn

from keystroke.config import load_app_config, utf8_stdio
from keystroke.server import create_app


def main() -> None:
    utf8_stdio()
    server = load_app_config()["server"]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=int(server["port"]))
    args = ap.parse_args()
    print(f"Open http://{server['host']}:{args.port}")
    uvicorn.run(create_app(), host=server["host"], port=args.port, log_level="info")


if __name__ == "__main__":
    main()
