"""Entry point: ``python -m mcp_server``.

Two transports. ``stdio`` is what a desktop client launches directly and is the
default. ``streamable-http`` is what runs behind the Cloudflare tunnel, where
there is no process for a client to spawn.

Configuration is resolved before the transport starts so that a bad environment
fails immediately and visibly. Under stdio in particular, anything written to
stdout after the transport opens corrupts the protocol stream - which is why
this reports errors on stderr and exits rather than trying to recover.
"""

import argparse
import sys

from mcp_server.config import ConfigError, Settings
from mcp_server.server import build_server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mcp_server",
        description="Expose the Scintilla corpus over the Model Context Protocol.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="stdio for a locally launched client, streamable-http to serve remotely.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Bind address for streamable-http. Defaults to loopback: the tunnel "
            "connects from the same host, so there is no reason to listen wider."
        ),
    )
    parser.add_argument("--port", type=int, default=8081, help="Port for streamable-http.")
    args = parser.parse_args(argv)

    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        print(f"scintilla-mcp: {exc}", file=sys.stderr)
        return 2

    server = build_server(settings)

    if args.transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(transport="streamable-http", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
