"""An MCP server exposing the Scintilla corpus to language-model clients.

The server is a client of the HTTP API, not of the database. That is a memory
decision before it is an architectural one: the embedding model is loaded once,
inside the API workers, and both the web frontend and this server reuse it. An
in-process MCP server would import torch and load a second copy of BGE, which
on a 12 GB host is the difference between comfortable and swapping.

It has the useful side effect of making this a second consumer of the same
published contract the frontend generates its types from.
"""
