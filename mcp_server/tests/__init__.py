"""Tests for the MCP server.

These never open a socket. The HTTP layer is exercised through
``httpx.MockTransport``, which runs the real request and response handling and
only replaces the wire, so status mapping and body decoding are genuinely
tested rather than mocked away.
"""
