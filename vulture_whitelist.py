"""Names a library uses that vulture can't see, so it wrongly reports them as dead code.

Listed in [tool.shape] dead_code_whitelist in pyproject.toml. This file is only parsed
by vulture, never run or imported. One line per name, each with the reason. Before
adding one, make sure nothing could call it: dead code is usually really dead.
"""

from uvicorn import Server

Server.should_exit  # uvicorn reads it to stop the server; backends/mcp_bridge.py only sets it
