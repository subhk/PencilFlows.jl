"""
Thin wrapper for the symbolic interface (deprecated entrypoint).

This file is kept for backward compatibility. The implementation now
lives under `src/symbolic/` and is included via `symbolic/main.jl`.
"""

@warn "`src/interfaces/symbolic_interface.jl` is deprecated; use `src/symbolic/main.jl` via the package entry instead."

include("../symbolic/main.jl")