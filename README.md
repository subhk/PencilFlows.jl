# PencilFlows.jl

[![CI](https://github.com/subhk/PencilFlows.jl/workflows/CI/badge.svg)](https://github.com/subhk/PencilFlows.jl/actions)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://subhk.github.io/PencilFlows.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://subhk.github.io/PencilFlows.jl/dev)

A high-performance Julia package for solving fluid dynamics problems using spectral methods with MPI parallelization via PencilArrays and PencilFFTs.

## Features

- **Parallel spectral methods** using PencilArrays/PencilFFTs for 2D domain decomposition
- **Navier-Stokes solver** with rotation (Coriolis) and stratification
- **Flexible boundary conditions** including time-dependent and spatially-varying
- **Multiple Poisson solvers** (FFT-based and multigrid)
- **Symbolic PDE interface** for rapid prototyping
- **Optimized performance** with SIMD and memory pooling

## Quick Start

### Installation

```julia
using Pkg
Pkg.add(url="https://github.com/subhk/PencilFlows.jl")
```

### Basic Example

```julia
using PencilFlows
using MPI

MPI.Init()

# Create decomposition
Nx, Ny, Nz = 64, 64, 64
decomp = init_pencil_decomposition(Nx, Ny, Nz)

# Set up grid and fields
grid = (dx=2π/Nx, dy=2π/Ny, dz=1.0, Nz=Nz)
fields = create_distributed_fields(decomp)

# Initialize simulation
u, v, w = ... # velocity fields
p = ...       # pressure field

# Time stepping
dt = 0.01
for step in 1:1000
    u, v, w, p = predictor_corrector_step!(u, v, w, p, t, dt;
                                          decomp, grid, fields, ...)
end

MPI.Finalize()
```

## Documentation

- **[Testing Guide](.github/TESTING.md)** - How to run tests
- **[Examples](examples/)** - Complete working examples
- **[API Documentation](docs/)** - Detailed API reference

## Testing

```bash
# Run all tests
julia --project=. runtests.jl

# Quick smoke test
julia --project=. tests/test_smoke.jl

# With MPI
mpiexec -n 4 julia --project=. runtests.jl
```

## Contributing

Contributions are welcome! Please ensure:
- Tests pass: `julia --project=. runtests.jl`
- Code follows style guidelines
- Add tests for new features

## Acknowledgments

Built with [PencilArrays.jl](https://github.com/jipolanco/PencilArrays.jl) and [PencilFFTs.jl](https://github.com/jipolanco/PencilFFTs.jl).
