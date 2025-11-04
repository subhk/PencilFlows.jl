# PencilFlows.jl

**High-Performance Spectral Methods for Fluid Dynamics Simulations**

PencilFlows.jl is a Julia package for solving partial differential equations (PDEs) in fluid dynamics using spectral methods with efficient MPI parallelization via pencil decomposition.

## Features

- **Spectral Methods**: High-order accuracy using FFT-based derivatives
- **MPI Parallelization**: Efficient 2D pencil decomposition for distributed computing
- **Advanced Solvers**: Multigrid Poisson solvers for pressure computation
- **Flexible Boundary Conditions**: Support for no-slip, free-slip, stress-free, Robin, and custom BCs
- **IMEX Time Stepping**: Implicit-explicit Runge-Kutta schemes for stiff equations
- **Symbolic Interface**: Intuitive equation specification using symbolic expressions
- **Non-uniform Grids**: Support for stretched grids and boundary layer resolution

## Quick Example

```julia
using PencilFlows

# Initialize MPI
using MPI
MPI.Init()

# Create a 3D grid
Nx, Ny, Nz = 128, 128, 64
Lx, Ly, H = 4π, 4π, 2.0

# Set up pencil decomposition
decomp = init_pencil_decomposition(Nx, Ny, Nz)

# Create distributed fields
fields = create_distributed_fields(decomp)

# Your simulation code here...

MPI.Finalize()
```

## Why PencilFlows.jl?

### High Performance
- Optimized FFT operations using FFTW
- Cache-friendly data layouts
- SIMD vectorization support
- Efficient MPI communication patterns

### Flexibility
- Support for multiple physics modules (Coriolis, buoyancy, etc.)
- Extensible boundary condition system
- Custom spatial field initialization
- Modular solver components

### Ease of Use
- Clean, intuitive API
- Comprehensive error checking
- Detailed documentation
- Example-driven learning

## Installation

See the [Installation Guide](installation.md) for detailed instructions.

## Getting Help

- **Documentation**: Browse the user guide and API reference
- **Issues**: Report bugs or request features on [GitHub](https://github.com/subhk/PencilFlows.jl/issues)
- **Examples**: Check out the examples directory for practical use cases

## Citation

If you use PencilFlows.jl in your research, please cite:

```bibtex
@software{pencilflows2025,
  author = {Kar, Subhajit},
  title = {PencilFlows.jl: High-Performance Spectral Methods for Fluid Dynamics},
  year = {2025},
  url = {https://github.com/subhk/PencilFlows.jl}
}
```

## License

PencilFlows.jl is released under the MIT License.

## Contents

```@contents
Pages = [
    "installation.md",
    "quickstart.md",
]
Depth = 2
```
