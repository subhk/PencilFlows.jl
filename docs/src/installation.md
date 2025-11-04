# Installation

## Prerequisites

Before installing PencilFlows.jl, you need:

1. **Julia** (version 1.10 or later)
   - Download from [julialang.org](https://julialang.org/downloads/)
   - Follow the installation instructions for your operating system

2. **MPI Library** (for parallel computing)
   - **macOS**: `brew install open-mpi`
   - **Ubuntu/Debian**: `sudo apt-get install libopenmpi-dev`
   - **Windows**: Download MS-MPI from [Microsoft](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi)

## Installing PencilFlows.jl

### From Julia REPL

Open Julia and type:

```julia
using Pkg
Pkg.add(url="https://github.com/subhk/PencilFlows.jl.git")
```

### Development Installation

If you want to modify the code:

```julia
using Pkg
Pkg.develop(url="https://github.com/subhk/PencilFlows.jl.git")
```

This will clone the repository to `~/.julia/dev/PencilFlows`.

## Verifying Installation

Test that everything works:

```julia
using PencilFlows
println("PencilFlows.jl loaded successfully!")
```

## Setting Up MPI

### Configure MPI.jl

After installing MPI on your system, configure Julia to use it:

```julia
using Pkg
Pkg.add("MPIPreferences")
using MPIPreferences
MPIPreferences.use_system_binary()
```

### Test MPI

Create a test file `test_mpi.jl`:

```julia
using MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

println("Hello from process $rank of $size")

MPI.Finalize()
```

Run it with:

```bash
mpiexec -n 4 julia test_mpi.jl
```

You should see output from 4 processes.

## Troubleshooting

### MPI Issues

If you get MPI-related errors:

1. **Check MPI installation**:
   ```bash
   which mpiexec  # Should show path to mpiexec
   ```

2. **Reconfigure MPI.jl**:
   ```julia
   using MPIPreferences
   MPIPreferences.use_system_binary(force=true)
   ```

3. **Rebuild packages**:
   ```julia
   using Pkg
   Pkg.build("MPI")
   ```

### Package Conflicts

If you encounter package conflicts:

```julia
using Pkg
Pkg.resolve()
Pkg.update()
```

### Getting Help

- Check the [GitHub Issues](https://github.com/subhk/PencilFlows.jl/issues)
- Ask questions in [Julia Discourse](https://discourse.julialang.org/)

## Next Steps

Once installation is complete, head to the [Quick Start](quickstart.md) guide to run your first simulation!
