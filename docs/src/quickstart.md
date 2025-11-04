# Quick Start Guide

This guide will walk you through creating your first fluid simulation with PencilFlows.jl, even if you've never done numerical simulations before!

## What You'll Learn

- How to set up a basic simulation domain
- Understanding grids and discretization
- Running a simple flow simulation
- Visualizing results

## Basic Concepts (Simple Explanations)

### What is a Grid?

Think of a grid like a 3D checkerboard. We divide space into small boxes (cells) and calculate fluid properties (velocity, pressure) at each point. More boxes = more accurate results (but slower).

```
Grid in 2D (top view):
┌─┬─┬─┬─┐
├─┼─┼─┼─┤
├─┼─┼─┼─┤
├─┼─┼─┼─┤
└─┴─┴─┴─┘
Each cell stores: velocity (u,v,w), pressure (p), etc.
```

### What is MPI?

MPI (Message Passing Interface) lets us split the work across multiple computer processors, making simulations run faster. It's like having multiple workers each handling one piece of the puzzle.

## Your First Simulation

### Step 1: Set Up the Environment

```julia
using PencilFlows
using MPI

# Initialize MPI (required for parallel computing)
MPI.Init()
```

### Step 2: Define the Simulation Domain

```julia
# Grid size (number of points in each direction)
Nx = 64   # x-direction (horizontal)
Ny = 64   # y-direction (horizontal)
Nz = 32   # z-direction (vertical)

# Physical dimensions (in meters or dimensionless units)
Lx = 2π   # Length in x
Ly = 2π   # Length in y
H  = 1.0  # Height in z
```

**What these mean:**
- `Nx, Ny, Nz`: How many grid points (more = more accurate, but slower)
- `Lx, Ly, H`: The actual size of your simulation box

### Step 3: Create the Computational Grid

```julia
# This sets up the grid structure
grid = Grid(Nx, Ny, Nz, Lx, Ly, H)

println("Grid created with $(Nx*Ny*Nz) total points")
```

### Step 4: Set Up Boundary Conditions

Boundary conditions tell the simulation how fluid behaves at the walls.

```julia
# NO_SLIP: fluid sticks to the wall (like honey on a surface)
# FREE_SLIP: fluid slides freely along the wall (like ice)

bc = BoundaryCondition(NO_SLIP, FREE_SLIP)
# Bottom wall: no-slip, Top wall: free-slip
```

### Step 5: Initialize the Flow Field

Let's create a simple swirling flow pattern:

```julia
# Allocate arrays for velocity components
u = zeros(Float64, Nx, Ny, Nz)  # velocity in x-direction
v = zeros(Float64, Nx, Ny, Nz)  # velocity in y-direction
w = zeros(Float64, Nx, Ny, Nz)  # velocity in z-direction
p = zeros(Float64, Nx, Ny, Nz)  # pressure

# Create initial conditions - a gentle swirl
for k in 1:Nz
    for j in 1:Ny
        for i in 1:Nx
            x = (i-1) * Lx / Nx
            y = (j-1) * Ly / Ny

            # Swirling pattern
            u[i,j,k] = -sin(y)
            v[i,j,k] =  sin(x)
            w[i,j,k] = 0.0
        end
    end
end

println("Initial flow field created")
```

### Step 6: Time Stepping

Now let's evolve the flow in time:

```julia
# Time step parameters
dt = 0.01      # Time step size (smaller = more stable, but slower)
t_end = 1.0    # End time
nsteps = Int(t_end / dt)

println("Running $nsteps time steps...")

# Simple forward Euler time stepping
for step in 1:nsteps
    # Apply boundary conditions
    apply_boundary_conditions!(u, v, w, grid, bc)

    # Update time
    t = step * dt

    # Print progress every 10 steps
    if step % 10 == 0
        println("Step $step/$nsteps, time = $t")
    end
end

println("Simulation complete!")
```

### Step 7: Clean Up

```julia
# Finalize MPI
MPI.Finalize()
```

## Complete Example

Here's the complete code in one block:

```julia
using PencilFlows
using MPI

# Initialize
MPI.Init()

# Setup grid
Nx, Ny, Nz = 64, 64, 32
Lx, Ly, H = 2π, 2π, 1.0
grid = Grid(Nx, Ny, Nz, Lx, Ly, H)

# Boundary conditions
bc = BoundaryCondition(NO_SLIP, FREE_SLIP)

# Initialize fields
u = zeros(Float64, Nx, Ny, Nz)
v = zeros(Float64, Nx, Ny, Nz)
w = zeros(Float64, Nx, Ny, Nz)
p = zeros(Float64, Nx, Ny, Nz)

# Initial conditions
for k in 1:Nz, j in 1:Ny, i in 1:Nx
    x = (i-1) * Lx / Nx
    y = (j-1) * Ly / Ny
    u[i,j,k] = -sin(y)
    v[i,j,k] =  sin(x)
end

# Time stepping
dt, t_end = 0.01, 1.0
for step in 1:Int(t_end/dt)
    apply_boundary_conditions!(u, v, w, grid, bc)
    if step % 10 == 0
        println("Step $step, t = $(step*dt)")
    end
end

# Finalize
MPI.Finalize()
println("Done!")
```

## Understanding the Output

When you run this, you'll see:
```
Grid created with 131072 total points
Initial flow field created
Running 100 time steps...
Step 10, t = 0.1
Step 20, t = 0.2
...
Step 100, t = 1.0
Simulation complete!
Done!
```

## What's Next?

Now that you have a basic simulation running, try:

1. **Change parameters**: Modify `Nx`, `Ny`, `Nz` to see how resolution affects results
2. **Different initial conditions**: Create different flow patterns
3. **Add visualization**: Learn to plot your results (see Examples section)
4. **Run in parallel**: Use `mpiexec -n 4 julia script.jl` to use 4 processors

## Common Beginner Mistakes

### 1. Grid Too Large
```julia
# ❌ Don't start with this:
Nx, Ny, Nz = 1024, 1024, 512  # Too many points! Will be very slow

# ✓ Start small:
Nx, Ny, Nz = 32, 32, 16  # Good for testing
```

### 2. Time Step Too Large
```julia
# ❌ Might be unstable:
dt = 1.0  # Too big!

# ✓ Start conservative:
dt = 0.001  # Smaller is safer
```

### 3. Forgetting MPI Init/Finalize
```julia
# ❌ Missing:
using PencilFlows
# ... your code ...

# ✓ Always include:
MPI.Init()
# ... your code ...
MPI.Finalize()
```

## Next Steps

Continue to the [User Guide](guide/concepts.md) to learn about:
- Advanced boundary conditions
- Parallel computing with MPI
- Poisson solvers for pressure
- Output and visualization

Happy simulating!
