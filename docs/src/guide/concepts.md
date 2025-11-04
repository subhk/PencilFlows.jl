# Core Concepts

This page explains the fundamental concepts in PencilFlows.jl in simple terms.

## What is Computational Fluid Dynamics (CFD)?

Imagine you want to predict how air flows around an airplane wing or how water swirls down a drain. CFD uses computers to solve mathematical equations that describe fluid motion.

### The Navier-Stokes Equations (Simplified)

These equations describe how fluids move. Don't worry about the math - PencilFlows handles it for you! But conceptually:

```
∂u/∂t = -u·∇u - ∇p + ν∇²u
```

This says: **How velocity changes = (advection) + (pressure) + (viscosity)**

- **Advection**: Fluid carrying itself along (like a river current)
- **Pressure**: Forces from pressure differences (like air from a balloon)
- **Viscosity**: Internal friction (like honey being thicker than water)

## Spectral Methods (The "How")

### Traditional vs. Spectral

**Traditional Finite Difference** (like measuring with a ruler):
```
f'(x) ≈ [f(x+h) - f(x)] / h
```
Simple but less accurate.

**Spectral Methods** (like decomposing sound into frequencies):
```
f(x) = Σ aₙ sin(nx) + bₙ cos(nx)
```
More accurate but requires periodic or special boundaries.

### Why Spectral?

```julia
# Spectral method: One FFT gives ALL derivatives!
using FFTW
f_hat = fft(f)           # Transform to frequency space
df_hat = im * k .* f_hat  # Derivative = multiply by ik
df = real(ifft(df_hat))   # Transform back

# Compare to finite difference:
# df[i] = (f[i+1] - f[i-1]) / (2*dx)  # Do this for EVERY point
```

**Benefits**:
- Very accurate (exponential convergence)
- Fast (FFT is O(N log N))
- Natural for periodic problems

## Pencil Decomposition (Parallelization Strategy)

### The Challenge

Imagine a 3D cube of data divided among 4 processors. How do we compute derivatives?

```
Original 3D grid (64×64×64):
[Processor layout shown as 2D slices]
```

### Pencil Decomposition Solution

We organize data in "pencils" - long, thin sticks through the domain:

```
X-Pencils (oriented along x):
═══════════════
Processor 0: ███████░░░░░░░░
Processor 1: ░░░░░░░███████░
Processor 2: ░░░░░░░░░░░░░███

Each processor owns complete "sticks" in x-direction
```

**Key Idea**:
- To compute x-derivatives → Use X-pencils (data is contiguous)
- To compute y-derivatives → Transpose to Y-pencils
- To compute z-derivatives → Transpose to Z-pencils

### Why This Works

```julia
# Computing d/dx requires data along x
# X-pencil has all x-data locally → No communication needed!

# Example:
decomp = init_pencil_decomposition(128, 128, 64)
fields = create_distributed_fields(decomp)

# Compute x-derivative (fast - local operation)
ddx!(dudx, u, decomp)  # Data already in X-pencils

# For y-derivative, we first transpose:
transpose!(u_y, decomp.transform_z_to_y, u_z)  # Communication
ddy!(dudy, u_y, decomp)  # Then local operation
```

## Grid Structure

### Uniform vs. Non-Uniform Grids

**Uniform Grid** (evenly spaced):
```
z: |---|---|---|---|---|  (Δz constant)
   0  0.2 0.4 0.6 0.8 1.0
```

**Non-Uniform Grid** (stretched):
```
z: |-|-----|-----------|  (finer near boundaries)
   0 0.1   0.4        1.0
```

Use non-uniform grids for:
- Boundary layers (flow near walls)
- Sharp gradients
- Better resolution where needed

```julia
# Create a boundary-layer grid
dz = create_boundary_layer_grid(Nz, H, δ_bl)
grid = Grid(Nx, Ny, Nz, Lx, Ly, H; dz=dz)
```

## Time Stepping

### Explicit vs. Implicit

**Explicit** (calculate next step directly):
```julia
u_new = u_old + dt * f(u_old)
```
- Simple
- Limited by stability (small dt required)

**Implicit** (solve for next step):
```julia
u_new = u_old + dt * f(u_new)  # u_new appears on both sides!
```
- More stable (larger dt allowed)
- Requires solving equations

### IMEX (Best of Both Worlds)

**IM**plicit for stiff terms + **EX**plicit for others:

```julia
# Navier-Stokes split:
# Explicit: Advection (fast, nonlinear)
# Implicit: Diffusion (slow, linear)

u_new = u_old + dt * [advection(u_old)] + dt * [diffusion(u_new)]
#                      ↑ Explicit            ↑ Implicit
```

## Boundary Conditions

### Types Available

**1. No-Slip**: Fluid velocity = wall velocity (usually zero)
```
Wall:  u = 0, v = 0, w = 0
Used for: Solid walls, viscous flows
```

**2. Free-Slip**: No friction at wall
```
Wall: tangential velocity free, normal velocity = 0
Used for: Frictionless walls, symmetry planes
```

**3. Stress-Free**: No stress at boundary
```
Wall: ∂u/∂z = 0, ∂v/∂z = 0, w = 0
Used for: Free surfaces, far-field boundaries
```

**4. Periodic**: Wraps around
```
u(x=0) = u(x=Lx)
Used for: Horizontal directions in atmospheric flows
```

## Poisson Solver (Pressure Computation)

### Why We Need It

From Navier-Stokes, we get a Poisson equation for pressure:
```
∇²p = f(velocity)
```

This ensures the velocity field is divergence-free (incompressible):
```
∇·u = 0  (fluid doesn't compress)
```

### Multigrid Method (Fast Solver)

Instead of solving on one grid, we use multiple resolution levels:

```
Fine grid:    ████████████████  (accurate but slow)
                  ↓↑
Medium grid:  ████████          (faster, approximate)
                  ↓↑
Coarse grid:  ████              (very fast, rough)
```

**Process**:
1. Start on fine grid
2. Can't solve quickly? Go to coarser grid
3. Solve on coarse grid (fast)
4. Interpolate back to fine grid
5. Repeat until converged

```julia
# Using multigrid
mg_plan = make_mg_poisson_distributed_auto(Nx, Ny, Nz, grid, comm)
mg_solve_distributed!(p, rhs, mg_plan)  # Fast!
```

## Data Layout and Performance

### Cache-Friendly Access

```julia
# ✓ Good: Access in memory order
for k in 1:Nz
    for j in 1:Ny
        for i in 1:Nx  # Innermost loop on contiguous dimension
            u[i,j,k] = ...
        end
    end
end

# ❌ Bad: Jumping around in memory
for i in 1:Nx
    for j in 1:Ny
        for k in 1:Nz  # Random access pattern
            u[i,j,k] = ...
        end
    end
end
```

### Memory Usage Estimation

For a simulation with grid `Nx × Ny × Nz`:

```julia
# Per field (velocity component, pressure, etc.):
memory_per_field = Nx * Ny * Nz * 8 bytes  # Float64

# Typical simulation needs ~10-20 fields:
total_memory = 15 * Nx * Ny * Nz * 8 bytes

# Example: 128³ grid
# 15 * 128³ * 8 bytes = ~400 MB
```

## Putting It All Together

A typical PencilFlows simulation:

```julia
# 1. Setup
decomp = init_pencil_decomposition(Nx, Ny, Nz)
fields = create_distributed_fields(decomp)
poisson = make_mg_poisson_distributed_auto(...)

# 2. Initialize
initialize_velocity!(fields, initial_condition)

# 3. Time loop
for step in 1:nsteps
    # Compute derivatives (using pencil transposes)
    compute_horizontal_derivatives_2d!(...)
    compute_z_derivatives_2d!(...)

    # Solve for pressure (using multigrid)
    mg_solve_distributed!(p, rhs, poisson)

    # Advance in time (using IMEX)
    time_step!(fields, dt, stepper)

    # Output
    if step % output_freq == 0
        write_state(...)
    end
end
```

## Next Topics

- [Boundary Conditions](boundary_conditions.md) - Detailed BC implementation
- [MPI Parallelization](mpi.md) - Using multiple processors
- [Poisson Solvers](poisson.md) - Deep dive into pressure solvers
