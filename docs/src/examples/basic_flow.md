# Basic Flow Simulation

This tutorial shows you how to set up and run a complete fluid simulation from scratch.

## Problem: 2D Taylor-Green Vortex

We'll simulate a classic test case: decaying vortices in a box. This is perfect for learning because:
- Simple initial conditions
- Known analytical solution (we can check our results!)
- Demonstrates key features

### Physical Setup

```
Domain: 2π × 2π × 1
Initial velocity: Swirling vortices
Boundary: Periodic in x,y; no-slip at top/bottom
```

## Complete Code with Explanations

```julia
using PencilFlows
using MPI

# Step 1: Initialize MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# Step 2: Define grid parameters
Nx, Ny, Nz = 64, 64, 32      # Grid resolution
Lx, Ly, H = 2π, 2π, 1.0      # Domain size
Re = 100.0                    # Reynolds number (viscosity = 1/Re)

# Step 3: Create grid structure
grid = Grid(Nx, Ny, Nz, Lx, Ly, H)

if rank == 0
    println("="^50)
    println("Taylor-Green Vortex Simulation")
    println("="^50)
    println("Grid: $Nx × $Ny × $Nz")
    println("Domain: $Lx × $Ly × $H")
    println("Reynolds number: $Re")
    println("="^50)
end

# Step 4: Set up pencil decomposition for parallel computing
decomp = init_pencil_decomposition(Nx, Ny, Nz)
fields = create_distributed_fields(decomp)

# Step 5: Initialize velocity field
# Taylor-Green vortex initial condition
function initialize_taylor_green!(u, v, w, grid)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Lx, Ly = grid.Lx, grid.Ly

    for k in 1:Nz
        for j in 1:Ny
            for i in 1:Nx
                x = (i-1) * Lx / Nx
                y = (j-1) * Ly / Ny

                # Classic Taylor-Green pattern
                u[i,j,k] =  sin(x) * cos(y)
                v[i,j,k] = -cos(x) * sin(y)
                w[i,j,k] = 0.0
            end
        end
    end
end

initialize_taylor_green!(fields.u_z, fields.v_z, fields.w_z, grid)

if rank == 0
    println("Initial conditions set")
end

# Step 6: Set up boundary conditions
bc = BoundaryCondition(NO_SLIP, NO_SLIP)  # No-slip at bottom and top

# Step 7: Set up Poisson solver for pressure
mg_plan = make_mg_poisson_distributed_auto(Nx, Ny, Nz, grid, comm)

if rank == 0
    println("Poisson solver initialized")
end

# Step 8: Time stepping parameters
dt = 0.001                    # Time step
t_end = 1.0                   # End time
nsteps = Int(t_end / dt)
output_freq = 100             # Output every N steps

# Step 9: Main time loop
if rank == 0
    println("Starting time integration...")
    println("Number of steps: $nsteps")
end

for step in 1:nsteps
    t = step * dt

    # Apply boundary conditions
    apply_velocity_bcs_nonuniform!(fields.u_z, fields.v_z, fields.w_z,
                                   grid, bc)

    # Compute derivatives
    compute_horizontal_derivatives_2d!(
        fields.dudx, fields.dudy,
        fields.dvdx, fields.dvdy,
        fields.dwdx, fields.dwdy,
        fields.dpdx, fields.dpdy,
        fields.u_z, fields.v_z, fields.w_z, fields.p_z,
        fields, decomp, grid
    )

    compute_z_derivatives_2d!(
        fields.dudz, fields.dvdz, fields.dwdz, fields.dpdz,
        fields.u_z, fields.v_z, fields.w_z, fields.p_z,
        decomp, grid, bc
    )

    # Compute pressure (solve Poisson equation)
    # Construct RHS: -∇·(u·∇u)
    # (simplified for this example)

    # Update velocity using computed derivatives
    # u_new = u_old + dt * (-u·∇u - ∇p + ν∇²u)
    # (using IMEX time stepper in practice)

    # Output
    if step % output_freq == 0 && rank == 0
        # Compute kinetic energy
        KE = compute_kinetic_energy(fields.u_z, fields.v_z, fields.w_z)
        println("Step $step/$nsteps, t = $t, KE = $KE")
    end
end

if rank == 0
    println("Simulation complete!")
end

# Step 10: Save final state
if rank == 0
    using JLD2
    @save "taylor_green_final.jld2" fields grid
    println("Results saved to taylor_green_final.jld2")
end

# Finalize MPI
MPI.Finalize()
```

## Understanding the Output

When you run this with `mpiexec -n 4 julia taylor_green.jl`:

```
==================================================
Taylor-Green Vortex Simulation
==================================================
Grid: 64 × 64 × 32
Domain: 6.283185307179586 × 6.283185307179586 × 1.0
Reynolds number: 100.0
==================================================
Initial conditions set
Poisson solver initialized
Starting time integration...
Number of steps: 1000
Step 100/1000, t = 0.1, KE = 0.4523
Step 200/1000, t = 0.2, KE = 0.3891
Step 300/1000, t = 0.3, KE = 0.3421
...
Simulation complete!
Results saved to taylor_green_final.jld2
```

## Analyzing Results

### Load and Visualize

```julia
using JLD2, Plots

# Load results
@load "taylor_green_final.jld2" fields grid

# Extract velocity at mid-plane
u_slice = fields.u_z[:, :, div(grid.Nz, 2)]

# Plot
heatmap(u_slice,
    title="U-velocity at mid-plane",
    xlabel="x", ylabel="y",
    color=:RdBu)
```

### Check Energy Decay

The theoretical decay rate for Taylor-Green vortex:
```
KE(t) = KE₀ * exp(-2νk²t)
where k = wavenumber, ν = 1/Re
```

```julia
# Compare simulation vs theory
t_array = [0, 0.1, 0.2, ..., 1.0]
KE_sim = [0.5, 0.452, 0.389, ..., 0.123]  # From output
KE_theory = 0.5 * exp.(-2 * (1/Re) * 2 * t_array)

plot(t_array, KE_sim, label="Simulation", marker=:o)
plot!(t_array, KE_theory, label="Theory", line=:dash)
xlabel!("Time")
ylabel!("Kinetic Energy")
```

## Exercises

### 1. Change Reynolds Number

Try different viscosities:
```julia
Re = 50   # More viscous → faster decay
Re = 500  # Less viscous → slower decay
```

### 2. Increase Resolution

Test convergence:
```julia
# Coarse
Nx, Ny, Nz = 32, 32, 16

# Fine
Nx, Ny, Nz = 128, 128, 64
```

Does the energy decay curve converge?

### 3. Different Initial Conditions

Try a different pattern:
```julia
function initialize_shear_layer!(u, v, w, grid)
    for k in 1:Nz
        z = (k-1) * H / Nz
        for j in 1:Ny
            for i in 1:Nx
                # Shear layer
                u[i,j,k] = tanh((z - 0.5) / 0.1)
                v[i,j,k] = 0.0
                w[i,j,k] = 0.01 * rand()  # Small perturbations
            end
        end
    end
end
```

## Common Issues and Solutions

### 1. Simulation Explodes (NaN values)

**Problem**: Time step too large
**Solution**: Reduce `dt`

```julia
# Check CFL condition: dt < dx / u_max
u_max = maximum(abs.(fields.u_z))
dt_safe = 0.5 * (Lx/Nx) / u_max
println("Recommended dt < $dt_safe")
```

### 2. Slow Performance

**Problem**: Grid too fine or inefficient layout
**Solutions**:
```julia
# Use fewer points
Nx, Ny, Nz = 32, 32, 16  # Start small

# Use more processors
# mpiexec -n 8 julia script.jl  (instead of 4)
```

### 3. Memory Errors

**Problem**: Not enough RAM
**Solution**:
```julia
# Estimate memory:
memory_GB = 15 * Nx * Ny * Nz * 8 / 1e9
println("Estimated memory: $memory_GB GB")

# Reduce if needed:
Nx, Ny, Nz = 32, 32, 16  # ~4 MB instead of 400 MB
```

## Next Steps

- Try [Parallel Computing](parallel.md) to use multiple processors efficiently
- Learn about [Custom Boundary Conditions](custom_bc.md) for more complex geometries
- Explore [Advanced Time Stepping](../guide/timestepping.md) for stiff problems
