# Core Functions

## Grid Setup

`Grid`

Creates a computational grid for the simulation domain.

**Arguments:**
- `Nx, Ny, Nz`: Number of grid points in each direction
- `Lx, Ly, H`: Physical dimensions
- `dz`: (Optional) Custom vertical grid spacing for non-uniform grids

**Example:**
```julia
# Uniform grid
grid = Grid(64, 64, 32, 2π, 2π, 1.0)

# Non-uniform grid (finer near boundaries)
dz_custom = create_boundary_layer_grid(32, 1.0, 0.1)
grid = Grid(64, 64, 32, 2π, 2π, 1.0; dz=dz_custom)
```

## Pencil Decomposition

### `init_pencil_decomposition`

Initializes 2D pencil decomposition for parallel computing.

**Arguments:**
- `Nx, Ny, Nz`: Grid dimensions
- `P1, P2`: (Optional) Process grid layout

**Returns:** `PencilDecomposition` object

**Example:**
```julia
# Automatic process grid selection
decomp = init_pencil_decomposition(128, 128, 64)

# Manual process grid (e.g., 2×4 = 8 processes)
decomp = init_pencil_decomposition(128, 128, 64; P1=2, P2=4)
```

### `create_distributed_fields`

Creates distributed arrays for velocity, pressure, and working storage.

**Arguments:**
- `decomp`: PencilDecomposition object

**Returns:** Named tuple with field arrays

**Example:**
```julia
fields = create_distributed_fields(decomp)
# Access fields:
# fields.u_z, fields.v_z, fields.w_z  (velocity in Z-pencils)
# fields.u_x, fields.v_x, fields.w_x  (velocity in X-pencils)
# fields.u_hat_x, ...                 (spectral coefficients)
```

## Derivatives

### `compute_horizontal_derivatives_2d!`

Computes horizontal (x, y) derivatives using FFTs.

**Example:**
```julia
compute_horizontal_derivatives_2d!(
    dudx, dudy, dvdx, dvdy, dwdx, dwdy, dpdx, dpdy,
    u_z, v_z, w_z, p_z,
    fields, decomp, grid
)
```

### `compute_z_derivatives_2d!`

Computes vertical (z) derivatives using finite differences.

**Example:**
```julia
compute_z_derivatives_2d!(
    dudz, dvdz, dwdz, dpdz,
    u_z, v_z, w_z, p_z,
    decomp, grid, bc
)
```

## Transform Operations

Basic derivative operations for testing and simple cases. Use the helper
functions `ddx!`, `ddy!`, `d2dx2!`, and `d2dy2!` to compute first and second
derivatives along the horizontal directions.

**Example:**
```julia
# First derivative in x
ddx!(df, f, plan)

# Second derivative in x
d2dx2!(d2f, f, plan)
```

## Boundary Conditions

### `BoundaryCondition`

Defines boundary conditions at top and bottom walls.

**Constructor:**
```julia
BoundaryCondition(bottom_type, top_type)
```

**Types:**
- `NO_SLIP`: u = v = w = 0
- `FREE_SLIP`: ∂u/∂z = ∂v/∂z = 0, w = 0
- `STRESS_FREE`: ∂u/∂z = ∂v/∂z = ∂w/∂z = 0
- `PERIODIC_Z`: Wrap-around
- `ROBIN`: Custom linear combination

**Example:**
```julia
# No-slip bottom, free-slip top
bc = BoundaryCondition(NO_SLIP, FREE_SLIP)

# Time-dependent prescribed velocity
bc = BoundaryCondition(
    PRESCRIBED_VELOCITY, FREE_SLIP,
    bottom_vel = (t -> sin(2π*t), 0.0, 0.0),  # (u, v, w)
    top_vel = (0.0, 0.0, 0.0)
)
```

## Utility Functions

### `compute_kinetic_energy`

Computes total kinetic energy.

```julia
KE = compute_kinetic_energy(u, v, w)
# Returns: 0.5 * ∫(u² + v² + w²) dV
```

### `compute_rms_field`

Root-mean-square of a field.

```julia
u_rms = compute_rms_field(u)
# Returns: √(⟨u²⟩)
```

### `compute_mean_field`

Spatial average.

```julia
u_mean = compute_mean_field(u)
# Returns: ⟨u⟩
```
