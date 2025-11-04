# Spatial Fields Guide for PencilFlows.jl

## Overview

PencilFlows.jl now supports spatially-varying, time-independent velocity fields and stratification/temperature fields. These can be functions of spatial coordinates (x, y, z) and are useful for:

- Initial conditions with realistic profiles
- Background flows (e.g., channel flow, atmospheric boundary layers)
- Stratification profiles (linear, exponential, ocean thermoclines)
- Forcing terms in the equations

## Quick Start

```julia
using PencilFlows

# Create a parabolic channel flow
channel_flow = create_channel_flow(1.0, 1.0; profile=:parabolic)

# Create linear stratification
NÂ² = 1.0
linear_strat = StratificationField(constant_stratification(NÂ²), :temperature)

# Evaluate at specific points
u, v, w = evaluate_spatial_field(channel_flow, 0.0, 0.0, 0.5)
T = evaluate_spatial_field(linear_strat, 0.0, 0.0, 0.5)
```

## Core Types

### VelocityField

Represents a spatially-varying velocity field U(x,y,z) = (u(x,y,z), v(x,y,z), w(x,y,z)).

```julia
# Create from a function
velocity_func = (x, y, z) -> (sin(z), 0.0, 0.0)  # Returns (u, v, w)
vel_field = VelocityField(velocity_func, :u, "Sinusoidal u-velocity")

# Evaluate at coordinates
u, v, w = evaluate_spatial_field(vel_field, 1.0, 2.0, 0.5)
```

### StratificationField

Represents a spatially-varying stratification/temperature field T(x,y,z) or Ï(x,y,z).

```julia
# Create from a function
strat_func = (x, y, z) -> -0.01 * z  # Linear temperature profile
strat_field = StratificationField(strat_func, :temperature, "Linear stratification")

# Evaluate at coordinates
T = evaluate_spatial_field(strat_field, 0.0, 0.0, 0.5)
```

## Predefined Velocity Profiles

### 1. Linear Shear

```julia
# U(z) = U‚ + dU_dz * z
vel_field = VelocityField(linear_shear(1.0, 0.5; direction=:x), :u)
```

### 2. Quadratic Profile

```julia
# U(z) = U‚ + a*z + b*zÂ²
vel_field = VelocityField(quadratic_profile(0.5, 1.0, -0.5; direction=:x), :u)
```

### 3. Exponential Profile

```julia
# U(z) = U‚ * exp(-decay_rate * (z - z‚))
vel_field = VelocityField(exponential_profile(1.0, 2.0, 0.0; direction=:x), :u)
```

### 4. Sinusoidal Field

```julia
# 1D: U(z) = A * sin(k*z + Ï†)
vel_field = VelocityField(sinusoidal_field(0.1, 2Ï; direction=:x), :u)

# 3D: U(x,y,z) = A * sin(kx*x + ky*y + kz*z + Ï†)
k = (Ï, 2Ï, Ï/2)  # (kx, ky, kz)
vel_field = VelocityField(sinusoidal_field(0.1, k; direction=:x, phase=Ï/4), :u)
```

### 5. Channel Flow Profiles

```julia
# Parabolic: u(z) = U_max * 4 * (z/H) * (1 - z/H)
parabolic_flow = create_channel_flow(2.0, 1.0; profile=:parabolic)

# Linear shear: u(z) = U_max * (2z/H - 1)
linear_flow = create_channel_flow(2.0, 1.0; profile=:linear)
```

### 6. Atmospheric Boundary Layer

```julia
# Logarithmic profile: u(z) = u_ref * ln(z/z‚) / ln(z_ref/z‚)
atm_profile = create_atmospheric_profile(10.0, 10.0, 0.1; profile=:logarithmic)

# Power law: u(z) = u_ref * (z/z_ref)^Î
atm_profile = create_atmospheric_profile(10.0, 10.0, 0.1; profile=:power_law, Î=0.1)
```

## Predefined Stratification Profiles

### 1. Constant Stratification

```julia
# Linear temperature: T(z) = -NÂ² * z
NÂ² = 1.0
strat_field = StratificationField(constant_stratification(NÂ²), :temperature)
```

### 2. Linear Stratification

```julia
# NÂ²(z) = NÂ²‚ + dNÂ²_dz * z †’ T(z) = -NÂ²‚ * z - 0.5 * dNÂ²_dz * zÂ²
strat_field = StratificationField(linear_stratification(1.0, 0.1), :temperature)
```

### 3. Exponential Stratification

```julia
# NÂ²(z) = NÂ²‚ * exp(-(z-z‚)/decay_scale)
strat_field = StratificationField(exponential_stratification(0.01, 30.0), :temperature)
```

### 4. Tanh Stratification (Thermocline)

```julia
# Smooth transition between two stratification values
strat_field = StratificationField(tanh_stratification(0.01, 0.2, 0.5), :temperature)
```

### 5. Ocean Stratification

```julia
# Realistic ocean profile with mixed layer and thermocline
ocean_strat = create_ocean_stratification(0.0001, 0.001, 50.0, 20.0)
```

## Integration with PencilFlows Simulations

### 1. Initialize Fields

```julia
# Initialize velocity components with spatial fields
initialize_velocity_field!(u, v, w, velocity_field, grid)

# Initialize stratification/temperature
initialize_stratification_field!(b, stratification_field, grid)
```

### 2. Add Forcing Terms

Use spatial fields as forcing terms in the momentum and buoyancy equations:

```julia
# In momentum_rhs! calls
momentum_rhs!(Ru, Rv, Rw, u, v, w, b;
              Î½=Î½, fplane=fplane,
              fields=fields, decomp=decomp, grid=grid, bc=bc,
              nlin_ws=nlin_ws, ws=ws,
              velocity_forcing=background_flow,
              forcing_strength=0.1)

# In buoyancy_rhs! calls  
buoyancy_rhs!(Rb, u, v, w, b;
              Îº=Îº, N2=N2,
              fields=fields, decomp=decomp, grid=grid, bc=bc,
              ws=ws,
              stratification_forcing=background_stratification,
              stratification_strength=0.01)
```

### 3. Time-Stepping with Spatial Fields

```julia
# Example time-stepping loop
for step in 1:nsteps
    predictor_corrector_step!(u, v, w, b, prob_params...,
                             velocity_forcing=channel_flow,
                             forcing_strength=restoring_strength,
                             stratification_forcing=linear_strat,
                             stratification_strength=strat_strength)
end
```

## Profile Generation and Analysis

### Generate Data for Plotting

```julia
# Generate profile data
z_range = 0.0:0.01:1.0
z_coords, u_values = plot_field_profile(velocity_field, z_range; component=:u)
z_coords, T_values = plot_field_profile(stratification_field, z_range)

# Use with your favorite plotting package
using Plots
plot(u_values, z_coords, xlabel="u(z)", ylabel="z")
plot(T_values, z_coords, xlabel="T(z)", ylabel="z")
```

### Field Statistics

```julia
# Get field information
field_stats = compute_field_statistics(velocity_field, grid)
println("Field description: $(field_stats[:description])")
println("Field type: $(field_stats[:type])")
```

## Custom Field Functions

### Create Your Own Velocity Field

```julia
# Define a custom function f(x,y,z) †’ (u,v,w)
custom_func = function(x, y, z)
    u = x * cos(Ï * z)
    v = y * sin(Ï * z)  
    w = 0.1 * sin(2Ï * z)
    return (u, v, w)
end

custom_field = VelocityField(custom_func, :uvw, "Custom 3D velocity field")
```

### Create Your Own Stratification Field

```julia
# Define a custom stratification function f(x,y,z) †’ T
custom_strat_func = (x, y, z) -> begin
    # Example: stratification varies with horizontal position
    base_strat = -z  # Linear background
    perturbation = 0.1 * sin(2Ï * x / Lx) * exp(-z/H)
    return base_strat + perturbation
end

custom_strat = StratificationField(custom_strat_func, :temperature, "Horizontally-varying stratification")
```

## Example: Rayleigh-BÃ©nard with Background Flow

```julia
using PencilFlows

# Problem setup
Nx, Ny, Nz = 128, 128, 64
Lx, Ly, H = 4Ï, 4Ï, 1.0

# Create background shear flow
shear_flow = VelocityField(linear_shear(0.1, 1.0; direction=:x), :u, "Background shear")

# Create unstable stratification  
Ra = 1e6
unstable_strat = StratificationField(constant_stratification(-Ra), :buoyancy, "Unstable stratification")

# Initialize simulation arrays...
# (grid setup, decomposition, etc.)

# Initialize with spatial fields
initialize_velocity_field!(u, v, w, shear_flow, grid)
initialize_stratification_field!(b, unstable_strat, grid)

# Time stepping with background flow maintained
restoring_strength = 0.01
for step in 1:1000
    predictor_corrector_step!(u, v, w, b, ...,
                             velocity_forcing=shear_flow,
                             forcing_strength=restoring_strength)
end
```

## Best Practices

1. **Field Evaluation**: Test your field functions at a few points before using in simulations
2. **Smooth Profiles**: Use smooth functions to avoid numerical issues
3. **Boundary Compatibility**: Ensure your spatial fields are compatible with boundary conditions
4. **Forcing Strength**: Start with small forcing strengths to avoid numerical instabilities
5. **Profile Verification**: Always plot your profiles to verify they look correct

## Performance Notes

- Spatial field evaluation is done at every grid point during initialization/forcing
- For performance-critical applications, consider pre-computing fields on arrays
- SIMD-friendly functions will be faster for large grids
- The forcing terms are applied after the main PDE terms, so they represent source/sink terms

## Troubleshooting

### Common Issues

1. **Field not applied**: Check that forcing_strength > 0
2. **Wrong component**: Ensure VelocityField component matches usage (:u, :v, :w)
3. **Coordinate mismatch**: Verify grid coordinates match field function expectations
4. **Boundary conflicts**: Spatial fields may conflict with imposed boundary conditions

### Debug Tips

```julia
# Test field evaluation
test_points = [(0.0, 0.0, 0.0), (Lx/2, Ly/2, H/2), (Lx, Ly, H)]
for (x, y, z) in test_points
    u, v, w = evaluate_spatial_field(vel_field, x, y, z)
    println("At ($x, $y, $z): u=$u, v=$v, w=$w")
end

# Check field statistics
field_stats = compute_field_statistics(field, grid)
```

This completes the spatial fields functionality for PencilFlows.jl! The system now supports flexible, spatially-varying velocity and stratification fields that can be used for realistic initial conditions and background forcing.
