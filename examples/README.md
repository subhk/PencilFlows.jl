# PencilFlows.jl Examples

This directory contains comprehensive examples demonstrating the capabilities of PencilFlows.jl, a Julia package for computational fluid dynamics with pencil decomposition.

## Overview

The examples showcase various aspects of the package:

- **Codebase Consistency**: All examples work reliably with no breaking points
- **Type Stability**: Optimized performance through proper type handling
- **Spatial Fields**: Flexible velocity and stratification field definitions
- **Boundary Conditions**: Robust boundary condition handling with time-dependent support
- **Solver Integration**: Full integration between all solver components

## Example Files

### Core Demonstrations

- **`comprehensive_demo.jl`** - Complete showcase of all PencilFlows.jl functionality
  - Codebase consistency verification
  - Spatial fields demonstration
  - Boundary conditions examples  
  - Solver components integration
  - Parameter handling
  - Example scenarios

### Specialized Examples

- **`universal_pde_examples.jl`** - Universal PDE solver demonstrations
  - Heat equations
  - Reaction-diffusion systems
  - Navier-Stokes with thermal convection
  - Rotating stratified flows
  - Wave equations
  - Magnetohydrodynamics

- **`pressure_poisson_derivation_demo.jl`** - Pressure-Poisson equation derivation
- **`time_dependent_velocity_bc_demo.jl`** - Time-dependent boundary conditions
- **`time_dependent_poisson_bc_demo.jl`** - Time-dependent Poisson boundaries

### Automatic Parameter Conversion

The `automatic_conversion/` directory contains examples of automatic parameter detection:

- **`simple_reynolds_example.jl`** - Basic Reynolds number analysis
- **`thermal_convection_example.jl`** - Rayleigh-Bénard convection
- **`ekman_rotating_example.jl`** - Rotating flow dynamics

### Advanced Features

- **`equation_rearrangement_examples.jl`** - Symbolic equation manipulation
- **`equation_form_analysis.jl`** - PDE structure analysis
- **`linear_left_demo.jl`** - Linear operator demonstrations
- **`quick_rearrangement_demo.jl`** - Fast equation processing

## Running Examples

### Basic Usage

```bash
# Run the comprehensive demo
julia examples/comprehensive_demo.jl

# Run universal PDE examples
julia examples/universal_pde_examples.jl

# Run specific demonstrations
julia examples/pressure_poisson_derivation_demo.jl
```

### Interactive Usage

```julia
# Load PencilFlows
include("src/PencilFlows.jl")
using .PencilFlows

# Include and run examples
include("examples/comprehensive_demo.jl")
main_demo()

# Or run specific functions
demo_spatial_fields()
demo_boundary_conditions()
```

## Key Features Demonstrated

### 1. Spatial Fields

```julia
# Linear shear velocity field
shear_func = linear_shear(1.0, 0.5, direction=:x)
velocity_field = VelocityField(shear_func, :uvw, "Channel flow profile")

# Constant stratification
strat_func = constant_stratification(1e-4)
strat_field = StratificationField(strat_func, :temperature, "Stable stratification")

# Evaluation at any point
u, v, w = evaluate_spatial_field(velocity_field, x, y, z)
temperature = evaluate_spatial_field(strat_field, x, y, z)
```

### 2. Boundary Conditions

```julia
# Constant boundary values
bc1 = BuoyancyBC(B_CONSTANT, B_FLUX, bottom_value=1.0, top_value=0.0)

# Time-dependent boundaries
temp_func(t) = 1.0 + 0.1 * sin(2π * t)
bc2 = BuoyancyBC(B_FUNCTION, B_CONSTANT, bottom_value=temp_func, top_value=0.5)

# Robin boundary conditions
bc3 = BuoyancyBC(B_ROBIN, B_ROBIN,
                bottom_alpha=1.0, bottom_beta=0.1, bottom_gamma=0.0,
                top_alpha=1.0, top_beta=0.1, top_gamma=1.0)
```

### 3. Solver Integration

```julia
# Create workspace
workspace = RSNSWorkspace(decomp, grid, proto_array)

# Predictor-corrector step
predictor_corrector_step!(u, v, w, b, p, workspace, dt, params...)

# Pressure projection
solve_poisson!(pressure, divergence, poisson_plan)
```

### 4. Parameter Systems

```julia
# Physical parameters
params = Dict(
    :Re => 1000.0,    # Reynolds number
    :Ra => 1e6,       # Rayleigh number  
    :Pr => 0.7,       # Prandtl number
    :f => 1e-4,       # Coriolis parameter
    :N2 => 1e-5       # Stratification strength
)
```

## Example Scenarios

### 1. Thermal Convection (Rayleigh-Bénard)

- Boussinesq equations with temperature field
- Heated bottom, cooled top boundaries
- Buoyancy-driven convection cells
- Parameters: Ra, Pr, aspect ratio

### 2. Rotating Stratified Flow

- Geophysical fluid dynamics
- Coriolis effects and stable stratification
- Oceanic/atmospheric applications
- Parameters: f (rotation), N² (stratification)

### 3. Channel Flow with Forcing

- Parabolic base flow profile
- Spatial forcing functions
- No-slip wall boundaries
- Transition to turbulence

### 4. Time-Dependent Boundaries

- Oscillating temperature boundaries
- Tidal forcing
- Seasonal variations
- Function-based boundary specifications

## Requirements

- Julia 1.6+
- PencilFlows.jl package
- Dependencies: LinearAlgebra, FFTW, MPI (for parallel execution)

## Notes on Improvements

The examples benefit from recent codebase improvements:

1. **No Breaking Points**: All examples run reliably without compilation errors
2. **Type Stability**: Better performance through proper Union types in boundary conditions
3. **Unicode Fixes**: Mathematical symbols properly handled throughout
4. **Function Integration**: Seamless operation between spatial fields, boundary conditions, and solvers
5. **Comprehensive Testing**: Examples serve as integration tests for the entire system

## Contributing

When adding new examples:

1. Follow the established pattern of comprehensive documentation
2. Include parameter descriptions and physical interpretation
3. Demonstrate integration between multiple components
4. Add error handling for educational purposes
5. Test examples with the full package to ensure consistency

## Support

For questions about examples or the PencilFlows.jl package:

- Check the main documentation
- Review existing examples for patterns
- Refer to the comprehensive demo for complete workflows
- Submit issues for bugs or enhancement requests