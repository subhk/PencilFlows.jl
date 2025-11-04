# Automatic Parameter Conversion Examples

This folder contains examples demonstrating PencilFlows.jl's automatic parameter conversion system.

##  **What is Automatic Parameter Conversion?**

Instead of manually computing physical parameters like viscosity (ν) from dimensionless numbers like Reynolds number (Re), PencilFlows.jl automatically:

1. **Analyzes your equations** to identify dimensionless parameters
2. **Converts them to physical parameters** using standard formulas
3. **Integrates converted parameters** throughout all solver components
4. **Optimizes solver settings** based on parameter regimes

##  **Examples**

### `simple_reynolds_example.jl`
- **What**: Basic Reynolds number �� viscosity conversion
- **Physics**: 2D incompressible Navier-Stokes flow
- **Conversion**: `ν = U_ref × L_ref / Re`
- **Application**: General fluid dynamics

### `ekman_rotating_example.jl`  
- **What**: Ekman number �� viscosity in rotating flows
- **Physics**: Rotating Navier-Stokes with Coriolis effects
- **Conversion**: `ν = Ek × f × L²`
- **Application**: Geophysical flows (ocean/atmosphere)

### `thermal_convection_example.jl`
- **What**: Multi-parameter conversion (Re, Ra, Pr)
- **Physics**: Thermal convection with buoyancy
- **Conversions**: `ν = UL/Re`, `κ = ν/Pr`, thermal coupling
- **Application**: Heat transfer, atmospheric convection

##  **Usage Pattern**

All examples follow the same simple pattern:

```julia
# 1. Write equations with dimensionless parameters
equations = [
    "dt(u) = -u*dx(u) - dx(p) + (1/Re)*lap(u)",  # Re appears here
    "dx(u) = 0"
]

# 2. Solve automatically (PencilFlows.jl handles all conversions)
solution = quick_solve(equations, Re=1000.0, dt=0.001)
```

##  **Supported Conversions**

| Input Parameter | Physical Parameter | Formula | Application |
|----------------|-------------------|---------|-------------|
| Re (Reynolds) | ν (viscosity) | `ν = UL/Re` | All flows |
| Ek (Ekman) | ν (viscosity) | `ν = Ek×f×L²` | Rotating flows |
| Pr (Prandtl) | κ (thermal diff.) | `κ = ν/Pr` | Heat transfer |
| Ra (Rayleigh) | Thermal coupling | Validates Ra | Convection |
| Ro (Rossby) | f (Coriolis) | `f = U/(Ro×L)` | Geophysical |
| Ri (Richardson) | N² (stratification) | Stability analysis | Stratified flows |

##  **Benefits**

- **Zero Setup**: Write equations �� get solver
- **No Manual Conversion**: All parameter math is automatic  
- **Consistent Parameters**: Same values used throughout solver stack
- **Regime Optimization**: Solver adapts to parameter values
- **Error Prevention**: No unit mistakes or conversion errors

##  **Running Examples**

```bash
# Run individual examples
julia simple_reynolds_example.jl
julia ekman_rotating_example.jl 
julia thermal_convection_example.jl

# Or run all examples
julia -e 'for f in readdir("."); f[end-2:end] == ".jl" && include(f); end'
```

## � **Learn More**

- See `../demos/demo_integrated_system.jl` for complete demonstrations
- Check `../docs/development/` for technical documentation
- Look at `../tests/` for validation and testing examples
