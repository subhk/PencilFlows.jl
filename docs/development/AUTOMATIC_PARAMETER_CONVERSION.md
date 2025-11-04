# Automatic Parameter Conversion in PencilFlows.jl

## Overview

PencilFlows.jl now features **intelligent automatic parameter conversion** that analyzes your symbolic equations, identifies constants vs variables, and builds the complete problem setup automatically. No manual parameter configuration required!

## How It Works

### 1. **Equation Analysis** 
The system automatically parses your equations to identify:
- **Variables**: Physical fields (u, v, w, p, T, b, etc.)
- **Constants**: Physical parameters (Re, Pr, Ra, ν, f, N2, etc.) 
- **Operators**: Differential operators (dt, dx, lap, div, etc.)
- **Physics Type**: Navier-Stokes, thermal convection, rotating flow, etc.

### 2. **Automatic Parameter Conversion**
Based on detected constants, the system automatically:
- Converts between different parameter representations
- Computes missing physical parameters
- Sets appropriate default values
- Validates parameter consistency

### 3. **Complete Problem Setup**
The system automatically configures:
- Computational domain and discretization
- Boundary conditions based on physics
- Optimal solver selection 
- Time stepping schemes

## Usage Examples

### Example 1: Rayleigh-Bénard Thermal Convection

```julia
using PencilFlows

# Just write your physics equations!
equations = [
    "dt(u) = -u*dx(u) - w*dz(u) - dx(p) + (1/Re)*lap(u)",
    "dt(w) = -u*dx(w) - w*dz(w) - dz(p) + (1/Re)*lap(w) + Ra*Pr*T",
    "dt(T) = -u*dx(T) - w*dz(T) + (1/(Re*Pr))*lap(T)",
    "dx(u) + dz(w) = 0"
]

# Automatic analysis and problem setup - no manual work!
prob = build_problem_from_equations(equations)

# System automatically detects:
# - Variables: u, w, T, p
# - Constants: Re, Ra, Pr  
# - Physics: Thermal convection
# - Sets up: 2D domain, thermal BCs, pressure projection

solution = solve!(prob, dt=0.001, max_iter=10000)
```

### Example 2: Rotating Stratified Ocean Flow  

```julia
# 3D rotating stratified equations
equations = [
    "dt(u) = -u*dx(u) - v*dy(u) - w*dz(u) - dx(p) + nu*lap(u) + f*v",
    "dt(v) = -u*dx(v) - v*dy(v) - w*dz(v) - dy(p) + nu*lap(v) - f*u",
    "dt(w) = -u*dx(w) - v*dy(w) - w*dz(w) - dz(p) + nu*lap(w) + b", 
    "dt(b) = -u*dx(b) - v*dy(b) - w*dz(b) + kappa*lap(b) + N2*w",
    "dx(u) + dy(v) + dz(w) = 0"
]

prob = build_problem_from_equations(equations)

# System automatically detects:
# - Variables: u, v, w, p, b
# - Constants: nu, f, kappa, N2
# - Physics: 3D rotating stratified Boussinesq
# - Computes: Re = UL/nu, Ek = nu/(f*L²), Pr = nu/kappa, etc.

solution = solve!(prob, dt=0.01, max_iter=5000)
```

### Example 3: Multiple Parameter Specification Methods

The system handles different ways of specifying the same physics:

```julia
# Method 1: Reynolds number
equations_Re = ["dt(u) = -u*dx(u) - dx(p) + (1/Re)*lap(u)"]
# System computes: ν = UL/Re

# Method 2: Direct viscosity  
equations_nu = ["dt(u) = -u*dx(u) - dx(p) + nu*lap(u)"]
# System computes: Re = UL/ν

# Method 3: Ekman number (rotating flows)
equations_Ek = ["dt(u) = -u*dx(u) - dx(p) + Ek*f*L2*lap(u) + f*v"]
# System computes: ν = Ek × f × L²

# All methods produce equivalent setups automatically!
```

## Supported Parameter Conversions

### **Reynolds Number ↔ Viscosity**
- `Re = UL/ν` ↔ `ν = UL/Re`
- Automatic bidirectional conversion

### **Ekman Number ↔ Viscosity** (Rotating Flows)
- `Ek = ν/(fL²)` ↔ `ν = Ek × f × L²`
- Requires rotation parameter f

### **Prandtl Number ↔ Thermal Diffusivity**
- `Pr = ν/κ` ↔ `κ = ν/Pr`
- Links momentum and thermal diffusion

### **Rayleigh Number** (Thermal Convection)
- `Ra = gΔTL³/(νκ)`
- Automatically computed from thermal parameters

### **Rossby Number** (Rotating Flows) 
- `Ro = U/(fL)`
- Measures importance of rotation

### **Richardson Number** (Stratified Flows)
- `Ri = N²L²/U²`
- Measures stratification effects

## Automatic Physics Detection

The system infers physics type and characteristics:

### **Equation Types Detected:**
- **Incompressible Navier-Stokes**: Pressure + divergence-free constraint
- **Thermal Convection**: Temperature coupling with buoyancy
- **Rotating Flow**: Coriolis terms present
- **Stratified Flow**: Buoyancy/density coupling
- **Magnetohydrodynamics**: Magnetic field equations (future)

### **Solver Selection:**
- **Navier-Stokes** → Predictor-corrector with pressure projection
- **Thermal Convection** → Add thermal coupling and boundary conditions
- **Rotating Flow** → Rotation-aware CFL constraints
- **High Re** → Consider semi-implicit methods

### **Boundary Conditions:**
- **Velocity**: No-slip (u=v=w=0) at walls
- **Temperature**: Hot/cold walls for convection
- **Pressure**: Neumann conditions for projection
- **Periodic**: Automatic detection from domain

## Advanced Features

### **Parameter Override**
```julia
equations = [...]

prob = build_problem_from_equations(
    equations,
    parameter_overrides=Dict(
        :Re => 10000.0,    # High Reynolds number
        :Pr => 0.7,        # Air properties
        :Ra => 1e6         # Supercritical convection
    )
)
```

### **Analysis Without Building**
```julia
# Just analyze equations without building problem
analysis = analyze_and_suggest(equations)

# Provides parameter suggestions and physics insights
```

### **Ultra-Quick Solve**
```julia
# Analyze, build, and solve in one call
solution, prob = quick_solve(equations, dt=0.001, max_iter=1000)
```

## Real Applications

### **Oceanic Mesoscale Dynamics**
```julia
equations = [
    "dt(u) = -u*dx(u) - v*dy(u) - w*dz(u) - dx(p) + (1/Re)*lap(u) + (1/Ro)*v",
    "dt(v) = -u*dx(v) - v*dy(v) - w*dz(v) - dy(p) + (1/Re)*lap(v) - (1/Ro)*u", 
    "dt(w) = -u*dx(w) - v*dy(w) - w*dz(w) - dz(p) + (1/Re)*lap(w) + Ri*b",
    "dt(b) = -u*dx(b) - v*dy(b) - w*dz(b) + (1/(Re*Pr))*lap(b) + N2*w"
]
# Typical values: Re~10^4, Ro~0.1, Ri~0.25, Pr~7
```

### **Atmospheric Convection**
```julia
equations = [
    "dt(u) = -u*dx(u) - w*dz(u) - dx(p) + (1/Re)*lap(u)",
    "dt(w) = -u*dx(w) - w*dz(w) - dz(p) + (1/Re)*lap(w) + Ra*Pr*T",
    "dt(T) = -u*dx(T) - w*dz(T) + (1/(Re*Pr))*lap(T)"
]
# Typical values: Ra~10^6, Re~10^4, Pr~0.7
```

### **Laboratory Rotating Tank**
```julia
equations = [
    "dt(u) = -u*dx(u) - dx(p) + Ek*Omega*L2*lap(u) + Omega*v",
    "dt(v) = -u*dx(v) - dy(p) + Ek*Omega*L2*lap(v) - Omega*u"
]
# System computes: f = 2Ω, ν = Ek × f × L²
```

## Benefits

### **For Users:**
- **No Manual Setup**: Just write physics equations
- **Flexible Input**: Multiple equivalent parameter specifications
- **Automatic Conversion**: Seamless parameter transformations
- **Physics-Aware**: Intelligent defaults and validation
- **Production Ready**: Complete solver setup automatically

### **For Researchers:**
- **Rapid Prototyping**: Focus on physics, not implementation
- **Parameter Studies**: Easy to vary dimensionless numbers
- **Reproducible**: Consistent parameter handling across studies
- **Educational**: Clear connection between equations and numerics

### **For Applications:**
- **Geophysical Flows**: Ocean/atmosphere dynamics
- **Engineering**: Heat transfer, mixing, turbulence
- **Astrophysical**: Stellar/planetary fluid dynamics
- **Laboratory**: Experimental validation studies

## Summary

PencilFlows.jl's automatic parameter conversion system transforms equation-based modeling:

**Before**: Manual parameter setup, solver configuration, BC specification
**After**: Write equations → Get complete working solver

This makes PencilFlows.jl accessible to researchers who want to focus on physics rather than numerical implementation details, while providing the flexibility and performance needed for production scientific computing.
