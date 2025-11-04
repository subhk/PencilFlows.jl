# Universal PDE Solver - PencilFlows.jl

The Universal PDE Solver is a revolutionary interface for PencilFlows.jl that allows you to input **any system of PDEs as strings** and automatically analyzes, builds, and solves the problem with optimal numerical methods.

## ULTIMATE Key Features

- **Automatic Equation Analysis**: Input equations as strings, system analyzes structure automatically
- **Dynamic Problem Building**: Automatically sets up domain, discretization, and boundary conditions
- **Intelligent Solver Selection**: Chooses optimal time integration schemes based on equation structure
- **Multi-Physics Support**: Handles fluid dynamics, heat transfer, reaction-diffusion, waves, and more
- **Parameter Detection**: Automatically identifies and sets default values for physical parameters
- **Adaptive Configuration**: Adjusts numerical methods based on physics type and parameters

## BENEFITS Quick Start

### Basic Usage

```julia
using PencilFlows

# Define any system of PDEs as strings
equations = [
    "dt(u) = -u*dx(u) - v*dy(u) - dx(p) + (1/Re)*lap(u)",
    "dt(v) = -u*dx(v) - v*dy(v) - dy(p) + (1/Re)*lap(v)",
    "dx(u) + dy(v) = 0"
]

# Solve automatically!
solution, prob, analysis = solve_pde_system(equations, 
    parameters=Dict(:Re => 1000))
```

### Ultra-Quick Interface

```julia
# For rapid prototyping - minimal setup
solution, prob, analysis = quick_pde([
    "dt(T) = alpha*lap(T)",
    "dt(c) = D*lap(c) + k*T*c"
])
```

## EXAMPLES Examples

### 1. Navier-Stokes with Thermal Convection

```julia
equations = [
    "dt(u) = -u*dx(u) - v*dy(u) - w*dz(u) - dx(p) + (1/Re)*lap(u)",
    "dt(v) = -u*dx(v) - v*dy(v) - w*dz(v) - dy(p) + (1/Re)*lap(v)",
    "dt(w) = -u*dx(w) - v*dy(w) - w*dz(w) - dz(p) + (1/Re)*lap(w) + Ra*Pr*T",
    "dt(T) = -u*dx(T) - v*dy(T) - w*dz(T) + (1/Re/Pr)*lap(T)",
    "dx(u) + dy(v) + dz(w) = 0"
]

solution, prob, analysis = solve_pde_system(equations,
    parameters=Dict(:Re => 1000, :Ra => 1e6, :Pr => 0.7),
    time_span=50.0,
    domain=(4π, 4π, 1.0),
    resolution=(128, 128, 64))
```

**What happens automatically:**
- [PASS] Detects 3D thermal convection system
- [PASS] Sets up Fourier (x,y) + finite difference (z) discretization
- [PASS] Applies no-slip velocity and thermal boundary conditions
- [PASS] Selects predictor-corrector time integration
- [PASS] Configures CFL constraints for stability

### 2. Rotating Stratified Flow

```julia
equations = [
    "dt(u) = -u*dx(u) - v*dy(u) - w*dz(u) - dx(p) + nu*lap(u) + f*v",
    "dt(v) = -u*dx(v) - v*dy(v) - w*dz(v) - dy(p) + nu*lap(v) - f*u",
    "dt(w) = -u*dx(w) - v*dy(w) - w*dz(w) - dz(p) + nu*lap(w) + b",
    "dt(b) = -u*dx(b) - v*dy(b) - w*dz(b) + kappa*lap(b) + N2*w",
    "dx(u) + dy(v) + dz(w) = 0"
]

solution, prob, analysis = solve_pde_system(equations,
    parameters=Dict(:nu => 1e-4, :kappa => 1e-4, :f => 1e-4, :N2 => 1e-5))
```

**What happens automatically:**
- [PASS] Detects rotating stratified Boussinesq system
- [PASS] Identifies rotation (f) and stratification (N2, b) effects
- [PASS] Selects IMEX time integration for multiple time scales
- [PASS] Configures rotation-aware CFL conditions

### 3. Reaction-Diffusion System

```julia
# Gray-Scott pattern formation model
equations = [
    "dt(u) = Du*lap(u) - u*v*v + F*(1-u)",
    "dt(v) = Dv*lap(v) + u*v*v - (F+k)*v"
]

solution, prob, analysis = solve_pde_system(equations,
    parameters=Dict(:Du => 2e-5, :Dv => 1e-5, :F => 0.04, :k => 0.06),
    domain=(2.5, 2.5),
    resolution=(256, 256),
    dt=:auto)  # Automatic time step selection
```

**What happens automatically:**
- [PASS] Detects reaction-diffusion system
- [PASS] Identifies stiff reaction terms
- [PASS] Enables adaptive time stepping
- [PASS] Configures appropriate boundary conditions

### 4. Wave Equation

```julia
# Second-order wave equation as first-order system
equations = [
    "dt(u) = v",
    "dt(v) = c²*lap(u)"
]

solution, prob, analysis = solve_pde_system(equations,
    parameters=Dict(:c => 1.0))
```

**What happens automatically:**
- [PASS] Detects wave equation structure
- [PASS] Selects Leapfrog time integration (optimal for waves)
- [PASS] Configures wave CFL condition

## ANALYSIS Analysis Mode

Use `analyze_pde_system()` to understand your equations without solving:

```julia
equations = [
    "dt(phi) = D*lap(phi) + alpha*phi - beta*phi³",
    "dt(psi) = nu*lap(psi) + gamma*phi*psi"
]

analysis = analyze_pde_system(equations)
```

**Output includes:**
- PARAMETER Variable and parameter identification
- ANALYSIS Equation structure analysis (spatial terms, nonlinear terms, coupling)
- SCIENCE Physics type inference
- NUMBERS Numerical method recommendations
- SOLVER Solver configuration suggestions

## SOLVER Advanced Configuration

### Custom Parameters

```julia
solution, prob, analysis = solve_pde_system(equations,
    parameters=Dict(
        :Re => 5000.0,      # High Reynolds number
        :Pr => 0.1,         # Liquid metal Prandtl number
        :Ra => 1e7          # Highly supercritical
    ))
```

### Custom Domain and Resolution

```julia
solution, prob, analysis = solve_pde_system(equations,
    domain=(10.0, 5.0, 2.0),       # Custom domain size
    resolution=(256, 128, 64),      # Custom resolution
    boundary_conditions=[           # Custom BCs
        "bottom(u) = sin(x)",
        "top(T) = 0.5"
    ])
```

### Time Integration Control

```julia
solution, prob, analysis = solve_pde_system(equations,
    dt=0.001,              # Fixed time step
    time_span=100.0,       # Total simulation time
    output_frequency=50    # Save every 50 steps
)
```

##  What Happens Automatically

### 1. Equation Analysis
- **Symbol Extraction**: Identifies all variables (u, v, w, p, T, etc.) and parameters (Re, Ra, nu, etc.)
- **Structure Analysis**: Categorizes terms as spatial derivatives, nonlinear terms, coupling terms
- **Physics Inference**: Determines equation type (Navier-Stokes, reaction-diffusion, etc.)

### 2. Problem Building
- **Domain Setup**: Configures appropriate coordinate systems and basis functions
- **Parameter Management**: Sets reasonable default values for all detected parameters
- **Boundary Conditions**: Applies physics-appropriate boundary conditions

### 3. Solver Configuration
- **Discretization Selection**: Chooses Fourier/finite-difference methods per coordinate
- **Time Integration**: Selects optimal scheme (RK4, IMEX, predictor-corrector, etc.)
- **Stability Analysis**: Configures CFL constraints and adaptive time stepping

### 4. Performance Optimization
- **Memory Management**: Optimizes array allocations and workspace usage
- **Parallelization**: Configures MPI decomposition for large problems
- **Output Management**: Sets up efficient data storage and visualization

## CONSISTENCY Supported Equation Types

The universal solver automatically recognizes and optimally configures:

| Equation Type | Example Variables | Special Features |
|---------------|------------------|------------------|
| **Navier-Stokes** | u, v, w, p | Predictor-corrector, pressure projection |
| **Thermal Convection** | u, v, w, p, T | Boussinesq approximation, thermal BCs |
| **Rotating Flow** | u, v, w, p + rotation | IMEX for fast rotation, geostrophic balance |
| **Stratified Flow** | u, v, w, p, b | Multiple time scales, stability analysis |
| **Reaction-Diffusion** | u, v, w, ... | Adaptive time stepping, stiff solvers |
| **Wave Equations** | u, v (displacement/velocity) | Leapfrog integration, wave CFL |
| **Heat Equation** | T, u, φ, ... | Implicit diffusion solvers |
| **Custom Systems** | Any variables | Automatic analysis and configuration |

## INTEGRATION Design Philosophy

The Universal PDE Solver follows these principles:

1. **Zero Configuration by Default**: Just provide equations, get optimal solver
2. **Intelligent Defaults**: Automatically choose best methods for each physics type
3. **Easy Customization**: Override any aspect when needed
4. **Transparent Analysis**: Show exactly what decisions were made and why
5. **Performance First**: Always select numerically optimal approaches
6. **Extensible**: Easy to add new equation types and solution methods

## PERFORMANCE Performance Features

- **Automatic CFL Analysis**: Computes optimal time steps based on all active constraints
- **IMEX Splitting**: Automatically identifies stiff/non-stiff terms for optimal efficiency
- **Adaptive Methods**: Enables adaptive time stepping when beneficial
- **Memory Optimization**: Minimizes memory usage through intelligent workspace management
- **Parallel Scaling**: Automatically configures MPI decomposition

## PHYSICS Research Applications

The universal solver is designed for:

- **Rapid Prototyping**: Test new PDE formulations quickly
- **Multi-Physics Modeling**: Couple different physics automatically  
- **Parameter Studies**: Easy parameter sweeps with optimal solvers
- **Educational Use**: Learn PDE solving without numerical method details
- **Code Validation**: Compare different formulations of the same physics

##  Integration with Existing PencilFlows

The universal interface seamlessly integrates with existing PencilFlows functionality:

```julia
# Use universal interface to build, then access full PencilFlows features
solution, prob, analysis = solve_pde_system(equations)

# Access the built problem for custom operations
field_data = gather_field(prob, :u)
kinetic_energy = compute_kinetic_energy(prob)

# Add custom diagnostics
add_analysis_task!(prob, "kinetic_energy", compute_kinetic_energy)
```

## DEMO Interactive Demo

Try the interactive demonstration:

```julia
# Run comprehensive examples
demo_universal_interface()

# See all supported equation types  
run_all_examples()
```

##  Contributing

The universal solver is designed to be easily extensible:

1. **New Physics Types**: Add recognition patterns in `enhanced_equation_analysis.jl`
2. **New Solvers**: Add stepper types in `adaptive_stepper_selection.jl`
3. **New Syntax**: Extend equation parsing in equation parsing modules
4. **New Examples**: Add demonstration cases in `universal_pde_examples.jl`

---

**The Universal PDE Solver transforms PencilFlows.jl into a general-purpose PDE solving environment where users can focus on physics rather than numerical implementation details.**