# PencilFlows.jl Pretty Printing Guide

PencilFlows.jl now includes a comprehensive styled output system that makes equation building and problem setup visually appealing and informative.

## Features

The pretty printing system provides:

- **Styled Headers**: Beautiful ASCII art header with project branding
- **Live Equation Summaries**: Real-time display of equations as you build them
- **Domain Visualization**: Clear display of computational domains and discretization
- **Progress Tracking**: Timestamped build progress with colored output
- **Professional Presentation**: HPC-appropriate styling for research and development

## Available Functions

### 1. `pencilflow_header()`

Displays the full PencilFlows.jl header with ASCII art:

```julia
using PencilFlows
pencilflow_header()
```

Output:
```
••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••—
•‘                                                              •‘
•‘   ____                _ _ _____ _                 _ _        •‘
•‘  |  _ \ ___ _ __   ___(_) |  ___| | _____      __| | |       •‘
•‘  | |_) / _ \ '_ \ / __| | | |_  | |/ _ \ \ /\ / /| | |       •‘
•‘  |  __/  __/ | | | (__| | |  _| | | (_) \ V  V / | | |       •‘
•‘  |_|   \___|_| |_|\___|_|_|_|   |_|\___/ \_/\_(_)|_|_|       •‘
•‘                                                              •‘
•‘           Pseudospectral PDE Solver for Julia                •‘
•‘        Built on PencilArrays & PencilFFTs for HPC            •‘
•‘                                                              •‘
••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••
```

### 2. `pencilflow_banner()`

Displays a compact banner shown automatically during equation building:

```julia
pencilflow_banner()
```

Output:
```

‚                      PencilFlows.jl                          ‚
‚              Equation Building in Progress...                ‚
˜
```

**Note**: This banner appears automatically when you first call `add_equation!()` or `add_bc!()` to let users know they're using PencilFlows.jl.

### 3. `reset_banner!()`

Reset the banner display for a new session:

```julia
reset_banner!()  # Allow banner to be shown again
```

### 4. `equation_summary(prob)`

Shows a formatted summary of the current equation system:

```julia
prob = SymbolicProblem()
# ... add equations, boundary conditions, parameters ...
equation_summary(prob)
```

Output:
```

‚                    EQUATION SYSTEM                         ‚
¤
‚ Fields: u, v, w, p, T                                       ‚
‚                                                             ‚
‚ Governing Equations:                                        ‚
‚   1. dt(u) - (1/Re)*lap(u) = -u*dx(u) - v*dy(u) + f*v     ‚
‚   2. dt(v) - (1/Re)*lap(v) = -u*dx(v) - v*dy(v) - f*u     ‚
‚   3. dx(u) + dy(v) + dz(w) = 0                             ‚
‚                                                             ‚
‚ Boundary Conditions:                                        ‚
‚   ¢ bottom(u) = 0                                          ‚
‚   ¢ bottom(v) = 0                                          ‚
‚   ¢ top(T) = 1.0                                           ‚
‚                                                             ‚
‚ Parameters:                                                 ‚
‚   ¢ Re = 1000.0                                            ‚
‚   ¢ Ra = 1000000.0                                         ‚
‚   ¢ Pr = 0.7                                               ‚
˜
```

### 5. `domain_summary(prob)`

Displays computational domain information:

```julia
domain_summary(prob)
```

Output:
```

‚                 COMPUTATIONAL DOMAIN                       ‚
¤
‚ x: Fourier on [0.00 to 6.28] with 128 points               ‚
‚ y: Fourier on [0.00 to 6.28] with 128 points               ‚
‚ z: FiniteDiff on [0.00 to 1.00] with 64 points             ‚
˜
```

### 6. `show_build_progress(stage, details="")`

Shows timestamped build progress:

```julia
show_build_progress("Domain setup", "3D periodic-bounded domain")
show_build_progress("Grid generation", "128 Ã— 128 Ã— 64 points")
show_build_progress("Problem built successfully!")
```

Output:
```
[14:23:45] >> Domain setup - 3D periodic-bounded domain
[14:23:46] >> Grid generation - 128 Ã— 128 Ã— 64 points  
[14:23:47] >> Problem built successfully!
```

## Automatic Integration

The pretty printing is automatically used when:

### Adding Equations

```julia
add_equation!(prob, "dt(u) - nu*lap(u) = -u*dx(u) - v*dy(u)")
```

Automatically shows:
- Progress message with timestamp
- Updated equation summary with all current equations

### Adding Boundary Conditions

```julia
add_bc!(prob, "bottom(u) = 0")
```

Automatically shows:
- Progress message for the new boundary condition

### Building Problems

```julia
build_problem!(prob, Nx=128, Ny=128, Nz=64)
```

Automatically shows:
- PencilFlows.jl header
- Complete equation summary
- Domain configuration
- Build progress steps
- Final completion summary

## Color Scheme

The output uses a consistent color scheme:

- **Blue**: Borders, labels, structure
- **Green**: Field names, successful operations
- **Yellow**: Basis types, parameters
- **Magenta**: Boundary conditions
- **White**: Equation text, general information
- **Cyan**: Headers, domain titles
- **Light Black**: Timestamps

## Example: Complete Workflow

```julia
using PencilFlows

# The header appears automatically when building
prob = SymbolicProblem()

# Add fields
u = Field(:u); v = Field(:v); p = Field(:p)
prob.fields = [u, v, p]

# Add parameters
prob.parameters[:Re] = 1000.0

# Each equation shows progress and updates summary
add_equation!(prob, "dt(u) - (1/Re)*lap(u) = -u*dx(u) - v*dy(u) - dx(p)")
add_equation!(prob, "dt(v) - (1/Re)*lap(v) = -u*dx(v) - v*dy(v) - dy(p)")
add_equation!(prob, "dx(u) + dy(v) = 0")

# Boundary conditions with progress
add_bc!(prob, "bottom(u) = 0")
add_bc!(prob, "top(u) = 1.0")

# Domain setup
x_basis = Fourier(:x, (0.0, 2Ï))
y_basis = Fourier(:y, (0.0, 2Ï))
z_basis = FiniteDifference(:z, (0.0, 1.0))
prob.domain = Domain((x_basis, 128), (y_basis, 128), (z_basis, 64))

# Build with full progress display
build_problem!(prob, Nx=128, Ny=128, Nz=64)
```

This creates a professional, informative output that's perfect for:

- Research presentations
- Educational demonstrations  
- Interactive development
- HPC workflow documentation
- Debugging and validation

## Customization

All functions respect the terminal's color capabilities and will gracefully fall back to plain text if colors aren't supported. The Unicode box drawing characters are carefully chosen for maximum compatibility.

The system is designed to be informative but not overwhelming - it enhances the development experience without cluttering the output. All emoji symbols have been removed for professional HPC environments.
