# Universal PDE Interface for PencilFlows.jl
# ========================================
# This provides the ultimate general interface where users can input any set of PDEs
# and the system automatically analyzes, builds, and solves the problem.

using Printf

"""
    solve_pde_system(equations::Vector{String}; kwargs...)

Universal interface to solve any system of PDEs. Just provide the equations as strings!

# Arguments
- `equations::Vector{String}`: Vector of PDE equations in string form
- `domain::Tuple = (2π, 2π, 1.0)`: Domain size for each spatial dimension  
- `resolution::Tuple = (64, 64, 32)`: Grid resolution for each spatial dimension
- `parameters::Dict = Dict()`: Manual parameter overrides
- `boundary_conditions::Vector{String} = []`: Custom boundary conditions
- `time_span::Float64 = 10.0`: Total simulation time
- `dt::Float64 = 0.001`: Time step (or :auto for automatic)
- `output_frequency::Int = 100`: How often to save output
- `verbose::Bool = true`: Print detailed progress

# Returns
- `solution`: Solution data structure
- `problem`: The built problem object  
- `system_analysis`: Analysis of the PDE system

# Examples

## 1. Navier-Stokes with Thermal Convection
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
    time_span=50.0)
```

## 2. Rotating Stratified Flow
```julia
equations = [
    "dt(u) = -u*dx(u) - v*dy(u) - w*dz(u) - dx(p) + nu*lap(u) + f*v",
    "dt(v) = -u*dx(v) - v*dy(v) - w*dz(v) - dy(p) + nu*lap(v) - f*u", 
    "dt(w) = -u*dx(w) - v*dy(w) - w*dz(w) - dz(p) + nu*lap(w) + b",
    "dt(b) = -u*dx(b) - v*dy(b) - w*dz(b) + kappa*lap(b) + N2*w",
    "dx(u) + dy(v) + dz(w) = 0"
]

solution, prob, analysis = solve_pde_system(equations,
    parameters=Dict(:nu => 1e-4, :kappa => 1e-4, :f => 1.0, :N2 => 1.0))
```

## 3. Custom Reaction-Diffusion System  
```julia
equations = [
    "dt(u) = D1*lap(u) + a*u - b*u*v",
    "dt(v) = D2*lap(v) + c*u*v - d*v"
]

solution, prob, analysis = solve_pde_system(equations,
    parameters=Dict(:D1 => 0.1, :D2 => 0.05, :a => 1.0, :b => 1.0, :c => 1.0, :d => 1.0),
    domain=(10.0, 10.0), resolution=(128, 128))
```

## 4. Wave Equation
```julia
equations = ["d2t(u) = c²*lap(u)"]

solution, prob, analysis = solve_pde_system(equations,
    parameters=Dict(:c => 1.0))
```
"""
function solve_pde_system(equations::Vector{String}; 
                         domain::Tuple = (2π, 2π, 1.0),
                         resolution::Tuple = (64, 64, 32), 
                         parameters::Dict = Dict(),
                         boundary_conditions::Vector{String} = String[],
                         time_span::Float64 = 10.0,
                         dt::Union{Float64, Symbol} = 0.001,
                         output_frequency::Int = 100,
                         verbose::Bool = true)
    
    if verbose
        print_universal_header()
        println("Input Equations:")
        for (i, eq) in enumerate(equations)
            println("  $i. $eq")
        end
    end
    
    # Step 1: Analyze the PDE system
    if verbose println("\nSTEP 1: ANALYZING PDE SYSTEM") end
    prob, system_analysis = build_general_pde_problem(equations,
        domain_size=domain, resolution=resolution, parameter_overrides=parameters)
    
    # Step 2: Add custom boundary conditions if provided
    if !isempty(boundary_conditions)
        if verbose println("\nSTEP 2: ADDING CUSTOM BOUNDARY CONDITIONS") end
        for bc in boundary_conditions
            add_bc!(prob, bc)
            verbose && println("   Added BC: $bc")
        end
    end
    
    # Step 3: Build the numerical problem
    if verbose println("\nSTEP 3: BUILDING NUMERICAL DISCRETIZATION") end
    build_problem!(prob)
    
    # Step 4: Configure time integration
    if verbose println("\nSTEP 4: CONFIGURING TIME INTEGRATION") end
    if dt == :auto
        dt = compute_automatic_timestep(prob, system_analysis)
        verbose && println("   Automatic timestep: dt = $dt")
    end
    
    # Step 5: Run simulation
    if verbose println("\nSTEP 5: RUNNING SIMULATION") end
    solution = run_simulation(prob, system_analysis, time_span, dt, output_frequency, verbose)
    
    if verbose
        println("\nSIMULATION COMPLETE!")
        print_solution_summary(solution, system_analysis)
    end
    
    return solution, prob, system_analysis
end

"""
    quick_pde(equations::Vector{String}; kwargs...)

Ultra-quick interface - just provide equations and get results with minimal setup.
"""
function quick_pde(equations::Vector{String}; kwargs...)
    return solve_pde_system(equations; time_span=1.0, dt=0.01, verbose=false, kwargs...)
end

"""
    analyze_pde_system(equations::Vector{String})

Analyze a PDE system without solving it - useful for understanding the structure.
"""
function analyze_pde_system(equations::Vector{String})
    println("ANALYZING PDE SYSTEM (No Solving)")
    println("="^40)
    
    system_analysis = analyze_general_pde_system(equations)
    provide_analysis_insights(system_analysis)
    
    return system_analysis
end

"""
    print_universal_header()

Print the universal PDE solver header.
"""
function print_universal_header()
    println()
    println("="^60)
    println("UNIVERSAL PDE SOLVER - PencilFlows.jl")
    println("="^60)
    println("Automatically analyze and solve ANY system of PDEs!")
    println()
end

"""
    compute_automatic_timestep(prob, system_analysis)

Compute an appropriate time step based on the problem characteristics.
"""
function compute_automatic_timestep(prob, system_analysis)
    # Start with a conservative estimate
    dt = 0.001
    
    # Adjust based on physics type
    if system_analysis.physics_type in [:navier_stokes_2d, :navier_stokes_3d, :thermal_convection_2d, :thermal_convection_3d]
        # CFL constraint for advection
        dt = min(dt, 0.1)  # Conservative for nonlinear terms
        
        # Viscous constraint
        if haskey(system_analysis.parameters, :Re)
            Re = system_analysis.parameters[:Re]
            dt = min(dt, 0.01 / Re)  # Rough viscous time step
        end
        
        # Rotation constraint
        if haskey(system_analysis.parameters, :f)
            f = system_analysis.parameters[:f]
            dt = min(dt, 0.1 / f)  # Inertial oscillation time scale
        end
    elseif system_analysis.physics_type == :heat_equation
        # Diffusive constraint
        dt = 0.001
    elseif system_analysis.physics_type == :wave_equation
        # CFL for wave equation
        dt = 0.01
    end
    
    return dt
end

"""
    run_simulation(prob, system_analysis, time_span, dt, output_frequency, verbose)

Run the actual simulation with progress monitoring.
"""
function run_simulation(prob, system_analysis, time_span, dt, output_frequency, verbose)
    max_iter = Int(ceil(time_span / dt))
    
    if verbose
        println("  Simulation Parameters:")
        println("    • Total time: $time_span")
        println("    • Time step: $dt")
        println("    • Total iterations: $max_iter")
        println("    • Output every: $output_frequency steps")
        println()
    end
    
    # Run the simulation (using existing PencilFlows solve!)
    solution = solve!(prob, dt=dt, max_iter=max_iter)
    
    return solution
end

"""
    print_solution_summary(solution, system_analysis)

Print a summary of the solution results.
"""
function print_solution_summary(solution, system_analysis)
    println("="^60)
    println("SOLUTION SUMMARY")
    println("="^60)
    println("• Physics: $(system_analysis.physics_type)")
    println("• Variables solved: $(join(sort(collect(system_analysis.variables)), ", "))")
    println("• Final time reached: $(get(solution, :final_time, "N/A"))")
    println("• Solution status: $(get(solution, :status, "Complete"))")
    
    if haskey(solution, :diagnostics)
        diag = solution[:diagnostics]
        println("\nKey Diagnostics:")
        for (key, value) in diag
            println("  • $key: $value")
        end
    end
    
    println("\nSolution completed successfully!")
    println("="^60)
end

"""
    provide_analysis_insights(system_analysis)

Provide insights and recommendations based on system analysis.
"""
function provide_analysis_insights(system_analysis)
    println("\nINSIGHTS AND RECOMMENDATIONS")
    println("="^40)
    
    # Physics insights
    println("Physics Type: $(system_analysis.physics_type)")
    provide_physics_specific_insights(system_analysis.physics_type)
    
    # Parameter insights  
    if !isempty(system_analysis.parameters)
        println("\nParameters:")
        for (param, value) in system_analysis.parameters
            insight = get_parameter_insight(param, value)
            println("  • $param = $value - $insight")
        end
    end
    
    # Numerical recommendations
    println("\nNumerical Recommendations:")
    provide_numerical_recommendations(system_analysis)
    
    println("\n" * "="^40)
end

"""
    provide_physics_specific_insights(physics_type::Symbol)

Provide insights specific to the physics type.
"""
function provide_physics_specific_insights(physics_type::Symbol)
    insights = Dict(
        :navier_stokes_2d => "2D incompressible fluid flow - expect vortex dynamics",
        :navier_stokes_3d => "3D incompressible fluid flow - complex turbulent structures possible",
        :thermal_convection_2d => "Thermal convection - expect convection rolls and circulation",
        :thermal_convection_3d => "3D thermal convection - convection cells and plumes",
        :heat_equation => "Pure diffusion - smooth temperature evolution",
        :reaction_diffusion => "Pattern formation possible - Turing instabilities",
        :wave_equation => "Wave propagation - oscillatory solutions",
        :general_pde_system => "Custom PDE system - analyze stability carefully"
    )
    
    insight = get(insights, physics_type, "Custom physics - analyze behavior carefully")
    println("  $insight")
end

"""
    get_parameter_insight(param::Symbol, value::Float64)

Get insight about a specific parameter value.
"""
function get_parameter_insight(param::Symbol, value::Float64)
    param_insights = Dict(
        :Re => value < 100 ? "Low Re (laminar)" : value < 1000 ? "Moderate Re" : value < 10000 ? "High Re (transitional)" : "Very high Re (turbulent)",
        :Ra => value < 1708 ? "Subcritical (stable)" : value < 10000 ? "Supercritical (steady convection)" : "Turbulent convection",
        :Pr => value < 0.1 ? "Liquid metals" : value < 1 ? "Gases" : value < 10 ? "Water-like" : "High Pr fluids",
        :f => value < 1e-5 ? "Weak rotation" : value < 1e-3 ? "Moderate rotation" : "Strong rotation",
        :nu => value < 1e-6 ? "Very low viscosity" : value < 1e-3 ? "Low viscosity" : "Moderate viscosity",
        :kappa => value < 1e-6 ? "Very low diffusivity" : value < 1e-3 ? "Low diffusivity" : "Moderate diffusivity"
    )
    
    return get(param_insights, param, "parameter in typical range")
end

"""
    provide_numerical_recommendations(system_analysis)

Provide numerical method recommendations.
"""
function provide_numerical_recommendations(system_analysis)
    # Resolution recommendations
    println("  Resolution:")
    if system_analysis.physics_type in [:navier_stokes_2d, :navier_stokes_3d]
        Re = get(system_analysis.parameters, :Re, 1000)
        if Re < 1000
            println("    - Current resolution should be adequate for Re = $Re")
        else
            println("    - Consider higher resolution for Re = $Re")
        end
    end
    
    # Time step recommendations
    println("  Time stepping:")
    if system_analysis.physics_type in [:navier_stokes_2d, :navier_stokes_3d, :thermal_convection_2d, :thermal_convection_3d]
        println("    - Use predictor-corrector or projection methods")
        println("    - Watch CFL condition for stability")
    else
        println("    - Explicit methods should work well")
    end
end

"""
    demo_universal_interface()

Demonstrate the universal interface with several example systems.
"""
function demo_universal_interface()
    print_universal_header()
    println("DEMONSTRATION MODE - Multiple PDE Systems")
    println()
    
    # Demo 1: Simple diffusion
    println("Demo 1: Heat Equation")
    equations1 = ["dt(T) = alpha*lap(T)"]
    analyze_pde_system(equations1)
    
    println("\n" * "-"^50 * "\n")
    
    # Demo 2: Reaction-diffusion
    println("Demo 2: Reaction-Diffusion System")  
    equations2 = [
        "dt(u) = D1*lap(u) + a*u - b*u*v",
        "dt(v) = D2*lap(v) + c*u*v - d*v"
    ]
    analyze_pde_system(equations2)
    
    println("\n" * "-"^50 * "\n")
    
    # Demo 3: Navier-Stokes
    println("Demo 3: Navier-Stokes with Thermal Convection")
    equations3 = [
        "dt(u) = -u*dx(u) - v*dy(u) - w*dz(u) - dx(p) + (1/Re)*lap(u)",
        "dt(v) = -u*dx(v) - v*dy(v) - w*dz(v) - dy(p) + (1/Re)*lap(v)",  
        "dt(w) = -u*dx(w) - v*dy(w) - w*dz(w) - dz(p) + (1/Re)*lap(w) + Ra*Pr*T",
        "dt(T) = -u*dx(T) - v*dy(T) - w*dz(T) + (1/Re/Pr)*lap(T)",
        "dx(u) + dy(v) + dz(w) = 0"
    ]
    analyze_pde_system(equations3)
    
    println("\nDEMONSTRATION COMPLETE!")
    println("Now you can use solve_pde_system() with your own equations!")
end

# Export the main interface functions
export solve_pde_system, quick_pde, analyze_pde_system, demo_universal_interface
