# Smart Interface for PencilFlow.jl
# =================================
# High-level interface that automatically analyzes equations and builds problems

"""
    build_problem_from_equations(equations::Vector{String}; kwargs...)

Smart interface that automatically analyzes symbolic equations, identifies all
constants and variables, and builds a complete problem setup ready for solving.

# Arguments
- `equations::Vector{String}`: Vector of equation strings in symbolic form
- `domain_size::Tuple = (2π, 2π, 1.0)`: Domain dimensions  
- `resolution::Tuple = (64, 64, 32)`: Grid resolution
- `parameter_overrides::Dict = Dict()`: Manual parameter value overrides

# Returns
- `prob::SymbolicProblem`: Fully configured problem ready for solve!()

# Examples

## Automatic Rayleigh-Bx©nard Convection
```julia
equations = [
    "dt(u) = -u*dx(u) - v*dz(u) - dx(p) + (1/Re)*lap(u)",
    "dt(v) = -u*dx(v) - v*dz(v) - dz(p) + (1/Re)*lap(v) + Ra*Pr*T",
    "dt(T) = -u*dx(T) - v*dz(T) + (1/Re/Pr)*lap(T)",
    "div(u, v) = 0"
]

prob = build_problem_from_equations(equations)
# Automatically detects: Ra, Pr, Re constants and sets up thermal convection

solution = solve!(prob, dt=0.001, max_iter=10000)
```

## Automatic Rotating Stratified Flow
```julia
equations = [
    "dt(u) = -u*dx(u) - v*dy(u) - w*dz(u) - dx(p) + nu*lap(u) + f*v",
    "dt(v) = -u*dx(v) - v*dy(v) - w*dz(v) - dy(p) + nu*lap(v) - f*u",
    "dt(w) = -u*dx(w) - v*dy(w) - w*dz(w) - dz(p) + nu*lap(w) + b",
    "dt(b) = -u*dx(b) - v*dy(b) - w*dz(b) + kappa*lap(b) + N2*w",
    "div(u, v, w) = 0"
]

# Automatically detects: nu, f, kappa, N2 and sets up rotating stratified flow
prob = build_problem_from_equations(equations)
solution = solve!(prob, dt=0.01, max_iter=5000)
```

## Manual Parameter Override
```julia
equations = ["dt(u) = -dx(p) + (1/Re)*lap(u)", "div(u) = 0"]

prob = build_problem_from_equations(
    equations, 
    parameter_overrides=Dict(:Re => 10000.0)  # High Reynolds number
)
```
"""
function build_problem_from_equations(equations::Vector{String}; 
                                    domain_size::Tuple = (2π, 2π, 1.0),
                                    resolution::Tuple = (64, 64, 32),
                                    parameter_overrides::Dict = Dict())
    
    println(" SMART PROBLEM BUILDER")
    println("="^50)
    
    # Create empty problem
    prob = SymbolicProblem()
    
    # Use automatic analysis to build the problem
    auto_build_problem!(prob, equations)
    
    # Apply any manual parameter overrides
    if !isempty(parameter_overrides)
        println("\n Applying parameter overrides...")
        for (param, value) in parameter_overrides
            prob.parameters[param] = Float64(value)
            println("    Override: $param = $value")
        end
    end
    
    # Show final parameter summary
    show_final_parameter_summary(prob)
    
    return prob
end

"""
    analyze_and_suggest(equations::Vector{String})

Analyze equations and provide suggestions for parameter values and setup
without actually building the problem.
"""
function analyze_and_suggest(equations::Vector{String})
    println(" EQUATION ANALYSIS & SUGGESTIONS")
    println("="^50)
    
    # Analyze the equations
    analysis = analyze_symbolic_equations(equations)
    
    # Provide parameter suggestions
    println("\n PARAMETER SUGGESTIONS:")
    suggest_parameter_values(analysis)
    
    # Provide physics insights
    println("\n PHYSICS ANALYSIS:")
    provide_physics_insights(analysis)
    
    # Provide solver recommendations  
    println("\n  SOLVER RECOMMENDATIONS:")
    provide_solver_recommendations(analysis)
    
    return analysis
end

"""
    suggest_parameter_values(analysis::EquationAnalysis)

Suggest appropriate parameter values based on equation analysis.
"""
function suggest_parameter_values(analysis::EquationAnalysis)
    suggestions = Dict{Symbol, Dict{Symbol, Any}}()
    
    # Reynolds number suggestions
    if :Re in keys(analysis.constants)
        suggestions[:Re] = Dict(
            :low_re => 100.0,
            :moderate_re => 1000.0,
            :high_re => 10000.0,
            :turbulent => 100000.0
        )
        println("  Reynolds Number (Re):")
        println("    ¢ Low Re (laminar): 100")
        println("    ¢ Moderate Re: 1,000") 
        println("    ¢ High Re (transitional): 10,000")
        println("    ¢ Turbulent: 100,000+")
    end
    
    # Rayleigh number suggestions
    if :Ra in keys(analysis.constants)
        suggestions[:Ra] = Dict(
            :subcritical => 1000.0,
            :critical => 1708.0,
            :supercritical => 10000.0,
            :turbulent => 1e6
        )
        println("  Rayleigh Number (Ra):")
        println("    ¢ Subcritical (no convection): < 1,708")
        println("    ¢ Critical (onset): ~ 1,708")
        println("    ¢ Supercritical (steady): 10,000")
        println("    ¢ Turbulent convection: 1,000,000+")
    end
    
    # Ekman number suggestions
    if :Ek in keys(analysis.constants)
        suggestions[:Ek] = Dict(
            :geophysical => 1e-5,
            :laboratory => 1e-3,
            :weakly_rotating => 1e-1
        )
        println("  Ekman Number (Ek):")
        println("    ¢ Geophysical flows: 1e-5")
        println("    ¢ Laboratory rotating tank: 1e-3")
        println("    ¢ Weakly rotating: 1e-1")
    end
    
    # Prandtl number suggestions
    if :Pr in keys(analysis.constants)
        suggestions[:Pr] = Dict(
            :liquid_metals => 0.01,
            :gases => 0.7,
            :water => 7.0,
            :oils => 100.0
        )
        println("  Prandtl Number (Pr):")
        println("    ¢ Liquid metals: 0.01")
        println("    ¢ Gases (air): 0.7")
        println("    ¢ Water: 7.0")
        println("    ¢ Oils: 100+")
    end
end

"""
    provide_physics_insights(analysis::EquationAnalysis)

Provide insights about the physics based on equation analysis.
"""
function provide_physics_insights(analysis::EquationAnalysis)
    println("Physics Type: $(analysis.equation_type)")
    
    if analysis.equation_type in [:navier_stokes_2d, :navier_stokes_3d]
        println("¢ Classical Navier-Stokes equations")
        println("¢ Fluid momentum conservation with viscous effects")
        println("¢ Pressure enforces incompressibility constraint")
    elseif analysis.equation_type in [:boussinesq_2d, :boussinesq_3d]
        println("¢ Boussinesq approximation for buoyancy-driven flows")
        println("¢ Thermal/density coupling drives convection")
        println("¢ Buoyancy acts as source term in momentum equations")
    elseif analysis.equation_type == :thermal_convection
        println("¢ Pure thermal convection (Rayleigh-Bx©nard type)")
        println("¢ Temperature differences drive circulation")
    end
    
    if analysis.has_rotation
        println("¢ Rotation effects present (Coriolis forces)")
        println("¢ Flow will exhibit geostrophic balance tendencies")
        println("¢ Rossby waves and rotation-modified turbulence expected")
    end
    
    if analysis.has_stratification
        println("¢ Stratification effects present")
        println("¢ Gravity waves and buoyancy oscillations possible")
        println("¢ Vertical motion suppressed in stable regions")
    end
end

"""
    provide_solver_recommendations(analysis::EquationAnalysis)

Provide solver and numerical method recommendations.
"""
function provide_solver_recommendations(analysis::EquationAnalysis)
    if analysis.equation_type in [:navier_stokes_2d, :navier_stokes_3d, :boussinesq_2d, :boussinesq_3d]
        println("¢ Use predictor-corrector with pressure projection")
        println("¢ Spectral methods in horizontal directions")
        println("¢ Finite differences in vertical direction")
        println("¢ Time step limited by CFL condition")
    end
    
    if analysis.has_rotation
        println("¢ Consider rotation-modified CFL condition")
        println("¢ May need implicit treatment for fast rotation")
    end
    
    if analysis.has_viscosity
        println("¢ Viscous time step constraint: dt < dx^2/ν")
        println("¢ Consider semi-implicit methods for high Re")
    end
    
    # Resolution recommendations
    println("Recommended Resolutions:")
    if analysis.equation_type == :thermal_convection
        println("  ¢ Low Ra: 642 or 64x*32")
        println("  ¢ High Ra: 2562 or 512x*256")
    elseif analysis.has_rotation
        println("  ¢ Include Rossby deformation radius")
        println("  ¢ Typical: 128*³ to 256*³")
    else
        println("  ¢ Standard: 64*³ to 128*³")
        println("  ¢ High Re: 256*³ or higher")
    end
end

"""
    show_final_parameter_summary(prob::SymbolicProblem)

Display final parameter summary after all processing.
"""
function show_final_parameter_summary(prob::SymbolicProblem)
    if isempty(prob.parameters)
        return
    end
    
    println("\n FINAL PARAMETER VALUES:")
    println("-"^30)
    
    # Group parameters by type
    dimensionless = []
    physical = []
    reference_scales = []
    
    for (param, value) in prob.parameters
        param_str = string(param)
        if param in [:Re, :Pr, :Ra, :Ek, :Ro, :Ri]
            push!(dimensionless, (param, value))
        elseif param in [:L_ref, :U_ref, :T_ref, :DT_ref]
            push!(reference_scales, (param, value))
        else
            push!(physical, (param, value))
        end
    end
    
    if !isempty(dimensionless)
        println("Dimensionless Numbers:")
        for (param, value) in sort(dimensionless)
            println("  $param = $value")
        end
    end
    
    if !isempty(physical)
        println("Physical Parameters:")  
        for (param, value) in sort(physical)
            println("  $param = $value")
        end
    end
    
    if !isempty(reference_scales)
        println("Reference Scales:")
        for (param, value) in sort(reference_scales)
            println("  $param = $value")
        end
    end
end

"""
    quick_solve(equations::Vector{String}; dt=0.001, max_iter=1000, kwargs...)

Ultra-quick interface: analyze equations, build problem, and solve in one call.
"""
function quick_solve(equations::Vector{String}; dt=0.001, max_iter=1000, kwargs...)
    println(" QUICK SOLVE MODE")
    println("="^30)
    
    # Build problem automatically
    prob = build_problem_from_equations(equations; kwargs...)
    
    # Build numerical discretization  
    build_problem!(prob)
    
    # Solve
    println("\n Starting simulation...")
    solution = solve!(prob, dt=dt, max_iter=max_iter)
    
    println(" Simulation complete!")
    return solution, prob
end

# Export the smart interface functions
export build_problem_from_equations, analyze_and_suggest, quick_solve
