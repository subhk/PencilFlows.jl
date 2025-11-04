# Flexible Symbolic Interface for PencilFlows.jl
# Integrates flexible parameter detection with the main symbolic system
# Note: flexible_parameter_detection.jl is included by the main module

"""
    FlexibleSymbolicProblem

A PencilFlows problem that can handle arbitrary variable and parameter names.
Only x, y, z, t are reserved as coordinates and time.
"""
mutable struct FlexibleSymbolicProblem
    equations::Vector{String}
    variables::Set{Symbol}
    parameters::Dict{Symbol, Float64}
    domain::Dict{Symbol, Any}
    boundary_conditions::Vector{Dict{Symbol, Any}}
    analysis::Union{FlexibleEquationAnalysis, Nothing}
    
    function FlexibleSymbolicProblem()
        new(String[], Set{Symbol}(), Dict{Symbol, Float64}(), 
            Dict{Symbol, Any}(), Dict{Symbol, Any}[], nothing)
    end
end

"""
    add_flexible_equation!(prob::FlexibleSymbolicProblem, equation::String)

Add an equation with arbitrary variable and parameter names.
System automatically detects and categorizes all symbols.
"""
function add_flexible_equation!(prob::FlexibleSymbolicProblem, equation::String)
    push!(prob.equations, equation)
    
    # Re-analyze equations to update variables and parameters
    if !isempty(prob.equations)
        prob.analysis = analyze_flexible_equations(prob.equations)
        
        # Update variables
        prob.variables = prob.analysis.variables
        
        # Update parameters (preserve existing values, add new ones with defaults)
        new_parameters = prob.analysis.parameters
        for param in new_parameters
            if !haskey(prob.parameters, param)
                # Use intelligent default guessing
                default_val = guess_parameter_default(param, prob.analysis)
                prob.parameters[param] = default_val
                println("  Auto-added parameter: $param = $default_val")
            end
        end
    end
    
    println("Added equation: $equation")
    return prob
end

"""
    set_flexible_parameter!(prob::FlexibleSymbolicProblem, param::Symbol, value::Float64)

Set a parameter value. Parameter doesn't need to be pre-defined.
"""
function set_flexible_parameter!(prob::FlexibleSymbolicProblem, param::Symbol, value::Float64)
    prob.parameters[param] = value
    println("Set parameter: $param = $value")
    return prob
end

"""
    auto_detect_physics!(prob::FlexibleSymbolicProblem)

Automatically detect physics type and suggest appropriate numerical methods.
"""
function auto_detect_physics!(prob::FlexibleSymbolicProblem)
    if prob.analysis === nothing
        println("No equations to analyze")
        return
    end
    
    physics = prob.analysis.equation_structure
    
    println("Automatic Physics Detection:")
    println("   Equations: $(physics[:num_equations])")
    println("   Variables: $(physics[:num_variables])")
    println("   Parameters: $(physics[:num_parameters])")
    println("   Dimensions: $(physics[:dimensionality])D")
    
    # Suggest numerical methods based on detected physics
    if physics[:has_time_evolution]
        println("   Time evolution detected → Use IMEX time stepping")
    end
    
    if physics[:has_diffusion]
        println("   Diffusion detected -> Use implicit treatment for stability")
        
        # Identify diffusion parameters
        if !isempty(physics[:likely_viscosity])
            println("     - Likely viscosity parameters: $(physics[:likely_viscosity])")
        end
        if !isempty(physics[:likely_diffusivity])
            println("     - Likely diffusivity parameters: $(physics[:likely_diffusivity])")
        end
    end
    
    if physics[:has_advection] 
        println("   Advection detected -> Use explicit treatment with dealiasing")
    end
    
    if physics[:has_incompressibility]
        println("   Incompressibility detected -> Use pressure projection")
    end
    
    # Suggest domain setup
    suggested_domain = suggest_domain_setup(physics)
    println("   Suggested domain: $suggested_domain")
    
    return physics
end

"""
    suggest_domain_setup(physics::Dict) -> Dict

Suggest appropriate domain setup based on detected physics.
"""
function suggest_domain_setup(physics::Dict)
    domain = Dict{Symbol, Any}()
    
    # Suggest spatial resolution based on dimensionality
    if physics[:dimensionality] == 1
        domain[:grid] = "1D: Nx=128"
        domain[:boundary] = "periodic or dirichlet"
    elseif physics[:dimensionality] == 2  
        domain[:grid] = "2D: Nx=64, Ny=64"
        domain[:boundary] = "periodic in x,y or mixed"
    elseif physics[:dimensionality] == 3
        domain[:grid] = "3D: Nx=32, Ny=32, Nz=32"
        domain[:boundary] = "periodic in x,y; varied in z"
    end
    
    # Suggest domain size based on diffusion
    if physics[:has_diffusion]
        domain[:size] = "L ~ sqrt(diffusion_time * diffusivity)"
    end
    
    return domain
end

"""
    build_flexible_problem!(prob::FlexibleSymbolicProblem; 
                           domain_size::Tuple{Float64,Float64,Float64}=(1.0,1.0,1.0),
                           grid_points::Tuple{Int,Int,Int}=(32,32,32))

Build the complete PencilFlows problem from flexible equations.
"""
function build_flexible_problem!(prob::FlexibleSymbolicProblem;
                                domain_size::Tuple{Float64,Float64,Float64}=(1.0,1.0,1.0),
                                grid_points::Tuple{Int,Int,Int}=(32,32,32))
    
    if isempty(prob.equations)
        error("No equations defined. Use add_flexible_equation! first.")
    end
    
    println("Building Flexible PencilFlows Problem...")
    
    # Auto-detect physics
    physics = auto_detect_physics!(prob)
    
    # Set up domain
    prob.domain[:Lx], prob.domain[:Ly], prob.domain[:Lz] = domain_size
    prob.domain[:Nx], prob.domain[:Ny], prob.domain[:Nz] = grid_points
    
    println("   Domain: $(domain_size) with grid $(grid_points)")
    
    # Validate parameter consistency
    validate_flexible_parameters!(prob)
    
    # Generate solution strategy
    strategy = generate_solution_strategy(prob)
    println("   Solution strategy: $strategy")
    
    return prob
end

"""
    validate_flexible_parameters!(prob::FlexibleSymbolicProblem)

Validate that all parameters have reasonable values and suggest improvements.
"""
function validate_flexible_parameters!(prob::FlexibleSymbolicProblem)
    println("Parameter Validation:")
    
    for (param, value) in prob.parameters
        if value <= 0 && param in prob.analysis.equation_structure[:likely_viscosity]
            println("   $param = $value (viscosity should be > 0)")
        elseif value <= 0 && param in prob.analysis.equation_structure[:likely_diffusivity]
            println("   $param = $value (diffusivity should be > 0)")
        elseif abs(value) > 1e6
            println("   $param = $value (very large value, check units)")
        elseif abs(value) < 1e-12
            println("   $param = $value (very small value, check units)")
        else
            println("   $param = $value")
        end
    end
end

"""
    generate_solution_strategy(prob::FlexibleSymbolicProblem) -> Dict

Generate appropriate solution strategy based on detected physics.
"""
function generate_solution_strategy(prob::FlexibleSymbolicProblem)
    strategy = Dict{Symbol, Any}()
    physics = prob.analysis.equation_structure
    
    # Time stepping
    if physics[:has_diffusion] && physics[:has_advection]
        strategy[:timestepper] = "IMEX (Implicit-Explicit)"
        strategy[:implicit_terms] = physics[:likely_viscosity] 
        strategy[:explicit_terms] = "advection"
    elseif physics[:has_diffusion]
        strategy[:timestepper] = "Backward Euler or BDF"
    else
        strategy[:timestepper] = "Forward Euler or RK4"
    end
    
    # Spatial discretization
    if physics[:dimensionality] >= 2
        strategy[:spatial] = "Spectral (FFT) in horizontal, finite differences in vertical"
    else
        strategy[:spatial] = "Finite differences or spectral"
    end
    
    # Linear solvers
    if physics[:has_incompressibility]
        strategy[:pressure_solver] = "FFT-based Poisson solver"
    end
    
    if physics[:has_diffusion]
        strategy[:diffusion_solver] = "Helmholtz solver with tridiagonal in vertical"
    end
    
    return strategy
end

"""
    show_flexible_problem_summary(prob::FlexibleSymbolicProblem)

Display comprehensive summary of the flexible problem.
"""
function show_flexible_problem_summary(prob::FlexibleSymbolicProblem)
    println("Flexible PencilFlows Problem Summary")
    println("=" ^ 50)
    
    if !isempty(prob.equations)
        println("Equations:")
        for (i, eq) in enumerate(prob.equations)
            println("   $i. $eq")
        end
    end
    
    if !isempty(prob.variables)
        println("\nVariables ($(length(prob.variables))):")
        for var in sort(collect(prob.variables))
            println("   - $var")
        end
    end
    
    if !isempty(prob.parameters)
        println("\nParameters ($(length(prob.parameters))):")
        for (param, value) in sort(collect(prob.parameters))
            println("   - $param = $value")
        end
    end
    
    if !isempty(prob.domain)
        println("\nDomain:")
        for (key, value) in prob.domain
            println("   - $key: $value")
        end
    end
    
    if prob.analysis !== nothing
        physics = prob.analysis.equation_structure
        println("\nPhysics Analysis:")
        println("   - Dimensionality: $(physics[:dimensionality])D")
        println("   - Time evolution: $(physics[:has_time_evolution])")
        println("   - Diffusion: $(physics[:has_diffusion])")
        println("   - Advection: $(physics[:has_advection])")
        println("   - Incompressibility: $(physics[:has_incompressibility])")
    end
    
    println("\n" * "=" ^ 50)
end

# Export the flexible interface
export FlexibleSymbolicProblem, add_flexible_equation!, set_flexible_parameter!
export auto_detect_physics!, build_flexible_problem!, show_flexible_problem_summary
