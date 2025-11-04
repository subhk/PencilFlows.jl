# Automatic Parameter Detection and Conversion System
# ===================================================
# This module automatically analyzes symbolic equations to identify constants,
# variables, and physical parameters, then builds the appropriate solver setup.

using Symbolics
using LinearAlgebra

"""
    EquationAnalysis

Structure to hold the analysis of symbolic equations, including identified
constants, variables, and inferred physical parameters.
"""
struct EquationAnalysis
    # Identified symbols
    variables::Set{Symbol}           # Field variables (u, v, w, p, b, T, etc.)
    constants::Dict{Symbol, Any}     # Physical constants and their values/expressions
    operators::Set{Symbol}           # Differential operators (dx, dy, dz, dt, lap, etc.)
    
    # Equation structure
    time_derivatives::Dict{Symbol, Any}    # dt(u) = ..., dt(T) = ..., etc.
    spatial_terms::Dict{Symbol, Vector{Any}}  # Spatial derivatives for each variable
    nonlinear_terms::Dict{Symbol, Vector{Any}}  # Nonlinear terms for each variable
    coupling_terms::Dict{Symbol, Vector{Any}}   # Cross-variable coupling
    
    # Inferred physics
    equation_type::Symbol            # :navier_stokes, :boussinesq, :thermal_convection, etc.
    dimensionality::Int              # 2D or 3D
    has_rotation::Bool               # Coriolis terms present
    has_stratification::Bool         # Buoyancy/temperature coupling
    has_viscosity::Bool              # Viscous terms present
    has_diffusion::Bool              # Thermal/scalar diffusion
    
    # Inferred parameters
    physical_parameters::Dict{Symbol, Any}  # Detected physical parameters
    dimensionless_groups::Dict{Symbol, Any} # Inferred dimensionless numbers
end

"""
    analyze_symbolic_equations(equations::Vector{String})

Automatically analyze a set of symbolic equations to identify all constants,
variables, and physical parameters, then infer the appropriate solver setup.
"""
function analyze_symbolic_equations(equations::Vector{String})
    println(">> Analyzing symbolic equations automatically...")
    
    # Parse all equations
    parsed_equations = []
    variables = Set{Symbol}()
    constants = Dict{Symbol, Any}()
    operators = Set{Symbol}()
    
    for (i, eq_str) in enumerate(equations)
        println("  Equation $i: $eq_str")
        
        # Parse the equation
        eq_data = parse_single_equation(eq_str)
        push!(parsed_equations, eq_data)
        
        # Collect symbols
        union!(variables, eq_data.variables)
        merge!(constants, eq_data.constants)
        union!(operators, eq_data.operators)
    end
    
    vars_str = join(variables, ", ")
    consts_str = join(keys(constants), ", ")
    println("  >> Found $(length(variables)) variables: $vars_str")
    println("  >> Found $(length(constants)) constants: $consts_str")
    
    # Analyze equation structure
    time_derivs, spatial_terms, nonlinear_terms, coupling_terms = analyze_equation_structure(parsed_equations)
    
    # Infer physics type and characteristics
    physics = infer_physics_type(variables, constants, time_derivs, spatial_terms, coupling_terms)
    
    # Infer physical parameters and dimensionless numbers
    phys_params, dimensionless = infer_physical_parameters(constants, physics)
    
    return EquationAnalysis(
        variables, constants, operators,
        time_derivs, spatial_terms, nonlinear_terms, coupling_terms,
        physics.equation_type, physics.dimensionality,
        physics.has_rotation, physics.has_stratification,
        physics.has_viscosity, physics.has_diffusion,
        phys_params, dimensionless
    )
end

"""
    parse_single_equation(eq_str::String)

Parse a single equation string to identify variables, constants, and operators.
"""
function parse_single_equation(eq_str::String)
    # Remove spaces and split by '='
    clean_eq = replace(eq_str, " " => "")
    lhs, rhs = split(clean_eq, '=')
    
    variables = Set{Symbol}()
    constants = Dict{Symbol, Any}()
    operators = Set{Symbol}()
    
    # Parse both sides
    for side in [lhs, rhs]
        vars, consts, ops = extract_symbols_from_expression(side)
        union!(variables, vars)
        merge!(constants, consts)
        union!(operators, ops)
    end
    
    return (
        equation_string = eq_str,
        lhs = lhs,
        rhs = rhs,
        variables = variables,
        constants = constants,
        operators = operators
    )
end

"""
    extract_symbols_from_expression(expr_str::String)

Extract variables, constants, and operators from an expression string.
"""
function extract_symbols_from_expression(expr_str::String)
    variables = Set{Symbol}()
    constants = Dict{Symbol, Any}()
    operators = Set{Symbol}()
    
    # Known field variables (these are always variables, never constants)
    field_variables = Set([:u, :v, :w, :p, :b, :T, :q, :ω, :ψ, :φ])
    
    # Known operators (these are always operators, never constants)
    known_operators = Set([:dt, :dx, :dy, :dz, :lap, :div, :grad, :curl, :d2dx2, :d2dy2, :d2dz2])
    
    # Common physical constants that are typically constants
    physical_constants = Dict(
        :Re => "Reynolds number",
        :Pr => "Prandtl number", 
        :Ra => "Rayleigh number",
        :Ek => "Ekman number",
        :Ro => "Rossby number",
        :Ri => "Richardson number",
        :nu => "kinematic viscosity",
        :nu_unicode => "kinematic viscosity",
        :kappa => "thermal diffusivity",
        :kappa_unicode => "thermal diffusivity",
        :f => "Coriolis parameter",
        :N => "buoyancy frequency",
        :N2 => "buoyancy frequency squared",
        :g => "gravitational acceleration",
        :alpha => "thermal expansion coefficient",
        :beta => "haline contraction coefficient",
        :beta_unicode => "haline contraction coefficient",
        :Omega => "rotation rate",
        :Omega_unicode => "rotation rate"
    )
    
    # Tokenize the expression (simple approach)
    # This is a simplified parser - in practice you'd use a proper expression parser
    tokens = extract_tokens(expr_str)
    
    for token in tokens
        sym = Symbol(token)
        
        if sym in field_variables
            push!(variables, sym)
        elseif sym in known_operators
            push!(operators, sym)
        elseif haskey(physical_constants, sym)
            constants[sym] = physical_constants[sym]
        elseif is_likely_variable(token)
            push!(variables, sym)
        elseif is_likely_constant(token)
            constants[sym] = "inferred constant"
        end
    end
    
    return variables, constants, operators
end

"""
    extract_tokens(expr_str::String)

Extract individual symbol tokens from an expression string.
"""
function extract_tokens(expr_str::String)
    # Remove common operators and parentheses to isolate symbols
    cleaned = expr_str
    for op in ["(", ")", "+", "-", "*", "/", "^", ".", ","]
        cleaned = replace(cleaned, op => " ")
    end
    
    # Split by whitespace and filter out numbers and empty strings
    tokens = filter(t -> !isempty(t) && !all(isdigit, t) && t != ".", split(cleaned))
    
    return tokens
end

"""
    is_likely_variable(token::String)

Determine if a token is likely a field variable based on naming patterns.
"""
function is_likely_variable(token::String)
    # Variables often have subscripts or are derivatives of field variables
    return (
        occursin("_", token) ||  # u_x, T_z, etc.
        startswith(token, "d") || # du, dT, etc.
        length(token) == 1 ||     # Single letter variables
        token in ["velocity", "temperature", "pressure", "buoyancy"]
    )
end

"""
    is_likely_constant(token::String)

Determine if a token is likely a physical constant based on naming patterns.
"""
function is_likely_constant(token::String)
    # Constants often are Greek letters, have specific patterns, or are numbers
    return (
        length(token) > 1 &&
        (islowercase(token[1])) &&
        !occursin("_", token) &&  # No subscripts typically
        !startswith(token, "d")   # Not derivatives
    )
end

"""
    analyze_equation_structure(parsed_equations)

Analyze the structure of parsed equations to identify different types of terms.
"""
function analyze_equation_structure(parsed_equations)
    time_derivatives = Dict{Symbol, Any}()
    spatial_terms = Dict{Symbol, Vector{Any}}()
    nonlinear_terms = Dict{Symbol, Vector{Any}}()
    coupling_terms = Dict{Symbol, Vector{Any}}()
    
    for eq in parsed_equations
        # Identify time derivatives (dt(...) terms)
        time_var = identify_time_derivative(eq.lhs)
        if time_var !== nothing
            time_derivatives[time_var] = eq.rhs
            
            # Initialize term collections for this variable
            spatial_terms[time_var] = []
            nonlinear_terms[time_var] = []
            coupling_terms[time_var] = []
            
            # Parse RHS terms
            rhs_terms = parse_rhs_terms(eq.rhs, time_var, eq.variables)
            
            append!(spatial_terms[time_var], rhs_terms.spatial)
            append!(nonlinear_terms[time_var], rhs_terms.nonlinear)
            append!(coupling_terms[time_var], rhs_terms.coupling)
        end
    end
    
    return time_derivatives, spatial_terms, nonlinear_terms, coupling_terms
end

"""
    identify_time_derivative(lhs::String)

Identify which variable is being differentiated with respect to time.
"""
function identify_time_derivative(lhs::String)
    # Look for dt(variable) pattern
    if startswith(lhs, "dt(") && endswith(lhs, ")")
        var_str = lhs[4:end-1]  # Extract variable name
        return Symbol(var_str)
    end
    return nothing
end

"""
    parse_rhs_terms(rhs::String, primary_var::Symbol, all_vars::Set{Symbol})

Parse the right-hand side to categorize different types of terms.
"""
function parse_rhs_terms(rhs::String, primary_var::Symbol, all_vars::Set{Symbol})
    spatial = []
    nonlinear = []
    coupling = []
    
    # Split RHS by + and - (keeping signs)
    terms = split_terms_with_signs(rhs)
    
    for term in terms
        if contains_spatial_derivatives(term)
            push!(spatial, term)
        elseif contains_products_of_variables(term, all_vars)
            push!(nonlinear, term)
        elseif contains_other_variables(term, primary_var, all_vars)
            push!(coupling, term)
        else
            push!(spatial, term)  # Default to spatial if unclear
        end
    end
    
    return (spatial=spatial, nonlinear=nonlinear, coupling=coupling)
end

"""
    split_terms_with_signs(expr::String)

Split an expression by + and - while preserving the signs.
"""
function split_terms_with_signs(expr::String)
    terms = []
    current_term = ""
    
    for (i, char) in enumerate(expr)
        if char in ['+', '-'] && i > 1
            push!(terms, strip(current_term))
            current_term = string(char)
        else
            current_term *= char
        end
    end
    
    if !isempty(current_term)
        push!(terms, strip(current_term))
    end
    
    return filter(t -> !isempty(t), terms)
end

"""
    contains_spatial_derivatives(term::String)

Check if a term contains spatial derivative operators.
"""
function contains_spatial_derivatives(term::String)
    spatial_ops = ["dx", "dy", "dz", "lap", "d2dx2", "d2dy2", "d2dz2", "grad", "div", "curl"]
    return any(op -> occursin(op, term), spatial_ops)
end

"""
    contains_products_of_variables(term::String, variables::Set{Symbol})

Check if a term contains products of field variables (nonlinear terms).
"""
function contains_products_of_variables(term::String, variables::Set{Symbol})
    var_count = 0
    for var in variables
        if occursin(string(var), term)
            var_count += 1
        end
    end
    return var_count >= 2 || occursin("*", term)
end

"""
    contains_other_variables(term::String, primary_var::Symbol, all_vars::Set{Symbol})

Check if a term contains variables other than the primary variable (coupling).
"""
function contains_other_variables(term::String, primary_var::Symbol, all_vars::Set{Symbol})
    other_vars = setdiff(all_vars, Set([primary_var]))
    return any(var -> occursin(string(var), term), other_vars)
end

"""
    infer_physics_type(variables, constants, time_derivatives, spatial_terms, coupling_terms)

Infer the type of physics and characteristics based on equation analysis.
"""
function infer_physics_type(variables, constants, time_derivatives, spatial_terms, coupling_terms)
    # Determine equation type based on variables and structure
    equation_type = :unknown
    dimensionality = 3  # Default to 3D
    has_rotation = false
    has_stratification = false
    has_viscosity = false
    has_diffusion = false
    
    # Check for Navier-Stokes equations
    if :u in variables && :v in variables
        if :w in variables
            equation_type = :navier_stokes_3d
            dimensionality = 3
        else
            equation_type = :navier_stokes_2d
            dimensionality = 2
        end
    end
    
    # Check for thermal/buoyancy variables
    if :T in variables || :b in variables ||  in variables
        if equation_type == :navier_stokes_3d
            equation_type = :boussinesq_3d
        elseif equation_type == :navier_stokes_2d
            equation_type = :boussinesq_2d
        else
            equation_type = :thermal_convection
        end
        has_stratification = true
        has_diffusion = true
    end
    
    # Check for rotation (Coriolis terms)
    rotation_indicators = [:f, :Omega, :Omega_unicode, :coriolis]
    has_rotation = any(ind -> haskey(constants, ind), rotation_indicators)
    
    # Check for viscosity
    viscosity_indicators = [:nu, :nu_unicode, :Re, :viscosity]
    has_viscosity = any(ind -> haskey(constants, ind), viscosity_indicators)
    
    # Check for diffusion
    diffusion_indicators = [:kappa, :kappa_unicode, :Pr, :diffusivity, :alpha]
    has_diffusion = any(ind -> haskey(constants, ind), diffusion_indicators)
    
    return (
        equation_type = equation_type,
        dimensionality = dimensionality,
        has_rotation = has_rotation,
        has_stratification = has_stratification,
        has_viscosity = has_viscosity,
        has_diffusion = has_diffusion
    )
end

"""
    infer_physical_parameters(constants, physics_info)

Infer physical parameters and dimensionless numbers from the detected constants.
"""
function infer_physical_parameters(constants, physics_info)
    physical_parameters = Dict{Symbol, Any}()
    dimensionless_groups = Dict{Symbol, Any}()
    
    # Direct physical parameters
    physical_mappings = Dict(
        :nu => :kinematic_viscosity,
        :nu_unicode => :kinematic_viscosity,
        :kappa => :thermal_diffusivity,
        :kappa_unicode => :thermal_diffusivity,
        :f => :coriolis_parameter,
        :N2 => :stratification,
        :g => :gravity,
        :alpha => :thermal_expansion,
        :Omega => :rotation_rate,
        :Omega_unicode => :rotation_rate
    )
    
    for (const_sym, const_desc) in constants
        if haskey(physical_mappings, const_sym)
            physical_parameters[physical_mappings[const_sym]] = const_sym
        end
    end
    
    # Dimensionless numbers
    dimensionless_mappings = Dict(
        :Re => :reynolds_number,
        :Pr => :prandtl_number,
        :Ra => :rayleigh_number,
        :Ek => :ekman_number,
        :Ro => :rossby_number,
        :Ri => :richardson_number
    )
    
    for (const_sym, const_desc) in constants
        if haskey(dimensionless_mappings, const_sym)
            dimensionless_groups[dimensionless_mappings[const_sym]] = const_sym
        end
    end
    
    return physical_parameters, dimensionless_groups
end

"""
    auto_build_problem!(prob::SymbolicProblem, equations::Vector{String})

Automatically build a problem by analyzing the equations and setting up
all parameters, boundary conditions, and solver options.
"""
function auto_build_problem!(prob::SymbolicProblem, equations::Vector{String})
    println(">> Auto-building problem from equations...")
    
    # Analyze equations
    analysis = analyze_symbolic_equations(equations)
    
    # Add all detected constants as parameters
    println(">> Setting up parameters...")
    for (const_sym, description) in analysis.constants
        if !haskey(prob.parameters, const_sym)
            # Set default values based on physical reasoning
            default_val = get_default_parameter_value(const_sym, analysis.equation_type)
            add_parameter!(prob, const_sym, default_val)
            println("    Added parameter $const_sym = $default_val ($description)")
        end
    end
    
    # Set up domain based on dimensionality
    setup_automatic_domain!(prob, analysis)
    
    # Add equations to problem
    for eq in equations
        add_equation!(prob, eq)
    end
    
    # Set up boundary conditions based on physics type
    setup_automatic_boundary_conditions!(prob, analysis)
    
    # Configure solver options
    setup_automatic_solver_options!(prob, analysis)
    
    # Store equation analysis for later use by solvers
    if prob.discretization === nothing
        # Create minimal discretization structure to hold analysis
        prob.discretization = (equation_analysis = analysis,)
    else
        # Add analysis to existing discretization
        prob.discretization = merge(prob.discretization, (equation_analysis = analysis,))
    end
    
    println(">> Problem built automatically!")
    print_problem_summary(prob, analysis)
    
    return prob
end

"""
    get_default_parameter_value(param_sym::Symbol, equation_type::Symbol)

Get reasonable default values for parameters based on equation type.
"""
function get_default_parameter_value(param_sym::Symbol, equation_type::Symbol)
    # Default values based on typical applications
    defaults = Dict(
        :Re => 1000.0,    # Moderate Reynolds number
        :Pr => 1.0,       # Pr = 1 for many fluids
        :Ra => 1e5,       # Supercritical Rayleigh number
        :Ek => 1e-3,      # Laboratory-scale Ekman number
        :Ro => 1.0,       # Moderate Rossby number
        :Ri => 0.25,      # Stable stratification
        :nu => 1e-3,      # Typical kinematic viscosity
        :nu_unicode => 1e-3,       # Typical kinematic viscosity
        :kappa => 1e-3,   # Typical thermal diffusivity
        :kappa_unicode => 1e-3,       # Typical thermal diffusivity
        :f => 1e-4,       # Earth-like Coriolis parameter
        :N2 => 1e-5,      # Typical oceanic stratification
        :g => 9.81,       # Earth gravity
        :alpha => 2e-4,   # Typical thermal expansion
        :Omega => 7.27e-5, # Earth rotation rate
        :Omega_unicode => 7.27e-5     # Earth rotation rate
    )
    
    return get(defaults, param_sym, 1.0)
end

"""
    setup_automatic_domain!(prob::SymbolicProblem, analysis::EquationAnalysis)

Set up computational domain based on equation analysis.
"""
function setup_automatic_domain!(prob::SymbolicProblem, analysis::EquationAnalysis)
    if analysis.dimensionality == 3
        # 3D domain
        x = Coordinate(:x, basis=Fourier(), domain=(-1.0, 1.0))
        y = Coordinate(:y, basis=Fourier(), domain=(-1.0, 1.0))
        z = Coordinate(:z, basis=FiniteDifference(), domain=(0.0, 1.0))
        domain = Domain([x, y, z])
    else
        # 2D domain
        x = Coordinate(:x, basis=Fourier(), domain=(-1.0, 1.0))
        z = Coordinate(:z, basis=FiniteDifference(), domain=(0.0, 1.0))
        domain = Domain([x, z])
    end
    
    set_domain!(prob, domain)
    println("    Set up $(analysis.dimensionality)D domain")
end

"""
    setup_automatic_boundary_conditions!(prob::SymbolicProblem, analysis::EquationAnalysis)

Set up boundary conditions based on physics type and variables.
"""
function setup_automatic_boundary_conditions!(prob::SymbolicProblem, analysis::EquationAnalysis)
    println(">> Setting up boundary conditions...")
    
    # No-slip boundary conditions for velocity
    if :u in analysis.variables
        add_bc!(prob, "bottom(u) = 0")
        add_bc!(prob, "top(u) = 0")
        println("    Added no-slip BCs for u")
    end
    
    if :v in analysis.variables
        add_bc!(prob, "bottom(v) = 0")
        add_bc!(prob, "top(v) = 0")
        println("    Added no-slip BCs for v")
    end
    
    if :w in analysis.variables
        add_bc!(prob, "bottom(w) = 0")
        add_bc!(prob, "top(w) = 0")
        println("    Added no-penetration BCs for w")
    end
    
    # Thermal boundary conditions
    if :T in analysis.variables
        if analysis.equation_type == :thermal_convection || analysis.equation_type in [:boussinesq_2d, :boussinesq_3d]
            add_bc!(prob, "bottom(T) = 1")  # Hot bottom
            add_bc!(prob, "top(T) = 0")     # Cold top
            println("    Added thermal BCs for T")
        end
    end
    
    # Buoyancy boundary conditions
    if :b in analysis.variables
        add_bc!(prob, "bottom(dz(b)) = 0")  # Neumann for buoyancy
        add_bc!(prob, "top(dz(b)) = 0")
        println("    Added buoyancy BCs")
    end
end

"""
    setup_automatic_solver_options!(prob::SymbolicProblem, analysis::EquationAnalysis)

Configure solver options based on physics type.
"""
function setup_automatic_solver_options!(prob::SymbolicProblem, analysis::EquationAnalysis)
    println(">> Configuring solver options...")
    
    # Store solver preferences in problem metadata
    if prob.metadata === nothing
        prob.metadata = Dict{Symbol, Any}()
    end
    
    # Choose time stepping scheme based on physics
    if analysis.equation_type in [:navier_stokes_2d, :navier_stokes_3d, :boussinesq_2d, :boussinesq_3d]
        prob.metadata[:time_scheme] = :predictor_corrector
        prob.metadata[:use_pressure_projection] = true
        println("    Selected predictor-corrector with pressure projection")
    else
        prob.metadata[:time_scheme] = :LSRK4
        println("    Selected low-storage Runge-Kutta")
    end
    
    # Set CFL constraints based on physics
    if analysis.has_rotation && haskey(analysis.constants, :f)
        prob.metadata[:cfl_rotation] = true
        println("    Enabled rotation CFL constraint")
    end
    
    if analysis.has_viscosity
        prob.metadata[:cfl_viscous] = true
        println("    Enabled viscous CFL constraint")
    end
end

"""
    print_problem_summary(prob::SymbolicProblem, analysis::EquationAnalysis)

Print a summary of the automatically configured problem.
"""
function print_problem_summary(prob::SymbolicProblem, analysis::EquationAnalysis)
    println("\n" * "="^60)
    println(" AUTOMATIC PROBLEM SETUP SUMMARY")
    println("="^60)
    
    println(" Equation Type: $(analysis.equation_type)")
    println(" Dimensionality: $(analysis.dimensionality)D")
    println(" Rotation: $(analysis.has_rotation ? "Yes" : "No")")
    println(" Stratification: $(analysis.has_stratification ? "Yes" : "No")")
    println(" Viscosity: $(analysis.has_viscosity ? "Yes" : "No")")
    println(" Diffusion: $(analysis.has_diffusion ? "Yes" : "No")")

    println("\n Variables ($(length(analysis.variables))): $(join(analysis.variables, ", "))")
    println("   Constants ($(length(analysis.constants))): $(join(keys(analysis.constants), ", "))")

    if !isempty(analysis.physical_parameters)
        println("\n Physical Parameters:")
        for (param, symbol) in analysis.physical_parameters
            value = get(prob.parameters, symbol, "not set")
            println("    $param ($symbol) = $value")
        end
    end
    
    if !isempty(analysis.dimensionless_groups)
        println("\n Dimensionless Numbers:")
        for (group, symbol) in analysis.dimensionless_groups
            value = get(prob.parameters, symbol, "not set")
            println("    $group ($symbol) = $value")
        end
    end
    
    println("\n� Boundary Conditions: $(length(prob.boundary_conditions)) conditions set")
    println(" Time Integration: $(get(prob.metadata, :time_scheme, "default"))")
    
    println("\n Ready for solve!(prob) with automatic parameter conversion!")
    println("="^60)
end

# Export the main functions
export analyze_symbolic_equations, auto_build_problem!, EquationAnalysis