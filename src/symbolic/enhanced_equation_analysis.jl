# Enhanced General PDE Analysis and Problem Building System
# =========================================================
# This module creates a fully general PDE solver that can handle arbitrary 
# systems of equations and automatically build the appropriate solver setup.

using Symbolics, LinearAlgebra
using Printf

"""
    GeneralPDESystem

Structure to hold a general PDE system with arbitrary equations and variables.
"""
mutable struct GeneralPDESystem
    # Raw equation specifications
    equations::Vector{String}
    variables::Set{Symbol}
    parameters::Dict{Symbol, Float64}
    
    # Analysis results
    equation_structure::Dict{Symbol, Any}
    coupling_matrix::Matrix{Bool}
    derivative_orders::Dict{Symbol, Dict{Symbol, Int}}  # var -> {space_dir -> max_order}
    temporal_structure::Dict{Symbol, Symbol}  # var -> time_derivative_type
    
    # Physical interpretation
    physics_type::Symbol
    dimensionality::Int
    characteristic_scales::Dict{Symbol, Any}
    
    # Solver configuration
    discretization_method::Dict{Symbol, Symbol}  # coord -> :fourier, :finite_diff, :spectral
    boundary_condition_types::Dict{Symbol, Vector{Symbol}}
    time_integration_scheme::Symbol
    
    function GeneralPDESystem(equations::Vector{String})
        new(equations, Set{Symbol}(), Dict{Symbol, Float64}(),
            Dict{Symbol, Any}(), zeros(Bool, 0, 0), Dict{Symbol, Dict{Symbol, Int}}(),
            Dict{Symbol, Symbol}(), :unknown, 3, Dict{Symbol, Any}(),
            Dict{Symbol, Symbol}(), Dict{Symbol, Vector{Symbol}}(), :auto)
    end
end

"""
    analyze_general_pde_system(equations::Vector{String})

Comprehensively analyze any system of PDEs to extract all structure and build
the appropriate numerical solver automatically.
"""
function analyze_general_pde_system(equations::Vector{String})
    println(" ANALYZING GENERAL PDE SYSTEM")
    println("="^50)
    
    system = GeneralPDESystem(equations)
    
    # Step 1: Parse and extract all symbols
    extract_symbols_from_equations!(system)
    
    # Step 2: Analyze equation structure
    analyze_equation_structure!(system)
    
    # Step 3: Build coupling matrix
    build_coupling_matrix!(system)
    
    # Step 4: Analyze derivative orders and spatial structure
    analyze_spatial_structure!(system)
    
    # Step 5: Infer physics and choose methods
    infer_physics_and_methods!(system)
    
    # Step 6: Suggest solver configuration
    configure_solver!(system)
    
    print_analysis_summary(system)
    return system
end

"""
    extract_symbols_from_equations!(system::GeneralPDESystem)

Extract all variables and parameters from the equation strings.
"""
function extract_symbols_from_equations!(system::GeneralPDESystem)
    println(" Extracting symbols from equations...")
    
    all_symbols = Set{Symbol}()
    parameters = Dict{Symbol, Float64}()
    
    for (i, eq) in enumerate(system.equations)
        println("  Equation $i: $eq")
        
        # Parse the equation
        clean_eq = replace(eq, " " => "")
        lhs, rhs = split(clean_eq, "=")
        
        # Extract symbols from both sides
        lhs_symbols = extract_symbols_from_string(lhs)
        rhs_symbols = extract_symbols_from_string(rhs)
        
        union!(all_symbols, lhs_symbols, rhs_symbols)
    end
    
    # Classify symbols
    variables, params = classify_symbols(all_symbols)
    system.variables = variables
    system.parameters = params
    
    println("   Found $(length(variables)) variables: $(join(sort(collect(variables)), ", "))")
    println("   Found $(length(params)) parameters: $(join(sort(collect(keys(params))), ", "))")
end

"""
    extract_symbols_from_string(expr::String)

Extract all symbol tokens from an expression string.
"""
function extract_symbols_from_string(expr::String)
    symbols = Set{Symbol}()
    
    # Remove operators and parentheses
    cleaned = expr
    for op in ["(", ")", "+", "-", "*", "/", "^", ".", ",", "="]
        cleaned = replace(cleaned, op => " ")
    end
    
    # Extract tokens
    tokens = filter(t -> !isempty(t) && !all(isdigit, t) && !contains(t, "."), split(cleaned))
    
    # Convert to symbols, filtering out derivatives and operators
    for token in tokens
        if !is_differential_operator(token) && !isnumeric_token(token)
            push!(symbols, Symbol(token))
        end
    end
    
    return symbols
end

"""
    is_differential_operator(token::String)

Check if a token represents a differential operator.
"""
function is_differential_operator(token::String)
    differential_ops = ["dt", "dx", "dy", "dz", "d2x", "d2y", "d2z", "lap", "div", "grad", "curl", "d2dx2", "d2dy2", "d2dz2"]
    return token in differential_ops || startswith(token, "d") && length(token) <= 3
end

"""
    isnumeric_token(token::String)

Check if a token is numeric.
"""
function isnumeric_token(token::String)
    try
        parse(Float64, token)
        return true
    catch
        return false
    end
end

"""
    classify_symbols(all_symbols::Set{Symbol})

Classify symbols into variables and parameters based on naming conventions.
"""
function classify_symbols(all_symbols::Set{Symbol})
    variables = Set{Symbol}()
    parameters = Dict{Symbol, Float64}()
    
    # Known field variables
    field_vars = Set([:u, :v, :w, :p, :T, :b, :ρ, :φ, :ψ, :q, :ω, :S, :c, :n, :h])
    
    # Known parameters with default values
    param_defaults = Dict(
        :Re => 1000.0, :Pr => 1.0, :Ra => 1e5, :Ek => 1e-3, :Ro => 1.0,
        :nu => 1e-3, :ν => 1e-3, :kappa => 1e-3, :κ => 1e-3,
        :f => 1e-4, :N2 => 1e-5, :g => 9.81, :alpha => 2e-4, :β => 2e-4,
        :sigma => 1.0, :gamma => 1.4, :Pe => 1000.0, :Da => 1e-3,
        :Ha => 100.0, :Pm => 1.0, :Le => 1.0, :Sc => 1.0,
        :A => 1.0, :B => 1.0, :C => 1.0, :D => 1.0, :E => 1.0
    )
    
    for sym in all_symbols
        if sym in field_vars || (length(string(sym)) == 1 && islowercase(string(sym)[1]))
            push!(variables, sym)
        elseif haskey(param_defaults, sym)
            parameters[sym] = param_defaults[sym]
        elseif is_likely_parameter(string(sym))
            parameters[sym] = 1.0  # Default value
        else
            # Default to variable if unclear
            push!(variables, sym)
        end
    end
    
    return variables, parameters
end

"""
    is_likely_parameter(s::String)

Determine if a string likely represents a parameter.
"""
function is_likely_parameter(s::String)
    return (
        length(s) > 1 &&
        (islowercase(s[1]) || s[1] in ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'κ', 'λ', 'μ', 'ν', 'ρ', 'σ', 'τ', 'φ', 'χ', 'ψ', 'ω']) &&
        !occursin("_", s)
    )
end

"""
    analyze_equation_structure!(system::GeneralPDESystem)

Analyze the mathematical structure of each equation.
"""
function analyze_equation_structure!(system::GeneralPDESystem)
    println(" Analyzing equation structure...")
    
    for (i, eq) in enumerate(system.equations)
        println("  Analyzing equation $i...")
        
        clean_eq = replace(eq, " " => "")
        lhs, rhs = split(clean_eq, "=")
        
        # Identify the primary variable (from time derivative on LHS)
        primary_var = identify_primary_variable(lhs)
        if primary_var !== nothing
            system.equation_structure[primary_var] = Dict(
                :lhs => lhs,
                :rhs => rhs,
                :equation_index => i,
                :time_derivative_present => contains_time_derivative(lhs),
                :spatial_terms => extract_spatial_terms(rhs),
                :nonlinear_terms => extract_nonlinear_terms(rhs, system.variables),
                :coupling_terms => extract_coupling_terms(rhs, primary_var, system.variables),
                :source_terms => extract_source_terms(rhs, system.variables)
            )
            
            if contains_time_derivative(lhs)
                system.temporal_structure[primary_var] = identify_time_derivative_type(lhs)
            end
        end
    end
    
    println("   Analyzed $(length(system.equation_structure)) equations")
end

"""
    identify_primary_variable(lhs::String)

Identify the primary variable being evolved in an equation.
"""
function identify_primary_variable(lhs::String)
    # Look for dt(var) or d_t(var) patterns
    if occursin("dt(", lhs) || occursin("d_t(", lhs)
        start_idx = findfirst("(", lhs)[1] + 1
        end_idx = findlast(")", lhs)[1] - 1
        var_str = lhs[start_idx:end_idx]
        return Symbol(var_str)
    end
    
    # Look for bare variables on LHS
    tokens = filter(t -> !isempty(t), split(replace(lhs, r"[()+=*/^-]" => " ")))
    for token in tokens
        if !is_differential_operator(token)
            return Symbol(token)
        end
    end
    
    return nothing
end

"""
    identify_time_derivative_type(lhs::String)

Identify the type of time derivative.
"""
function identify_time_derivative_type(lhs::String)
    if occursin("d2t", lhs) || occursin("d²t", lhs)
        return :second_order_time
    elseif occursin("dt", lhs) || occursin("d_t", lhs)
        return :first_order_time
    else
        return :algebraic
    end
end

"""
    contains_time_derivative(expr::String)

Check if expression contains time derivatives.
"""
function contains_time_derivative(expr::String)
    time_patterns = ["dt(", "d_t(", "∂_t", "d2t", "d²t"]
    return any(pattern -> occursin(pattern, expr), time_patterns)
end

"""
    extract_spatial_terms(rhs::String)

Extract terms containing spatial derivatives.
"""
function extract_spatial_terms(rhs::String)
    spatial_ops = ["dx", "dy", "dz", "lap", "div", "grad", "curl", "d2dx2", "d2dy2", "d2dz2", "∇", "Δ"]
    terms = String[]
    
    # Split into terms (simple approach)
    parts = split(rhs, r"[+-]")
    for part in parts
        if any(op -> occursin(op, part), spatial_ops)
            push!(terms, strip(part))
        end
    end
    
    return terms
end

"""
    extract_nonlinear_terms(rhs::String, variables::Set{Symbol})

Extract nonlinear terms (products of variables).
"""
function extract_nonlinear_terms(rhs::String, variables::Set{Symbol})
    terms = String[]
    
    # Look for products of variables
    parts = split(rhs, r"[+-]")
    for part in parts
        var_count = 0
        for var in variables
            if occursin(string(var), part)
                var_count += 1
            end
        end
        if var_count >= 2 || occursin("*", part)
            push!(terms, strip(part))
        end
    end
    
    return terms
end

"""
    extract_coupling_terms(rhs::String, primary_var::Symbol, variables::Set{Symbol})

Extract terms that couple to other variables.
"""
function extract_coupling_terms(rhs::String, primary_var::Symbol, variables::Set{Symbol})
    terms = String[]
    other_vars = setdiff(variables, Set([primary_var]))
    
    parts = split(rhs, r"[+-]")
    for part in parts
        for var in other_vars
            if occursin(string(var), part) && !occursin("*", part)
                push!(terms, strip(part))
                break
            end
        end
    end
    
    return terms
end

"""
    extract_source_terms(rhs::String, variables::Set{Symbol})

Extract source terms (constant or parameter-dependent).
"""
function extract_source_terms(rhs::String, variables::Set{Symbol})
    terms = String[]
    
    parts = split(rhs, r"[+-]")
    for part in parts
        has_variables = any(var -> occursin(string(var), part), variables)
        has_spatial_ops = any(op -> occursin(op, part), ["dx", "dy", "dz", "lap", "div", "grad"])
        
        if !has_variables && !has_spatial_ops && !isempty(strip(part))
            push!(terms, strip(part))
        end
    end
    
    return terms
end

"""
    build_coupling_matrix!(system::GeneralPDESystem)

Build a matrix showing which variables are coupled to which.
"""
function build_coupling_matrix!(system::GeneralPDESystem)
    println(" Building variable coupling matrix...")
    
    vars = sort(collect(system.variables))
    n_vars = length(vars)
    system.coupling_matrix = zeros(Bool, n_vars, n_vars)
    
    for (i, var1) in enumerate(vars)
        for (j, var2) in enumerate(vars)
            if haskey(system.equation_structure, var1)
                eq_data = system.equation_structure[var1]
                rhs = eq_data[:rhs]
                if occursin(string(var2), rhs)
                    system.coupling_matrix[i, j] = true
                end
            end
        end
    end
    
    coupled_pairs = sum(system.coupling_matrix) - length(vars)  # Subtract diagonal
    println("   Built coupling matrix: $(coupled_pairs) coupling relationships")
end

"""
    analyze_spatial_structure!(system::GeneralPDESystem)

Analyze the spatial derivative structure of the system.
"""
function analyze_spatial_structure!(system::GeneralPDESystem)
    println(" Analyzing spatial structure...")
    
    spatial_coords = Set{Symbol}()
    max_orders = Dict{Symbol, Int}()
    
    for var in system.variables
        system.derivative_orders[var] = Dict{Symbol, Int}()
        
        if haskey(system.equation_structure, var)
            rhs = system.equation_structure[var][:rhs]
            
            # Check for derivatives in each direction
            for coord in [:x, :y, :z]
                order = get_max_derivative_order(rhs, coord)
                if order > 0
                    system.derivative_orders[var][coord] = order
                    push!(spatial_coords, coord)
                    max_orders[coord] = max(get(max_orders, coord, 0), order)
                end
            end
        end
    end
    
    system.dimensionality = length(spatial_coords)
    println("   Detected $(system.dimensionality)D system with coordinates: $(join(spatial_coords, ", "))")
    println("   Maximum derivative orders: $(max_orders)")
end

"""
    get_max_derivative_order(expr::String, coord::Symbol)

Get the maximum order of derivatives with respect to a coordinate.
"""
function get_max_derivative_order(expr::String, coord::Symbol)
    max_order = 0
    coord_str = string(coord)
    
    # Check for various derivative patterns
    patterns = [
        "d$(coord_str)" => 1,
        "d2$(coord_str)" => 2,
        "d2d$(coord_str)2" => 2,
        "∂$(coord_str)" => 1,
        "∂²$(coord_str)" => 2
    ]
    
    for (pattern, order) in patterns
        if occursin(pattern, expr)
            max_order = max(max_order, order)
        end
    end
    
    # Check for Laplacian (includes all second derivatives)
    if occursin("lap", expr) || occursin("Δ", expr) || occursin("∇²", expr)
        max_order = max(max_order, 2)
    end
    
    return max_order
end

"""
    infer_physics_and_methods!(system::GeneralPDESystem)

Infer the type of physics and suggest appropriate numerical methods.
"""
function infer_physics_and_methods!(system::GeneralPDESystem)
    println("Inferring physics and numerical methods...")
    
    # Classify physics based on variables and structure
    if :u in system.variables && :v in system.variables
        if :T in system.variables || :b in system.variables
            system.physics_type = system.dimensionality == 3 ? :thermal_convection_3d : :thermal_convection_2d
        else
            system.physics_type = system.dimensionality == 3 ? :navier_stokes_3d : :navier_stokes_2d
        end
    elseif :T in system.variables
        system.physics_type = :heat_equation
    elseif :c in system.variables
        system.physics_type = :reaction_diffusion
    elseif :φ in system.variables
        system.physics_type = :wave_equation
    else
        system.physics_type = :general_pde_system
    end
    
    # Suggest discretization methods based on physics type
    suggest_discretization_methods!(system)
    
    println("   Inferred physics: $(system.physics_type)")
end

"""
    suggest_discretization_methods!(system::GeneralPDESystem)

Suggest appropriate discretization methods for each coordinate.
"""
function suggest_discretization_methods!(system::GeneralPDESystem)
    # Default suggestions based on common practice
    if system.dimensionality >= 2
        # Horizontal directions: periodic -> Fourier, bounded -> finite differences
        system.discretization_method[:x] = :fourier  # Assuming periodic
        if system.dimensionality >= 3
            system.discretization_method[:y] = :fourier  # Assuming periodic
        end
        # Vertical direction: usually bounded -> finite differences
        system.discretization_method[:z] = :finite_difference
    else
        # 1D case
        system.discretization_method[:x] = :finite_difference
    end
    
    println("   Suggested discretization methods: $(system.discretization_method)")
end

"""
    configure_solver!(system::GeneralPDESystem)

Configure solver parameters based on the analyzed system.
"""
function configure_solver!(system::GeneralPDESystem)
    println("  Configuring solver...")
    
    # Choose time integration scheme
    has_time_derivatives = any(eq -> haskey(eq, :time_derivative_present) && eq[:time_derivative_present], 
                              values(system.equation_structure))
    
    if has_time_derivatives
        if system.physics_type in [:navier_stokes_2d, :navier_stokes_3d, :thermal_convection_2d, :thermal_convection_3d]
            system.time_integration_scheme = :predictor_corrector
        else
            system.time_integration_scheme = :runge_kutta_4
        end
    else
        system.time_integration_scheme = :steady_state
    end
    
    # Suggest boundary conditions
    suggest_boundary_conditions!(system)
    
    println("   Selected time integration: $(system.time_integration_scheme)")
end

"""
    suggest_boundary_conditions!(system::GeneralPDESystem)

Suggest appropriate boundary conditions based on physics.
"""
function suggest_boundary_conditions!(system::GeneralPDESystem)
    for var in system.variables
        bcs = Symbol[]
        
        if var in [:u, :v, :w]  # Velocity components
            push!(bcs, :no_slip)
        elseif var == :T  # Temperature
            push!(bcs, :fixed_temperature)
        elseif var == :p  # Pressure
            push!(bcs, :neumann)
        elseif var == :b  # Buoyancy
            push!(bcs, :neumann)
        else
            push!(bcs, :periodic)  # Default
        end
        
        system.boundary_condition_types[var] = bcs
    end
end

"""
    print_analysis_summary(system::GeneralPDESystem)

Print a comprehensive summary of the analysis.
"""
function print_analysis_summary(system::GeneralPDESystem)
    println("\n" * "="^60)
    println(" GENERAL PDE SYSTEM ANALYSIS COMPLETE")
    println("="^60)
    
    println(" System Overview:")
    println("  • Physics Type: $(system.physics_type)")
    println("  • Dimensionality: $(system.dimensionality)D")
    println("  • Variables ($(length(system.variables))): $(join(sort(collect(system.variables)), ", "))")
    println("  • Parameters ($(length(system.parameters))): $(join(sort(collect(keys(system.parameters))), ", "))")
    
    println("\n Equation Structure:")
    for (var, eq_data) in system.equation_structure
        temporal_type = get(system.temporal_structure, var, :algebraic)
        println("  • $(var): $(temporal_type) equation")
        println("    - Spatial terms: $(length(eq_data[:spatial_terms]))")
        println("    - Nonlinear terms: $(length(eq_data[:nonlinear_terms]))")
        println("    - Coupling terms: $(length(eq_data[:coupling_terms]))")
    end
    
    println("\n Solver Configuration:")
    println("  • Time Integration: $(system.time_integration_scheme)")
    println("  • Discretization Methods:")
    for (coord, method) in system.discretization_method
        println("    - $(coord): $(method)")
    end
    
    println("\n Ready for automatic problem building!")
    println("="^60)
end

"""
    build_general_pde_problem(equations::Vector{String}; kwargs...)

Main interface function to build a general PDE problem from equations.
"""
function build_general_pde_problem(equations::Vector{String}; 
                                  domain_size::Tuple = (2π, 2π, 1.0),
                                  resolution::Tuple = (64, 64, 32),
                                  parameter_overrides::Dict = Dict())
    
    println(" BUILDING GENERAL PDE PROBLEM")
    println("="^40)
    
    # Analyze the system
    system = analyze_general_pde_system(equations)
    
    # Create PencilFlows problem
    prob = SymbolicProblem()
    
    # Set up domain
    setup_domain_from_analysis!(prob, system, domain_size, resolution)
    
    # Add parameters
    add_parameters_from_analysis!(prob, system, parameter_overrides)
    
    # Add equations
    for eq in equations
        add_equation!(prob, eq)
    end
    
    # Add boundary conditions
    add_boundary_conditions_from_analysis!(prob, system)
    
    println("\n General PDE problem built successfully!")
    return prob, system
end

"""
    setup_domain_from_analysis!(prob, system, domain_size, resolution)

Set up the computational domain based on system analysis.
"""
function setup_domain_from_analysis!(prob, system, domain_size, resolution)
    bases = []
    points = []
    
    coords = [:x, :y, :z][1:system.dimensionality]
    
    for (i, coord) in enumerate(coords)
        method = get(system.discretization_method, coord, :fourier)
        
        if method == :fourier
            basis = Fourier(coord, (-domain_size[i]/2, domain_size[i]/2))
        else
            basis = FiniteDifference(coord, (0.0, domain_size[i]))
        end
        
        push!(bases, basis)
        push!(points, resolution[i])
    end
    
    domain = Domain([(bases[i], points[i]) for i in 1:length(bases)]...)
    set_domain!(prob, domain)
    
    println("   Set up $(system.dimensionality)D domain")
end

"""
    add_parameters_from_analysis!(prob, system, overrides)

Add parameters to the problem from system analysis.
"""
function add_parameters_from_analysis!(prob, system, overrides)
    for (param, default_val) in system.parameters
        value = get(overrides, param, default_val)
        add_parameter!(prob, param, value)
    end
    
    println("   Added $(length(system.parameters)) parameters")
end

"""
    add_boundary_conditions_from_analysis!(prob, system)

Add appropriate boundary conditions based on system analysis.
"""
function add_boundary_conditions_from_analysis!(prob, system)
    bc_count = 0
    
    for (var, bc_types) in system.boundary_condition_types
        for bc_type in bc_types
            if bc_type == :no_slip
                add_bc!(prob, "bottom($var) = 0")
                add_bc!(prob, "top($var) = 0")
                bc_count += 2
            elseif bc_type == :fixed_temperature && var == :T
                add_bc!(prob, "bottom($var) = 1")
                add_bc!(prob, "top($var) = 0")
                bc_count += 2
            elseif bc_type == :neumann
                add_bc!(prob, "bottom(dz($var)) = 0")
                add_bc!(prob, "top(dz($var)) = 0")
                bc_count += 2
            end
        end
    end
    
    println("   Added $bc_count boundary conditions")
end

# Export the main functions
export GeneralPDESystem, analyze_general_pde_system, build_general_pde_problem
