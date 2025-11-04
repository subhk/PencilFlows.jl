# Improved IVP Equation Parser for PencilFlows.jl
# Self-contained parser without external dependencies

using Symbolics, LinearAlgebra

"""
    ImprovedIVPParser

Advanced IVP equation parser with better operator handling,
validation, and structured problem formulation.
"""
mutable struct ImprovedIVPParser
    equations::Vector{String}
    variables::Set{Symbol}
    parameters::Dict{Symbol, Float64}
    boundary_conditions::Vector{String}
    domain::Dict{Symbol, Any}
    namespace::Dict{Symbol, Any}
    parsed_equations::Vector{Dict{String, Any}}
    
    function ImprovedIVPParser()
        new(String[], Set{Symbol}(), Dict{Symbol, Float64}(), String[],
            Dict{Symbol, Any}(), Dict{Symbol, Any}(), Dict{String, Any}[])
    end
end

"""
    set_domain!(parser::ImprovedIVPParser; 
                coords=[:x, :y, :z],
                domain_size=(2π, 2π, 2.0),
                grid_points=(64, 64, 32),
                basis_types=[:fourier, :fourier, :chebyshev])

Set up the computational domain.
"""
function set_domain!(parser::ImprovedIVPParser;
                    coords=[:x, :y, :z],
                    domain_size=(2π, 2π, 2.0),
                    grid_points=(64, 64, 32),
                    basis_types=[:fourier, :fourier, :chebyshev])
    
    parser.domain[:coordinates] = coords
    parser.domain[:domain_size] = domain_size
    parser.domain[:grid_points] = grid_points
    parser.domain[:basis_types] = basis_types
    
    # Add coordinates to namespace
    for coord in coords
        parser.namespace[coord] = coord
    end
    
    @info "Domain configured" coords=coords domain_size=domain_size grid_points=grid_points
    return parser
end

"""
    add_variable!(parser::ImprovedIVPParser, var::Symbol)

Add a variable to the problem.
"""
function add_variable!(parser::ImprovedIVPParser, var::Symbol)
    push!(parser.variables, var)
    parser.namespace[var] = var
    @debug "Added variable: $var"
    return parser
end

"""
    set_parameter!(parser::ImprovedIVPParser, param::Symbol, value::Float64)

Set a parameter value.
"""
function set_parameter!(parser::ImprovedIVPParser, param::Symbol, value::Float64)
    parser.parameters[param] = value
    parser.namespace[param] = value
    @debug "Set parameter: $param = $value"
    return parser
end

"""
    parse_equation!(parser::ImprovedIVPParser, equation::String)

Parse an equation with improved operator handling and validation.
"""
function parse_equation!(parser::ImprovedIVPParser, equation::String)
    try
        # Clean and normalize equation
        clean_eq = normalize_equation(equation)
        
        # Split into LHS and RHS
        lhs_str, rhs_str = split_equation(clean_eq)
        
        # Parse expressions
        lhs_expr = parse_expression(lhs_str)
        rhs_expr = parse_expression(rhs_str)
        
        # Extract variables and parameters
        eq_vars = extract_symbols(lhs_expr) ∪ extract_symbols(rhs_expr)
        for var in eq_vars
            if is_variable(var)
                add_variable!(parser, var)
            elseif is_parameter(var)
                if !haskey(parser.parameters, var)
                    default_val = guess_parameter_default(var)
                    set_parameter!(parser, var, default_val)
                end
            end
        end
        
        # Validate equation structure for IVP
        validate_ivp_equation(lhs_expr, rhs_expr, equation)
        
        # Store parsed equation
        eq_data = Dict{String, Any}(
            "original" => equation,
            "clean" => clean_eq,
            "lhs" => lhs_expr,
            "rhs" => rhs_expr,
            "lhs_str" => lhs_str,
            "rhs_str" => rhs_str,
            "variables" => eq_vars,
            "index" => length(parser.equations) + 1
        )
        
        push!(parser.equations, clean_eq)
        push!(parser.parsed_equations, eq_data)
        
        @info "Equation parsed successfully" equation=equation
        return true
        
    catch e
        @error "Failed to parse equation: $equation" exception=e
        return false
    end
end

"""
    add_boundary_condition!(parser::ImprovedIVPParser, bc::String)

Add a boundary condition.
"""
function add_boundary_condition!(parser::ImprovedIVPParser, bc::String)
    try
        # Parse and validate BC
        bc_data = parse_boundary_condition(bc)
        
        # Extract any new variables/parameters
        if haskey(bc_data, :variable)
            add_variable!(parser, bc_data[:variable])
        end
        
        push!(parser.boundary_conditions, bc)
        @info "Boundary condition added" bc=bc
        return true
        
    catch e
        @error "Failed to add boundary condition: $bc" exception=e
        return false
    end
end

"""
    normalize_equation(equation::String) -> String

Clean and normalize equation string.
"""
function normalize_equation(equation::String)
    # Remove extra whitespace
    clean = strip(equation)
    
    # Convert ** to ^ (Python to Julia exponentiation)
    clean = replace(clean, "**" => "^")
    
    # Normalize operators
    clean = replace(clean, "lap(" => "laplacian(")
    clean = replace(clean, "div(" => "divergence(")
    
    return clean
end

"""
    split_equation(equation::String) -> Tuple{String, String}

Split equation into LHS and RHS, handling nested parentheses.
"""
function split_equation(equation::String)
    # Find top-level equals signs by tracking parenthetical level
    paren_level = 0
    equals_positions = Int[]
    
    for (i, char) in enumerate(equation)
        if char == '('
            paren_level += 1
        elseif char == ')'
            paren_level -= 1
        elseif char == '=' && paren_level == 0
            push!(equals_positions, i)
        end
    end
    
    if length(equals_positions) != 1
        throw(ArgumentError("Equation must have exactly one top-level equals sign: $equation"))
    end
    
    eq_pos = equals_positions[1]
    lhs = strip(equation[1:eq_pos-1])
    rhs = strip(equation[eq_pos+1:end])
    
    return lhs, rhs
end

"""
    parse_expression(expr_str::String)

Parse expression string into symbolic form.
"""
function parse_expression(expr_str::String)
    if isempty(strip(expr_str))
        return :(0)
    end
    
    # Handle special cases
    if expr_str == "0"
        return :(0)
    end
    
    try
        # Use Julia's Meta.parse for robust parsing
        parsed = Meta.parse(expr_str)
        return parsed
        
    catch e
        @warn "Parse error in expression: $expr_str"
        return Symbol(expr_str)  # Fallback to symbol
    end
end

"""
    extract_symbols(expr) -> Set{Symbol}

Extract all symbols from a parsed expression.
"""
function extract_symbols(expr)
    symbols = Set{Symbol}()
    
    if isa(expr, Symbol)
        push!(symbols, expr)
    elseif isa(expr, Expr)
        for arg in expr.args
            union!(symbols, extract_symbols(arg))
        end
    end
    
    return symbols
end

"""
    is_variable(sym::Symbol) -> Bool

Determine if symbol is a field variable.
"""
function is_variable(sym::Symbol)
    # Common field variables
    field_vars = [:u, :v, :w, :p, :b, :T, :theta, :rho, :psi, :phi, :omega]
    
    # Not coordinates or known parameters
    not_coords = sym ∉ [:x, :y, :z, :t]
    not_params = sym ∉ [:nu, :kappa, :f, :Ra, :Pr, :Re, :Pe, :Ri]
    
    return (sym in field_vars) || (not_coords && not_params && length(string(sym)) <= 3)
end

"""
    is_parameter(sym::Symbol) -> Bool

Determine if symbol is a parameter.
"""
function is_parameter(sym::Symbol)
    # Common parameters
    params = [:nu, :kappa, :alpha, :beta, :gamma, :f, :Ra, :Pr, :Re, :Pe, :Ri, :Ro, :Ek]
    
    return sym in params || (!is_variable(sym) && sym ∉ [:x, :y, :z, :t])
end

"""
    guess_parameter_default(param::Symbol) -> Float64

Guess reasonable default value for parameter.
"""
function guess_parameter_default(param::Symbol)
    defaults = Dict(
        :nu => 1e-4,      # viscosity
        :kappa => 1e-5,   # thermal diffusivity
        :alpha => 1e-3,   # thermal expansion
        :f => 1e-4,       # Coriolis parameter
        :Ra => 1e6,       # Rayleigh number
        :Pr => 1.0,       # Prandtl number
        :Re => 1000.0,    # Reynolds number
        :Pe => 1000.0,    # Peclet number
        :Ri => 0.1,       # Richardson number
        :Ro => 0.1,       # Rossby number
        :Ek => 1e-4       # Ekman number
    )
    
    return get(defaults, param, 1.0)
end

"""
    validate_ivp_equation(lhs, rhs, original::String)

Validate equation structure for proper IVP formulation.
Ensures linear terms on LHS, nonlinear terms on RHS.
"""
function validate_ivp_equation(lhs, rhs, original::String)
    # Check for time derivative on LHS
    if !contains_time_derivative(lhs)
        @warn "No time derivative found on LHS - may not be suitable for IVP: $original"
    end
    
    # Check for time derivatives on RHS (should be avoided)
    if contains_time_derivative(rhs)
        throw(ArgumentError("Time derivatives should not appear on RHS: $original"))
    end
    
    # Validate linear/nonlinear term placement
    lhs_terms = extract_terms_from_expr(lhs)
    rhs_terms = extract_terms_from_expr(rhs)
    
    # Check LHS terms should be linear
    for term in lhs_terms
        if !is_linear_term(term) && !contains_time_derivative(term)
            @warn "Nonlinear term on LHS (should be moved to RHS): $(expr_to_string(term)) in equation: $original"
        end
    end
    
    # Check RHS terms should be nonlinear or forcing
    for term in rhs_terms
        if is_linear_spatial_term(term)
            @warn "Linear spatial term on RHS (consider moving to LHS for implicit treatment): $(expr_to_string(term)) in equation: $original"
        end
    end
    
    return true
end

"""
    contains_time_derivative(expr) -> Bool

Check if expression contains a time derivative dt().
"""
function contains_time_derivative(expr)
    if isa(expr, Expr) && expr.head == :call
        if length(expr.args) >= 1 && expr.args[1] == :dt
            return true
        end
        # Check recursively
        for arg in expr.args
            if contains_time_derivative(arg)
                return true
            end
        end
    end
    
    return false
end

"""
    extract_terms_from_expr(expr) -> Vector

Extract individual terms from an expression (handling + and -).
"""
function extract_terms_from_expr(expr)
    terms = []
    
    if isa(expr, Expr) && expr.head == :call && length(expr.args) >= 2
        op = expr.args[1]
        if op == :+ || op == :-
            # Addition/subtraction - extract all terms
            for i in 2:length(expr.args)
                append!(terms, extract_terms_from_expr(expr.args[i]))
            end
        else
            # Single term
            push!(terms, expr)
        end
    else
        # Single term (symbol, number, etc.)
        push!(terms, expr)
    end
    
    return terms
end

"""
    is_linear_term(term) -> Bool

Check if a term represents a linear operator (suitable for LHS).
Linear terms: dt(u), laplacian(u), dx(p), f*v, -f*u, etc.
"""
function is_linear_term(term)
    if isa(term, Expr) && term.head == :call && length(term.args) >= 1
        op = term.args[1]
        
        # Time derivatives are linear
        if op == :dt
            return true
        end
        
        # Spatial linear operators
        if op in [:dx, :dy, :dz, :laplacian, :lap, :divergence, :div, :gradient, :grad]
            return true
        end
        
        # Multiplication - check if it's linear
        if op == :* && length(term.args) >= 3
            # Linear if: constant * linear_op, parameter * field, etc.
            return is_linear_multiplication(term)
        end
        
        # Unary minus
        if op == :- && length(term.args) == 2
            return is_linear_term(term.args[2])
        end
        
    elseif isa(term, Symbol)
        # Pure field variables are linear
        return is_variable(term)
    end
    
    return false
end

"""
    is_linear_spatial_term(term) -> Bool

Check if term is a linear spatial operator (diffusion, pressure gradient, Coriolis).
These should typically be on LHS for implicit treatment.
"""
function is_linear_spatial_term(term)
    if isa(term, Expr) && term.head == :call && length(term.args) >= 1
        op = term.args[1]
        
        # Diffusion terms: nu*laplacian(u), kappa*laplacian(b)
        if op == :laplacian || op == :lap
            return true
        end
        
        # Pressure gradients: dx(p), dy(p), dz(p)  
        if op in [:dx, :dy, :dz] && length(term.args) >= 2
            field = term.args[2]
            if isa(field, Symbol) && (field == :p || field == :P)
                return true
            end
        end
        
        # Coriolis terms: f*v, -f*u
        if op == :* && length(term.args) >= 3
            return is_coriolis_term(term)
        end
        
        # Viscous terms: nu*laplacian(u)
        if op == :* && length(term.args) >= 3
            return is_viscous_term(term)
        end
    end
    
    return false
end

"""
    is_linear_multiplication(term) -> Bool

Check if multiplication term is linear (constant * linear_operator).
"""
function is_linear_multiplication(term)
    if isa(term, Expr) && term.head == :call && term.args[1] == :* && length(term.args) >= 3
        
        factors = term.args[2:end]
        
        # Count parameters vs variables vs operators
        n_params = 0
        n_vars = 0  
        n_ops = 0
        
        for factor in factors
            if isa(factor, Symbol)
                if is_parameter(factor)
                    n_params += 1
                elseif is_variable(factor)
                    n_vars += 1
                end
            elseif isa(factor, Expr) && factor.head == :call
                # Operator like laplacian(u), dx(u)
                n_ops += 1
            end
        end
        
        # Linear if: parameter * variable, parameter * operator(variable)
        # Nonlinear if: variable * variable, multiple variables
        return n_vars <= 1 && n_ops <= 1
    end
    
    return false
end

"""
    is_coriolis_term(term) -> Bool

Check if term is a Coriolis term: f*u, f*v, -f*u, etc.
"""
function is_coriolis_term(term)
    if isa(term, Expr) && term.head == :call && term.args[1] == :* && length(term.args) >= 3
        factors = term.args[2:end]
        
        has_coriolis_param = any(f -> isa(f, Symbol) && f == :f, factors)
        has_velocity = any(f -> isa(f, Symbol) && f in [:u, :v, :w], factors)
        
        return has_coriolis_param && has_velocity && length(factors) == 2
    end
    
    return false
end

"""
    is_viscous_term(term) -> Bool

Check if term is a viscous diffusion term: nu*laplacian(u), etc.
"""
function is_viscous_term(term)
    if isa(term, Expr) && term.head == :call && term.args[1] == :* && length(term.args) >= 3
        factors = term.args[2:end]
        
        has_viscosity = any(f -> isa(f, Symbol) && f in [:nu, :kappa, :alpha], factors)
        has_diffusion_op = any(f -> isa(f, Expr) && f.head == :call && 
                             f.args[1] in [:laplacian, :lap], factors)
        
        return has_viscosity && has_diffusion_op && length(factors) == 2
    end
    
    return false
end

"""
    expr_to_string(expr) -> String

Convert expression back to readable string.
"""
function expr_to_string(expr)
    if isa(expr, Symbol)
        return string(expr)
    elseif isa(expr, Number)
        return string(expr)
    elseif isa(expr, Expr) && expr.head == :call
        op = expr.args[1]
        args = expr.args[2:end]
        
        if op in [:+, :-, :*, :/, :^] && length(args) == 2
            left = expr_to_string(args[1])
            right = expr_to_string(args[2])
            return "($left $op $right)"
        elseif length(args) == 1
            arg = expr_to_string(args[1])
            return "$op($arg)"
        else
            arg_strs = [expr_to_string(arg) for arg in args]
            return "$op($(join(arg_strs, ", ")))"
        end
    else
        return string(expr)
    end
end

"""
    parse_boundary_condition(bc::String) -> Dict

Parse boundary condition string.
"""
function parse_boundary_condition(bc::String)
    bc_data = Dict{Symbol, Any}()
    
    # Simple pattern matching for common BC types
    if occursin("=", bc)
        lhs, rhs = split(bc, "=", limit=2)
        lhs = strip(lhs)
        rhs = strip(rhs)
        
        # Extract field and coordinate info
        if occursin("(", lhs) && occursin(")", lhs)
            # Pattern: field(coord=value) = rhs
            field_match = match(r"(\w+)\((\w+)=([^)]+)\)", lhs)
            if field_match !== nothing
                bc_data[:type] = :dirichlet
                bc_data[:variable] = Symbol(field_match.captures[1])
                bc_data[:coordinate] = Symbol(field_match.captures[2])
                bc_data[:position] = parse(Float64, field_match.captures[3])
                bc_data[:value] = parse(Float64, rhs)
            end
        elseif startswith(lhs, "d")
            # Pattern: dz(field)(coord=value) = rhs (Neumann)
            bc_data[:type] = :neumann
            # More complex parsing would be needed
        end
    end
    
    return bc_data
end

"""
    validate_problem(parser::ImprovedIVPParser) -> Bool

Validate the complete IVP problem.
"""
function validate_problem(parser::ImprovedIVPParser) -> Bool
    errors = String[]
    
    # Check basic requirements
    if isempty(parser.equations)
        push!(errors, "No equations defined")
    end
    
    if isempty(parser.variables)
        push!(errors, "No variables detected")
    end
    
    if isempty(parser.domain)
        push!(errors, "Domain not configured")
    end
    
    # Check equation-variable balance
    n_eqs = length(parser.equations)
    n_vars = length(parser.variables)
    
    if n_eqs != n_vars
        push!(errors, "Equation count ($n_eqs) ≠ variable count ($n_vars)")
    end
    
    # Validate each equation
    for (i, eq_data) in enumerate(parser.parsed_equations)
        try
            lhs = eq_data["lhs"]
            rhs = eq_data["rhs"]
            validate_ivp_equation(lhs, rhs, eq_data["original"])
        catch e
            push!(errors, "Equation $i validation failed: $(e.msg)")
        end
    end
    
    if !isempty(errors)
        @error "Problem validation failed" errors=errors
        return false
    end
    
    @info "Problem validation successful"
    return true
end

"""
    get_summary(parser::ImprovedIVPParser) -> Dict

Get comprehensive problem summary.
"""
function get_summary(parser::ImprovedIVPParser)
    summary = Dict{String, Any}()
    
    summary["num_equations"] = length(parser.equations)
    summary["num_variables"] = length(parser.variables)
    summary["num_parameters"] = length(parser.parameters)
    summary["num_bcs"] = length(parser.boundary_conditions)
    
    summary["variables"] = sort(collect(parser.variables))
    summary["parameters"] = Dict(string(k) => v for (k,v) in parser.parameters)
    
    if !isempty(parser.domain)
        summary["domain"] = parser.domain
    end
    
    summary["equations"] = [eq_data["original"] for eq_data in parser.parsed_equations]
    summary["boundary_conditions"] = parser.boundary_conditions
    
    return summary
end

"""
    build_system_matrices(parser::ImprovedIVPParser) -> Tuple

Build system matrices for IVP: M*dt(u) + L*u = F
"""
function build_system_matrices(parser::ImprovedIVPParser)
    if !validate_problem(parser)
        throw(ArgumentError("Problem validation failed"))
    end
    
    n_vars = length(parser.variables)
    n_eqs = length(parser.equations)
    
    # Placeholder matrices (would need proper discretization)
    M = sparse(I, n_vars, n_vars)  # Mass matrix
    L = spzeros(n_vars, n_vars)    # Stiffness matrix  
    F = zeros(n_vars)              # Forcing vector
    
    @info "System matrices built" M_size=size(M) L_size=size(L) F_size=length(F)
    
    return M, L, F
end

# Export improved parser interface
"""
    suggest_equation_restructure(parser::ImprovedIVPParser, eq_index::Int)

Suggest how to restructure an equation for proper IVP form.
Linear terms should be on LHS, nonlinear terms on RHS.
"""
function suggest_equation_restructure(parser::ImprovedIVPParser, eq_index::Int)
    if eq_index > length(parser.parsed_equations)
        return "Invalid equation index"
    end
    
    eq_data = parser.parsed_equations[eq_index]
    original = eq_data["original"]
    lhs = eq_data["lhs"] 
    rhs = eq_data["rhs"]
    
    println("Equation $eq_index: $original")
    println("Current form: $(eq_data["lhs_str"]) = $(eq_data["rhs_str"])")
    
    # Extract and classify terms
    lhs_terms = extract_terms_from_expr(lhs)
    rhs_terms = extract_terms_from_expr(rhs)
    
    # Classify terms
    linear_lhs = []
    nonlinear_lhs = []
    linear_rhs = []
    nonlinear_rhs = []
    
    for term in lhs_terms
        if is_linear_term(term) || contains_time_derivative(term)
            push!(linear_lhs, term)
        else
            push!(nonlinear_lhs, term)
        end
    end
    
    for term in rhs_terms
        if is_linear_spatial_term(term)
            push!(linear_rhs, term)
        else
            push!(nonlinear_rhs, term)
        end
    end
    
    # Suggest restructuring
    suggestions = String[]
    
    if !isempty(nonlinear_lhs)
        push!(suggestions, "Move nonlinear terms to RHS: $(join([expr_to_string(t) for t in nonlinear_lhs], ", "))")
    end
    
    if !isempty(linear_rhs)
        push!(suggestions, "Consider moving linear terms to LHS for implicit treatment: $(join([expr_to_string(t) for t in linear_rhs], ", "))")
    end
    
    if !isempty(suggestions)
        println("Suggestions for better IVP structure:")
        for (i, suggestion) in enumerate(suggestions)
            println("  $i. $suggestion")
        end
        
        # Suggest restructured equation
        new_lhs_terms = vcat(linear_lhs, linear_rhs)
        new_rhs_terms = vcat(nonlinear_rhs, [:(-($(t))) for t in nonlinear_lhs])
        
        if !isempty(new_lhs_terms) && !isempty(new_rhs_terms)
            new_lhs_str = join([expr_to_string(t) for t in new_lhs_terms], " + ")
            new_rhs_str = join([expr_to_string(t) for t in new_rhs_terms], " + ")
            
            println("Suggested restructured form:")
            println("  $new_lhs_str = $new_rhs_str")
            println("  (Linear terms on LHS, nonlinear terms on RHS)")
        end
    else
        println("[OK] Equation structure is already optimal for IVP")
    end
    
    return suggestions
end

export ImprovedIVPParser, set_domain!, add_variable!, set_parameter!
export parse_equation!, add_boundary_condition!, validate_problem
export get_summary, build_system_matrices, suggest_equation_restructure