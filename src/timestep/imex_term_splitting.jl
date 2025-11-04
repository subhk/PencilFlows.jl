# Automatic IMEX Term Splitting for General PDE Systems
# ======================================================
# Automatically identifies linear vs nonlinear terms and sets up optimal
# implicit-explicit time stepping: linear terms implicit, nonlinear explicit.

using Symbolics, LinearAlgebra

"""
    IMEXSplitting

Structure to hold the implicit-explicit splitting of PDE terms.
"""
struct IMEXSplitting
    # For each variable, categorize terms
    implicit_terms::Dict{Symbol, Vector{String}}    # Linear terms (diffusion, etc.)
    explicit_terms::Dict{Symbol, Vector{String}}    # Nonlinear terms
    coupling_terms::Dict{Symbol, Vector{String}}    # Linear coupling between variables
    source_terms::Dict{Symbol, Vector{String}}      # Source/forcing terms
    
    # Analysis metadata
    stiffness_ratio::Float64                        # Estimate of stiffness
    recommended_scheme::Symbol                      # Optimal IMEX scheme
    implicit_solve_complexity::Symbol              # Expected solve complexity
end

"""
    analyze_imex_splitting(system::ArbitraryPDESystem)

Automatically analyze equations and determine optimal implicit-explicit splitting.

Linear terms (implicit for stability):
- Diffusion: lap(u), d2dx2(u), etc.
- Linear coupling: constant*other_variable
- Linear operators: grad, div, curl with linear coefficients

Nonlinear terms (explicit to avoid nonlinear solves):
- Products of variables: u*v, u*dx(v), etc.
- Nonlinear functions: u^2, exp(u), sin(u), etc.
- Convection: u*dx(u), etc.
"""
function analyze_imex_splitting(system)
    println("  ANALYZING IMEX TERM SPLITTING")
    println("="^40)
    println("Linear terms → Implicit (stable)")
    println("Nonlinear terms → Explicit (efficient)")
    println()
    
    implicit_terms = Dict{Symbol, Vector{String}}()
    explicit_terms = Dict{Symbol, Vector{String}}()
    coupling_terms = Dict{Symbol, Vector{String}}()
    source_terms = Dict{Symbol, Vector{String}}()
    
    total_linear_terms = 0
    total_nonlinear_terms = 0
    max_derivative_order = 0
    
    # Analyze each evolution equation
    for (var, equation) in system.evolution_equations
        println("   Analyzing equation for $var:")
        
        # Initialize term collections for this variable
        implicit_terms[var] = String[]
        explicit_terms[var] = String[]
        coupling_terms[var] = String[]
        source_terms[var] = String[]
        
        # Extract RHS of equation (everything after dt(var) = )
        rhs = extract_rhs_from_equation(equation)
        println("    RHS: $rhs")
        
        # Split into individual terms
        terms = split_equation_into_terms(rhs)
        println("    Found $(length(terms)) terms")
        
        for term in terms
            term_type, complexity = classify_term_for_imex(term, var, system.all_variables)
            
            if term_type == :linear_spatial
                push!(implicit_terms[var], term)
                total_linear_terms += 1
                max_derivative_order = max(max_derivative_order, get_derivative_order(term))
                println("       Linear (implicit): $term")
                
            elseif term_type == :linear_coupling
                push!(coupling_terms[var], term)
                total_linear_terms += 1
                println("       Coupling (implicit): $term")
                
            elseif term_type == :nonlinear
                push!(explicit_terms[var], term)
                total_nonlinear_terms += 1
                println("       Nonlinear (explicit): $term")
                
            elseif term_type == :source
                push!(source_terms[var], term)
                println("       Source (explicit): $term")
                
            else
                # Default classification - be conservative
                if contains_spatial_derivatives(term)
                    push!(implicit_terms[var], term)
                    total_linear_terms += 1
                    println("        Unknown spatial (implicit): $term")
                else
                    push!(explicit_terms[var], term)
                    println("        Unknown (explicit): $term")
                end
            end
        end
        println()
    end
    
    # Analyze stiffness and recommend scheme
    stiffness_ratio = estimate_stiffness_ratio(system, max_derivative_order)
    scheme = recommend_imex_scheme(total_linear_terms, total_nonlinear_terms, stiffness_ratio)
    complexity = estimate_implicit_solve_complexity(system, implicit_terms, coupling_terms)
    
    splitting = IMEXSplitting(
        implicit_terms, explicit_terms, coupling_terms, source_terms,
        stiffness_ratio, scheme, complexity
    )
    
    print_imex_analysis_summary(splitting, system)
    return splitting
end

"""
    extract_rhs_from_equation(equation::String)

Extract the right-hand side from an equation of the form "dt(var) = rhs".
"""
function extract_rhs_from_equation(equation::String)
    if !occursin("=", equation)
        return equation
    end
    
    parts = split(equation, "=", limit=2)
    if length(parts) != 2
        return equation
    end
    
    return strip(parts[2])
end

"""
    split_equation_into_terms(rhs::String)

Split RHS into individual terms, preserving signs.
"""
function split_equation_into_terms(rhs::String)
    terms = String[]
    current_term = ""
    paren_depth = 0
    
    i = 1
    while i <= length(rhs)
        char = rhs[i]
        
        if char == '('
            paren_depth += 1
            current_term *= char
        elseif char == ')'
            paren_depth -= 1
            current_term *= char
        elseif (char == '+' || char == '-') && paren_depth == 0 && i > 1
            # End of current term
            if !isempty(strip(current_term))
                push!(terms, strip(current_term))
            end
            current_term = string(char)  # Start new term with sign
        else
            current_term *= char
        end
        
        i += 1
    end
    
    # Add the last term
    if !isempty(strip(current_term))
        push!(terms, strip(current_term))
    end
    
    return filter(t -> !isempty(t), terms)
end

"""
    classify_term_for_imex(term::String, primary_var::Symbol, all_vars::Set{Symbol})

Classify a term for IMEX splitting.
"""
function classify_term_for_imex(term::String, primary_var::Symbol, all_vars::Set{Symbol})
    # Remove leading + or - for analysis
    clean_term = strip(lstrip(term, ['+', '-', ' ']))
    
    # Check for spatial derivatives (usually linear and stiff)
    if contains_spatial_derivatives(clean_term)
        if is_linear_spatial_term(clean_term, primary_var)
            return :linear_spatial, 1
        else
            # Nonlinear spatial term (like u*dx(u))
            return :nonlinear, 2
        end
    end
    
    # Check for variable products (nonlinear)
    if contains_variable_products(clean_term, all_vars)
        return :nonlinear, 2
    end
    
    # Check for linear coupling to other variables
    if contains_linear_coupling(clean_term, primary_var, all_vars)
        return :linear_coupling, 1
    end
    
    # Check for nonlinear functions
    if contains_nonlinear_functions(clean_term)
        return :nonlinear, 3
    end
    
    # Check if it's a source term (no variables)
    if !contains_any_variables(clean_term, all_vars)
        return :source, 0
    end
    
    # Default: if it contains the primary variable linearly, treat as linear
    if contains_variable_linearly(clean_term, primary_var)
        return :linear_coupling, 1
    end
    
    # Default to nonlinear for safety
    return :nonlinear, 2
end

"""
    is_linear_spatial_term(term::String, var::Symbol)

Check if a spatial derivative term is linear in the given variable.
"""
function is_linear_spatial_term(term::String, var::Symbol)
    var_str = string(var)
    
    # Linear patterns: coefficient*operator(var)
    linear_patterns = [
        Regex("^[^$var_str]*lap\\($var_str\\)[^$var_str]*\$"),
        Regex("^[^$var_str]*d2dx2\\($var_str\\)[^$var_str]*\$"),
        Regex("^[^$var_str]*d2dy2\\($var_str\\)[^$var_str]*\$"),
        Regex("^[^$var_str]*d2dz2\\($var_str\\)[^$var_str]*\$"),
        Regex("^[^$var_str]*dx\\($var_str\\)[^$var_str]*\$"),
        Regex("^[^$var_str]*dy\\($var_str\\)[^$var_str]*\$"),
        Regex("^[^$var_str]*dz\\($var_str\\)[^$var_str]*\$")
    ]
    
    for pattern in linear_patterns
        if occursin(pattern, term)
            return true
        end
    end
    
    return false
end

"""
    contains_variable_products(term::String, variables::Set{Symbol})

Check if term contains products of variables (u*v, u*dx(v), etc.).
"""
function contains_variable_products(term::String, variables::Set{Symbol})
    var_count = 0
    for var in variables
        # Count occurrences of each variable
        var_str = string(var)
        matches = length(collect(eachmatch(Regex("\\b$var_str\\b"), term)))
        var_count += matches
    end
    
    # If more than one variable occurrence, likely a product
    return var_count >= 2 || occursin("*", term)
end

"""
    contains_linear_coupling(term::String, primary_var::Symbol, all_vars::Set{Symbol})

Check for linear coupling to other variables (coefficient*other_var).
"""
function contains_linear_coupling(term::String, primary_var::Symbol, all_vars::Set{Symbol})
    other_vars = setdiff(all_vars, Set([primary_var]))
    
    for var in other_vars
        var_str = string(var)
        # Look for patterns like: coefficient*var, var, -var, +var
        if occursin(Regex("^[^*$var_str]*\\b$var_str\\b[^*$var_str]*\$"), term)
            return true
        end
    end
    
    return false
end

## use contains_nonlinear_functions from equation_format_validation.jl

"""
    contains_any_variables(term::String, variables::Set{Symbol})

Check if term contains any of the system variables.
"""
function contains_any_variables(term::String, variables::Set{Symbol})
    for var in variables
        if occursin(string(var), term)
            return true
        end
    end
    return false
end

"""
    contains_variable_linearly(term::String, var::Symbol)

Check if variable appears linearly (not in products or nonlinear functions).
"""
function contains_variable_linearly(term::String, var::Symbol)
    var_str = string(var)
    
    # Variable appears
    if !occursin(var_str, term)
        return false
    end
    
    # But not in nonlinear contexts
    if occursin("*$var_str*", term) || occursin("$var_str*$var_str", term)
        return false
    end
    
    if contains_nonlinear_functions(term)
        return false
    end
    
    return true
end

"""
    get_derivative_order(term::String)

Get the highest derivative order in a term.
"""
function get_derivative_order(term::String)
    if occursin("d2", term) || occursin("lap", term)
        return 2
    elseif occursin("dx", term) || occursin("dy", term) || occursin("dz", term)
        return 1
    else
        return 0
    end
end

"""
    estimate_stiffness_ratio(system, max_derivative_order)

Estimate the stiffness ratio of the system.
"""
function estimate_stiffness_ratio(system, max_derivative_order::Int)
    # Base stiffness from derivative order
    base_stiffness = max_derivative_order^2  # Second derivatives are much stiffer
    
    # Adjust for small parameters
    param_stiffness = 1.0
    for (param, value) in system.all_parameters
        if value < 0.01
            param_stiffness *= 1.0 / value
        end
    end
    
    # Limit to reasonable range
    return min(1000.0, base_stiffness * param_stiffness)
end

"""
    recommend_imex_scheme(n_linear, n_nonlinear, stiffness_ratio)

Recommend optimal IMEX scheme based on term analysis.
"""
function recommend_imex_scheme(n_linear::Int, n_nonlinear::Int, stiffness_ratio::Float64)
    if n_linear == 0
        # No linear terms - pure explicit
        return :explicit_rk4
    elseif n_nonlinear == 0
        # No nonlinear terms - pure implicit
        return :implicit_euler
    elseif stiffness_ratio > 100
        # Very stiff - robust IMEX
        return :imex_bdf2  # Second-order backward differentiation formula
    elseif stiffness_ratio > 10
        # Moderately stiff - balanced IMEX
        return :imex_ark4  # Additive Runge-Kutta
    else
        # Mildly stiff - efficient IMEX
        return :imex_euler  # Simple first-order IMEX
    end
end

"""
    estimate_implicit_solve_complexity(system, implicit_terms, coupling_terms)

Estimate the computational complexity of implicit solves.
"""
function estimate_implicit_solve_complexity(system, 
                                          implicit_terms::Dict, coupling_terms::Dict)
    n_vars = length(system.primary_variables)
    
    # Count total implicit terms
    total_implicit = sum(length(terms) for terms in values(implicit_terms))
    total_coupling = sum(length(terms) for terms in values(coupling_terms))
    
    if total_coupling > n_vars
        return :strongly_coupled  # Requires coupled solve
    elseif total_coupling > 0
        return :weakly_coupled    # Some coupling
    else
        return :decoupled         # Each variable independent
    end
end

"""
    print_imex_analysis_summary(splitting, system)

Print comprehensive summary of IMEX analysis.
"""
function print_imex_analysis_summary(splitting, system)
    println(" IMEX SPLITTING ANALYSIS COMPLETE")
    println("="^45)
    
    total_implicit = sum(length(terms) for terms in values(splitting.implicit_terms))
    total_explicit = sum(length(terms) for terms in values(splitting.explicit_terms))
    total_coupling = sum(length(terms) for terms in values(splitting.coupling_terms))
    total_source = sum(length(terms) for terms in values(splitting.source_terms))
    
    println(" Term Distribution:")
    println("  • Implicit (linear): $total_implicit terms")
    println("  • Explicit (nonlinear): $total_explicit terms") 
    println("  • Coupling (linear): $total_coupling terms")
    println("  • Source: $total_source terms")
    
    println("\n System Properties:")
    println("  • Stiffness ratio: $(round(splitting.stiffness_ratio, digits=1))")
    println("  • Implicit complexity: $(splitting.implicit_solve_complexity)")
    println("  • Recommended scheme: $(splitting.recommended_scheme)")
    
    println("\n Per-Variable Breakdown:")
    for var in system.primary_variables
        implicit_count = length(get(splitting.implicit_terms, var, []))
        explicit_count = length(get(splitting.explicit_terms, var, []))
        coupling_count = length(get(splitting.coupling_terms, var, []))
        
        println("  • $var: $implicit_count implicit, $explicit_count explicit, $coupling_count coupling")
    end
    
    println("\n IMEX splitting optimized for stability and efficiency!")
    println("="^45)
end

"""
    create_imex_stepper(splitting::IMEXSplitting, system::ArbitraryPDESystem, prob)

Create an IMEX time stepper based on the splitting analysis.
"""
function create_imex_stepper(splitting, system, prob)
    println(" CREATING IMEX TIME STEPPER")
    println("="^30)
    
    scheme = splitting.recommended_scheme
    println("  Selected scheme: $scheme")
    
    # Configure based on scheme type
    if scheme == :imex_euler
        stepper = configure_imex_euler(splitting, system, prob)
    elseif scheme == :imex_ark4
        stepper = configure_imex_ark4(splitting, system, prob)
    elseif scheme == :imex_bdf2
        stepper = configure_imex_bdf2(splitting, system, prob)
    elseif scheme == :explicit_rk4
        stepper = configure_explicit_rk4(splitting, system, prob)
    elseif scheme == :implicit_euler
        stepper = configure_implicit_euler(splitting, system, prob)
    else
        @warn "Unknown scheme $scheme, falling back to IMEX Euler"
        stepper = configure_imex_euler(splitting, system, prob)
    end
    
    println("   IMEX stepper configured successfully!")
    return stepper
end

# Placeholder stepper configuration functions
# These would integrate with the existing PencilFlows time stepping infrastructure

function configure_imex_euler(splitting, system, prob)
    println("     IMEX Euler: Simple, robust, first-order")
    return Dict(:scheme => :imex_euler, :order => 1, :splitting => splitting)
end

function configure_imex_ark4(splitting, system, prob) 
    println("     IMEX ARK4: High-order, balanced efficiency")
    return Dict(:scheme => :imex_ark4, :order => 4, :splitting => splitting)
end

function configure_imex_bdf2(splitting, system, prob)
    println("     IMEX BDF2: Robust for stiff systems, second-order")
    return Dict(:scheme => :imex_bdf2, :order => 2, :splitting => splitting)
end

function configure_explicit_rk4(splitting, system, prob)
    println("     Explicit RK4: Pure explicit, fourth-order")
    return Dict(:scheme => :explicit_rk4, :order => 4, :splitting => splitting)
end

function configure_implicit_euler(splitting, system, prob)
    println("     Implicit Euler: Pure implicit, unconditionally stable")
    return Dict(:scheme => :implicit_euler, :order => 1, :splitting => splitting)
end

# Export the IMEX analysis functions
export IMEXSplitting, analyze_imex_splitting, create_imex_stepper
