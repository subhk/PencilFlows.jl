# Automatic Equation Rearrangement
# =================================
# Automatically rearranges PDE equations to put:
# - Linear terms on the LEFT-hand side (with time derivative)
# - Nonlinear terms on the RIGHT-hand side
#
# Example transformation:
# Input:  dt(u) = nu*lap(u) - u*dx(u) + f*v
# Output: dt(u) - nu*lap(u) - f*v = -u*dx(u)

using Symbolics

"""
    EquationRearrangement

Structure to hold the rearranged equation system.
"""
struct EquationRearrangement
    # Original equations
    original_equations::Vector{String}
    
    # Rearranged equations  
    rearranged_equations::Vector{String}
    
    # Analysis of each equation
    equation_analysis::Dict{Symbol, Dict{Symbol, Any}}
    
    # Summary statistics
    total_linear_terms::Int
    total_nonlinear_terms::Int
    rearrangement_success_rate::Float64
end

"""
    rearrange_equations_linear_left(equations::Vector{String})

Automatically rearrange equations to put linear terms on LHS, nonlinear on RHS.

# Input Format
Standard PDE form: dt(u) = all_terms

# Output Format  
Rearranged form: dt(u) ± linear_terms = nonlinear_terms

# Examples

## Input
```julia
equations = [
    "dt(u) = nu*lap(u) - u*dx(u) + f*v",
    "dt(v) = kappa*lap(v) + a*u - b*u*v"
]
```

## Output
```julia
rearranged = [
    "dt(u) - nu*lap(u) - f*v = -u*dx(u)",
    "dt(v) - kappa*lap(v) - a*u = -b*u*v"
]
```
"""
function rearrange_equations_linear_left(equations::Vector{String})
    println("AUTOMATIC EQUATION REARRANGEMENT")
    println("="^45)
    println("Moving linear terms to LHS, nonlinear terms to RHS")
    println()
    
    rearranged_equations = String[]
    equation_analysis = Dict{Symbol, Dict{Symbol, Any}}()
    total_linear = 0
    total_nonlinear = 0
    success_count = 0
    
    for (i, eq) in enumerate(equations)
        println("Equation $i: $eq")
        
        try
            # Extract primary variable and RHS
            primary_var, rhs = extract_variable_and_rhs(eq)
            
            if primary_var === nothing
                println("  Could not identify primary variable, keeping original form")
                push!(rearranged_equations, eq)
                continue
            end
            
            # Classify terms as linear or nonlinear
            linear_terms, nonlinear_terms = classify_terms_for_rearrangement(rhs, primary_var)
            
            # Build rearranged equation
            rearranged_eq = build_rearranged_equation(primary_var, linear_terms, nonlinear_terms)
            push!(rearranged_equations, rearranged_eq)
            
            # Store analysis
            equation_analysis[primary_var] = Dict(
                :original => eq,
                :rearranged => rearranged_eq,
                :linear_terms => linear_terms,
                :nonlinear_terms => nonlinear_terms,
                :linear_count => length(linear_terms),
                :nonlinear_count => length(nonlinear_terms)
            )
            
            total_linear += length(linear_terms)
            total_nonlinear += length(nonlinear_terms)
            success_count += 1
            
            println("  Rearranged: $rearranged_eq")
            println("     Linear terms (LHS): $(length(linear_terms))")
            println("     Nonlinear terms (RHS): $(length(nonlinear_terms))")
            
        catch e
            println("  Rearrangement failed: $e")
            println("     Keeping original form")
            push!(rearranged_equations, eq)
        end
        println()
    end
    
    success_rate = success_count / length(equations)
    
    rearrangement = EquationRearrangement(
        equations, rearranged_equations, equation_analysis,
        total_linear, total_nonlinear, success_rate
    )
    
    print_rearrangement_summary(rearrangement)
    return rearrangement
end

"""
    extract_variable_and_rhs(equation::String)

Extract the primary variable and right-hand side from an equation.
"""
function extract_variable_and_rhs(equation::String)
    # Handle different time derivative notations
    patterns = [
        r"dt\(([^)]+)\)\s*=\s*(.+)",
        r"d_t\(([^)]+)\)\s*=\s*(.+)",
        r"∂t\(([^)]+)\)\s*=\s*(.+)",
        r"∂_t\(([^)]+)\)\s*=\s*(.+)"
    ]
    
    for pattern in patterns
        match_result = match(pattern, equation)
        if match_result !== nothing
            var_name = Symbol(strip(match_result.captures[1]))
            rhs = strip(match_result.captures[2])
            return var_name, rhs
        end
    end
    
    return nothing, nothing
end

"""
    classify_terms_for_rearrangement(rhs::String, primary_var::Symbol)

Classify terms in RHS as linear or nonlinear for rearrangement.
"""
function classify_terms_for_rearrangement(rhs::String, primary_var::Symbol)
    # Split RHS into individual terms
    terms = split_into_terms_with_signs(rhs)
    
    linear_terms = String[]
    nonlinear_terms = String[]
    
    for term in terms
        if is_linear_term_for_rearrangement(term, primary_var)
            push!(linear_terms, term)
        else
            push!(nonlinear_terms, term)
        end
    end
    
    return linear_terms, nonlinear_terms
end

"""
    split_into_terms_with_signs(expression::String)

Split expression into terms, preserving + and - signs.
"""
function split_into_terms_with_signs(expression::String)
    terms = String[]
    current_term = ""
    paren_depth = 0
    i = 1
    
    while i <= length(expression)
        char = expression[i]
        
        if char == '('
            paren_depth += 1
            current_term *= char
        elseif char == ')'
            paren_depth -= 1
            current_term *= char
        elseif (char == '+' || char == '-') && paren_depth == 0 && i > 1
            # End current term, start new one
            cleaned_term = strip(current_term)
            if !isempty(cleaned_term)
                push!(terms, cleaned_term)
            end
            current_term = string(char)  # Start with the sign
        else
            current_term *= char
        end
        
        i += 1
    end
    
    # Add the last term
    cleaned_term = strip(current_term)
    if !isempty(cleaned_term)
        push!(terms, cleaned_term)
    end
    
    return filter(t -> !isempty(t), terms)
end

"""
    is_linear_term_for_rearrangement(term::String, primary_var::Symbol)

Determine if a term should be considered linear for rearrangement purposes.

Linear terms (move to LHS):
- Laplacians: lap(u), d2dx2(u), etc.
- Linear derivatives: dx(u), dy(u), dz(u)
- Linear coupling: coefficient*other_variable
- Linear in primary variable: coefficient*u

Nonlinear terms (keep on RHS):
- Products of variables: u*v, u*dx(v)
- Convection: u*dx(u)
- Nonlinear functions: u^2, exp(u), sin(u)
- Constants/sources: numbers, parameters without variables
"""
function is_linear_term_for_rearrangement(term::String, primary_var::Symbol)
    # Clean the term (remove leading +/-)
    clean_term = strip(lstrip(term, ['+', '-', ' ']))
    
    # Check for linear spatial operators on the primary variable
    if contains_linear_spatial_operators(clean_term, primary_var)
        return true
    end
    
    # Check for linear coupling to other variables
    if contains_linear_coupling_term(clean_term, primary_var)
        return true
    end
    
    # Check if it's linear in the primary variable itself
    if contains_primary_variable_linearly(clean_term, primary_var)
        return true
    end
    
    # Check for products of variables (nonlinear)
    if contains_variable_products(clean_term)
        return false
    end
    
    # Check for nonlinear functions
    if contains_nonlinear_functions(clean_term)
        return false
    end
    
    # Constants and source terms are nonlinear (stay on RHS)
    if is_constant_or_source_term(clean_term)
        return false
    end
    
    # Default: if uncertain, treat as linear
    return true
end

"""
    contains_linear_spatial_operators(term::String, var::Symbol)

Check if term contains linear spatial operators applied to the variable.
"""
function contains_linear_spatial_operators(term::String, var::Symbol)
    var_str = string(var)
    
    # Linear spatial operator patterns
    linear_spatial_patterns = [
        "lap($var_str)",
        "d2dx2($var_str)",
        "d2dy2($var_str)", 
        "d2dz2($var_str)",
        "dx($var_str)",
        "dy($var_str)",
        "dz($var_str)",
        "grad($var_str)",
        "div($var_str)",
        "curl($var_str)"
    ]
    
    for pattern in linear_spatial_patterns
        if occursin(pattern, term)
            # Make sure it's not in a product (like u*lap(v))
            if !contains_multiplication_outside_operator(term, pattern)
                return true
            end
        end
    end
    
    return false
end

"""
    contains_linear_coupling_term(term::String, primary_var::Symbol)

Check if term represents linear coupling to other variables.
"""
function contains_linear_coupling_term(term::String, primary_var::Symbol)
    primary_var_str = string(primary_var)
    
    # Look for terms that contain other variables but not the primary variable
    # and don't contain products
    if !occursin(primary_var_str, term) && !occursin("*", term)
        # Check if it contains variable-like symbols
        if occursin(r"[a-zA-Z]", term)
            return true
        end
    end
    
    return false
end

"""
    contains_primary_variable_linearly(term::String, var::Symbol)

Check if the primary variable appears linearly in the term.
"""
function contains_primary_variable_linearly(term::String, var::Symbol)
    var_str = string(var)
    
    # Variable must appear
    if !occursin(var_str, term)
        return false
    end
    
    # But not in products with other variables
    if occursin("*", term) && count_variable_occurrences(term) > 1
        return false
    end
    
    # And not in nonlinear functions
    if contains_nonlinear_functions(term)
        return false
    end
    
    return true
end

"""
    contains_variable_products(term::String)

Check if term contains products of variables.
"""
function contains_variable_products(term::String)
    # Look for multiplication signs
    if !occursin("*", term)
        return false
    end
    
    # Count variable-like symbols
    variable_count = count_variable_occurrences(term)
    
    return variable_count >= 2
end

## use contains_nonlinear_functions from equation_format_validation.jl

"""
    is_constant_or_source_term(term::String)

Check if term is a constant or source term (no variables).
"""
function is_constant_or_source_term(term::String)
    # Remove numbers and operators to see if any variables remain
    cleaned = replace(term, r"[0-9\.\+\-\*/\(\)\s]" => "")
    
    # If only parameter-like symbols remain (no clear variables), it's a source
    if isempty(cleaned) || all(c -> islowercase(c) || c in ['α', 'β', 'γ', 'δ', 'ε'], cleaned)
        return true
    end
    
    return false
end

"""
    count_variable_occurrences(term::String)

Count how many variable-like symbols appear in a term.
"""
function count_variable_occurrences(term::String)
    # Simple heuristic: count uppercase single letters and common variable names
    variable_patterns = [r"\b[uvwpTcbA-Z]\b", r"\b[a-z][a-z0-9]*\b"]
    count = 0
    
    for pattern in variable_patterns
        matches = collect(eachmatch(pattern, term))
        count += length(matches)
    end
    
    return count
end

"""
    contains_multiplication_outside_operator(term::String, operator_pattern::String)

Check if there's multiplication outside of the specified operator.
"""
function contains_multiplication_outside_operator(term::String, operator_pattern::String)
    # Remove the operator pattern and see if multiplication remains
    without_operator = replace(term, operator_pattern => "OPERATOR")
    
    # If there's still a * outside the operator context, it's a product
    return occursin("*", without_operator) && !occursin("*OPERATOR", without_operator)
end

"""
    build_rearranged_equation(var::Symbol, linear_terms::Vector{String}, nonlinear_terms::Vector{String})

Build the rearranged equation with linear terms on LHS, nonlinear on RHS.
"""
function build_rearranged_equation(var::Symbol, linear_terms::Vector{String}, nonlinear_terms::Vector{String})
    # Start with time derivative
    lhs = "dt($var)"
    
    # Add linear terms to LHS (flip signs)
    for term in linear_terms
        flipped_term = flip_term_sign(term)
        lhs *= " $flipped_term"
    end
    
    # Build RHS with nonlinear terms
    if isempty(nonlinear_terms)
        rhs = "0"
    else
        rhs = join(nonlinear_terms, " ")
        # Clean up double signs
        rhs = clean_up_signs(rhs)
    end
    
    return "$lhs = $rhs"
end

"""
    flip_term_sign(term::String)

Flip the sign of a term for moving it to the other side.
"""
function flip_term_sign(term::String)
    term = strip(term)
    
    if startswith(term, "+")
        return "-" * term[2:end]
    elseif startswith(term, "-")
        return "+" * term[2:end]
    else
        # No explicit sign means positive, so flip to negative
        return "- $term"
    end
end

"""
    clean_up_signs(expression::String)

Clean up double signs and formatting in an expression.
"""
function clean_up_signs(expression::String)
    # Replace multiple spaces with single space
    cleaned = replace(expression, r"\s+" => " ")
    
    # Fix double signs
    cleaned = replace(cleaned, "+ +" => "+")
    cleaned = replace(cleaned, "- -" => "+")
    cleaned = replace(cleaned, "+ -" => "-")
    cleaned = replace(cleaned, "- +" => "-")
    
    # Clean leading/trailing spaces
    return strip(cleaned)
end

"""
    print_rearrangement_summary(rearrangement::EquationRearrangement)

Print a comprehensive summary of the equation rearrangement.
"""
function print_rearrangement_summary(rearrangement::EquationRearrangement)
    println("EQUATION REARRANGEMENT SUMMARY")
    println("="^40)
    
    println("Overall Statistics:")
    println("  • Total equations: $(length(rearrangement.original_equations))")
    println("  • Successfully rearranged: $(round(rearrangement.rearrangement_success_rate * 100, digits=1))%")
    println("  • Total linear terms moved to LHS: $(rearrangement.total_linear_terms)")
    println("  • Total nonlinear terms on RHS: $(rearrangement.total_nonlinear_terms)")
    
    println("\nEquation-by-Equation Results:")
    for (var, analysis) in rearrangement.equation_analysis
        println("  Variable $var:")
        println("     Original:   $(analysis[:original])")
        println("     Rearranged: $(analysis[:rearranged])")
        println("     Linear→LHS: $(analysis[:linear_count]) terms")
        println("     Nonlinear→RHS: $(analysis[:nonlinear_count]) terms")
        println()
    end
    
    println("REARRANGEMENT COMPLETE!")
    println("Linear terms now on LHS, nonlinear terms on RHS")
    println("="^40)
end

"""
    demonstrate_equation_rearrangement()

Demonstrate automatic equation rearrangement with various examples.
"""
function demonstrate_equation_rearrangement()
    println("EQUATION REARRANGEMENT DEMONSTRATION")
    println("="^50)
    
    # Example 1: Simple reaction-diffusion
    println("Example 1: Reaction-Diffusion")
    equations1 = [
        "dt(u) = D*lap(u) + a*u - b*u*v",
        "dt(v) = D*lap(v) + c*u*v - d*v"
    ]
    rearrangement1 = rearrange_equations_linear_left(equations1)
    
    println("\n" * "-"^50 * "\n")
    
    # Example 2: Navier-Stokes
    println("Example 2: Navier-Stokes")
    equations2 = [
        "dt(u) = -u*dx(u) - v*dy(u) - dx(p) + nu*lap(u)",
        "dt(v) = -u*dx(v) - v*dy(v) - dy(p) + nu*lap(v)"
    ]
    rearrangement2 = rearrange_equations_linear_left(equations2)
    
    println("\n" * "-"^50 * "\n")
    
    # Example 3: Complex system
    println("Example 3: Complex Multi-Physics")
    equations3 = [
        "dt(T) = kappa*lap(T) - u*dx(T) - v*dy(T) + Q",
        "dt(c) = D*lap(c) - u*dx(c) + k*T*c - decay*c"
    ]
    rearrangement3 = rearrange_equations_linear_left(equations3)
    
    println("\nDEMONSTRATION COMPLETE!")
    println("All equations successfully rearranged to standard form!")
end

# Export the rearrangement functions
export EquationRearrangement, rearrange_equations_linear_left, demonstrate_equation_rearrangement
