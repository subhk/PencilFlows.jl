# Equation Format Validation
# ==========================
# Validates that user-provided equations follow the required format:
# - Linear terms on LEFT-hand side (with time derivative)
# - Nonlinear and non-constant terms on RIGHT-hand side
# - Throws errors if format is incorrect

using Printf

"""
    EquationFormatError

Custom error type for incorrect equation format.
"""
struct EquationFormatError <: Exception
    message::String
    equation::String
    equation_index::Int
    violations::Vector{String}
end

function Base.showerror(io::IO, e::EquationFormatError)
    println(io, "ERROR: EQUATION FORMAT ERROR")
    println(io, "Equation $(e.equation_index): $(e.equation)")
    println(io, "Required format: dt(var) + linear_terms = nonlinear_terms")
    println(io, "\nViolations found:")
    for violation in e.violations
        println(io, "  • $violation")
    end
    println(io, "\n$(e.message)")
end

"""
    validate_equation_format(equations::Vector{String})

Validate that all equations follow the required format:
- Linear terms on LEFT-hand side
- Nonlinear and non-constant terms on RIGHT-hand side

# Required Format
```
dt(variable) ± linear_terms = nonlinear_terms
```

# Examples of CORRECT format:
```julia
correct_equations = [
    "dt(u) - nu*lap(u) - f*v = -u*dx(u)",
    "dt(v) - kappa*lap(v) + a*u = -b*u*v + source",
    "dt(T) - alpha*lap(T) = -u*dx(T) - v*dy(T) + Q"
]
```

# Examples of INCORRECT format (will throw errors):
```julia
incorrect_equations = [
    "dt(u) = nu*lap(u) - u*dx(u)",              # Linear term on RHS
    "dt(v) - u*dx(v) = kappa*lap(v)",           # Nonlinear on LHS, linear on RHS
    "dt(T) + u*dx(T) = alpha*lap(T) + source"   # Mixed: nonlinear on LHS
]
```
"""
function validate_equation_format(equations::Vector{String})
    println("VALIDATING EQUATION FORMAT")
    println("="^40)
    println("Required: Linear terms <- LEFT | RIGHT -> Nonlinear terms")
    println()
    
    all_valid = true
    
    for (i, equation) in enumerate(equations)
        println("Equation $i: $equation")
        
        try
            validate_single_equation_format(equation, i)
            println("   Format is CORRECT")
        catch e
            if e isa EquationFormatError
                println("   Format is INCORRECT")
                all_valid = false
                rethrow(e)
            else
                println("    Could not parse equation")
                rethrow(e)
            end
        end
        println()
    end
    
    if all_valid
        println(" ALL EQUATIONS HAVE CORRECT FORMAT!")
        println(" Linear terms on LEFT, nonlinear terms on RIGHT")
        println("="^40)
    end
    
    return true
end

"""
    validate_single_equation_format(equation::String, eq_index::Int)

Validate format of a single equation.
"""
function validate_single_equation_format(equation::String, eq_index::Int)
    violations = String[]
    
    # Split equation into LHS and RHS
    if !occursin("=", equation)
        push!(violations, "Equation must contain '=' sign")
        throw(EquationFormatError("No equals sign found", equation, eq_index, violations))
    end
    
    parts = split(equation, "=", limit=2)
    if length(parts) != 2
        push!(violations, "Equation must have exactly one '=' sign")
        throw(EquationFormatError("Multiple equals signs", equation, eq_index, violations))
    end
    
    lhs = String(strip(parts[1]))
    rhs = String(strip(parts[2]))
    
    # Validate LHS: should start with time derivative and contain only linear terms
    validate_lhs_format(lhs, violations, eq_index)
    
    # Validate RHS: should contain only nonlinear and non-constant terms
    validate_rhs_format(rhs, violations, eq_index)
    
    # If violations found, throw error
    if !isempty(violations)
        message = "Equation format does not follow required pattern: dt(var) ± linear_terms = nonlinear_terms"
        throw(EquationFormatError(message, equation, eq_index, violations))
    end
    
    return true
end

"""
    validate_lhs_format(lhs::String, violations::Vector{String}, eq_index::Int)

Validate that LHS contains only time derivative and linear terms.
"""
function validate_lhs_format(lhs::String, violations::Vector{String}, eq_index::Int)
    # Must start with time derivative
    if !contains_time_derivative_at_start(lhs)
        push!(violations, "LHS must start with time derivative dt(variable)")
    end
    
    # Extract terms after time derivative
    remaining_terms = extract_non_time_derivative_terms(lhs)
    
    # Check each remaining term on LHS
    for term in remaining_terms
        term_str = String(term)
        if is_nonlinear_term(term_str)
            push!(violations, "Nonlinear term '$term_str' found on LHS (should be on RHS)")
        elseif is_source_or_constant_term(term_str)
            push!(violations, "Constant/source term '$term_str' found on LHS (should be on RHS)")
        elseif !is_linear_term(term_str)
            push!(violations, "Unrecognized term '$term_str' on LHS (should be linear)")
        end
    end
end

"""
    validate_rhs_format(rhs::String, violations::Vector{String}, eq_index::Int)

Validate that RHS contains only nonlinear and non-constant terms.
"""
function validate_rhs_format(rhs::String, violations::Vector{String}, eq_index::Int)
    # Skip validation if RHS is just "0"
    if strip(rhs) in ["0", "0.0"]
        return
    end
    
    # Split RHS into terms
    rhs_terms = split_expression_into_terms(rhs)
    
    # Check each term on RHS
    for term in rhs_terms
        term_str = String(term)
        if is_linear_spatial_term(term_str)
            push!(violations, "Linear spatial term '$term_str' found on RHS (should be on LHS)")
        elseif is_linear_coupling_term(term_str)
            push!(violations, "Linear coupling term '$term_str' found on RHS (should be on LHS)")
        end
        # Note: nonlinear terms and sources are allowed on RHS
    end
end

"""
    contains_time_derivative_at_start(lhs::String)

Check if LHS starts with a time derivative.
"""
function contains_time_derivative_at_start(lhs::String)
    time_derivative_patterns = [
        r"^\s*dt\([^)]+\)",
        r"^\s*d_t\([^)]+\)",
        r"^\s*∂t\([^)]+\)",
        r"^\s*∂_t\([^)]+\)"
    ]
    
    return any(pattern -> occursin(pattern, lhs), time_derivative_patterns)
end

"""
    extract_non_time_derivative_terms(lhs::String)

Extract all terms from LHS except the time derivative.
"""
function extract_non_time_derivative_terms(lhs::String)
    # Remove the time derivative part
    lhs_without_dt = replace(lhs, r"dt\([^)]+\)" => "")
    lhs_without_dt = replace(lhs_without_dt, r"d_t\([^)]+\)" => "")
    lhs_without_dt = replace(lhs_without_dt, r"∂t\([^)]+\)" => "")
    lhs_without_dt = replace(lhs_without_dt, r"∂_t\([^)]+\)" => "")
    
    # Split into terms
    return split_expression_into_terms(lhs_without_dt)
end

"""
    split_expression_into_terms(expr::String)

Split expression into individual terms with signs.
"""
function split_expression_into_terms(expr::String)
    if isempty(strip(expr))
        return String[]
    end
    
    terms = String[]
    current_term = ""
    paren_depth = 0
    
    i = 1
    while i <= length(expr)
        char = expr[i]
        
        if char == '('
            paren_depth += 1
            current_term *= char
        elseif char == ')'
            paren_depth -= 1
            current_term *= char
        elseif (char == '+' || char == '-') && paren_depth == 0 && i > 1
            # End current term, start new one
            cleaned = strip(current_term)
            if !isempty(cleaned)
                push!(terms, cleaned)
            end
            current_term = string(char)
        else
            current_term *= char
        end
        
        i += 1
    end
    
    # Add last term
    cleaned = strip(current_term)
    if !isempty(cleaned)
        push!(terms, cleaned)
    end
    
    return filter(t -> !isempty(t), terms)
end

"""
    is_linear_term(term::String)

Check if a term is linear (allowed on LHS).
"""
function is_linear_term(term::String)
    clean_term = String(strip(lstrip(term, ['+', '-', ' '])))
    
    return (is_linear_spatial_term(clean_term) || 
            is_linear_coupling_term(clean_term) ||
            is_linear_variable_term(clean_term))
end

"""
    is_linear_spatial_term(term::String)

Check if term is a linear spatial derivative.
"""
function is_linear_spatial_term(term::String)
    # Clean term for better pattern matching - preserve sign
    clean = strip(term)
    # Remove leading signs for pattern matching
    clean_nosign = strip(lstrip(clean, ['+', '-', ' ']))
    
    # First check if it's a convection term (nonlinear) - this takes priority
    if contains_convection_terms(String(clean_nosign))
        return false
    end
    
    # Patterns for linear spatial derivatives with optional coefficients
    linear_spatial_patterns = [
        r"(^|.*\*)?lap\([a-zA-Z_][a-zA-Z0-9_]*\)$",
        r"(^|.*\*)?d2dx2\([a-zA-Z_][a-zA-Z0-9_]*\)$",
        r"(^|.*\*)?d2dy2\([a-zA-Z_][a-zA-Z0-9_]*\)$",
        r"(^|.*\*)?d2dz2\([a-zA-Z_][a-zA-Z0-9_]*\)$",
        r"(^|.*\*)?dx\([a-zA-Z_][a-zA-Z0-9_]*\)$",
        r"(^|.*\*)?dy\([a-zA-Z_][a-zA-Z0-9_]*\)$",
        r"(^|.*\*)?dz\([a-zA-Z_][a-zA-Z0-9_]*\)$"
    ]
    
    for pattern in linear_spatial_patterns
        if occursin(pattern, clean_nosign)
            # Check if it's a simple coefficient*derivative pattern
            if occursin("*", clean_nosign)
                # Split and check: should be coefficient*function(variable)
                parts = split(clean_nosign, "*", limit=2)
                if length(parts) == 2
                    coeff = strip(parts[1])
                    func_part = strip(parts[2])
                    # Coefficient should be a parameter/number, not a variable*variable
                    if !occursin("*", coeff) && !occursin("(", coeff)
                        return true
                    end
                end
            else
                # Just a pure derivative like lap(u)
                return true
            end
        end
    end
    
    return false
end

"""
    is_linear_coupling_term(term::String)

Check if term represents linear coupling to other variables.
"""
function is_linear_coupling_term(term::String)
    clean = String(strip(lstrip(term, ['+', '-', ' '])))
    
    # Pattern: coefficient*variable or just variable  
    # Examples: a*u, nu*v, 3.14*T, just u
    if occursin(r"^([a-zA-Z0-9_.]+\*)?[a-zA-Z_][a-zA-Z0-9_]*$", clean)
        # Make sure it's not variable*variable multiplication
        if !contains_variable_multiplication(clean)
            return true
        end
    end
    
    return false
end

"""
    is_linear_variable_term(term::String)

Check if term is linear in a single variable.
"""
function is_linear_variable_term(term::String)
    # Pattern: coefficient*variable or just variable
    return occursin(r"^[^*]*[a-zA-Z_][a-zA-Z0-9_]*$", term) && !contains_variable_multiplication(term)
end

"""
    is_nonlinear_term(term::String)

Check if term is nonlinear (not allowed on LHS).
"""
function is_nonlinear_term(term::String)
    clean_term = String(strip(lstrip(term, ['+', '-', ' '])))
    
    return (contains_variable_multiplication(clean_term) ||
            contains_nonlinear_functions(clean_term) ||
            contains_convection_terms(clean_term))
end

"""
    is_source_or_constant_term(term::String)

Check if term is a source or constant (not allowed on LHS).
"""
function is_source_or_constant_term(term::String)
    clean_term = String(strip(lstrip(term, ['+', '-', ' '])))
    
    # If it doesn't contain variables but contains parameters/numbers, it's a source
    if !contains_variables(clean_term) && (contains_parameters(clean_term) || is_numeric(clean_term))
        return true
    end
    
    return false
end

"""
    contains_variable_multiplication(term::String)

Check if term contains products of variables.
"""
function contains_variable_multiplication(term::String)
    # Look for patterns like: variable*variable, but exclude coefficient*variable patterns
    if !occursin("*", term)
        return false
    end
    
    # Split by * and check what's on each side
    parts = split(term, "*")
    if length(parts) < 2
        return false
    end
    
    # For coefficient*variable patterns, only count if we have multiple actual variables
    # Common coefficient patterns: lowercase letters (nu, alpha), numbers, Greek letters
    # Variable patterns: usually single uppercase or mixed case (u, v, T, phi)
    
    function_names = ["lap", "dx", "dy", "dz", "d2dx2", "d2dy2", "d2dz2", "exp", "sin", "cos", "log", "sqrt"]
    actual_variable_count = 0
    
    for part in parts
        part_clean = String(strip(part))
        
        # Skip function calls
        if occursin("(", part_clean)
            continue
        end
        
        # Skip pure numbers
        if occursin(r"^[0-9.]+$", part_clean)
            continue
        end
        
        # Skip common coefficient patterns (lowercase parameters)
        if occursin(r"^[a-z][a-z0-9]*$", part_clean) && !(part_clean in function_names)
            continue  # This is likely a coefficient/parameter
        end
        
        # This is likely an actual field variable
        if occursin(r"^[a-zA-Z_][a-zA-Z0-9_]*$", part_clean) && !(part_clean in function_names)
            actual_variable_count += 1
        end
    end
    
    # True multiplication if we have 2+ actual variables (not including coefficients)
    return actual_variable_count >= 2
end

"""
    contains_nonlinear_functions(term::String)

Check for nonlinear mathematical functions.
"""
function contains_nonlinear_functions(term::String)
    nonlinear_patterns = [
        r"\^[2-9]",    # Powers
        r"exp\(",      # Exponential
        r"sin\(",      # Trigonometric
        r"cos\(",
        r"log\(",      # Logarithmic
        r"sqrt\("      # Square root
    ]
    
    return any(pattern -> occursin(pattern, term), nonlinear_patterns)
end

"""
    contains_convection_terms(term::String)

Check for convection terms like u*dx(u).
"""
function contains_convection_terms(term::String)
    # Pattern: variable*derivative_of_variable
    return occursin(r"[a-zA-Z_][a-zA-Z0-9_]*\*d[xyz]\([a-zA-Z_][a-zA-Z0-9_]*\)", term)
end

"""
    contains_variables(term::String)

Check if term contains any variables.
"""
function contains_variables(term::String)
    return occursin(r"[a-zA-Z_][a-zA-Z0-9_]*", term)
end

"""
    contains_parameters(term::String)

Check if term contains parameters (lowercase symbols, Greek letters).
"""
function contains_parameters(term::String)
    # Simple heuristic: lowercase symbols that are likely parameters
    return occursin(r"[a-z][a-z0-9]*", term) || occursin(r"[α-ω]", term)
end

"""
    is_numeric(term::String)

Check if term is purely numeric.
"""
function is_numeric(term::String)
    try
        parse(Float64, term)
        return true
    catch
        return false
    end
end

"""
    demonstrate_format_validation()

Demonstrate the equation format validation with examples.
"""
function demonstrate_format_validation()
    println("EQUATION FORMAT VALIDATION DEMONSTRATION")
    println("="^55)
    
    # Test correct formats
    println(" TESTING CORRECT FORMATS:")
    correct_equations = [
        "dt(u) - nu*lap(u) - f*v = -u*dx(u)",
        "dt(v) - kappa*lap(v) + a*u = -b*u*v + source",
        "dt(T) - alpha*lap(T) = -u*dx(T) - v*dy(T) + Q"
    ]
    
    try
        validate_equation_format(correct_equations)
    catch e
        println("Unexpected error: $e")
    end
    
    println("\n" * "="*55)
    
    # Test incorrect formats
    println(" TESTING INCORRECT FORMATS:")
    incorrect_equations = [
        "dt(u) = nu*lap(u) - u*dx(u)",              # Linear term on RHS
        "dt(v) - u*dx(v) = kappa*lap(v)",           # Nonlinear on LHS, linear on RHS  
        "dt(T) + u*dx(T) = alpha*lap(T) + source"   # Mixed: nonlinear on LHS
    ]
    
    for (i, eq) in enumerate(incorrect_equations)
        println("\nTesting incorrect equation $i:")
        try
            validate_equation_format([eq])
        catch e
            if e isa EquationFormatError
                println(" Correctly caught format error!")
                println("Error details: $(e.message)")
            else
                println(" Unexpected error type: $e")
            end
        end
    end
end

# Export validation functions
export EquationFormatError, validate_equation_format, validate_single_equation_format
export demonstrate_format_validation