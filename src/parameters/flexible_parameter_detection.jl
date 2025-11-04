# Flexible Automatic Parameter Detection for PencilFlows.jl
# Supports ANY variable names and parameters - only x,y,z,t are fixed

"""
    FlexibleEquationAnalysis

Analysis result for equations with arbitrary variable and parameter names.
Only x,y,z,t are reserved as spatial coordinates and time.
"""
struct FlexibleEquationAnalysis
    equations::Vector{String}
    variables::Set{Symbol}           # All field variables (detected from dt(...))
    parameters::Set{Symbol}          # All parameters (everything else)
    operators::Set{Symbol}          # Differential operators
    spatial_coords::Set{Symbol}     # Always {:x, :y, :z}
    time_coord::Symbol              # Always :t
    equation_structure::Dict{Symbol, Any}
end

"""
    analyze_flexible_equations(equations::Vector{String})

Automatically detect variables, parameters, and structure from arbitrary equations.

# Rules:
- x, y, z, t are ALWAYS spatial coordinates and time
- Variables are detected from dt(...) terms (time derivatives)
- Everything else (except operators) is a parameter
- No hardcoded variable names

# Examples:
```julia
# Any variable names allowed
equations = [
    "dt(velocity) = -pressure_grad + diffusion*lap(velocity)", 
    "dt(temperature) = thermal_diffusion*lap(temperature)",
    "div(velocity) = 0"
]

# System automatically detects:
# Variables: {velocity, temperature, pressure_grad}
# Parameters: {diffusion, thermal_diffusion}
```
"""
function analyze_flexible_equations(equations::Vector{String})
    variables = Set{Symbol}()
    parameters = Set{Symbol}()
    operators = Set{Symbol}()
    
    # Fixed coordinates (never change)
    spatial_coords = Set([:x, :y, :z])
    time_coord = :t
    
    # Known differential operators (these are never parameters)
    known_operators = Set([
        :dt, :dx, :dy, :dz,            # First derivatives
        :d2dx2, :d2dy2, :d2dz2,        # Second derivatives
        :lap, :div, :grad, :curl,      # Vector operators
        :laplacian, :divergence, :gradient, :rotation
    ])
    
    equation_structure = Dict{Symbol, Any}()
    
    for (i, eq_str) in enumerate(equations)
        println(" Analyzing equation $i: $eq_str")
        
        # Parse equation into tokens
        tokens = extract_all_tokens(eq_str)
        
        for token_str in tokens
            if !isempty(token_str) && !isnumeric_string(token_str)
                sym = Symbol(token_str)
                
                # Skip fixed coordinates
                if sym in spatial_coords || sym == time_coord
                    continue
                end
                
                # Skip known operators
                if sym in known_operators
                    push!(operators, sym)
                    continue
                end
                
                # Check if this is a time derivative dt(...)
                if is_time_derivative_term(eq_str, token_str)
                    var_name = extract_variable_from_dt(eq_str, token_str)
                    if var_name != :nothing
                        push!(variables, var_name)
                        println(" Variable detected: $var_name (from time derivative)")
                    end
                # Check if this is inside spatial derivatives dx(...), dy(...), etc.
                elseif is_spatial_derivative_argument(eq_str, token_str)
                    var_name = Symbol(token_str)
                    push!(variables, var_name)
                    println(" Variable detected: $var_name (from spatial derivative)")
                # Everything else is a parameter
                else
                    push!(parameters, sym)
                    println(" Parameter detected: $sym")
                end
            end
        end
    end
    
    # Additional analysis: detect equation types and physics
    equation_structure = analyze_equation_physics(equations, variables, parameters)
    
    return FlexibleEquationAnalysis(
        equations, variables, parameters, operators,
        spatial_coords, time_coord, equation_structure
    )
end

"""
    extract_all_tokens(equation_str::String) -> Vector{String}

Extract all tokens from equation string, handling function calls and operators.
"""
function extract_all_tokens(equation_str::String)
    tokens = String[]
    
    # Remove common mathematical symbols and split
    clean_str = replace(equation_str, 
        r"[=+\-*/()^{}[\],]" => " ",
        r"\s+" => " "
    )
    
    # Split and collect non-empty tokens
    raw_tokens = split(clean_str)
    
    for token in raw_tokens
        token_str = string(token)
        if !isempty(strip(token_str))
            push!(tokens, strip(token_str))
        end
    end
    
    # Also extract function arguments from original string
    function_args = extract_function_arguments(equation_str)
    append!(tokens, function_args)
    
    return unique(tokens)
end

"""
    is_time_derivative_term(equation_str::String, token::String) -> Bool

Check if token appears in a time derivative dt(...).
"""
function is_time_derivative_term(equation_str::String, token::String)
    # Look for patterns like dt(token) or dt( token )
    pattern = "dt\\s*\\(\\s*" * token * "\\s*\\)"
    return occursin(Regex(pattern), equation_str)
end

"""
    extract_variable_from_dt(equation_str::String, token::String) -> Symbol

Extract variable name from dt(...) term.
"""
function extract_variable_from_dt(equation_str::String, token::String)
    # Match dt(variable_name)
    m = match(r"dt\s*\(\s*(\w+)\s*\)", equation_str)
    if m !== nothing
        return Symbol(m.captures[1])
    end
    return :nothing
end

"""
    is_spatial_derivative_argument(equation_str::String, token::String) -> Bool

Check if token appears as argument in spatial derivatives dx(...), dy(...), etc.
"""
function is_spatial_derivative_argument(equation_str::String, token::String)
    spatial_derivs = ["dx", "dy", "dz", "d2dx2", "d2dy2", "d2dz2", "lap", "div", "grad"]
    
    for deriv in spatial_derivs
        pattern = Regex(deriv * "\\s*\\(\\s*" * token * "\\s*\\)")
        if occursin(pattern, equation_str)
            return true
        end
    end
    
    return false
end

"""
    extract_function_arguments(equation_str::String) -> Vector{String}

Extract all arguments from function calls in the equation.
"""
function extract_function_arguments(equation_str::String)
    args = String[]
    
    # Find all function calls: function_name(argument)
    for m in eachmatch(r"(\w+)\s*\(\s*(\w+)\s*\)", equation_str)
        if length(m.captures) >= 2
            push!(args, m.captures[2])  # The argument
        end
    end
    
    return args
end

"""
    analyze_equation_physics(equations, variables, parameters) -> Dict

Analyze the physics type and characteristics from the detected variables and parameters.
"""
function analyze_equation_physics(equations::Vector{String}, variables::Set{Symbol}, parameters::Set{Symbol})
    physics = Dict{Symbol, Any}()
    
    # Count equations and variables
    physics[:num_equations]  = length(equations)
    physics[:num_variables]  = length(variables) 
    physics[:num_parameters] = length(parameters)
    
    # Detect physics type based on equation patterns
    physics[:has_time_evolution] = any(eq -> occursin("dt(", eq), equations)
    physics[:has_diffusion] = any(eq -> occursin("lap(", eq), equations)
    physics[:has_advection] = any(eq -> occursin(r"\w+\*d[xyz]\(", eq), equations)
    physics[:has_incompressibility] = any(eq -> occursin("div(", eq), equations)
    
    # Dimensionality detection
    has_x = any(eq -> occursin("dx(", eq), equations)
    has_y = any(eq -> occursin("dy(", eq), equations) 
    has_z = any(eq -> occursin("dz(", eq), equations)
    
    physics[:dimensionality] = sum([has_x, has_y, has_z])
    
    # Parameter classification (try to guess physical meaning)
    physics[:likely_viscosity] = find_likely_viscosity_parameters(parameters, equations)
    physics[:likely_diffusivity] = find_likely_diffusivity_parameters(parameters, equations)
    physics[:likely_forcing] = find_likely_forcing_parameters(parameters, equations)
    
    return physics
end

"""
    find_likely_viscosity_parameters(parameters, equations) -> Vector{Symbol}

Guess which parameters might be viscosity based on usage patterns.
"""
function find_likely_viscosity_parameters(parameters::Set{Symbol}, equations::Vector{String})
    likely_viscosity = Symbol[]
    
    for param in parameters
        param_str = string(param)
        
        # Look for parameter multiplying Laplacian terms
        for eq in equations
            if occursin("$param_str*lap(", eq) || occursin("lap($param_str*", eq)
                push!(likely_viscosity, param)
                break
            end
        end
    end
    
    return likely_viscosity
end

"""
    find_likely_diffusivity_parameters(parameters, equations) -> Vector{Symbol}

Guess which parameters might be thermal/scalar diffusivity.
"""
function find_likely_diffusivity_parameters(parameters::Set{Symbol}, equations::Vector{String})
    likely_diffusivity = Symbol[]
    
    for param in parameters
        param_str = string(param)
        
        # Look for patterns suggesting thermal diffusion
        for eq in equations
            if (occursin("$param_str*lap(", eq) && 
                (occursin("temperature", lowercase(eq)) || 
                 occursin("thermal", lowercase(eq)) ||
                 occursin("heat", lowercase(eq))))
                push!(likely_diffusivity, param)
                break
            end
        end
    end
    
    return likely_diffusivity
end

"""
    find_likely_forcing_parameters(parameters, equations) -> Vector{Symbol}

Guess which parameters might be forcing terms.
"""
function find_likely_forcing_parameters(parameters::Set{Symbol}, equations::Vector{String})
    likely_forcing = Symbol[]
    
    for param in parameters
        param_str = string(param)
        
        # Look for parameters that appear as standalone terms (not multiplying derivatives)
        for eq in equations
            if (occursin(param_str, eq) && 
                !occursin("$param_str*lap(", eq) &&
                !occursin("$param_str*d", eq))
                push!(likely_forcing, param)
                break
            end
        end
    end
    
    return likely_forcing
end

"""
    isnumeric_string(s::String) -> Bool

Check if string represents a number.
"""
function isnumeric_string(s::String)
    return tryparse(Float64, s) !== nothing
end

"""
    create_flexible_problem(equations::Vector{String}; default_values::Dict{Symbol,Float64}=Dict())

Create a problem from arbitrary equations with automatic parameter detection.

# Example:
```julia
equations = [
    "dt(fluid_velocity) = -my_viscosity*lap(fluid_velocity) + external_force",
    "dt(scalar_field) = thermal_coeff*lap(scalar_field)"
]

prob = create_flexible_problem(equations; 
    default_values = Dict(:my_viscosity => 0.01, :thermal_coeff => 0.005))
```
"""
function create_flexible_problem(equations::Vector{String}; default_values::Dict{Symbol,Float64}=Dict{Symbol,Float64}())
    analysis = analyze_flexible_equations(equations)
    
    println(" Flexible Problem Creation Summary:")
    println("   Variables: $(sort(collect(analysis.variables)))")
    println("   Parameters: $(sort(collect(analysis.parameters)))")
    println("   Physics: $(analysis.equation_structure)")
    
    # Create problem structure (simplified - would integrate with full PencilFlows)
    problem = Dict{Symbol, Any}(
        :equations => analysis.equations,
        :variables => analysis.variables,
        :parameters => Dict{Symbol, Float64}(),
        :physics => analysis.equation_structure
    )
    
    # Set parameter values
    for param in analysis.parameters
        if haskey(default_values, param)
            problem[:parameters][param] = default_values[param]
            println("  Set $param = $(default_values[param])")
        else
            # Use intelligent defaults based on likely parameter type
            default_val = guess_parameter_default(param, analysis)
            problem[:parameters][param] = default_val
            println("  Auto-set $param = $default_val (guessed)")
        end
    end
    
    return problem
end

"""
    guess_parameter_default(param::Symbol, analysis::FlexibleEquationAnalysis) -> Float64

Intelligently guess default values for parameters based on their likely physical meaning.
"""
function guess_parameter_default(param::Symbol, analysis::FlexibleEquationAnalysis)
    param_str = lowercase(string(param))
    
    # Viscosity-like parameters
    if (param in analysis.equation_structure[:likely_viscosity] ||
        occursin("visc", param_str) || occursin("nu", param_str))
        return 1e-3  # Typical kinematic viscosity
    end
    
    # Diffusivity-like parameters  
    if (param in analysis.equation_structure[:likely_diffusivity] ||
        occursin("diffus", param_str) || occursin("thermal", param_str) ||
        occursin("kappa", param_str))
        return 1e-3  # Typical thermal diffusivity
    end
    
    # Forcing-like parameters
    if (param in analysis.equation_structure[:likely_forcing] ||
        occursin("force", param_str) || occursin("source", param_str))
        return 1.0   # Unit forcing
    end
    
    # Reynolds-like numbers
    if occursin("reynolds", param_str) || occursin("re", param_str)
        return 1000.0
    end
    
    # Coriolis-like parameters
    if (occursin("coriolis", param_str) || occursin("rotation", param_str) ||
        param_str == "f" || occursin("omega", param_str))
        return 1e-4  # Earth-like rotation
    end
    
    # Default for unknown parameters
    return 1.0
end

# Export the flexible detection functions
export FlexibleEquationAnalysis, analyze_flexible_equations, create_flexible_problem
