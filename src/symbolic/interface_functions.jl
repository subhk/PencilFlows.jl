# Interface Functions for SymbolicProblem
# Provides compatibility with ProblemBuilder interface

"""
    add_equation!(prob::SymbolicProblem, equation::String)

Add an equation to the symbolic problem.
"""
function add_equation!(prob::SymbolicProblem, equation::String)
    # Parse the equation string into a symbolic expression
    parsed_eq = Meta.parse(equation)
    push!(prob.equations, parsed_eq)
    println("Added equation to symbolic problem: $equation")
    return prob
end

"""
    add_bc!(prob::SymbolicProblem, boundary_condition::String)

Add a boundary condition to the symbolic problem.
"""
function add_bc!(prob::SymbolicProblem, boundary_condition::String)
    # Parse boundary condition
    bc = parse_boundary_condition_string(boundary_condition)
    push!(prob.boundary_conditions, bc)
    println("Added boundary condition to symbolic problem: $boundary_condition")
    return prob
end

"""
    add_parameter!(prob::SymbolicProblem, name::Union{Symbol,String}, value::Union{Real,Function,String})

Add a parameter to the symbolic problem.
"""
function add_parameter!(prob::SymbolicProblem, name::Union{Symbol,String}, value::Union{Real,Function,String})
    param_symbol = name isa String ? Symbol(name) : name
    
    if value isa String
        # Try to parse as numeric first, then symbolic
        try
            numeric_value = parse(Float64, value)
            prob.parameters[param_symbol] = numeric_value
            println("Added numeric parameter to symbolic problem: $param_symbol = $numeric_value")
        catch
            # If parse_symbolic_expression is available, use it
            if @isdefined(parse_symbolic_expression)
                func = parse_symbolic_expression(value)
                if func !== nothing
                    # Store as a callable function - convert to Float64 for now
                    # TODO: Enhance SymbolicProblem to handle function parameters properly
                    println("Warning: Symbolic parameter converted to constant for SymbolicProblem: $param_symbol")
                    prob.parameters[param_symbol] = 1.0  # Placeholder
                else
                    error("Could not parse symbolic expression: $value")
                end
            else
                # Fallback: try to evaluate as expression
                try
                    # Simple evaluation for basic expressions
                    prob.parameters[param_symbol] = 1.0  # Safe fallback
                    println("Warning: String parameter simplified to constant for SymbolicProblem: $param_symbol")
                catch
                    error("Could not process parameter string: $value")
                end
            end
        end
    elseif value isa Function
        println("Warning: Function parameter converted to constant for SymbolicProblem: $param_symbol")
        prob.parameters[param_symbol] = 1.0  # Placeholder
    else
        prob.parameters[param_symbol] = Float64(value)
        println("Added parameter to symbolic problem: $param_symbol = $value")
    end
    
    return prob
end

"""
    add_parameter!(prob::SymbolicProblem; kwargs...)

Add multiple parameters using named arguments.
"""
function add_parameter!(prob::SymbolicProblem; kwargs...)
    for (name, value) in kwargs
        add_parameter!(prob, Symbol(name), value)
    end
    return prob
end

"""
    parse_boundary_condition_string(bc_str::String)

Parse a boundary condition string into a SymbolicBoundaryCondition object.
"""
function parse_boundary_condition_string(bc_str::String)
    # Simple parsing - can be enhanced later
    # Expected format: "left(u) = 0" or "right(dz(T)) = 1.0"
    
    if occursin("left(", bc_str)
        location = :left
        remaining = split(bc_str, "left(")[2]
    elseif occursin("right(", bc_str)
        location = :right
        remaining = split(bc_str, "right(")[2]
    elseif occursin("top(", bc_str)
        location = :top
        remaining = split(bc_str, "top(")[2]
    elseif occursin("bottom(", bc_str)
        location = :bottom
        remaining = split(bc_str, "bottom(")[2]
    else
        error("Could not parse boundary condition location from: $bc_str")
    end
    
    # Extract field and value
    if occursin(") =", remaining)
        parts = split(remaining, ") =")
        field_part = strip(parts[1])
        value_part = strip(parts[2])
        
        # Check if it's a derivative condition
        if startswith(field_part, "dz(") || startswith(field_part, "dx(") || startswith(field_part, "dy(")
            # Extract field name from derivative
            field_match = match(r"d[xyz]\(([^)]+)\)", field_part)
            if field_match !== nothing
                field = Symbol(field_match.captures[1])
                bc_type = :neumann
            else
                error("Could not parse derivative field from: $field_part")
            end
        else
            # Simple field
            field = Symbol(field_part)
            bc_type = :dirichlet
        end
        
        # Parse value
        try
            bc_value = parse(Float64, value_part)
        catch
            # Treat as expression
            bc_value = Meta.parse(value_part)
        end
        
        # Create SymbolicBoundaryCondition using the correct constructor
        if bc_value isa Real
            return SymbolicBoundaryCondition(field, location, bc_type, bc_value)
        else
            return SymbolicBoundaryCondition(field, location, bc_type, bc_value)
        end
    else
        error("Could not parse boundary condition format from: $bc_str")
    end
end

"""
    solve!(prob::SymbolicProblem; kwargs...)

Solve the symbolic problem.
"""
function solve!(prob::SymbolicProblem; kwargs...)
    println("Solving symbolic problem...")
    println("  Equations: $(length(prob.equations))")
    println("  Boundary conditions: $(length(prob.boundary_conditions))")
    println("  Parameters: $(length(prob.parameters))")
    
    # Build the problem first
    build_problem!(prob)
    
    # TODO: Integrate with actual solver
    println("Symbolic problem solved (placeholder)")
    return prob
end

# Note: parse_symbolic_expression is imported from equation_boundary_interface.jl
# This function is available when the interfaces are loaded together