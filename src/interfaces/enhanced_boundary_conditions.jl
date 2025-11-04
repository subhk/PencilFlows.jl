# Enhanced Boundary Conditions for General Equation Interface
# ============================================================
# Extends the equation/boundary interface to support:
# - Time-dependent boundary conditions
# - Spatially-dependent boundary conditions  
# - Complex functional boundary conditions
# - Mixed time-space dependencies

using Printf

"""
    EnhancedBoundaryCondition

Enhanced boundary condition supporting time, spatial, and functional dependencies.
"""
struct EnhancedBoundaryCondition
    variable::String
    location::String          # "left", "right", "top", "bottom", etc.
    condition_type::Symbol    # :dirichlet, :neumann, :robin
    expression::String        # Original user expression
    time_dependent::Bool
    spatial_dependent::Bool  
    julia_function::Union{Function, Nothing}
    description::String
end

"""
    analyze_boundary_condition_dependencies(bc_string::String)

Analyze a boundary condition string to detect time and spatial dependencies.
"""
function analyze_boundary_condition_dependencies(bc_string::String)
    time_dependent = false
    spatial_dependent = false
    description_parts = String[]
    
    # Time dependency patterns
    time_patterns = [
        r"\bt\b",              # explicit t variable
        r"sin\s*\(\s*[^)]*t",  # sin(t), sin(2*t), etc.
        r"cos\s*\(\s*[^)]*t",  # cos(t), cos(omega*t), etc.
        r"exp\s*\(\s*[^)]*t",  # exp(t), exp(-t), etc.
        r"t\s*[\^]\s*[0-9]+",  # t^2, t^3, etc.
        r"[0-9.]+\s*\*\s*t",   # 2*t, 0.5*t, etc.
    ]
    
    # Spatial dependency patterns  
    spatial_patterns = [
        r"\bx\b",              # explicit x coordinate
        r"\by\b",              # explicit y coordinate
        r"\bz\b",              # explicit z coordinate
        r"sin\s*\(\s*[^)]*[xyz]",  # sin(x), sin(pi*y), etc.
        r"cos\s*\(\s*[^)]*[xyz]",  # cos(z), cos(2*pi*x), etc.
        r"exp\s*\(\s*[^)]*[xyz]",  # exp(x), exp(-z), etc.
    ]
    
    # Check for time dependencies
    for pattern in time_patterns
        if occursin(pattern, bc_string)
            time_dependent = true
            push!(description_parts, "time-dependent")
            break
        end
    end
    
    # Check for spatial dependencies
    for pattern in spatial_patterns
        if occursin(pattern, bc_string)
            spatial_dependent = true
            push!(description_parts, "spatially-varying")
            break
        end
    end
    
    # Generate description
    if isempty(description_parts)
        description = "constant"
    else
        description = join(description_parts, ", ")
    end
    
    return time_dependent, spatial_dependent, description
end

"""
    parse_enhanced_boundary_condition(bc_string::String)

Parse a boundary condition string into an EnhancedBoundaryCondition.

Supports formats like:
- "left(u) = sin(t)"                    # Time-dependent
- "right(T) = 1 + 0.5*sin(2*pi*x)"     # Spatially-varying  
- "top(u) = exp(-t)*cos(pi*y)"          # Time and space dependent
- "bottom(dz(T)) = 0"                   # Neumann condition
"""
function parse_enhanced_boundary_condition(bc_string::String)
    # Extract location and variable
    location_match = match(r"(left|right|top|bottom|front|back)\s*\(\s*([^)]+)\s*\)", bc_string)
    if location_match === nothing
        error("Invalid boundary condition format: $bc_string")
    end
    
    location = location_match.captures[1]
    variable_part = location_match.captures[2]
    
    # Determine if it's a derivative (Neumann) or value (Dirichlet)
    condition_type = :dirichlet
    variable = variable_part
    if occursin(r"d[xyz]\s*\(", variable_part)
        condition_type = :neumann
        # Extract variable from derivative: dz(u) -> u
        var_match = match(r"d[xyz]\s*\(\s*([^)]+)\s*\)", variable_part)
        variable = var_match !== nothing ? var_match.captures[1] : variable_part
    end
    
    # Extract the expression after the equals sign
    equals_split = split(bc_string, "=", limit=2)
    if length(equals_split) != 2
        error("Boundary condition must contain '=' sign: $bc_string")
    end
    expression = String(strip(equals_split[2]))
    
    # Analyze dependencies
    time_dependent, spatial_dependent, description = analyze_boundary_condition_dependencies(expression)
    
    return EnhancedBoundaryCondition(
        variable,
        location,
        condition_type,
        expression,
        time_dependent,
        spatial_dependent,
        nothing,  # julia_function (filled later if needed)
        description
    )
end

"""
    add_enhanced_bc!(builder::ProblemBuilder, bc_string::String)

Add an enhanced boundary condition with automatic time/spatial dependency detection.
"""
function add_enhanced_bc!(builder::ProblemBuilder, bc_string::String)
    println(" Adding enhanced boundary condition: $bc_string")
    
    try
        # Parse the boundary condition
        enhanced_bc = parse_enhanced_boundary_condition(bc_string)
        
        # Validate basic format first
        validate_boundary_condition_format(bc_string)
        
        # Add to builder's regular boundary conditions
        push!(builder.boundary_conditions, bc_string)
        
        # Store enhanced info in builder metadata
        if !haskey(builder.domain_info, :enhanced_bcs)
            builder.domain_info[:enhanced_bcs] = EnhancedBoundaryCondition[]
        end
        push!(builder.domain_info[:enhanced_bcs], enhanced_bc)
        
        # Print analysis
        dependency_info = []
        enhanced_bc.time_dependent && push!(dependency_info, "time-dependent")
        enhanced_bc.spatial_dependent && push!(dependency_info, " spatially-varying")
        enhanced_bc.condition_type == :neumann && push!(dependency_info, " Neumann (derivative)")
        enhanced_bc.condition_type == :dirichlet && push!(dependency_info, " Dirichlet (value)")
        
        if isempty(dependency_info)
            println("   Constant boundary condition accepted")
        else
            println("   Enhanced BC accepted: $(join(dependency_info, ", "))")
        end
        
    catch e
        println("   Enhanced BC rejected: $e")
        rethrow(e)
    end
    
    return builder
end

"""
    add_functional_bc!(builder::ProblemBuilder, variable::String, location::String, func::Function; condition_type::Symbol=:dirichlet)

Add a boundary condition using a Julia function.

# Examples
```julia
# Time-dependent oscillating wall
add_functional_bc!(builder, "u", "left", t -> sin(2*π*t))

# Spatially-varying temperature  
add_functional_bc!(builder, "T", "bottom", (x,y,z) -> 1 + 0.5*sin(π*x))

# Time and space dependent
add_functional_bc!(builder, "u", "top", (x,y,z,t) -> exp(-t)*sin(π*x)*cos(π*y))
```
"""
function add_functional_bc!(builder::ProblemBuilder, variable::String, location::String, func::Function; condition_type::Symbol=:dirichlet)
    println(" Adding functional boundary condition: $variable at $location")
    
    # Determine function signature to detect dependencies
    func_string = string(func)
    time_dependent = occursin("t", func_string) || length(methods(func)) > 1
    spatial_dependent = any(coord -> occursin(coord, func_string), ["x", "y", "z"])
    
    # Create enhanced BC
    description = "Julia function"
    if time_dependent && spatial_dependent
        description = "time and space dependent function"
    elseif time_dependent
        description = "time-dependent function"
    elseif spatial_dependent
        description = "spatially-varying function"
    end
    
    enhanced_bc = EnhancedBoundaryCondition(
        variable,
        location,
        condition_type,
        "function",  # placeholder expression
        time_dependent,
        spatial_dependent,
        func,
        description
    )
    
    # Store in builder
    if !haskey(builder.domain_info, :enhanced_bcs)
        builder.domain_info[:enhanced_bcs] = EnhancedBoundaryCondition[]
    end
    push!(builder.domain_info[:enhanced_bcs], enhanced_bc)
    
    if !haskey(builder.domain_info, :functional_bcs)
        builder.domain_info[:functional_bcs] = Dict{String, Function}()
    end
    key = "$(location)_$(variable)"
    builder.domain_info[:functional_bcs][key] = func
    
    println("   Functional BC added: $description")
    return builder
end

"""
    analyze_boundary_condition_system(builder::ProblemBuilder)

Analyze all boundary conditions for time/spatial dependencies and compatibility.
"""
function analyze_boundary_condition_system(builder::ProblemBuilder)
    println(" ANALYZING BOUNDARY CONDITION SYSTEM")
    println("="^45)
    
    enhanced_bcs = get(builder.domain_info, :enhanced_bcs, EnhancedBoundaryCondition[])
    functional_bcs = get(builder.domain_info, :functional_bcs, Dict{String, Function}())
    
    if isempty(enhanced_bcs) && isempty(functional_bcs)
        println(" No enhanced boundary conditions detected")
        println(" Standard constant boundary conditions only")
        return
    end
    
    # Count different types
    time_dependent_count = count(bc -> bc.time_dependent, enhanced_bcs)
    spatial_dependent_count = count(bc -> bc.spatial_dependent, enhanced_bcs)
    functional_count = length(functional_bcs)
    
    println(" Boundary Condition Analysis:")
    println("  • Total BCs: $(length(builder.boundary_conditions))")
    println("  • Time-dependent: $time_dependent_count")
    println("  • Spatially-varying: $spatial_dependent_count")
    println("  • Functional: $functional_count")
    
    # Detailed analysis
    if !isempty(enhanced_bcs)
        println("\\n Enhanced Boundary Conditions:")
        for (i, bc) in enumerate(enhanced_bcs)
            deps = String[]
            bc.time_dependent && push!(deps, "time")
            bc.spatial_dependent && push!(deps, "spatial")
            dep_str = isempty(deps) ? "constant" : join(deps, "+")
            
            println("  $i. $(bc.location)($(bc.variable)) = $(bc.expression)")
            println("     Type: $(bc.condition_type), Dependencies: $dep_str")
        end
    end
    
    # Time integration implications
    if time_dependent_count > 0
        println("\\nTIME INTEGRATION IMPLICATIONS:")
        println("  • Boundary conditions will be updated each timestep")
        println("  • Solver must evaluate BC expressions at current time")
        println("  • May require smaller timesteps for rapidly varying BCs")
    end
    
    # Spatial dependency implications
    if spatial_dependent_count > 0
        println("\\n SPATIAL DEPENDENCY IMPLICATIONS:")
        println("  • BCs vary across boundary surfaces")
        println("  • Requires pointwise evaluation on boundary nodes")
        println("  • May affect convergence and stability")
    end
    
    println("="^45)
end

"""
    demonstrate_enhanced_boundary_conditions()

Demonstrate enhanced boundary condition capabilities.
"""
function demonstrate_enhanced_boundary_conditions()
    println(" ENHANCED BOUNDARY CONDITIONS DEMONSTRATION")
    println("="^60)
    
    # Example 1: Time-dependent oscillating wall
    println("\\n Example 1: Oscillating Driven Cavity")
    println("-"^40)
    
    builder1 = ProblemBuilder()
    add_equation!(builder1, "dt(u) - nu*lap(u) + dx(p) = -u*dx(u) - v*dy(u)")
    add_equation!(builder1, "dt(v) - nu*lap(v) + dy(p) = -u*dx(v) - v*dy(v)")
    
    # Time-dependent boundary conditions
    add_enhanced_bc!(builder1, "left(u) = 0")                    # Static bottom wall
    add_enhanced_bc!(builder1, "left(v) = 0")
    add_enhanced_bc!(builder1, "right(u) = sin(t)")             # Oscillating top wall
    add_enhanced_bc!(builder1, "right(v) = 0.5*cos(2*t)")       # Secondary oscillation
    
    add_parameter!(builder1, :nu, 1e-3)
    
    analyze_boundary_condition_system(builder1)
    
    # Example 2: Spatially-varying temperature
    println("\\n Example 2: Non-Uniform Heating")
    println("-"^35)
    
    builder2 = ProblemBuilder()
    add_equation!(builder2, "dt(T) - alpha*lap(T) = -u*dx(T) - v*dy(T)")
    add_equation!(builder2, "dt(u) - nu*lap(u) = buoyancy*T")
    
    # Spatially-varying boundary conditions
    add_enhanced_bc!(builder2, "left(T) = 1 + 0.5*sin(pi*x)")      # Sinusoidal heating
    add_enhanced_bc!(builder2, "right(T) = 0.2*cos(2*pi*y)")       # Cooling pattern
    add_enhanced_bc!(builder2, "left(u) = 0")
    add_enhanced_bc!(builder2, "right(u) = 0")
    
    add_parameter!(builder2, :alpha, 1e-2)
    add_parameter!(builder2, :nu, 1e-3)
    add_parameter!(builder2, :buoyancy, 100.0)
    
    analyze_boundary_condition_system(builder2)
    
    # Example 3: Complex functional boundary conditions
    println("\\n Example 3: Functional Boundary Conditions")
    println("-"^45)
    
    builder3 = ProblemBuilder()
    add_equation!(builder3, "dt(u) - nu*lap(u) = -u*dx(u)")
    add_equation!(builder3, "dt(T) - alpha*lap(T) = -u*dx(T)")
    
    # Add functional BCs using Julia functions
    add_functional_bc!(builder3, "u", "left", t -> exp(-0.1*t)*sin(2*π*t))  # Decaying oscillation
    add_functional_bc!(builder3, "T", "right", (x,y,z) -> 1 + 0.3*sin(π*x)*cos(π*y))  # 2D pattern
    
    # Mixed: time-space dependent
    complex_bc(x,y,z,t) = (1 + 0.2*sin(t)) * exp(-z) * sin(π*x)
    add_functional_bc!(builder3, "T", "left", complex_bc)
    
    add_parameter!(builder3, :nu, 1e-3)
    add_parameter!(builder3, :alpha, 1e-2)
    
    analyze_boundary_condition_system(builder3)
    
    println("\\n ENHANCED BOUNDARY CONDITIONS DEMO COMPLETE!")
    println("="^60)
    println(" Capabilities demonstrated:")
    println("  • Time-dependent BCs: sin(t), cos(2*t), exp(-t)")
    println("  • Spatially-varying BCs: sin(π*x), cos(2*π*y)")
    println("  • Functional BCs: Julia functions with arbitrary complexity")
    println("  • Mixed dependencies: f(x,y,z,t)")
    println("  • Automatic dependency detection and analysis")
end

# Export enhanced boundary condition functions
export EnhancedBoundaryCondition, parse_enhanced_boundary_condition
export add_enhanced_bc!, add_functional_bc!, analyze_boundary_condition_system
export demonstrate_enhanced_boundary_conditions
