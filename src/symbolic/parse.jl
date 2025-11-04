# Improved parsing utilities with better operator handling

@deprecated "Use ImprovedIVPParser.parse_equation! instead" 
function parse_equation(eq_str::String)
    @warn "parse_equation is deprecated. Use ImprovedIVPParser.parse_equation! for better parsing."
    # Fallback to simple Meta.parse for compatibility
    clean_eq = replace(eq_str, " " => "")
    clean_eq = replace(clean_eq, "**" => "^")  # Convert Python-style exponentiation
    
    lhs_str, rhs_str = split(clean_eq, "=")
    lhs_expr = Meta.parse(lhs_str)
    rhs_expr = Meta.parse(rhs_str)
    return :($lhs_expr - ($rhs_expr) == 0)
end

# Import new improved parser
include("improved_parser.jl")

# parse_boundary_condition is now defined in parse_bc.jl to avoid duplication

# contains_time_variable moved to parse_utils.jl to avoid duplication

