# Public API for equation and BC handling

function set_grid!(prob::SymbolicProblem, coord::Symbol, N::Int)
    prob.grid_points[coord] = N
    return prob
end

function add_equation!(prob::SymbolicProblem, equation::String)
    ensure_banner_shown!(prob)
    _validate_equation_structure!(prob, equation)
    eq_expr = parse_equation(equation)
    push!(prob.equations, eq_expr)
    show_build_progress("Added equation", equation)
    equation_summary(prob)
    return prob
end

function add_bc!(prob::SymbolicProblem, bc_spec::String)
    ensure_banner_shown!(prob)
    bc = parse_boundary_condition(bc_spec)
    push!(prob.boundary_conditions, bc)
    show_build_progress("Added boundary condition", bc_spec)
    return prob
end

