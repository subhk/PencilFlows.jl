# Boundary condition utilities

function add_time_dependent_bc!(prob::SymbolicProblem, field::Symbol, location::Symbol, func::Function; bc_type::Symbol=:dirichlet)
    bc = SymbolicBoundaryCondition(field, location, bc_type, func)
    push!(prob.boundary_conditions, bc)
    return prob
end

function evaluate_bc_value(bc::SymbolicBoundaryCondition, t_val::Real)
    if !bc.time_dependent
        return bc.value
    elseif isa(bc.value, Function)
        return bc.value(t_val)
    elseif isa(bc.value, Expr)
        expr_with_time = substitute_time_variable(bc.value, t_val)
        return eval(expr_with_time)
    else
        return bc.value
    end
end

