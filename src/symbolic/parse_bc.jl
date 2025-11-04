# Boundary condition parsing

function parse_boundary_condition(bc_str::String)
    clean_bc = replace(bc_str, " " => "")
    lhs_str, rhs_str = split(clean_bc, "=")
    lhs_expr = Meta.parse(lhs_str)
    if lhs_expr.head == :call
        location = lhs_expr.args[1]
        field_expr = lhs_expr.args[2]
        if isa(field_expr, Expr) && field_expr.head == :call
            field_name = field_expr.args[2]; bc_type = :neumann
        else
            field_name = field_expr; bc_type = :dirichlet
        end
    else
        error("Invalid boundary condition format: $bc_str")
    end
    rhs_try = try Meta.parse(rhs_str) catch; nothing end
    if rhs_try isa Expr && rhs_try.head == :call && rhs_try.args[1] == :robin
        args = rhs_try.args[2:end]
        length(args) == 3 || error("robin(a,b,c) requires 3 arguments")
        a, b, c = args[1], args[2], args[3]
        getval(ex) = contains_time_variable(ex) ? ex : (try Float64(eval(ex)) catch; ex end)
        value = (getval(a), getval(b), getval(c))
        return SymbolicBoundaryCondition(field_name, location, :robin, value)
    end
    try
        value = parse(Float64, rhs_str)
        return SymbolicBoundaryCondition(field_name, location, bc_type, value)
    catch
        rhs_expr = Meta.parse(rhs_str)
        if contains_time_variable(rhs_expr)
            return SymbolicBoundaryCondition(field_name, location, bc_type, rhs_expr)
        else
            try
                value = eval(rhs_expr)
                return SymbolicBoundaryCondition(field_name, location, bc_type, Float64(value))
            catch
                return SymbolicBoundaryCondition(field_name, location, bc_type, rhs_expr)
            end
        end
    end
end

