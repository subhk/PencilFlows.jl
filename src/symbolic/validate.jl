# Validation helpers for equation structure

function _validate_equation_structure!(prob::SymbolicProblem, eq::String)
    strict = prob.metadata !== nothing && get(prob.metadata, :enforce_lhs_rhs_strict, false)
    has_eq = occursin('=', eq); has_eq || return
    lhs, rhs = split(eq, '='); problems = String[]
    occursin(r"\bdt\s*\(", rhs) && push!(problems, "RHS contains time derivative dt(...)  move to LHS")
    linear_tokens = [r"\blap\s*\(", r"\bdx\s*\(", r"\bdy\s*\(", r"\bdz\s*\(", r"\bgrad\s*\("]
    for tok in linear_tokens
        occursin(tok, rhs) && push!(problems, "RHS contains linear operator ($(tok))  move to LHS")
    end
    nlin_hints = [r"u\s*\*\s*dx\s*\(", r"v\s*\*\s*dy\s*\(", r"w\s*\*\s*dz\s*\(", r"u\s*\*\s*grad\s*\("]
    for tok in nlin_hints
        occursin(tok, lhs) && push!(problems, "LHS appears to contain nonlinear advection ($(tok))  move to RHS")
    end
    params = prob.parameters
    occursin(r"\bnu\s*\*\s*lap\s*\(", rhs) && push!(problems, "Viscous term nu*lap(u) found on RHS  move to LHS")
    if haskey(params, :f) || haskey(params, :Omega) || haskey(params, :coriolis)
        occursin(r"\bf\s*\*\s*v\b", rhs) && push!(problems, "Coriolis term f*v on RHS  move to LHS")
        occursin(r"\bf\s*\*\s*u\b", rhs) && push!(problems, "Coriolis term f*u on RHS  move to LHS")
    end
    occursin(r"\bnu\s*\(\s*", lhs) && push!(problems, "Nonconstant coefficient nu(...) on LHS  move to RHS")
    if !isempty(problems)
        msg = join(problems, "\n  - ")
        if strict
            error("Equation does not satisfy LHS/RHS split rules:\n  - " * msg * "\n  Equation: " * eq)
        else
            @warn "Equation may violate LHS/RHS split rules:\n  - $(msg)\n  Equation: $(eq)"
        end
    end
    return true
end

