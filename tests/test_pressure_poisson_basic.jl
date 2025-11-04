println("Pressure Poisson Derivation: standard incompressible case")

momentum = [
    "dt(u) + u*dx(u) + v*dy(u) + w*dz(u) = -dx(p) + nu*lap(u)",
    "dt(v) + u*dx(v) + v*dy(v) + w*dz(v) = -dy(p) + nu*lap(v)",
    "dt(w) + u*dx(w) + v*dy(w) + w*dz(w) = -dz(p) + nu*lap(w)"
]

poisson = derive_pressure_poisson_equation(momentum; incompressibility = "div(u) = 0")
println("  equation: ", poisson.equation_string)

typeof(poisson.solver_requirements) == Dict{Symbol,Any} || error("solver_requirements not a Dict")
occursin("∇²", poisson.equation_string) || occursin("lap", poisson.equation_string) || error("Poisson LHS not detected in equation_string")

println("OK: Poisson derivation looks consistent")

