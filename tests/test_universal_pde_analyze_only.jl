println("Universal PDE: analyze-only sanity check")

equations = [
    "dt(u) = -u*dx(u) - v*dy(u) - dx(p) + (1/Re)*lap(u)",
    "dt(v) = -u*dx(v) - v*dy(v) - dy(p) + (1/Re)*lap(v)",
    "dx(u) + dy(v) = 0"
]

system = analyze_pde_system(equations)

# Basic expectations: physics type detected and variables present
haskey(system, :physics_type) || error(":physics_type missing in analysis result")
haskey(system, :variables) || error(":variables missing in analysis result")
length(system[:variables]) > 0 || error("No variables detected in analysis result")

println("OK: analyze_pde_system returned expected structure")

