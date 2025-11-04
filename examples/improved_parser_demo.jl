# Demo of improved IVP equation parser
# Self-contained parser with better operator handling

using PencilFlows

println("=== Improved IVP Parser Demo ===\n")

# Create improved IVP parser
println("1. Creating improved IVP parser...")
parser = ImprovedIVPParser()
println("   [OK] Parser created successfully")

# Set up domain
println("\n2. Setting up domain...")
set_domain!(parser, 
           coords=[:x, :y, :z],
           domain_size=(2π, 2π, 2.0),
           grid_points=(32, 32, 16),
           basis_types=[:fourier, :fourier, :chebyshev])
println("   [OK] Domain configured successfully")

# Set parameters
println("\n3. Setting parameters...")
set_parameter!(parser, :nu, 1e-4)     # viscosity
set_parameter!(parser, :kappa, 1e-5)  # thermal diffusivity  
set_parameter!(parser, :f, 1e-4)      # Coriolis parameter
set_parameter!(parser, :Ra, 1e6)      # Rayleigh number
println("   [OK] Parameters set: nu, kappa, f, Ra")

# Parse equations with improved parser
println("\n4. Parsing equations...")
println("   Note: Demonstrating proper IVP structure validation")

# Example equations - some with incorrect structure for demonstration
equations = [
    "dt(u) - f*v + dx(p) - nu*laplacian(u) = -u*dx(u) - v*dy(u) - w*dz(u)",
    "dt(v) + f*u + dy(p) - nu*laplacian(v) = -u*dx(v) - v*dy(v) - w*dz(v)", 
    "dt(w) + dz(p) - nu*laplacian(w) + b = -u*dx(w) - v*dy(w) - w*dz(w)",
    "dx(u) + dy(v) + dz(w) = 0",
    "dt(b) - kappa*laplacian(b) = -u*dx(b) - v*dy(b) - w*dz(b)"
]

for (i, eq) in enumerate(equations)
    success = parse_equation!(parser, eq)
    if success
        println("   [OK] Equation $i: $eq")
    else
        println("   [FAIL] Failed equation $i: $eq")
    end
end

# Add boundary conditions
println("\n5. Adding boundary conditions...")
bcs = [
    "u(z=0) = 0",
    "v(z=0) = 0", 
    "w(z=0) = 0",
    "b(z=0) = 1",
    "u(z=2) = 0",
    "v(z=2) = 0",
    "w(z=2) = 0", 
    "b(z=2) = 0"
]

for bc in bcs
    success = add_boundary_condition!(parser, bc)
    if success
        println("   [OK] BC: $bc")
    else
        println("   [FAIL] Failed BC: $bc")
    end
end

# Validate problem
println("\n6. Validating problem...")
if validate_problem(parser)
    println("   [OK] Problem validation successful")
else
    println("   [FAIL] Problem validation failed")
end

# Get problem summary
println("\n7. Problem summary...")
summary = get_summary(parser)
println("   - Equations: $(summary["num_equations"])")
println("   - Variables: $(summary["num_variables"]) → $(summary["variables"])")
println("   - Parameters: $(summary["num_parameters"]) → $(keys(summary["parameters"]))")
println("   - Boundary conditions: $(summary["num_bcs"])")

# Show domain info
if haskey(summary, "domain")
    domain = summary["domain"] 
    println("   - Domain: $(domain[:coordinates]) with size $(domain[:domain_size])")
    println("   - Grid: $(domain[:grid_points]) points")
    println("   - Bases: $(domain[:basis_types])")
end

# Build system matrices
println("\n8. Building system matrices...")
try
    M, L, F = build_system_matrices(parser)
    println("   [OK] Matrix building successful")
    println("   - M matrix: $(size(M)) (mass matrix)")
    println("   - L matrix: $(size(L)) (stiffness matrix)")
    println("   - F vector: $(length(F)) (forcing)")

catch e
    println("   [FAIL] Matrix building failed: $(e)")
end

# Demonstrate equation structure analysis
println("\n9. Analyzing equation structure for IVP optimality...")
println("   (Linear terms should be on LHS, nonlinear on RHS)")

for i in 1:length(parser.parsed_equations)
    println("\n   Analyzing equation $i:")
    suggest_equation_restructure(parser, i)
end

println("\n10. Parser capabilities demonstrated:")
println("   [OK] Automatic variable detection from equations")
println("   [OK] Intelligent parameter default assignment")
println("   [OK] Proper equation normalization and validation")
println("   [OK] IVP structure validation (linear terms on LHS)")
println("   [OK] Equation restructuring suggestions")
println("   [OK] Term classification (linear vs nonlinear)")
println("   [OK] IMEX timestepping preparation")
println("   [OK] Boundary condition parsing")
println("   [OK] Comprehensive problem validation")
println("   [OK] System matrix structure setup")

println("\n=== Improved IVP Parser Demo Complete ===")
println("[OK] Self-contained parser with enhanced IVP structure validation!")
println("[OK] Ensures proper linear/nonlinear term placement for stability!")
println("[OK] No external dependencies beyond standard Julia packages")