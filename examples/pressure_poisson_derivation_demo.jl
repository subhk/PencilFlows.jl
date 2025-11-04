# Demonstration of Automatic Pressure Poisson Equation Derivation
# ================================================================
# This example shows how PencilFlows.jl can automatically derive the correct
# pressure Poisson equation from user-provided momentum equations

using PencilFlows

"""
    demo_standard_navier_stokes()

Standard incompressible Navier-Stokes equations.
"""
function demo_standard_navier_stokes()
    println("EXAMPLE 1: Standard Incompressible Navier-Stokes")
    println("="^70)
    
    momentum_equations = [
        "dt(u) + u*dx(u) + v*dy(u) + w*dz(u) = -dx(p) + ν*lap(u)",
        "dt(v) + u*dx(v) + v*dy(v) + w*dz(v) = -dy(p) + ν*lap(v)", 
        "dt(w) + u*dx(w) + v*dy(w) + w*dz(w) = -dz(p) + ν*lap(w)"
    ]
    
    println("INPUT EQUATIONS:")
    for (i, eq) in enumerate(momentum_equations)
        println("  $(i). $(eq)")
    end
    println()
    
    poisson_eq = derive_pressure_poisson_equation(momentum_equations;
                                                 incompressibility="div(u) = 0",
                                                 time_discretization=:semi_implicit)
    
    solver_code = generate_poisson_solver_code(poisson_eq)
    println("GENERATED SOLVER CODE:")
    println(solver_code)
    
    return poisson_eq
end

"""
    demo_boussinesq_equations()

Boussinesq approximation with buoyancy.
"""
function demo_boussinesq_equations()
    println("\nEXAMPLE 2: Boussinesq Equations with Buoyancy")
    println("="^70)
    
    momentum_equations = [
        "dt(u) + u*dx(u) + v*dy(u) + w*dz(u) = -dx(p) + ν*lap(u)",
        "dt(v) + u*dx(v) + v*dy(v) + w*dz(v) = -dy(p) + ν*lap(v)", 
        "dt(w) + u*dx(w) + v*dy(w) + w*dz(w) = -dz(p) + ν*lap(w) + g*α*T"
    ]
    
    println("INPUT EQUATIONS:")
    for (i, eq) in enumerate(momentum_equations)
        println("  $(i). $(eq)")
    end
    println()
    
    poisson_eq = derive_pressure_poisson_equation(momentum_equations;
                                                 density_variation=:boussinesq,
                                                 time_discretization=:semi_implicit,
                                                 body_forces=["g*α*T"])
    
    return poisson_eq
end

"""
    demo_variable_density_flow()

Variable density flow with complex coefficients.
"""
function demo_variable_density_flow()
    println("\nEXAMPLE 3: Variable Density Flow")
    println("="^70)
    
    momentum_equations = [
        "dt(ρ*u) + dx(ρ*u*u) + dy(ρ*u*v) + dz(ρ*u*w) = -dx(p) + dx(μ*dx(u))",
        "dt(ρ*v) + dx(ρ*v*u) + dy(ρ*v*v) + dz(ρ*v*w) = -dy(p) + dy(μ*dy(v))", 
        "dt(ρ*w) + dx(ρ*w*u) + dy(ρ*w*v) + dz(ρ*w*w) = -dz(p) + dz(μ*dz(w)) - ρ*g"
    ]
    
    println("INPUT EQUATIONS:")
    for (i, eq) in enumerate(momentum_equations)
        println("  $(i). $(eq)")
    end
    println()
    
    poisson_eq = derive_pressure_poisson_equation(momentum_equations;
                                                 density_variation=:variable,
                                                 viscosity_model=:variable,
                                                 time_discretization=:implicit)
    
    return poisson_eq
end

"""
    demo_cylindrical_coordinates()

Navier-Stokes in cylindrical coordinates.
"""
function demo_cylindrical_coordinates()
    println("\nEXAMPLE 4: Cylindrical Coordinate System")
    println("="^70)
    
    momentum_equations = [
        "dt(u_r) + u_r*dr(u_r) + (u_θ/r)*dθ(u_r) + u_z*dz(u_r) - u_θ²/r = -dr(p) + ν*lap_cyl(u_r)",
        "dt(u_θ) + u_r*dr(u_θ) + (u_θ/r)*dθ(u_θ) + u_z*dz(u_θ) + u_r*u_θ/r = -(1/r)*dθ(p) + ν*lap_cyl(u_θ)", 
        "dt(u_z) + u_r*dr(u_z) + (u_θ/r)*dθ(u_z) + u_z*dz(u_z) = -dz(p) + ν*lap_cyl(u_z)"
    ]
    
    println("INPUT EQUATIONS:")
    for (i, eq) in enumerate(momentum_equations)
        println("  $(i). $(eq)")
    end
    println()
    
    poisson_eq = derive_pressure_poisson_equation(momentum_equations;
                                                 coordinate_system=:cylindrical,
                                                 incompressibility="(1/r)*dr(r*u_r) + (1/r)*dθ(u_θ) + dz(u_z) = 0")
    
    return poisson_eq
end

"""
    demo_custom_boundary_conditions()

Example with specific boundary condition analysis.
"""
function demo_custom_boundary_conditions()
    println("\nEXAMPLE 5: Custom Boundary Conditions")
    println("="^70)
    
    momentum_equations = [
        "dt(u) + u*dx(u) + v*dy(u) + w*dz(u) = -dx(p) + ν*lap(u)",
        "dt(v) + u*dx(v) + v*dy(v) + w*dz(v) = -dy(p) + ν*lap(v)", 
        "dt(w) + u*dx(w) + v*dy(w) + w*dz(w) = -dz(p) + ν*lap(w)"
    ]
    
    boundary_info = Dict{Symbol, Any}(
        :left => "inflow",
        :right => "outflow", 
        :bottom => "no_slip",
        :top => "free_slip",
        :front => "periodic",
        :back => "periodic"
    )
    
    println("INPUT EQUATIONS:")
    for (i, eq) in enumerate(momentum_equations)
        println("  $(i). $(eq)")
    end
    println("\nBOUNDARY CONDITIONS:")
    for (boundary, condition) in boundary_info
        println("  • $(boundary): $(condition)")
    end
    println()
    
    poisson_eq = derive_pressure_poisson_equation(momentum_equations;
                                                 boundary_info=boundary_info,
                                                 time_discretization=:semi_implicit)
    
    return poisson_eq
end

"""
    run_all_demos()

Run all demonstration examples.
"""
function run_all_demos()
    println("PRESSURE POISSON EQUATION DERIVATION DEMONSTRATIONS")
    println("="^80)
    println("This demo shows how PencilFlows.jl automatically derives the correct")
    println("pressure Poisson equation from user-provided momentum equations.")
    println()
    
    # Run all examples
    eq1 = demo_standard_navier_stokes()
    eq2 = demo_boussinesq_equations() 
    eq3 = demo_variable_density_flow()
    eq4 = demo_cylindrical_coordinates()
    eq5 = demo_custom_boundary_conditions()
    
    println("\n" * "="^80)
    println("ALL DEMONSTRATIONS COMPLETED")
    println("="^80)
    println("Each example shows:")
    println("  • Automatic detection of pressure terms")
    println("  • Derivation of appropriate Poisson equation")
    println("  • Boundary condition analysis")
    println("  • Solver method recommendation")
    println("  • Generated solver code")
    
    return [eq1, eq2, eq3, eq4, eq5]
end

# Run demonstrations if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_demos()
end
