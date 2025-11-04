# Demonstration: Linear-Left, Nonlinear-Right Pressure Derivation  
# =================================================================
# Shows automatic pressure Poisson derivation for the form:
# dt(u) + L(u) = N(u) + S    (Linear-Left, Nonlinear-Right)

include("../src/interfaces/linear_left_pressure_derivation.jl")
using Printf

"""
Demo 1: Standard IMEX Navier-Stokes (your preferred form)
"""
function demo_imex_navier_stokes()
    println("DEMO 1: IMEX NAVIER-STOKES (LINEAR-LEFT FORM)")
    println("="^80)
    
    # Your preferred form: Linear terms on left, nonlinear on right
    equations = [
        "dt(u) + dx(p) + ν*lap(u) = -u*dx(u) - v*dy(u) - w*dz(u) + f_x",
        "dt(v) + dy(p) + ν*lap(v) = -u*dx(v) - v*dy(v) - w*dz(v) + f_y", 
        "dt(w) + dz(p) + ν*lap(w) = -u*dx(w) - v*dy(w) - w*dz(w) + f_z + g"
    ]
    
    println("INPUT EQUATIONS (Linear-Left, Nonlinear-Right):")
    for (i, eq) in enumerate(equations)
        println("  $(i). $(eq)")
    end
    println()
    
    result = derive_pressure_poisson_linear_left(equations;
                                               time_discretization=:imex,
                                               linear_treatment=:implicit,
                                               nonlinear_treatment=:explicit)
    
    return result
end

"""
Demo 2: Variable coefficient case
"""
function demo_variable_coefficient_linear_left()
    println("\nDEMO 2: VARIABLE DENSITY (LINEAR-LEFT FORM)")
    println("="^80)
    
    equations = [
        "dt(u) + (1/ρ)*dx(p) + ν*lap(u) = -u*dx(u) - v*dy(u) - w*dz(u)",
        "dt(v) + (1/ρ)*dy(p) + ν*lap(v) = -u*dx(v) - v*dy(v) - w*dz(v)", 
        "dt(w) + (1/ρ)*dz(p) + ν*lap(w) = -u*dx(w) - v*dy(w) - w*dz(w) - g"
    ]
    
    println("INPUT EQUATIONS (Variable Density):")
    for (i, eq) in enumerate(equations)
        println("  $(i). $(eq)")
    end
    println()
    
    result = derive_pressure_poisson_linear_left(equations;
                                               density_variation=:variable,
                                               time_discretization=:imex)
    
    return result
end

"""
Demo 3: Advanced IMEX with complex linear terms
"""
function demo_advanced_imex()
    println("\nDEMO 3: ADVANCED IMEX (COMPLEX LINEAR TERMS)")
    println("="^80)
    
    equations = [
        "dt(u) + dx(p) + ν*lap(u) + σ*u = -u*dx(u) - v*dy(u) - w*dz(u) + cos(x)*sin(t)",
        "dt(v) + dy(p) + ν*lap(v) + σ*v = -u*dx(v) - v*dy(v) - w*dz(v) + sin(y)*cos(t)", 
        "dt(w) + dz(p) + ν*lap(w) + σ*w = -u*dx(w) - v*dy(w) - w*dz(w) - g*α*T"
    ]
    
    println("INPUT EQUATIONS (Linear damping + time-dependent forcing):")
    for (i, eq) in enumerate(equations)
        println("  $(i). $(eq)")
    end
    println()
    
    result = derive_pressure_poisson_linear_left(equations;
                                               time_discretization=:imex,
                                               boundary_info=Dict{Symbol, Any}(
                                                   :left => "inflow",
                                                   :right => "outflow",
                                                   :bottom => "no_slip",
                                                   :top => "no_slip"
                                               ))
    
    return result
end

"""
Demo 4: Comparison with standard form
"""
function demo_form_comparison()
    println("\nDEMO 4: COMPARISON OF EQUATION FORMS")
    println("="^80)
    
    # Same physics, different forms
    standard_form = [
        "dt(u) + u*dx(u) + v*dy(u) + w*dz(u) = -dx(p) + ν*lap(u)",
    ]
    
    linear_left_form = [
        "dt(u) + dx(p) + ν*lap(u) = -u*dx(u) - v*dy(u) - w*dz(u)",
    ]
    
    println("STANDARD FORM (Nonlinear-Left):")
    println("  $(standard_form[1])")
    println("  → Pressure projection method")
    println("  → Explicit advection, implicit pressure")
    println()
    
    println("LINEAR-LEFT FORM (your preferred):")
    println("  $(linear_left_form[1])")
    println("  → IMEX method")
    println("  → Implicit linear operators, explicit nonlinear")
    println()
    
    println("ADVANTAGES OF LINEAR-LEFT FORM:")
    println("  Natural IMEX splitting")
    println("  Pressure in implicit part (stable)")
    println("  Diffusion in implicit part (large timesteps)")
    println("  Optimal for spectral methods")
    println("  Efficient preconditioning")
    println()
    
    return nothing
end

"""
Generate IMEX solver code for linear-left form
"""
function generate_imex_solver_code()
    println("\nGENERATED IMEX SOLVER CODE:")
    println("-"^60)
    
    code = """
    # IMEX solver for linear-left form
    function imex_step_linear_left!(u, v, w, p, t, dt; grid, bc, params)
        # Stage 1: Explicit evaluation of nonlinear terms
        eval_nonlinear_rhs!(R_exp, u, v, w, params)  # Right-hand side
        
        # Stage 2: Implicit solve for linear terms  
        # (I + dt*∇p + dt*ν*∇²)u^{n+1} = u^n + dt*R_exp
        solve_implicit_linear!(u_new, u, R_exp, dt, params, bc)
        
        # Stage 3: Pressure projection (if needed)
        if needs_projection(bc)
            ∇²π = (1/dt)*∇·u_new
            solve_poisson!(π, ∇²π, grid, bc)
            u_new .-= dt .* ∇π
        end
        
        return u_new, v_new, w_new, p_new
    end
    """
    
    println(code)
    return code
end

"""
Run all demos
"""
function run_linear_left_demos()
    println("LINEAR-LEFT PRESSURE DERIVATION DEMONSTRATIONS")
    println("="^90)
    println("Form: ∂u/∂t + L(u) = N(u) + S")
    println("      ↳ Linear-Left  ↳ Nonlinear-Right")
    println()
    
    # Run demonstrations
    result1 = demo_imex_navier_stokes()
    result2 = demo_variable_coefficient_linear_left() 
    result3 = demo_advanced_imex()
    demo_form_comparison()
    generate_imex_solver_code()
    
    println("\n" * "="^90)
    println("ALL LINEAR-LEFT DEMONSTRATIONS COMPLETED")
    println("="^90)
    println("Key Benefits of Linear-Left Form:")
    println("  Optimal for IMEX (Implicit-Explicit) methods")
    println("  Large stable timesteps")
    println("  Efficient spectral/multigrid solvers")
    println("  Natural pressure treatment")
    
    return [result1, result2, result3]
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_linear_left_demos()
end
