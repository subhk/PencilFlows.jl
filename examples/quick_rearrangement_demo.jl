# Quick Demonstration: Linear Left, Nonlinear Right
# =================================================

include("../src/symbolic/equation_rearrangement.jl")

"""
    quick_demo()

Quick demonstration of automatic equation rearrangement.
"""
function quick_demo()
    println("QUICK DEMO: Linear -> Left, Nonlinear -> Right")
    println("="^55)
    
    # Example equations in standard form
    equations = [
        "dt(u) = D*lap(u) + a*u - b*u*v + source",
        "dt(v) = kappa*lap(v) - u*dx(v) + c*v - d*u*v",
        "dt(T) = alpha*lap(T) + beta*T - u*dx(T) - gamma*T*T"
    ]
    
    println("INPUT (Standard PDE form):")
    for (i, eq) in enumerate(equations)
        println("  $i. $eq")
    end
    
    println("\nAUTOMATIC REARRANGEMENT...")
    rearrangement = rearrange_equations_linear_left(equations)
    
    println("\nOUTPUT (Linear Left, Nonlinear Right):")
    for (i, eq) in enumerate(rearrangement.rearranged_equations)
        println("  $i. $eq")
    end
    
    println("\nPERFECT! Linear terms on LEFT, nonlinear on RIGHT!")
    return rearrangement
end

# Run the demo
if abspath(PROGRAM_FILE) == @__FILE__
    quick_demo()
end

export quick_demo
