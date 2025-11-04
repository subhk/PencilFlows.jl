# Time-Dependent Velocity Boundary Conditions Demo
# 
# This example demonstrates the new time and spatial dependent velocity BC capabilities

using PencilFlows

"""
Demo: Time-dependent velocity boundary conditions

This example shows how to set up velocity BCs that vary with time and/or spatial position.
"""
function velocity_bc_demo()
    println("=== Time/Spatial-Dependent Velocity BC Demo ===")
    
    # 1. Time-dependent oscillatory BC
    println("\n1. Time-dependent velocity BC:")
    println("   Bottom wall: u = sin(ωt), v = 0, w = 0")
    println("   Top wall: No-slip")
    
    ω = 2π  # oscillation frequency
    bc_time = BoundaryCondition(
        PRESCRIBED_VELOCITY, NO_SLIP,
        (t -> sin(ω*t), 0.0, 0.0),  # u oscillates at bottom
        (0.0, 0.0, 0.0)             # no-slip at top
    )
    
    # 2. Spatial-dependent velocity BC  
    println("\n2. Spatial-dependent velocity BC:")
    println("   Bottom wall: u = x*sin(t), v = y*cos(t), w = 0")
    println("   Top wall: Free-slip")
    
    bc_spatial = BoundaryCondition(
        PRESCRIBED_VELOCITY, FREE_SLIP,
        ((x,y,t) -> x*sin(t), (x,y,t) -> y*cos(t), 0.0),  # spatial variation
        (0.0, 0.0, 0.0)                                   # free-slip at top
    )
    
    # 3. Robin BC with time-dependent coefficients
    println("\n3. Robin velocity BC:")
    println("   Bottom wall: α(t)u + β∂u/∂z = γ(t)")
    println("   α(t) = 1 + 0.1*sin(t), γ(t) = cos(t)")
    
    bc_robin = BoundaryCondition(
        ROBIN, NO_SLIP,
        (0.0, 0.0, 0.0),           # velocity values (unused for Robin)
        (0.0, 0.0, 0.0),           # top is no-slip
        bottom_robin = (t -> 1 + 0.1*sin(t), 1.0, t -> cos(t)),  # α(t), β, γ(t)
        top_robin = (0.0, 1.0, 0.0)
    )
    
    # 4. Mixed function types
    println("\n4. Mixed BC types:")
    println("   u: time-dependent, v: spatial-dependent, w: constant")
    
    bc_mixed = BoundaryCondition(
        PRESCRIBED_VELOCITY, NO_SLIP,
        (t -> 0.1*sin(2π*t),              # u: time-dependent
         (i,j,t) -> 0.01*(i-10)*(j-10),   # v: grid index dependent  
         0.0),                            # w: constant
        (0.0, 0.0, 0.0)                   # top: no-slip
    )
    
    println("\n=== BC Examples Created Successfully ===")
    println("These BCs can now be used with apply_velocity_bcs_nonuniform!")
    println("Example usage:")
    println("  apply_velocity_bcs_nonuniform!(u, v, w, grid, bc_time, t; xnodes=x, ynodes=y)")
    
    return bc_time, bc_spatial, bc_robin, bc_mixed
end

# Run the demo if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    velocity_bc_demo()
end