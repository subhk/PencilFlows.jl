# Time-Dependent Poisson Boundary Conditions Demo
# 
# This example demonstrates time and spatial dependent boundary conditions for the Poisson solver

using PencilFlows

"""
Demo: Time-dependent Poisson boundary conditions

This example shows how to set up Poisson solver BCs that vary with time and/or spatial position.
"""
function poisson_bc_demo()
    println("=== Time/Spatial-Dependent Poisson BC Demo ===")
    
    # 1. Time-dependent Dirichlet BC
    println("\n1. Time-dependent Dirichlet BC:")
    println("   Bottom: π = sin(ωt), Top: π = 0")
    
    ω = 2π  # oscillation frequency
    bc_time_dirichlet = Dict(
        :bottom_type => :dirichlet,
        :top_type => :dirichlet,
        :bottom_value => (t -> sin(ω*t)),  # time-dependent
        :top_value => 0.0                   # constant
    )
    
    # 2. Spatial-dependent Neumann BC  
    println("\n2. Spatial-dependent Neumann BC:")
    println("   Bottom: ∂π/∂z = x*y*cos(t), Top: ∂π/∂z = 0")
    
    bc_spatial_neumann = Dict(
        :bottom_type => :neumann,
        :top_type => :neumann,
        :bottom_value => ((x,y,t) -> x*y*cos(t)),  # spatial and time variation
        :top_value => 0.0                           # constant flux
    )
    
    # 3. Robin BC with time-dependent coefficients
    println("\n3. Robin BC with time-dependent coefficients:")
    println("   Bottom: α(t)π + β∂π/∂z = γ(t)")
    println("   α(t) = 1 + 0.1*sin(t), γ(t) = cos(t)")
    
    bc_robin = Dict(
        :bottom_type => :robin,
        :top_type => :dirichlet,
        :bottom_alpha => (t -> 1 + 0.1*sin(t)),    # time-dependent α
        :bottom_beta => 1.0,                        # constant β  
        :bottom_gamma => (t -> cos(t)),             # time-dependent γ
        :top_value => 0.0                           # Dirichlet at top
    )
    
    # 4. Mixed spatial and temporal dependencies
    println("\n4. Mixed spatial/temporal dependencies:")
    println("   Bottom: π = A*sin(kx*x + ky*y + ωt)")
    println("   Top: ∂π/∂z = B*exp(-r²/σ²)*sin(ωt)")
    
    A, kx, ky, ω = 1.0, 2π, π, 2π
    B, σ = 0.5, 1.0
    
    bc_mixed = Dict(
        :bottom_type => :dirichlet,
        :top_type => :neumann,
        :bottom_value => ((x,y,t) -> A*sin(kx*x + ky*y + ω*t)),
        :top_value => ((x,y,t) -> begin
                          r² = x^2 + y^2  
                          B*exp(-r²/σ^2)*sin(ω*t)
                      end)
    )
    
    # 5. Grid-index based BC (useful for structured grids)
    println("\n5. Grid-index dependent BC:")
    println("   Bottom: π = f(i,j,t) where f depends on grid indices")
    
    bc_grid_index = Dict(
        :bottom_type => :dirichlet,
        :top_type => :dirichlet,
        :bottom_value => ((i,j,t) -> 0.1*sin(2π*t)*cos(π*i/10)*cos(π*j/10)),
        :top_value => 0.0
    )
    
    println("\n=== Poisson BC Examples Created Successfully ===")
    println("These BCs can now be used with the enhanced Poisson solver:")
    println("Example usage:")
    println("  plan = make_poisson_plan(rhs_z; decomp=decomp, grid=grid, bc_spec=bc_time_dirichlet)")
    println("  solve_poisson!(phi, rhs, plan; t=current_time, xnodes=x, ynodes=y)")
    
    return bc_time_dirichlet, bc_spatial_neumann, bc_robin, bc_mixed, bc_grid_index
end

# Run the demo if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    poisson_bc_demo()
end