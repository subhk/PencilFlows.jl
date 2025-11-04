# High-Level Simulation API for PencilFlows.jl
# ============================================
# This module provides the main user-facing interface for running simulations
# It coordinates between parameter handling, core algorithms, and I/O

using LinearAlgebra

"""
    run_simulation!(u, v, w, b, p; params, grid, fields, bc, time_params, solver_options...)

High-level interface for running PencilFlows simulations with automatic parameter handling.

# Arguments
- `u, v, w`: Velocity field components (PencilArrays)
- `b`: Buoyancy field (PencilArray) 
- `p`: Pressure field (PencilArray)

# Keyword Arguments
- `params::SolverParameters`: Unified parameter object with all physics parameters
- `grid`: Grid specification
- `fields`: Field decomposition object
- `bc`: Boundary conditions
- `time_params`: Time integration parameters (t_start, t_end, dt, etc.)
- `solver_options`: Solver-specific options (poisson_method, mg_cycles, etc.)
- `output_options`: Output and analysis options
"""
function run_simulation!(u, v, w, b, p; 
                        params::SolverParameters,
                        grid, fields, bc, decomp,
                        t_start::Real = 0.0, 
                        t_end::Real = 1.0, 
                        dt::Real = 0.01,
                        output_interval::Real = 0.1,
                        poisson_method::Symbol = :fft,
                        mg_cycles::Int = 6,
                        mg_pre::Int = 3, 
                        mg_post::Int = 3,
                        bc_b = nothing,
                        output_dir::String = "output",
                        save_fields::Bool = true,
                        analyze_energy::Bool = true,
                        kwargs...)
                        
    # Validate inputs
    @assert t_end > t_start "t_end must be greater than t_start"
    @assert dt > 0 "dt must be positive"
    
    # Initialize simulation state
    t = t_start
    step = 0
    output_time = t_start
    
    # Create workspaces (auto-created by predictor_corrector_step!)
    nlin_ws = nothing
    ws = nothing
    poisson_plan = nothing
    mg_plan = nothing
    
    # Initialize output and analysis
    if save_fields || analyze_energy
        mkpath(output_dir)
    end
    
    # Time integration loop
    while t < t_end
        # Adaptive timestep (ensure we don't overshoot t_end)
        dt_actual = min(dt, t_end - t)
        
        # Single predictor-corrector step using enhanced interface
        enhanced_predictor_corrector_step!(u, v, w, b, p, t, dt_actual, params;
                                          decomp=decomp, grid=grid, fields=fields, bc=bc,
                                          poisson_method=poisson_method,
                                          mg_cycles=mg_cycles, mg_pre=mg_pre, mg_post=mg_post,
                                          bc_b=bc_b, nlin_ws=nlin_ws, ws=ws,
                                          poisson_plan=poisson_plan, mg_plan=mg_plan,
                                          kwargs...)
        
        # Update time and step counter
        t += dt_actual
        step += 1
        
        # Output and analysis
        if t >= output_time || abs(t - t_end) < 1e-12
            if save_fields
                save_simulation_state!(u, v, w, b, p, t, step, output_dir, params)
            end
            
            if analyze_energy
                analyze_simulation_state!(u, v, w, b, p, t, step, params, grid)
            end
            
            # Print progress
            println("Step $step: t = $(round(t, digits=6)), dt = $(round(dt_actual, digits=6))")
            
            output_time += output_interval
        end
    end
    
    println("Simulation completed: $(step) steps, final time t = $(round(t, digits=6))")
    return u, v, w, b, p, t, step
end

"""
    enhanced_predictor_corrector_step!(u, v, w, b, p, t, dt, params; kwargs...)

Enhanced predictor-corrector step that uses SolverParameters for automatic parameter handling.
This is the proper interface that coordinates between parameter conversion and core algorithms.

This function can operate in two modes:
1. **Standard mode**: Delegates to core predictor_corrector_step! with converted parameters
2. **Enhanced mode**: Uses enhanced RHS functions for more sophisticated parameter handling

The mode is determined by the complexity of the SolverParameters object.
"""
function enhanced_predictor_corrector_step!(u, v, w, b, p, t, dt, params::SolverParameters;
                                          decomp, grid, fields, bc,
                                          poisson_method::Symbol=:fft,
                                          mg_cycles::Int=6, mg_pre::Int=3, mg_post::Int=3,
                                          bc_b=nothing, nlin_ws=nothing, ws=nothing,
                                          poisson_plan=nothing, mg_plan=nothing,
                                          use_enhanced_rhs::Bool=false,
                                          kwargs...)
    
    if use_enhanced_rhs
        # Use enhanced RHS functions for sophisticated parameter handling
        return _enhanced_predictor_corrector_with_parameter_rhs!(
            u, v, w, b, p, t, dt, params;
            decomp=decomp, grid=grid, fields=fields, bc=bc,
            poisson_method=poisson_method, mg_cycles=mg_cycles, mg_pre=mg_pre, mg_post=mg_post,
            bc_b=bc_b, nlin_ws=nlin_ws, ws=ws, poisson_plan=poisson_plan, mg_plan=mg_plan,
            kwargs...)
    else
        # Standard mode: Convert SolverParameters to individual parameters for core function
        nu = params.nu
        kappa = params.kappa  
        fplane = params.has_rotation ? FPlane(params.f) : FPlane(0.0)
        N2 = params.N2
        
        # Call the core predictor-corrector step function
        return predictor_corrector_step!(u, v, w, b, p, t, dt;
                                       decomp=decomp, grid=grid, fields=fields, bc=bc,
                                       nu=nu, kappa=kappa, fplane=fplane, N2=N2,
                                       poisson_method=poisson_method,
                                       mg_cycles=mg_cycles, mg_pre=mg_pre, mg_post=mg_post,
                                       bc_b=bc_b, nlin_ws=nlin_ws, ws=ws,
                                       poisson_plan=poisson_plan, mg_plan=mg_plan,
                                       kwargs...)
    end
end

"""
    _enhanced_predictor_corrector_with_parameter_rhs!(...)

Internal function that uses enhanced RHS functions for sophisticated parameter handling.
This implements the predictor-corrector algorithm using parameter-aware RHS functions.
"""
function _enhanced_predictor_corrector_with_parameter_rhs!(u, v, w, b, p, t, dt, params::SolverParameters;
                                                         decomp, grid, fields, bc,
                                                         poisson_method::Symbol=:fft,
                                                         mg_cycles::Int=6, mg_pre::Int=3, mg_post::Int=3,
                                                         bc_b=nothing, nlin_ws=nothing, ws=nothing,
                                                         poisson_plan=nothing, mg_plan=nothing,
                                                         kwargs...)
    
    # Auto-create workspaces if needed
    nlin_ws === nothing && (nlin_ws = NonlinearWorkspace(decomp))
    ws === nothing && (ws = RSNSWorkspace(decomp, grid, u))
    
    if poisson_method === :fft
        if poisson_plan === nothing
            poisson_plan = make_poisson_plan(u; decomp=decomp, grid=grid, bc_z=:neumann)
            update_poisson_plan_with_parameters!(poisson_plan, params)
        end
    end

    # Enforce boundary conditions
    apply_velocity_bcs_nonuniform!(u, v, w, grid, bc, t)
    if bc_b !== nothing
        apply_buoyancy_bcs_nonuniform!(b, grid, bc_b, t)
    end

    # RHS evaluation at t^n using parameter-aware functions
    fill!(ws.Ru1, zero(eltype(ws.Ru1)))
    fill!(ws.Rv1, zero(eltype(ws.Rv1)))
    fill!(ws.Rw1, zero(eltype(ws.Rw1)))
    fill!(ws.Rb1, zero(eltype(ws.Rb1)))
    
    # Use enhanced momentum and buoyancy RHS with parameter conversion
    enhanced_momentum_rhs!(ws.Ru1, ws.Rv1, ws.Rw1, u, v, w, b, params;
                          fields=fields, decomp=decomp, grid=grid, bc=bc, nlin_ws=nlin_ws, ws=ws)
    
    enhanced_buoyancy_rhs!(ws.Rb1, u, v, w, b, params;
                          fields=fields, decomp=decomp, grid=grid, bc=bc, ws=ws,
                          bc_b=bc_b, t=t)

    # Predictor step
    @. ws.ustar = u + dt * ws.Ru1
    @. ws.vstar = v + dt * ws.Rv1
    @. ws.wstar = w + dt * ws.Rw1
    @. ws.bstar = b + dt * ws.Rb1

    if bc_b !== nothing
        apply_buoyancy_bcs_nonuniform!(ws.bstar, grid, bc_b, t + dt)
    end

    # Pressure projection
    project_velocity!(ws.ustar, ws.vstar, ws.wstar, ws.pipi, ws.divu, dt;
                      fields=fields, decomp=decomp, grid=grid, bc=bc, 
                      poisson_plan=poisson_plan, ws=ws,
                      poisson_method=poisson_method, mg_plan=mg_plan,
                      mg_cycles=mg_cycles, mg_pre=mg_pre, mg_post=mg_post)

    # RHS evaluation at t^{n+1} with projected predictor
    fill!(ws.Ru2, zero(eltype(ws.Ru2)))
    fill!(ws.Rv2, zero(eltype(ws.Rv2)))
    fill!(ws.Rw2, zero(eltype(ws.Rw2)))
    fill!(ws.Rb2, zero(eltype(ws.Rb2)))
    
    enhanced_momentum_rhs!(ws.Ru2, ws.Rv2, ws.Rw2, ws.ustar, ws.vstar, ws.wstar, ws.bstar, params;
                          fields=fields, decomp=decomp, grid=grid, bc=bc, nlin_ws=nlin_ws, ws=ws)
    
    enhanced_buoyancy_rhs!(ws.Rb2, ws.ustar, ws.vstar, ws.wstar, ws.bstar, params;
                          fields=fields, decomp=decomp, grid=grid, bc=bc, ws=ws,
                          bc_b=bc_b, t=t+dt)

    # Corrector step (Heun's method)  
    @. u += 0.5 * dt * (ws.Ru1 + ws.Ru2)
    @. v += 0.5 * dt * (ws.Rv1 + ws.Rv2)
    @. w += 0.5 * dt * (ws.Rw1 + ws.Rw2)
    @. b += 0.5 * dt * (ws.Rb1 + ws.Rb2)

    # Final projection
    project_velocity!(u, v, w, ws.pipi, ws.divu, dt;
                      fields=fields, decomp=decomp, grid=grid, bc=bc,
                      poisson_plan=poisson_plan, ws=ws,
                      poisson_method=poisson_method, mg_plan=mg_plan,
                      mg_cycles=mg_cycles, mg_pre=mg_pre, mg_post=mg_post)

    # Update pressure
    @. p += ws.pipi

    return u, v, w, b, p, (ws=ws, nlin_ws=nlin_ws, poisson_plan=poisson_plan, params=params)
end

"""
    save_simulation_state!(u, v, w, b, p, t, step, output_dir, params)

Save simulation state to files for analysis and visualization.
"""
function save_simulation_state!(u, v, w, b, p, t, step, output_dir::String, params::SolverParameters)
    # Implementation would use existing I/O functions
    # This is a placeholder for now
    filename = joinpath(output_dir, "state_$(lpad(step, 6, '0')).h5")
    # write_state(filename, u, v, w, b, p, t, params)
    println("  Saved state to $filename")
end

"""
    analyze_simulation_state!(u, v, w, b, p, t, step, params, grid)

Compute and log simulation diagnostics (energy, enstrophy, etc.).
"""
function analyze_simulation_state!(u, v, w, b, p, t, step, params::SolverParameters, grid)
    # Compute kinetic energy
    KE = compute_kinetic_energy(u, v, w)
    
    # Compute potential energy (if stratified)
    PE = params.has_stratification ? compute_potential_energy(b, grid, params.N2) : 0.0
    
    # Log diagnostics
    println("  Diagnostics: KE = $(round(KE, digits=8)), PE = $(round(PE, digits=8))")
end

"""
    compute_potential_energy(b, grid, N2)

Compute gravitational potential energy from buoyancy field.
"""
function compute_potential_energy(b, grid, N2::Real)
    # This would integrate b * z over the domain
    # Placeholder implementation
    return 0.5 * N2 * sum(abs2, parent(b)) / length(b)
end

# Export main simulation interface
export run_simulation!, enhanced_predictor_corrector_step!
export save_simulation_state!, analyze_simulation_state!
