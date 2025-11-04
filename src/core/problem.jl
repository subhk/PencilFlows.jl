# ---------------------------------------------------------------------------
# File: problem.jl - Improved Problem Container with Advanced Timestep Integration
# ---------------------------------------------------------------------------

using LinearAlgebra, Statistics, MPI

# Import the timestep functionality (assuming it's available)
# include("timestep.jl")

# ---------------------------------------------------------------------------
# CFL utilities (calculator + simple adaptive rule)
# ---------------------------------------------------------------------------

"""
    CFLCalculator(velocity_extractor, grid; safety_factor=0.5, target_cfl=0.4)

Lightweight helper that estimates the CFL number from velocity fields. The
`velocity_extractor(scratches, state)` function must fill up to three scratch
arrays with u, v, (and optionally) w components compatible with `state`.
"""
mutable struct CFLCalculator
    velocity_extractor::Function
    grid::Any  # Keep as Any since Grid type is not well-defined in codebase
    safety_factor::Float64
    target_cfl::Float64
    scratch_arrays::Vector{AbstractArray}
end

function CFLCalculator(velocity_extractor, grid; safety_factor=0.5, target_cfl=0.4)
    CFLCalculator(velocity_extractor, grid, float(safety_factor), float(target_cfl), AbstractArray[])
end

"""Ensure scratch arrays exist and match the prototype state layout."""
function _ensure_scratch!(calc::CFLCalculator, state)
    nneeded = 3  # up to u,v,w
    while length(calc.scratch_arrays) < nneeded
        push!(calc.scratch_arrays, similar(state))
    end
    return calc.scratch_arrays
end

"""
    compute_cfl(calc::CFLCalculator, state, dt)

Compute an estimate of the CFL number given current `state` and `dt`.
"""
function compute_cfl(calc::CFLCalculator, state, dt)
    scratches = _ensure_scratch!(calc, state)
    # Fill scratches with velocity components via user-provided extractor
    calc.velocity_extractor(scratches, state)

    # Compute max speed component-wise (parallel-aware if available)
    umax = try
        parallel_maximum(scratches[1])
    catch
        maximum(abs, scratches[1])
    end
    vmax = try
        parallel_maximum(scratches[2])
    catch
        maximum(abs, scratches[2])
    end
    wmax = 0.0
    if length(size(scratches[3])) > 0
        wmax = try
            parallel_maximum(scratches[3])
        catch
            maximum(abs, scratches[3])
        end
    end
    vmax_tot = max(umax, vmax, wmax)

    # Grid spacings (fall back if names differ)
    g = calc.grid
    dx = hasproperty(g, :dx) ? g.dx : (hasproperty(g, :Lx) && hasproperty(g, :nx) ? g.Lx / g.nx : 1.0)
    dy = hasproperty(g, :dy) ? g.dy : (hasproperty(g, :Ly) && hasproperty(g, :ny) ? g.Ly / g.ny : dx)
    dz = hasproperty(g, :dz) ? (g.dz isa Number ? g.dz : (length(g.dz) > 0 ? minimum(g.dz) : dx)) :
         (hasproperty(g, :Lz) && hasproperty(g, :nz) ? g.Lz / g.nz : dy)

    local_min_h = minimum((dx, dy, dz))
    dt_cfl = local_min_h / (vmax_tot + eps())
    # Return CFL number (dt/dt_cfl)
    return float(dt) / dt_cfl
end

"""
    adaptive_timestep(calc::CFLCalculator, dt, cfl)

Simple proportional controller to adjust dt to meet the target CFL.
"""
function adaptive_timestep(calc::CFLCalculator, dt, cfl)
    cfl_target = calc.target_cfl
    cfl_safe   = cfl_target * calc.safety_factor
    cfl > 0 ? float(dt) * (cfl_safe / cfl) : float(dt)
end

# export Problem, init_problem, step!, advance!,
#        compute_cfl, set_dt!, get_timestep_info,
#        register_callback!, clear_callbacks!, run_callbacks!,
#        analyze_stability, get_memory_usage

# ---------------------------------------------------------------------------
# Advanced Problem Container
# ---------------------------------------------------------------------------

"""
    Problem{Phys,Grid,Stepper,CFL}

Advanced container for computational problems with integrated timestep management.

# Fields
- `physics`: Physics object containing state and RHS evaluation
- `grid`: Computational grid (1D, 2D, or 3D)
- `stepper`: Time stepping scheme from the timestep library
- `cfl_calc`: CFL calculator for adaptive time stepping
- `t`: Current simulation time
- `dt`: Current time step size
- `iteration`: Current iteration number
- `scheme`: Time stepping scheme identifier
- `parallel`: Whether running in parallel
- `adaptive`: Whether using adaptive time stepping
- `vars`: User-defined variables dictionary
- `callbacks`: Registered callback functions
- `stats`: Runtime statistics
"""
mutable struct Problem{Phys,Grid,Stepper,CFL}
    # Core components
    physics   :: Phys
    grid      :: Grid
    stepper   :: Stepper
    cfl_calc  :: CFL
    
    # Time integration state
    t         :: Float64
    dt        :: Float64
    dt_min    :: Float64
    dt_max    :: Float64
    iteration :: Int
    
    # Configuration
    scheme    :: Symbol
    parallel  :: Bool
    adaptive  :: Bool
    
    # User extensions
    vars      :: Dict{Symbol,Any}
    callbacks :: Vector{Function}
    
    # Statistics and monitoring
    stats     :: Dict{Symbol,Any}
    
    function Problem(physics::Phys, grid::Grid, stepper::Stepper, cfl_calc::CFL,
                    t::Float64, dt::Float64, dt_min::Float64, dt_max::Float64,
                    scheme::Symbol, parallel::Bool, adaptive::Bool) where {Phys,Grid,Stepper,CFL}
        
        new{Phys,Grid,Stepper,CFL}(
            physics, grid, stepper, cfl_calc,
            t, dt, dt_min, dt_max, 0,
            scheme, parallel, adaptive,
            Dict{Symbol,Any}(),
            Function[],
            Dict{Symbol,Any}(
                :total_steps => 0,
                :total_time => 0.0,
                :dt_changes => 0,
                :cfl_violations => 0,
                :max_cfl => 0.0,
                :min_dt => dt,
                :max_dt => dt
            )
        )
    end
end

# ---------------------------------------------------------------------------
# Problem Constructor with Advanced Options
# ---------------------------------------------------------------------------

"""
    init_problem(physics, grid; kwargs...)

Initialize an advanced Problem container with modern timestep integration.

# Arguments
- `physics`: Physics object with state and rhs! method
- `grid`: Computational grid object

# Keyword Arguments
- `dt::Real = 1e-3`: Initial time step size
- `dt_min::Real = 1e-10`: Minimum allowed time step
- `dt_max::Real = 1e-1`: Maximum allowed time step
- `scheme::Symbol = :LSRK4`: Time stepping scheme
- `adaptive::Bool = true`: Enable adaptive time stepping
- `cfl_target::Real = 0.4`: Target CFL number for adaptive stepping
- `safety_factor::Real = 0.5`: Safety factor for CFL calculations
- `parallel::Bool = false`: Whether running in parallel
- `low_storage::Bool = true`: Prefer low-storage schemes when available

# Supported Schemes
- Low-storage RK: `:LSRK4`, `:LSRK3SSP`, `:LSRK2Heun`
- Adams-Bashforth: `:LSAB1`, `:LSAB2`, `:LSAB3`, `:LSAB4`
- Classical: `:RK4`

# Returns
A `Problem` instance ready for time integration.

# Example
```julia
# Create physics and grid objects
physics = MyPhysics(...)
grid = create_grid(...)

# Initialize problem with adaptive LSRK4
prob = init_problem(physics, grid; 
                   scheme=:LSRK4, 
                   adaptive=true,
                   cfl_target=0.3)

# Advance simulation
advance!(prob; t_end=10.0)
```
"""
function init_problem(physics, grid;
                     dt::Real = 1e-3,
                     dt_min::Real = 1e-10,
                     dt_max::Real = 1e-1,
                     scheme::Symbol = :LSRK4,
                     adaptive::Bool = true,
                     cfl_target::Real = 0.4,
                     safety_factor::Real = 0.5,
                     parallel::Bool = false,
                     low_storage::Bool = true)

    # Validate inputs
    dt, dt_min, dt_max = float(dt), float(dt_min), float(dt_max)
    if dt_min >= dt_max
        throw(ArgumentError("dt_min must be < dt_max"))
    end
    if dt < dt_min || dt > dt_max
        throw(ArgumentError("dt must be in [dt_min, dt_max]"))
    end
    if cfl_target <= 0
        throw(ArgumentError("cfl_target must be positive"))
    end

    # Get state from physics
    if !hasproperty(physics, :state)
        throw(ArgumentError("Physics object must have a 'state' field"))
    end
    state = physics.state

    # Create memory-efficient stepper
    stepper = create_stepper(scheme, state; low_storage=low_storage)
    
    # Create CFL calculator if adaptive
    cfl_calc = if adaptive
        velocity_extractor = create_velocity_extractor(physics)
        CFLCalculator(velocity_extractor, grid; 
                     safety_factor=safety_factor, 
                     target_cfl=cfl_target)
    else
        nothing
    end

    # Create problem
    prob = Problem(physics, grid, stepper, cfl_calc,
                  0.0, dt, dt_min, dt_max,
                  scheme, parallel, adaptive)
    
    # Initialize statistics
    prob.stats[:scheme_order] = order(stepper)
    prob.stats[:is_low_storage] = is_low_storage(stepper)
    prob.stats[:memory_factor] = get_memory_factor(stepper)
    
    return prob
end

# ---------------------------------------------------------------------------
# Velocity Extractor Factory
# ---------------------------------------------------------------------------

"""
    create_velocity_extractor(physics) -> Function

Create a velocity extraction function based on physics object structure.
"""
function create_velocity_extractor(physics)
    # Check for common velocity field patterns
    if hasproperty(physics, :u) && hasproperty(physics, :v) && hasproperty(physics, :w)
        # 3D velocity fields
        return function extract_3d_velocities!(scratches, state)
            if length(scratches) >= 1; copy_array!(scratches[1], physics.u); end
            if length(scratches) >= 2; copy_array!(scratches[2], physics.v); end
            if length(scratches) >= 3; copy_array!(scratches[3], physics.w); end
        end
    elseif hasproperty(physics, :u) && hasproperty(physics, :v)
        # 2D velocity fields
        return function extract_2d_velocities!(scratches, state)
            if length(scratches) >= 1; copy_array!(scratches[1], physics.u); end
            if length(scratches) >= 2; copy_array!(scratches[2], physics.v); end
        end
    elseif hasproperty(physics, :velocity)
        # Single velocity array (assume first components are velocity)
        return function extract_packed_velocities!(scratches, state)
            vel = physics.velocity
            ndims = length(scratches)
            for i in 1:min(ndims, size(vel, 1))
                copy_array!(scratches[i], @view vel[i, ..])
            end
        end
    else
        # Fallback: assume state contains velocities in first components
        return function extract_state_velocities!(scratches, state)
            if isa(state, AbstractArray) && ndims(state) > 1
                for i in 1:min(length(scratches), size(state, 1))
                    copy_array!(scratches[i], @view state[i, ..])
                end
            else
                # Can't extract velocities, use zeros
                for scratch in scratches
                    zero_array!(scratch)
                end
            end
        end
    end
end

# ---------------------------------------------------------------------------
# Enhanced Time Stepping
# ---------------------------------------------------------------------------

"""
    step!(prob::Problem; verbose::Bool=false)

Perform one adaptive time step with comprehensive monitoring.

# Returns
- `success::Bool`: Whether the step was successful
- `dt_new::Float64`: New time step for next iteration
- `cfl::Float64`: Current CFL number (if adaptive)
"""
function step!(prob::Problem; verbose::Bool=false)
    start_time = time()
    
    # Create RHS closure
    function rhs!(dψ, ψ, t, context=nothing)
        if hasmethod(prob.physics.rhs!, (typeof(dψ), typeof(ψ), typeof(prob.grid), typeof(t)))
            prob.physics.rhs!(dψ, ψ, prob.grid, t)
        else
            prob.physics.rhs!(dψ, ψ, t)
        end
    end

    # Store initial state for potential rollback
    initial_state = copy(prob.physics.state)
    
    try
        # Perform time step
        if prob.adaptive && prob.cfl_calc !== nothing
            # Adaptive stepping with CFL monitoring
            current_cfl = compute_cfl(prob.cfl_calc, prob.physics.state, prob.dt)
            
            # Check for CFL violation
            if current_cfl > 2.0 * prob.cfl_calc.target_cfl
                prob.stats[:cfl_violations] += 1
                if verbose
                    @warn "Large CFL number detected: $current_cfl (target: $(prob.cfl_calc.target_cfl))"
                end
            end
            
            # Advance with stepper
            advance!(prob.stepper, prob.physics.state, prob.t, prob.dt, rhs!)
            
            # Compute new time step
            new_cfl = compute_cfl(prob.cfl_calc, prob.physics.state, prob.dt)
            dt_new = adaptive_timestep(prob.cfl_calc, prob.dt, new_cfl)
            
            # Clamp to allowed range
            dt_new = clamp(dt_new, prob.dt_min, prob.dt_max)
            
            # Update statistics
            prob.stats[:max_cfl] = max(prob.stats[:max_cfl], current_cfl)
            if dt_new != prob.dt
                prob.stats[:dt_changes] += 1
            end
            prob.stats[:min_dt] = min(prob.stats[:min_dt], dt_new)
            prob.stats[:max_dt] = max(prob.stats[:max_dt], dt_new)
            
            prob.dt = dt_new
            dt_next = dt_new
            cfl_result = new_cfl
        else
            # Fixed time step
            advance!(prob.stepper, prob.physics.state, prob.t, prob.dt, rhs!)
            dt_next = prob.dt
            cfl_result = 0.0
        end
        
        # Update time and iteration
        prob.t += prob.dt
        prob.iteration += 1
        prob.stats[:total_steps] += 1
        
        # Update timing statistics
        step_time = time() - start_time
        prob.stats[:total_time] += step_time
        
        if verbose
            @info "Step $(prob.iteration): t=$(prob.t), dt=$(prob.dt), CFL=$cfl_result"
        end
        
        return true, dt_next, cfl_result
        
    catch e
        # Rollback on failure
        copy_array!(prob.physics.state, initial_state)
        
        if verbose
            @error "Time step failed: $e"
        end
        
        # Try smaller time step if adaptive
        if prob.adaptive
            prob.dt *= 0.5
            prob.dt = max(prob.dt, prob.dt_min)
            if verbose
                @info "Reducing time step to $(prob.dt)"
            end
        end
        
        return false, prob.dt, 0.0
    end
end

# ---------------------------------------------------------------------------
# Advanced Integration Loop
# ---------------------------------------------------------------------------

"""
    advance!(prob::Problem; kwargs...)

Advanced time integration with comprehensive monitoring and output.

# Keyword Arguments
- `t_end::Real`: Final time
- `max_steps::Int = typemax(Int)`: Maximum number of steps
- `write_every::Int = 0`: Checkpoint frequency (0 = no checkpoints)
- `print_every::Int = 0`: Progress reporting frequency
- `basename::String = "output"`: Output file base name
- `output_format::Symbol = :jld2`: Output format (:jld2, :netcdf, :csv)
- `check_stability::Bool = true`: Monitor for instabilities
- `stability_threshold::Real = 1e10`: Threshold for detecting blowup
- `verbose::Bool = true`: Enable progress reporting

# Returns
- `success::Bool`: Whether integration completed successfully
- `final_stats::Dict`: Final integration statistics
"""
function advance!(prob::Problem;
                 t_end::Real,
                 max_steps::Int = typemax(Int),
                 write_every::Int = 0,
                 print_every::Int = 0,
                 basename::String = "output",
                 output_format::Symbol = :jld2,
                 check_stability::Bool = true,
                 stability_threshold::Real = 1e10,
                 verbose::Bool = true)

    # Validation
    if t_end <= prob.t
        throw(ArgumentError("t_end must be greater than current time $(prob.t)"))
    end

    # Initialize timing
    integration_start = time()
    
    if verbose
        println("Starting integration:")
        println("  Initial time: $(prob.t)")
        println("  Final time: $t_end")
        println("  Scheme: $(prob.scheme) (order $(order(prob.stepper)))")
        println("  Storage: $(is_low_storage(prob.stepper) ? "Low" : "Full")")
        println("  Adaptive: $(prob.adaptive)")
    end

    steps_taken = 0
    last_print_time = time()
    
    try
        while prob.t < t_end - 1e-12 && steps_taken < max_steps
            # Take one step
            success, dt_next, current_cfl = step!(prob; verbose=false)
            
            if !success
                @error "Time step failed at t=$(prob.t), iteration $(prob.iteration)"
                return false, prob.stats
            end
            
            steps_taken += 1
            
            # Stability check
            if check_stability
                max_val = maximum(abs, prob.physics.state)
                if max_val > stability_threshold
                    @error "Simulation appears unstable: max|state| = $max_val > $stability_threshold"
                    return false, prob.stats
                end
            end
            
            # Progress reporting
            if print_every > 0 && (prob.iteration % print_every == 0 || time() - last_print_time > 10.0)
                progress = (prob.t - 0.0) / (t_end - 0.0) * 100
                avg_dt = prob.t / prob.iteration
                eta = (t_end - prob.t) / avg_dt * (time() - integration_start) / prob.iteration
                
                if verbose
                    @info @sprintf("Progress: %.1f%% | t=%.3e | dt=%.3e | CFL=%.3f | ETA=%.1fs", 
                                  progress, prob.t, prob.dt, current_cfl, eta)
                end
                last_print_time = time()
            end
            
            # Checkpointing
            if write_every > 0 && (prob.iteration % write_every == 0)
                write_checkpoint(prob, basename, output_format)
            end
            
            # Run user callbacks
            run_callbacks!(prob)
            
            # Adjust time step to hit t_end exactly
            if prob.t + prob.dt > t_end
                prob.dt = t_end - prob.t
            end
        end
        
        # Final statistics
        integration_time = time() - integration_start
        prob.stats[:wall_time] = integration_time
        prob.stats[:steps_per_second] = steps_taken / integration_time
        prob.stats[:time_per_step] = integration_time / steps_taken
        
        if verbose
            println("\nIntegration completed successfully!")
            print_final_stats(prob)
        end
        
        return true, prob.stats
        
    catch e
        @error "Integration failed with error: $e"
        return false, prob.stats
    end
end

# ---------------------------------------------------------------------------
# CFL and Stability Analysis
# ---------------------------------------------------------------------------

"""
    compute_cfl(prob::Problem) -> Float64

Compute CFL number for current state with automatic dimension detection.
"""
function compute_cfl(prob::Problem)
    if prob.cfl_calc !== nothing
        return compute_cfl(prob.cfl_calc, prob.physics.state, prob.dt)
    else
        # Fallback implementation
        return compute_cfl_fallback(prob)
    end
end

"""
    compute_cfl_fallback(prob::Problem) -> Float64

Fallback CFL computation for legacy compatibility.
"""
function compute_cfl_fallback(prob::Problem)
    g = prob.grid
    
    # Check for 3D
    is3d = hasproperty(g, :nz) && hasproperty(g, :Lz)
    
    # Check for velocity fields
    hasUVW = hasproperty(prob.physics, :u) && hasproperty(prob.physics, :v) &&
             (!is3d || hasproperty(prob.physics, :w))
    
    if hasUVW
        u = prob.physics.u
        v = prob.physics.v
        
        umax = parallel_maximum(u)
        vmax = parallel_maximum(v)
        
        if is3d && hasproperty(prob.physics, :w)
            w = prob.physics.w
            wmax = parallel_maximum(w)
            dx = g.Lx / g.nx
            dy = g.Ly / g.ny
            dz = g.Lz / g.nz
            vmax_tot = max(umax, vmax, wmax)
            dt_cfl = 0.5 * min(dx, dy, dz) / (vmax_tot + eps())
        else
            dx = g.Lx / g.nx
            dy = g.Ly / g.ny
            vmax_tot = max(umax, vmax)
            dt_cfl = 0.5 * min(dx, dy) / (vmax_tot + eps())
        end
        
        return prob.dt / dt_cfl  # Return CFL number
    else
        return 0.0  # No velocity fields found
    end
end

"""
    analyze_stability(prob::Problem) -> NamedTuple

Analyze stability characteristics of the current scheme and problem.
"""
function analyze_stability(prob::Problem)
    # Basic stability info
    scheme_order = order(prob.stepper)
    is_ssp = prob.scheme in [:LSRK3SSP]  # Strong stability preserving
    
    # CFL-based stability
    current_cfl = compute_cfl(prob)
    theoretical_cfl_limit = get_theoretical_cfl_limit(prob.scheme)
    stability_margin = theoretical_cfl_limit / (current_cfl + eps())
    
    # State-based checks
    state_norm = sqrt(sum(abs2, prob.physics.state))
    state_max = maximum(abs, prob.physics.state)
    
    return (
        scheme = prob.scheme,
        order = scheme_order,
        is_ssp = is_ssp,
        current_cfl = current_cfl,
        cfl_limit = theoretical_cfl_limit,
        stability_margin = stability_margin,
        state_norm = state_norm,
        state_max = state_max,
        is_stable = stability_margin > 1.0 && state_max < 1e6
    )
end

"""Get theoretical CFL limit for time stepping schemes."""
function get_theoretical_cfl_limit(scheme::Symbol)
    limits = Dict(
        :LSRK4 => 2.0,
        :LSRK3SSP => 2.0,
        :LSRK2Heun => 1.0,
        :RK4 => 2.8,
        :LSAB1 => 1.0,
        :LSAB2 => 1.0,
        :LSAB3 => 0.6,
        :LSAB4 => 0.3
    )
    return get(limits, scheme, 1.0)
end

# ---------------------------------------------------------------------------
# Utilities and Information
# ---------------------------------------------------------------------------

"""
    set_dt!(prob::Problem, newdt::Real; update_limits::Bool=false)

Set time step with validation and optional limit updates.
"""
function set_dt!(prob::Problem, newdt::Real; update_limits::Bool=false)
    newdt = float(newdt)
    
    if update_limits
        prob.dt_min = min(prob.dt_min, newdt)
        prob.dt_max = max(prob.dt_max, newdt)
    end
    
    if newdt < prob.dt_min || newdt > prob.dt_max
        @warn "New dt=$newdt outside limits [$(prob.dt_min), $(prob.dt_max)]"
    end
    
    prob.dt = clamp(newdt, prob.dt_min, prob.dt_max)
    return prob.dt
end

"""
    get_timestep_info(prob::Problem) -> NamedTuple

Get comprehensive information about the time stepping configuration.
"""
function get_timestep_info(prob::Problem)
    return (
        scheme = prob.scheme,
        order = order(prob.stepper),
        is_low_storage = is_low_storage(prob.stepper),
        memory_factor = get_memory_factor(prob.stepper),
        current_dt = prob.dt,
        dt_range = (prob.dt_min, prob.dt_max),
        adaptive = prob.adaptive,
        current_time = prob.t,
        iteration = prob.iteration,
        cfl_target = prob.adaptive ? prob.cfl_calc.target_cfl : nothing
    )
end

"""
    get_memory_usage(prob::Problem) -> NamedTuple

Estimate memory usage of the time stepping components.
"""
function get_memory_usage(prob::Problem)
    state_size = sizeof(prob.physics.state)
    element_count = length(prob.physics.state)
    
    stepper_memory = memory_usage(prob.stepper, element_count, 8)
    
    total_memory = state_size + stepper_memory.total_bytes
    if prob.cfl_calc !== nothing
        cfl_memory = length(prob.cfl_calc.scratch_arrays) * state_size
        total_memory += cfl_memory
    else
        cfl_memory = 0
    end
    
    return (
        state_memory = state_size,
        stepper_memory = stepper_memory.total_bytes,
        cfl_memory = cfl_memory,
        total_memory = total_memory,
        memory_factor = stepper_memory.storage_factor
    )
end

"""Get memory factor for different steppers."""
function get_memory_factor(stepper::AbstractStepper)
    if isa(stepper, LSRKStepper)
        return 2.0
    elseif isa(stepper, LSABStepper)
        return Float64(stepper.order + 3)
    elseif isa(stepper, RK4Stepper)
        return 5.0
    else
        return 1.0
    end
end

# ---------------------------------------------------------------------------
# Callback System
# ---------------------------------------------------------------------------

"""
    register_callback!(prob::Problem, callback::Function)

Register a callback function to be called after each time step.
Callback signature: `callback(prob::Problem) -> nothing`
"""
function register_callback!(prob::Problem, callback::Function)
    push!(prob.callbacks, callback)
    return length(prob.callbacks)
end

"""
    clear_callbacks!(prob::Problem)

Remove all registered callbacks.
"""
function clear_callbacks!(prob::Problem)
    empty!(prob.callbacks)
    return nothing
end

"""
    run_callbacks!(prob::Problem)

Execute all registered callbacks.
"""
function run_callbacks!(prob::Problem)
    for callback in prob.callbacks
        try
            callback(prob)
        catch e
            @warn "Callback failed: $e"
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Output and Checkpointing
# ---------------------------------------------------------------------------

"""
    write_checkpoint(prob::Problem, basename::String, format::Symbol)

Write problem state to checkpoint file.
"""
function write_checkpoint(prob::Problem, basename::String, format::Symbol)
    filename = "$(basename)_$(lpad(prob.iteration, 6, '0'))"
    
    if format == :jld2
        # Would require JLD2.jl
        @info "JLD2 output not implemented, skipping checkpoint"
    elseif format == :netcdf
        # Would require NCDatasets.jl
        @info "NetCDF output not implemented, skipping checkpoint"
    elseif format == :csv
        # Simple CSV output for debugging
        open("$(filename).csv", "w") do io
            println(io, "iteration,time,dt,max_state")
            println(io, "$(prob.iteration),$(prob.t),$(prob.dt),$(maximum(abs, prob.physics.state))")
        end
    else
        @warn "Unknown output format: $format"
    end
end

"""
    print_final_stats(prob::Problem)

Print comprehensive final statistics.
"""
function print_final_stats(prob::Problem)
    stats = prob.stats
    
    println("Final Statistics:")
    println("  Total steps: $(stats[:total_steps])")
    println("  Wall time: $(stats[:wall_time]:.2f) s")
    println("  Steps/second: $(stats[:steps_per_second]:.1f)")
    println("  Time/step: $(stats[:time_per_step]*1000:.2f) ms")
    println("  dt range: [$(stats[:min_dt]:.2e), $(stats[:max_dt]:.2e)]")
    println("  dt changes: $(stats[:dt_changes])")
    println("  Max CFL: $(stats[:max_cfl]:.3f)")
    println("  CFL violations: $(stats[:cfl_violations])")
    
    memory_info = get_memory_usage(prob)
    println("  Memory factor: $(memory_info.memory_factor)x")
    println("  Total memory: $(memory_info.total_memory ÷ 1024^2:.1f) MB")
end
