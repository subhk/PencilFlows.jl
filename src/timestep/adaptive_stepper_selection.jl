# Adaptive Time Stepper Selection for General PDE Systems
# ========================================================
# This module automatically selects and configures appropriate time integration
# schemes based on the analyzed PDE system characteristics.

"""
    StepperRecommendation

Structure containing recommended time integration settings for a PDE system.
"""
struct StepperRecommendation
    primary_scheme::Symbol          # Main time integration scheme
    adaptive::Bool                  # Whether to use adaptive time stepping
    cfl_constraints::Vector{Symbol} # Active CFL constraints
    stability_requirements::Vector{Symbol} # Stability considerations
    implicit_variables::Vector{Symbol}     # Variables requiring implicit treatment
    explicit_variables::Vector{Symbol}     # Variables suitable for explicit treatment
    recommended_dt::Float64         # Suggested time step
    max_dt::Float64                # Maximum stable time step
    min_dt::Float64                # Minimum practical time step
end

"""
    select_optimal_stepper(system_analysis, user_preferences=Dict())

Automatically select the optimal time integration scheme based on PDE system analysis.

# Arguments
- `system_analysis`: Result from analyze_general_pde_system()  
- `user_preferences`: Optional Dict with user preferences (:scheme, :dt, :adaptive, etc.)

# Returns
- `StepperRecommendation`: Complete recommendation for time integration
"""
function select_optimal_stepper(system_analysis, user_preferences=Dict())
    println("  SELECTING OPTIMAL TIME INTEGRATION")
    println("="^45)
    
    # Initialize recommendation
    rec = StepperRecommendation(
        :auto, false, Symbol[], Symbol[], Symbol[], Symbol[],
        0.001, 0.1, 1e-6
    )
    
    # Analyze temporal structure
    temporal_analysis = analyze_temporal_characteristics(system_analysis)
    
    # Select primary scheme based on physics and structure
    primary_scheme = select_primary_scheme(system_analysis, temporal_analysis)
    
    # Determine IMEX splitting if needed
    imex_splitting = analyze_imex_splitting(system_analysis, temporal_analysis)
    
    # Assess CFL constraints
    cfl_constraints = assess_cfl_constraints(system_analysis)
    
    # Compute time step recommendations  
    dt_analysis = compute_timestep_recommendations(system_analysis, cfl_constraints)
    
    # Build final recommendation
    recommendation = StepperRecommendation(
        primary_scheme,
        get(user_preferences, :adaptive, should_use_adaptive(system_analysis)),
        cfl_constraints,
        temporal_analysis.stability_requirements,
        imex_splitting.implicit_vars,
        imex_splitting.explicit_vars,
        get(user_preferences, :dt, dt_analysis.recommended),
        dt_analysis.max_stable,
        dt_analysis.min_practical
    )
    
    print_stepper_recommendation(recommendation, system_analysis)
    return recommendation
end

"""
    analyze_temporal_characteristics(system_analysis)

Analyze the temporal characteristics of the PDE system.
"""
function analyze_temporal_characteristics(system_analysis)
    println("   Analyzing temporal characteristics...")
    
    # Check for different types of time derivatives
    has_first_order_time = false
    has_second_order_time = false
    has_mixed_order = false
    stiff_variables = Symbol[]
    fast_variables = Symbol[]
    stability_requirements = Symbol[]
    
    for (var, temporal_type) in system_analysis.temporal_structure
        if temporal_type == :first_order_time
            has_first_order_time = true
        elseif temporal_type == :second_order_time
            has_second_order_time = true
            has_mixed_order = has_first_order_time
        end
    end
    
    # Identify stiff and fast variables based on physics
    if system_analysis.physics_type in [:navier_stokes_2d, :navier_stokes_3d, :thermal_convection_2d, :thermal_convection_3d]
        # Pressure is typically stiff (incompressibility constraint)
        if :p in system_analysis.variables
            push!(stiff_variables, :p)
            push!(stability_requirements, :incompressibility)
        end
        
        # Velocity can have fast time scales due to viscosity at high Re
        if haskey(system_analysis.parameters, :Re)
            Re = system_analysis.parameters[:Re]
            if Re > 1000
                append!(fast_variables, [:u, :v, :w] ∩ system_analysis.variables)
                push!(stability_requirements, :high_reynolds)
            end
        end
    end
    
    # Check for rotation (creates fast oscillations)
    if haskey(system_analysis.parameters, :f)
        f = system_analysis.parameters[:f]
        if f > 1e-3
            append!(fast_variables, [:u, :v] ∩ system_analysis.variables)
            push!(stability_requirements, :fast_rotation)
        end
    end
    
    # Check for stiff chemistry/reactions
    if system_analysis.physics_type == :reaction_diffusion
        push!(stability_requirements, :stiff_reactions)
        # Assume all variables can be stiff in reaction-diffusion
        append!(stiff_variables, collect(system_analysis.variables))
    end
    
    println("     First-order time derivatives: $has_first_order_time")
    println("     Second-order time derivatives: $has_second_order_time") 
    println("     Stiff variables: $(join(stiff_variables, ", "))")
    println("     Fast variables: $(join(fast_variables, ", "))")
    
    return (
        has_first_order=has_first_order_time,
        has_second_order=has_second_order_time,
        has_mixed_order=has_mixed_order,
        stiff_variables=stiff_variables,
        fast_variables=fast_variables,
        stability_requirements=stability_requirements
    )
end

"""
    select_primary_scheme(system_analysis, temporal_analysis)

Select the primary time integration scheme.
"""
function select_primary_scheme(system_analysis, temporal_analysis)
    println("   Selecting primary integration scheme...")
    
    scheme = :runge_kutta_4  # Default
    
    # Choose based on physics type and characteristics
    if system_analysis.physics_type in [:navier_stokes_2d, :navier_stokes_3d, :thermal_convection_2d, :thermal_convection_3d]
        if :p in system_analysis.variables
            scheme = :predictor_corrector  # Handle incompressibility
            println("    → Predictor-corrector for incompressible flow")
        else
            scheme = :imex_runge_kutta    # IMEX for stiff terms
            println("    → IMEX Runge-Kutta for mixed stiffness")
        end
    elseif system_analysis.physics_type == :reaction_diffusion
        if length(temporal_analysis.stiff_variables) > 0
            scheme = :backward_euler      # Handle stiff reactions
            println("    → Backward Euler for stiff reactions")
        else
            scheme = :runge_kutta_4
            println("    → RK4 for non-stiff reaction-diffusion")
        end
    elseif system_analysis.physics_type == :wave_equation
        if temporal_analysis.has_second_order
            scheme = :leapfrog           # Good for second-order wave equations
            println("    → Leapfrog for wave equation")
        else
            scheme = :runge_kutta_4
            println("    → RK4 for first-order wave system")
        end
    elseif system_analysis.physics_type == :heat_equation
        scheme = :backward_euler         # Stable for pure diffusion
        println("    → Backward Euler for heat equation")
    else
        # General PDE system - choose based on characteristics
        if length(temporal_analysis.stiff_variables) > 0
            scheme = :imex_runge_kutta
            println("    → IMEX RK for general system with stiffness")
        else
            scheme = :runge_kutta_4
            println("    → RK4 for general explicit system")
        end
    end
    
    return scheme
end

"""
    analyze_imex_splitting(system_analysis, temporal_analysis)

Determine optimal implicit-explicit splitting for IMEX methods.
"""
function analyze_imex_splitting(system_analysis, temporal_analysis)
    println("    Analyzing IMEX splitting...")
    
    implicit_vars = Symbol[]
    explicit_vars = Symbol[]
    
    # Pressure is always implicit (Lagrange multiplier)
    if :p in system_analysis.variables
        push!(implicit_vars, :p)
    end
    
    # High-order spatial derivatives often need implicit treatment
    for (var, eq_data) in system_analysis.equation_structure
        if haskey(eq_data, :spatial_terms)
            spatial_terms = eq_data[:spatial_terms]
            
            # Check for high-order derivatives or stiff operators
            has_laplacian = any(term -> occursin("lap", term), spatial_terms)
            has_high_order = any(term -> occursin("d2", term), spatial_terms)
            
            if has_laplacian || has_high_order
                # Consider diffusive time scale
                if system_analysis.physics_type in [:heat_equation, :reaction_diffusion]
                    push!(implicit_vars, var)
                elseif haskey(system_analysis.parameters, :Re)
                    Re = system_analysis.parameters[:Re]
                    if Re > 1000  # High Re -> stiff viscous terms
                        push!(implicit_vars, var)
                    else
                        push!(explicit_vars, var)
                    end
                end
            else
                push!(explicit_vars, var)
            end
        end
    end
    
    # Remove duplicates and handle pressure separately
    implicit_vars = unique(implicit_vars)
    explicit_vars = unique(setdiff(explicit_vars, implicit_vars))
    
    println("     Implicit treatment: $(join(implicit_vars, ", "))")
    println("     Explicit treatment: $(join(explicit_vars, ", "))")
    
    return (implicit_vars=implicit_vars, explicit_vars=explicit_vars)
end

"""
    assess_cfl_constraints(system_analysis)

Assess which CFL constraints are active for the system.
"""
function assess_cfl_constraints(system_analysis)
    println("   Assessing CFL constraints...")
    
    constraints = Symbol[]
    
    # Advective CFL (from nonlinear terms)
    has_advection = false
    for (var, eq_data) in system_analysis.equation_structure
        if haskey(eq_data, :nonlinear_terms) && !isempty(eq_data[:nonlinear_terms])
            has_advection = true
            break
        end
    end
    
    if has_advection
        push!(constraints, :advective_cfl)
    end
    
    # Viscous CFL (from diffusive terms)
    if haskey(system_analysis.parameters, :Re) || haskey(system_analysis.parameters, :nu) || haskey(system_analysis.parameters, :ν)
        push!(constraints, :viscous_cfl)
    end
    
    # Thermal diffusion CFL
    if haskey(system_analysis.parameters, :Pr) || haskey(system_analysis.parameters, :kappa) || haskey(system_analysis.parameters, :κ)
        push!(constraints, :thermal_cfl)
    end
    
    # Rotation CFL (inertial oscillations)
    if haskey(system_analysis.parameters, :f)
        push!(constraints, :rotation_cfl)
    end
    
    # Wave CFL (for wave equations)
    if system_analysis.physics_type == :wave_equation
        push!(constraints, :wave_cfl)
    end
    
    println("     Active constraints: $(join(constraints, ", "))")
    return constraints
end

"""
    compute_timestep_recommendations(system_analysis, cfl_constraints)

Compute recommended time step based on system characteristics.
"""
function compute_timestep_recommendations(system_analysis, cfl_constraints)
    println("    Computing time step recommendations...")
    
    # Start with default values
    dt_recommended = 0.001
    dt_max = 0.1
    dt_min = 1e-8
    
    # Advective constraint
    if :advective_cfl in cfl_constraints
        dt_max = min(dt_max, 0.1)  # Conservative for nonlinear terms
        dt_recommended = min(dt_recommended, 0.01)
    end
    
    # Viscous constraint
    if :viscous_cfl in cfl_constraints
        if haskey(system_analysis.parameters, :Re)
            Re = system_analysis.parameters[:Re]
            dt_viscous = 0.1 / Re  # Rough estimate
            dt_max = min(dt_max, dt_viscous)
            dt_recommended = min(dt_recommended, dt_viscous * 0.1)
        elseif haskey(system_analysis.parameters, :nu)
            nu = system_analysis.parameters[:nu]
            dt_viscous = 0.01 / nu  # Assumes grid scale ~ 0.1
            dt_max = min(dt_max, dt_viscous)
            dt_recommended = min(dt_recommended, dt_viscous * 0.1)
        end
    end
    
    # Rotation constraint
    if :rotation_cfl in cfl_constraints && haskey(system_analysis.parameters, :f)
        f = system_analysis.parameters[:f]
        dt_rotation = 0.1 / f  # Inertial period constraint
        dt_max = min(dt_max, dt_rotation)
        dt_recommended = min(dt_recommended, dt_rotation * 0.1)
    end
    
    # Wave constraint  
    if :wave_cfl in cfl_constraints && haskey(system_analysis.parameters, :c)
        c = system_analysis.parameters[:c]
        dt_wave = 0.1 / c  # Assumes grid scale ~ 0.1
        dt_max = min(dt_max, dt_wave)
        dt_recommended = min(dt_recommended, dt_wave * 0.5)
    end
    
    # Ensure reasonable bounds
    dt_recommended = max(dt_min, min(dt_max * 0.1, dt_recommended))
    
    println("     Recommended dt: $dt_recommended")
    println("     Maximum stable dt: $dt_max")
    println("     Minimum practical dt: $dt_min")
    
    return (recommended=dt_recommended, max_stable=dt_max, min_practical=dt_min)
end

"""
    should_use_adaptive(system_analysis)

Determine if adaptive time stepping would be beneficial.
"""
function should_use_adaptive(system_analysis)
    # Adaptive stepping is beneficial for:
    # 1. Stiff systems (reaction-diffusion)
    # 2. Systems with multiple time scales
    # 3. Nonlinear systems with varying dynamics
    
    if system_analysis.physics_type == :reaction_diffusion
        return true  # Often stiff
    end
    
    if haskey(system_analysis.parameters, :Re)
        Re = system_analysis.parameters[:Re]
        if Re > 5000
            return true  # High Re can have varying time scales
        end
    end
    
    # Check for multiple time scales (rotation + other dynamics)
    has_rotation = haskey(system_analysis.parameters, :f)
    has_thermal = :T in system_analysis.variables || :b in system_analysis.variables
    
    if has_rotation && has_thermal
        return true  # Multiple time scales
    end
    
    return false  # Default to fixed time stepping
end

"""
    print_stepper_recommendation(recommendation, system_analysis)

Print a detailed summary of the time stepping recommendation.
"""
function print_stepper_recommendation(recommendation, system_analysis)
    println("\n  TIME INTEGRATION RECOMMENDATION")
    println("="^45)
    
    println(" Primary Scheme: $(recommendation.primary_scheme)")
    
    scheme_descriptions = Dict(
        :predictor_corrector => "Predictor-corrector with pressure projection for incompressible flow",
        :imex_runge_kutta => "Implicit-explicit Runge-Kutta for mixed stiffness",
        :runge_kutta_4 => "Fourth-order Runge-Kutta for general explicit systems",
        :backward_euler => "Backward Euler for stiff systems",
        :leapfrog => "Leapfrog for second-order wave equations"
    )
    
    if haskey(scheme_descriptions, recommendation.primary_scheme)
        println("   $(scheme_descriptions[recommendation.primary_scheme])")
    end
    
    println("\n Configuration:")
    println("   • Adaptive time stepping: $(recommendation.adaptive ? "Yes" : "No")")
    println("   • Recommended dt: $(recommendation.recommended_dt)")
    println("   • Maximum stable dt: $(recommendation.max_dt)")
    
    if !isempty(recommendation.cfl_constraints)
        println("\n Active CFL Constraints:")
        for constraint in recommendation.cfl_constraints
            println("   • $constraint")
        end
    end
    
    if !isempty(recommendation.implicit_variables)
        println("\n  IMEX Splitting:")
        println("   • Implicit: $(join(recommendation.implicit_variables, ", "))")
        println("   • Explicit: $(join(recommendation.explicit_variables, ", "))")
    end
    
    if !isempty(recommendation.stability_requirements)
        println("\n  Stability Considerations:")
        for req in recommendation.stability_requirements
            println("   • $req")
        end
    end
    
    println("\n Recommendation complete!")
    println("="^45)
end

"""
    create_stepper_from_recommendation(recommendation, system_analysis, prob)

Create and configure the actual time stepper based on the recommendation.
"""
function create_stepper_from_recommendation(recommendation, system_analysis, prob)
    println(" Creating time stepper from recommendation...")
    
    scheme = recommendation.primary_scheme
    
    if scheme == :predictor_corrector
        return create_predictor_corrector_stepper(prob, recommendation)
    elseif scheme == :imex_runge_kutta
        return create_imex_rk_stepper(prob, recommendation) 
    elseif scheme == :runge_kutta_4
        return create_rk4_stepper(prob, recommendation)
    elseif scheme == :backward_euler
        return create_backward_euler_stepper(prob, recommendation)
    elseif scheme == :leapfrog
        return create_leapfrog_stepper(prob, recommendation)
    else
        @warn "Unknown scheme $(scheme), falling back to RK4"
        return create_rk4_stepper(prob, recommendation)
    end
end

# Placeholder stepper creation functions (would integrate with existing PencilFlows steppers)
create_predictor_corrector_stepper(prob, rec) = println("   Created predictor-corrector stepper")
create_imex_rk_stepper(prob, rec) = println("   Created IMEX RK stepper") 
create_rk4_stepper(prob, rec) = println("   Created RK4 stepper")
create_backward_euler_stepper(prob, rec) = println("   Created backward Euler stepper")
create_leapfrog_stepper(prob, rec) = println("   Created leapfrog stepper")

# Export main functions
export StepperRecommendation, select_optimal_stepper, create_stepper_from_recommendation