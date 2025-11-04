# Integrated Parameter System for All PencilFlow.jl Solvers
# ==========================================================
# This module ensures automatic parameter conversion is seamlessly integrated
# into pressure solver, nonlinear terms, and predictor-corrector time stepping

using LinearAlgebra

"""
    SolverParameters

Unified parameter structure that holds all physical parameters in a consistent
format for use across pressure solver, nonlinear terms, and time stepping.
Automatically converts between different parameter representations.
"""
struct SolverParameters
    # Primary physical parameters (always available)
    nu::Float64          # kinematic viscosity
    kappa::Float64       # thermal/scalar diffusivity
    f::Float64           # Coriolis parameter
    N2::Float64          # buoyancy frequency squared

    # Computed dimensionless numbers (for reference and validation)
    Re::Float64          # Reynolds number
    Pr::Float64          # Prandtl number
    Ek::Float64          # Ekman number
    Ro::Float64          # Rossby number
    Ra::Float64          # Rayleigh number
    Ri::Float64          # Richardson number

    # Reference scales used for conversions
    L_ref::Float64       # reference length scale
    U_ref::Float64       # reference velocity scale
    T_ref::Float64       # reference time scale
    delta_T_ref::Float64 # reference temperature difference
    g::Float64           # gravitational acceleration
    alpha::Float64       # thermal expansion coefficient

    # Solver control flags
    has_rotation::Bool
    has_stratification::Bool
    has_thermal::Bool
    has_magnetic::Bool

    # Internal conversion metadata
    conversion_source::Dict{Symbol, Symbol}  # tracks how parameters were derived
end

"""
    create_solver_parameters(analysis::EquationAnalysis, param_dict::Dict{Symbol,Float64})

Create unified SolverParameters from equation analysis and user parameter specifications.
This is the bridge between automatic equation analysis and numerical solvers.
"""
function create_solver_parameters(analysis::EquationAnalysis, param_dict::Dict{Symbol,Float64})
    # Reference scales (with sensible defaults)
    L_ref = get(param_dict, :L_ref, 1.0)
    U_ref = get(param_dict, :U_ref, 1.0)
    T_ref = get(param_dict, :T_ref, L_ref / U_ref)
    delta_T_ref = get(param_dict, :DT_ref, get(param_dict, :Delta_T_ref, 1.0))
    g = get(param_dict, :g, 9.81)
    alpha = get(param_dict, :alpha, get(param_dict, :Î, 2e-4))

    # Track parameter conversion sources
    conversion_source = Dict{Symbol, Symbol}()

    # Primary parameter: viscosity (nu)
    nu, nu_source = determine_viscosity(param_dict, L_ref, U_ref)
    conversion_source[:nu] = nu_source

    # Primary parameter: thermal diffusivity (kappa)
    kappa, kappa_source = determine_thermal_diffusivity(param_dict, nu)
    conversion_source[:kappa] = kappa_source

    # Primary parameter: Coriolis parameter (f)
    f, f_source = determine_coriolis_parameter(param_dict)
    conversion_source[:f] = f_source

    # Primary parameter: stratification (N2)
    N2, N2_source = determine_stratification(param_dict)
    conversion_source[:N2] = N2_source

    # Compute all dimensionless numbers
    Re = U_ref * L_ref / nu
    Pr = nu / kappa
    Ek = (f > 0) ? nu / (f * L_ref^2) : Inf
    Ro = (f > 0) ? U_ref / (f * L_ref) : Inf
    Ri = (U_ref > 0) ? N2 * L_ref^2 / U_ref^2 : 0.0
    Ra = (kappa > 0 && nu > 0) ? g * alpha * delta_T_ref * L_ref^3 / (nu * kappa) : 0.0

    # Determine physics characteristics
    has_rotation = f > 0 || analysis.has_rotation
    has_stratification = N2 > 0 || analysis.has_stratification
    has_thermal = kappa != nu || analysis.has_diffusion || :T in analysis.variables || :b in analysis.variables
    has_magnetic = false  # Future extension

    return SolverParameters(
        nu, kappa, f, N2, Re, Pr, Ek, Ro, Ra, Ri,
        L_ref, U_ref, T_ref, delta_T_ref, g, alpha,
        has_rotation, has_stratification, has_thermal, has_magnetic,
        conversion_source
    )
end

"""
    determine_viscosity(param_dict, L_ref, U_ref)

Determine kinematic viscosity from various possible parameter specifications.
Returns (nu, source_symbol) indicating how nu was determined.
"""
function determine_viscosity(param_dict::Dict{Symbol,Float64}, L_ref::Float64, U_ref::Float64)
    # Method 1: Direct specification
    if haskey(param_dict, :nu)
        return param_dict[:nu], :nu
    elseif haskey(param_dict, :Îν) || haskey(param_dict, :Î½)
        return get(param_dict, :Îν, get(param_dict, :Î½, 0.0)), :nu_legacy
    elseif haskey(param_dict, :viscosity)
        return param_dict[:viscosity], :viscosity
    end

    # Method 2: Via Reynolds number
    if haskey(param_dict, :Re)
        nu = U_ref * L_ref / param_dict[:Re]
        return nu, :Re
    elseif haskey(param_dict, :Reynolds)
        nu = U_ref * L_ref / param_dict[:Reynolds]
        return nu, :Reynolds
    end

    # Method 3: Via Ekman number (requires rotation)
    if haskey(param_dict, :Ek) && (haskey(param_dict, :f) || haskey(param_dict, :Omega) || haskey(param_dict, :coriolis))
        f_val = get(param_dict, :f, get(param_dict, :Omega, get(param_dict, :coriolis, 0.0)))
        if haskey(param_dict, :Omega)
            f_val = 2 * param_dict[:Omega]  # f = 2*Omega
        end
        if f_val > 0
            nu = param_dict[:Ek] * f_val * L_ref^2
            return nu, :Ek
        end
    end

    # Method 4: Default based on typical Reynolds number
    nu_default = U_ref * L_ref / 1000.0  # Default Re = 1000
    return nu_default, :default
end

"""
    determine_thermal_diffusivity(param_dict, nu)

Determine thermal diffusivity from various parameter specifications.
"""
function determine_thermal_diffusivity(param_dict::Dict{Symbol,Float64}, nu::Float64)
    # Method 1: Direct specification
    if haskey(param_dict, :kappa)
        return param_dict[:kappa], :kappa
    elseif haskey(param_dict, :Îκ) || haskey(param_dict, :Îº)
        return get(param_dict, :Îκ, get(param_dict, :Îº, 0.0)), :kappa_legacy
    elseif haskey(param_dict, :diffusivity)
        return param_dict[:diffusivity], :diffusivity
    end

    # Method 2: Via Prandtl number
    if haskey(param_dict, :Pr)
        kappa = nu / param_dict[:Pr]
        return kappa, :Pr
    elseif haskey(param_dict, :Prandtl)
        kappa = nu / param_dict[:Prandtl]
        return kappa, :Prandtl
    end

    # Method 3: Default (Pr = 1)
    return nu, :default
end

"""
    determine_coriolis_parameter(param_dict)

Determine Coriolis parameter from various specifications.
"""
function determine_coriolis_parameter(param_dict::Dict{Symbol,Float64})
    if haskey(param_dict, :f)
        return param_dict[:f], :f
    elseif haskey(param_dict, :coriolis)
        return param_dict[:coriolis], :coriolis
    elseif haskey(param_dict, :Omega)
        f = 2 * param_dict[:Omega]  # f = 2*Omega
        return f, :Omega
    elseif haskey(param_dict, :Î©) || haskey(param_dict, :Ω)
        f = 2 * get(param_dict, :Î©, get(param_dict, :Ω, 0.0))
        return f, :Omega_legacy
    end

    # Default: no rotation
    return 0.0, :default
end

"""
    determine_stratification(param_dict)

Determine stratification parameter from various specifications.
"""
function determine_stratification(param_dict::Dict{Symbol,Float64})
    if haskey(param_dict, :N2)
        return param_dict[:N2], :N2
    elseif haskey(param_dict, :N_squared)
        return param_dict[:N_squared], :N_squared
    elseif haskey(param_dict, :buoyancy_frequency_squared)
        return param_dict[:buoyancy_frequency_squared], :buoyancy_frequency_squared
    end

    # Default: no stratification
    return 0.0, :default
end

"""
    update_poisson_plan_with_parameters!(plan, params::SolverParameters)

Update Poisson solver plan to use automatically converted parameters.
This ensures pressure solver uses consistent parameter values.
"""
function update_poisson_plan_with_parameters!(plan, params::SolverParameters)
    if plan === nothing
        return
    end

    # Update boundary condition specifications if they depend on parameters
    if plan.bc_spec !== nothing
        # For Robin BCs: alpha*u + beta*du/dz = gamma, coefficients might depend on parameters
        update_bc_coefficients_with_parameters!(plan.bc_spec, params)
    end

    # Store parameter information in plan for solver reference
    if !hasfield(typeof(plan), :solver_params)
        # If plan doesn't have solver_params field, store in metadata
        plan.solver_metadata = Dict(:params => params)
    end
end

"""
    update_bc_coefficients_with_parameters!(bc_spec, params::SolverParameters)

Update boundary condition coefficients to use converted parameters.
"""
function update_bc_coefficients_with_parameters!(bc_spec::Dict{Symbol,Any}, params::SolverParameters)
    # Bottom boundary
    if haskey(bc_spec, :bottom_type) && bc_spec[:bottom_type] == :robin
        if haskey(bc_spec, :bottom_alpha_expr)
            bc_spec[:bottom_alpha] = evaluate_parameter_expression(bc_spec[:bottom_alpha_expr], params)
        end
        if haskey(bc_spec, :bottom_beta_expr)
            bc_spec[:bottom_beta] = evaluate_parameter_expression(bc_spec[:bottom_beta_expr], params)
        end
        if haskey(bc_spec, :bottom_gamma_expr)
            bc_spec[:bottom_gamma] = evaluate_parameter_expression(bc_spec[:bottom_gamma_expr], params)
        end
    end

    # Top boundary
    if haskey(bc_spec, :top_type) && bc_spec[:top_type] == :robin
        if haskey(bc_spec, :top_alpha_expr)
            bc_spec[:top_alpha] = evaluate_parameter_expression(bc_spec[:top_alpha_expr], params)
        end
        if haskey(bc_spec, :top_beta_expr)
            bc_spec[:top_beta] = evaluate_parameter_expression(bc_spec[:top_beta_expr], params)
        end
        if haskey(bc_spec, :top_gamma_expr)
            bc_spec[:top_gamma] = evaluate_parameter_expression(bc_spec[:top_gamma_expr], params)
        end
    end
end

"""
    evaluate_parameter_expression(expr, params::SolverParameters)

Evaluate expressions that depend on parameters (e.g., "1/Re", "Ek*f").
"""
function evaluate_parameter_expression(expr, params::SolverParameters)
    if expr isa Real
        return Float64(expr)
    elseif expr isa String
        # Simple expression evaluation (could be extended with proper parser)
        expr_clean = replace(expr, " " => "")

        # Common parameter substitutions
        expr_clean = replace(expr_clean, "Re" => string(params.Re))
        expr_clean = replace(expr_clean, "Pr" => string(params.Pr))
        expr_clean = replace(expr_clean, "Ek" => string(params.Ek))
        expr_clean = replace(expr_clean, "Ro" => string(params.Ro))
        expr_clean = replace(expr_clean, "Ra" => string(params.Ra))
        expr_clean = replace(expr_clean, "nu" => string(params.nu))
        expr_clean = replace(expr_clean, "kappa" => string(params.kappa))
        expr_clean = replace(expr_clean, "f" => string(params.f))
        expr_clean = replace(expr_clean, "N2" => string(params.N2))

        # Evaluate the expression
        try
            return eval(Meta.parse(expr_clean))
        catch
            @warn "Could not evaluate parameter expression: $expr"
            return 1.0
        end
    else
        return Float64(expr)
    end
end

# NOTE: enhanced_predictor_corrector_step! moved to simulation_api.jl for proper architectural separation

"""
    enhanced_momentum_rhs!(Ru, Rv, Rw, u, v, w, b, params::SolverParameters; kwargs...)

Enhanced momentum RHS that uses automatically converted parameters.
"""
function enhanced_momentum_rhs!(Ru, Rv, Rw, u, v, w, b, params::SolverParameters;
                               fields, decomp, grid, bc, nlin_ws, ws, kwargs...)

    # Nonlinear advection terms using parameter-aware methods
    enhanced_nonlinear_advection!(Ru, Rv, Rw, u, v, w, params;
                                 fields=fields, decomp=decomp, grid=grid, bc=bc, nlin_ws=nlin_ws, ws=ws)

    # Viscous terms with converted viscosity
    enhanced_viscous_terms!(Ru, Rv, Rw, u, v, w, params;
                           fields=fields, decomp=decomp, grid=grid, ws=ws)

    # Coriolis terms with converted rotation parameter
    if params.has_rotation
        enhanced_coriolis_terms!(Ru, Rv, u, v, params; ws=ws)
    end

    # Buoyancy coupling with converted parameters
    if params.has_stratification
        enhanced_buoyancy_coupling!(Rw, b, params; ws=ws)
    end
end

"""
    enhanced_buoyancy_rhs!(Rb, u, v, w, b, params::SolverParameters; kwargs...)

Enhanced buoyancy RHS with automatic parameter conversion.
"""
function enhanced_buoyancy_rhs!(Rb, u, v, w, b, params::SolverParameters;
                               fields, decomp, grid, bc, ws, bc_b=nothing, t=0.0, kwargs...)

    # Advection terms
    enhanced_scalar_advection!(Rb, u, v, w, b, params;
                              fields=fields, decomp=decomp, grid=grid, bc=bc, ws=ws)

    # Diffusion terms with converted thermal diffusivity
    enhanced_thermal_diffusion!(Rb, b, params;
                               fields=fields, decomp=decomp, grid=grid, ws=ws)

    # Stratification source term
    if params.has_stratification
        enhanced_stratification_source!(Rb, w, params; ws=ws)
    end

    # Apply boundary conditions if specified
    if bc_b !== nothing
        apply_enhanced_buoyancy_bcs!(Rb, b, bc_b, params, t)
    end
end

"""
    enhanced_nonlinear_advection!(Ru, Rv, Rw, u, v, w, params; kwargs...)

Nonlinear advection terms with parameter-aware scaling and discretization.
"""
function enhanced_nonlinear_advection!(Ru, Rv, Rw, u, v, w, params::SolverParameters;
                                      fields, decomp, grid, bc, nlin_ws, ws, kwargs...)

    # Use existing compute_nonlinear_terms! function which is well-tested
    # This computes the advective nonlinear terms -(u·∇)u
    compute_nonlinear_terms!(Ru, Rv, Rw, u, v, w, fields, decomp, grid, bc, nlin_ws)
end

"""
    enhanced_viscous_terms!(Ru, Rv, Rw, u, v, w, params; kwargs...)

Viscous terms using automatically converted viscosity parameter.
"""
function enhanced_viscous_terms!(Ru, Rv, Rw, u, v, w, params::SolverParameters;
                                fields, decomp, grid, ws, kwargs...)

    # Compute Laplacian terms with converted viscosity
    horizontal_laplacian_2d!(ws.lap, u, fields, decomp, grid, ws)
    Ru_data = parent(Ru)
    lap_data = parent(ws.lap)
    @. Ru_data += params.nu * lap_data

    horizontal_laplacian_2d!(ws.lap, v, fields, decomp, grid, ws)
    Rv_data = parent(Rv)
    @. Rv_data += params.nu * lap_data

    horizontal_laplacian_2d!(ws.lap, w, fields, decomp, grid, ws)
    Rw_data = parent(Rw)
    @. Rw_data += params.nu * lap_data

    # Vertical viscous terms would go here for non-uniform grids
end

"""
    enhanced_coriolis_terms!(Ru, Rv, u, v, params; kwargs...)

Coriolis terms using converted Coriolis parameter.
"""
function enhanced_coriolis_terms!(Ru, Rv, u, v, params::SolverParameters; ws, kwargs...)
    if !params.has_rotation
        return
    end

    # f x u terms: Ru += f*v, Rv += -f*u
    Ru_data = parent(Ru)
    Rv_data = parent(Rv)
    u_data = parent(u)
    v_data = parent(v)

    @. Ru_data += params.f * v_data
    @. Rv_data -= params.f * u_data
end

"""
    enhanced_thermal_diffusion!(Rb, b, params; kwargs...)

Thermal diffusion using converted thermal diffusivity.
"""
function enhanced_thermal_diffusion!(Rb, b, params::SolverParameters;
                                    fields, decomp, grid, ws, kwargs...)

    # Thermal Laplacian with converted diffusivity
    horizontal_laplacian_2d!(ws.lap, b, fields, decomp, grid, ws)
    Rb_data = parent(Rb)
    lap_data = parent(ws.lap)
    @. Rb_data += params.kappa * lap_data
end

"""
    enhanced_stratification_source!(Rb, w, params; kwargs...)

Stratification source term using converted N2 parameter.
"""
function enhanced_stratification_source!(Rb, w, params::SolverParameters; ws, kwargs...)
    if !params.has_stratification
        return
    end

    # N2 w term in buoyancy equation
    Rb_data = parent(Rb)
    w_data = parent(w)
    @. Rb_data += params.N2 * w_data
end

# Note: Conservative vs. standard advection forms are handled by momentum_rhs_conservative! vs momentum_rhs!

function enhanced_scalar_advection!(Rb, u, v, w, b, params; fields, decomp, grid, bc, ws)
    # Scalar advection: -u·∇b using existing compute_scalar_advection! function
    # Use the existing NonlinearWorkspace-based implementation
    nlin_ws = NonlinearWorkspace(decomp)  # Create temporary workspace if needed
    compute_scalar_advection!(Rb, u, v, w, b, fields, decomp, grid, bc, nlin_ws)
end

function apply_enhanced_buoyancy_bcs!(Rb, b, bc_b, params, t)
    # Apply boundary conditions with parameter-dependent values
    # Implementation would extend existing BC application
end

# Export the enhanced functions
export SolverParameters, create_solver_parameters
export update_poisson_plan_with_parameters!
# Note: enhanced_predictor_corrector_step! exported from simulation_api.jl

