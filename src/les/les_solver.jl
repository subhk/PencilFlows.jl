# LES Solver Integration
# Connects the LES module to the main PencilFlows solver system

"""
    apply_sgs_model!(u, v, w, transform_plans, grid, bc, workspace, dt)

Apply the sub-grid scale model to modify the velocity field for LES.

This function is the main entry point for LES integration with the Navier-Stokes solver.
It computes the SGS stress tensor using spectral derivatives and applies it to the velocity equations.

Follows PencilFlows convention: (output_arrays, input_arrays, transforms, grid, bc, workspace, params)

# Arguments
- `u, v, w`: Velocity components [Nx, Ny, Nz] in Z-pencil orientation
- `transform_plans`: PencilFlows transform plans for spectral derivatives
- `grid`: PencilFlows grid object with domain size (Lx, Ly) and z-grid
- `bc`: PencilFlows boundary conditions
- `workspace`: LESWorkspace for computations
- `dt`: Time step size

# Usage
```julia
# During time stepping loop
if is_les_active()
    apply_sgs_model!(u, v, w, transform_plans, grid, bc, les_workspace, dt)
end
```
"""
function apply_sgs_model!(u, v, w, transform_plans, grid, bc, workspace::LESWorkspace{T}, dt) where T
    if !is_les_active()
        return nothing
    end
    
    config = get_les_config()
    if config === nothing
        @warn "LES is active but no configuration found"
        return nothing
    end
    
    @debug "Applying SGS model" model=typeof(config.sgs_model) filter_width=config.filter_width dt=dt grid_size=size(u)
    
    # Compute strain rate and rotation rate tensors using spectral derivatives
    compute_strain_rate_tensor!(workspace.strain_rate, u, v, w, transform_plans, grid, bc, workspace)
    compute_rotation_rate_tensor!(workspace.rotation_rate, u, v, w, transform_plans, grid, bc, workspace)
    
    # Apply filtering if explicit filtering is enabled
    if config.filter_type != :implicit
        apply_les_filter!(workspace.temp_field, u, config.filter_type, config.filter_width, workspace)
        copyto!(u, workspace.temp_field)
        
        apply_les_filter!(workspace.temp_field, v, config.filter_type, config.filter_width, workspace)
        copyto!(v, workspace.temp_field)
        
        apply_les_filter!(workspace.temp_field, w, config.filter_type, config.filter_width, workspace)
        copyto!(w, workspace.temp_field)
    end
    
    # Compute SGS stress tensor
    compute_sgs_stress!(workspace.sgs_stress, config.sgs_model, 
                       workspace.strain_rate, workspace.rotation_rate,
                       config.filter_width, workspace)
    
    # Apply SGS stress to velocity equations (divergence of stress tensor)
    apply_sgs_stress_divergence!(u, v, w, workspace.sgs_stress, grid, dt)
    
    return nothing
end

"""
    apply_sgs_stress_divergence!(u, v, w, τ, grid, dt)

Apply the divergence of the SGS stress tensor to the velocity field.

The SGS stress appears in the momentum equations as:
∂uᵢ/∂t + ... = ... + ∂τᵢⱼ/∂xⱼ

# Arguments
- `u, v, w`: Velocity components to be modified [Nx, Ny, Nz]
- `τ`: SGS stress tensor [Nx, Ny, Nz, 6] (symmetric storage)
- `grid`: Grid information for finite differences
- `dt`: Time step for integration
"""
function apply_sgs_stress_divergence!(u, v, w, τ, grid, dt)
    Nx, Ny, Nz = size(u)
    
    # Get grid spacing
    dx = isa(grid.dx, AbstractArray) ? grid.dx : fill(grid.dx, Nx)
    dy = isa(grid.dy, AbstractArray) ? grid.dy : fill(grid.dy, Ny)
    dz = isa(grid.dz, AbstractArray) ? grid.dz : fill(grid.dz, Nz)
    
    # Apply ∂τ₁ⱼ/∂xⱼ to u-momentum equation
    @inbounds for k in 1:Nz
        for j in 1:Ny
            for i in 1:Nx
                # ∂τ₁₁/∂x + ∂τ₁₂/∂y + ∂τ₁₃/∂z
                dtau_dx = compute_stress_derivative_x(τ, i, j, k, 1, dx, Nx)  # τ₁₁
                dtau_dy = compute_stress_derivative_y(τ, i, j, k, 4, dy, Ny)  # τ₁₂
                dtau_dz = compute_stress_derivative_z(τ, i, j, k, 5, dz, Nz)  # τ₁₃
                
                u[i,j,k] += dt * (dtau_dx + dtau_dy + dtau_dz)
            end
        end
    end
    
    # Apply ∂τ₂ⱼ/∂xⱼ to v-momentum equation  
    @inbounds for k in 1:Nz
        for j in 1:Ny
            for i in 1:Nx
                # ∂τ₂₁/∂x + ∂τ₂₂/∂y + ∂τ₂₃/∂z
                dtau_dx = compute_stress_derivative_x(τ, i, j, k, 4, dx, Nx)  # τ₁₂ = τ₂₁
                dtau_dy = compute_stress_derivative_y(τ, i, j, k, 2, dy, Ny)  # τ₂₂
                dtau_dz = compute_stress_derivative_z(τ, i, j, k, 6, dz, Nz)  # τ₂₃
                
                v[i,j,k] += dt * (dtau_dx + dtau_dy + dtau_dz)
            end
        end
    end
    
    # Apply ∂τ₃ⱼ/∂xⱼ to w-momentum equation
    @inbounds for k in 1:Nz
        for j in 1:Ny
            for i in 1:Nx
                # ∂τ₃₁/∂x + ∂τ₃₂/∂y + ∂τ₃₃/∂z
                dtau_dx = compute_stress_derivative_x(τ, i, j, k, 5, dx, Nx)  # τ₁₃ = τ₃₁
                dtau_dy = compute_stress_derivative_y(τ, i, j, k, 6, dy, Ny)  # τ₂₃ = τ₃₂
                dtau_dz = compute_stress_derivative_z(τ, i, j, k, 3, dz, Nz)  # τ₃₃
                
                w[i,j,k] += dt * (dtau_dx + dtau_dy + dtau_dz)
            end
        end
    end
    
    return nothing
end

"""
    compute_stress_derivative_x(τ, i, j, k, component, dx, Nx)

Compute ∂τ/∂x using second-order central differences with periodic boundary conditions.
"""
@inline function compute_stress_derivative_x(τ, i, j, k, component, dx, Nx)
    im = i == 1 ? Nx : i-1
    ip = i == Nx ? 1 : i+1
    
    return (τ[ip, j, k, component] - τ[im, j, k, component]) / (2 * dx[i])
end

"""
    compute_stress_derivative_y(τ, i, j, k, component, dy, Ny)

Compute ∂τ/∂y using second-order central differences with periodic boundary conditions.
"""
@inline function compute_stress_derivative_y(τ, i, j, k, component, dy, Ny)
    jm = j == 1 ? Ny : j-1
    jp = j == Ny ? 1 : j+1
    
    return (τ[i, jp, k, component] - τ[i, jm, k, component]) / (2 * dy[j])
end

"""
    compute_stress_derivative_z(τ, i, j, k, component, dz, Nz)

Compute ∂τ/∂z using second-order central differences.
Assumes wall boundary conditions in z-direction.
"""
@inline function compute_stress_derivative_z(τ, i, j, k, component, dz, Nz)
    if k == 1
        # Forward difference at lower boundary
        return (τ[i, j, k+1, component] - τ[i, j, k, component]) / dz[k]
    elseif k == Nz
        # Backward difference at upper boundary
        return (τ[i, j, k, component] - τ[i, j, k-1, component]) / dz[k-1]
    else
        # Central difference in interior
        return (τ[i, j, k+1, component] - τ[i, j, k-1, component]) / (2 * dz[k])
    end
end

"""
    les_timestep_hook!(u, v, w, p, transform_plans, grid, params, workspace, t, dt)

Hook function to be called during time stepping when LES is active.

This function integrates LES into the main time stepping loop by:
1. Computing SGS stress based on resolved velocity field using spectral derivatives
2. Applying SGS stress to momentum equations
3. Optionally applying explicit filtering

# Arguments
- `u, v, w, p`: Velocity and pressure fields [Nx, Ny, Nz] in Z-pencil orientation
- `transform_plans`: PencilFlows transform plans for spectral derivatives
- `grid`: Grid information with domain size and z-grid
- `params`: Simulation parameters
- `workspace`: Combined workspace (should include LES workspace)
- `t`: Current simulation time
- `dt`: Time step size
"""
function les_timestep_hook!(u, v, w, p, transform_plans, grid, params, workspace, t, dt)
    if !is_les_active()
        return nothing
    end
    
    # Extract LES workspace from combined workspace
    les_workspace = get_les_workspace(workspace)
    if les_workspace === nothing
        @warn "LES is active but no LES workspace found"
        return nothing
    end
    
    # Apply SGS model using spectral derivatives
    apply_sgs_model!(u, v, w, transform_plans, grid, les_workspace, dt)
    
    return nothing
end

"""
    get_les_workspace(workspace) -> Union{Nothing, LESWorkspace}

Extract LES workspace from the main simulation workspace.

This function provides a standard interface for accessing the LES workspace
from different main workspace types.
"""
function get_les_workspace(workspace)
    # Check if workspace has a les_workspace field
    if hasfield(typeof(workspace), :les_workspace)
        return workspace.les_workspace
    end
    
    # Check if workspace is directly a LES workspace
    if workspace isa LESWorkspace
        return workspace
    end
    
    # Check if workspace is a Dict and has LES entry
    if workspace isa Dict && haskey(workspace, :les)
        return workspace[:les]
    end
    
    return nothing
end

"""
    setup_les_integration!(solver, les_config; workspace_type=Float64)

Set up LES integration with the main PencilFlows solver.

This function:
1. Activates LES with the provided configuration
2. Creates LES workspace if needed
3. Sets up integration hooks

# Arguments
- `solver`: Main PencilFlows solver object
- `les_config`: LES configuration
- `workspace_type`: Type for LES workspace arrays (default Float64)

# Returns
- LES workspace for manual management (if needed)

# Example
```julia
# Create LES configuration
sgs_model = SmagorinskyLilly(0.17)
les_config = LESConfiguration(sgs_model, 0.1; filter_type=:box)

# Setup LES integration
les_workspace = setup_les_integration!(solver, les_config)
```
"""
function setup_les_integration!(solver, les_config; workspace_type=Float64)
    # Activate LES
    activate_les!(les_config)
    
    # Get grid dimensions from solver
    Nx, Ny, Nz = size(solver.grid.x), size(solver.grid.y), size(solver.grid.z)
    
    # Create LES workspace
    les_workspace = create_les_workspace(workspace_type, Nx, Ny, Nz)
    
    # Add LES workspace to main workspace if possible
    if hasfield(typeof(solver.workspace), :les_workspace)
        solver.workspace.les_workspace = les_workspace
    elseif solver.workspace isa Dict
        solver.workspace[:les] = les_workspace
    else
        @info "Cannot automatically integrate LES workspace. Manual integration required."
    end
    
    return les_workspace
end

"""
    compute_les_diagnostics(u, v, w, workspace::LESWorkspace, config)

Compute LES diagnostic quantities for analysis and validation.

# Returns
- Dictionary with diagnostic quantities:
  - `:eddy_viscosity`: Spatially-averaged eddy viscosity
  - `:sgs_dissipation`: Sub-grid scale dissipation rate
  - `:resolved_tke`: Resolved turbulent kinetic energy
  - `:strain_magnitude`: Average strain rate magnitude
"""
function compute_les_diagnostics(u, v, w, workspace::LESWorkspace{T}, config) where T
    Nx, Ny, Nz = size(u)
    
    # Compute strain rate tensor (if not already computed)
    # Note: In practice this would be computed during the SGS model application
    
    diagnostics = Dict{Symbol, T}()
    
    # Average eddy viscosity
    diagnostics[:eddy_viscosity] = sum(workspace.eddy_viscosity) / (Nx * Ny * Nz)
    
    # Resolved turbulent kinetic energy
    tke = zero(T)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        tke += 0.5 * (u[i,j,k]^2 + v[i,j,k]^2 + w[i,j,k]^2)
    end
    diagnostics[:resolved_tke] = tke / (Nx * Ny * Nz)
    
    # Average strain rate magnitude
    strain_mag = zero(T)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        S = workspace.strain_rate
        s_mag_sq = 2 * (S[i,j,k,1]^2 + S[i,j,k,2]^2 + S[i,j,k,3]^2) +
                  4 * (S[i,j,k,4]^2 + S[i,j,k,5]^2 + S[i,j,k,6]^2)
        strain_mag += sqrt(s_mag_sq)
    end
    diagnostics[:strain_magnitude] = strain_mag / (Nx * Ny * Nz)
    
    # SGS dissipation rate (τij * Sij)
    sgs_dissipation = zero(T)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        τ = workspace.sgs_stress
        S = workspace.strain_rate
        # τij * Sij = τ11*S11 + τ22*S22 + τ33*S33 + 2*(τ12*S12 + τ13*S13 + τ23*S23)
        dissip = τ[i,j,k,1]*S[i,j,k,1] + τ[i,j,k,2]*S[i,j,k,2] + τ[i,j,k,3]*S[i,j,k,3] +
                2*(τ[i,j,k,4]*S[i,j,k,4] + τ[i,j,k,5]*S[i,j,k,5] + τ[i,j,k,6]*S[i,j,k,6])
        sgs_dissipation += dissip
    end
    diagnostics[:sgs_dissipation] = sgs_dissipation / (Nx * Ny * Nz)
    
    return diagnostics
end