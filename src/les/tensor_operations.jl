# Tensor Operations for Large Eddy Simulation
# Efficient computation of strain rate and rotation rate tensors using spectral derivatives

using ..PencilFlows: compute_horizontal_derivatives_fd!, dz_derivative_nonuniform_with_bcs!, BoundaryCondition, NO_SLIP, FREE_SLIP

"""
    compute_strain_rate_tensor!(S, u, v, w, transform_plans, grid, bc, workspace)

Compute the strain rate tensor Sᵢⱼ = ½(∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ) using spectral derivatives.

Uses Fourier spectral derivatives for horizontal (x,y) directions and PencilFlows 
finite differences for the vertical (z) direction.

Follows PencilFlows convention: (output, inputs, transforms, grid, bc, workspace)

The strain rate tensor is symmetric, so we store only 6 components:
- S[:,:,:,1] = S₁₁ = ∂u/∂x
- S[:,:,:,2] = S₂₂ = ∂v/∂y  
- S[:,:,:,3] = S₃₃ = ∂w/∂z
- S[:,:,:,4] = S₁₂ = ½(∂u/∂y + ∂v/∂x)
- S[:,:,:,5] = S₁₃ = ½(∂u/∂z + ∂w/∂x)
- S[:,:,:,6] = S₂₃ = ½(∂v/∂z + ∂w/∂y)

# Arguments
- `S`: Output strain rate tensor [Nx, Ny, Nz, 6] in Z-pencil orientation
- `u, v, w`: Velocity components [Nx, Ny, Nz] in Z-pencil orientation
- `transform_plans`: PencilFlows transform plans for spectral derivatives
- `grid`: PencilFlows grid object with domain size and z-grid
- `bc`: PencilFlows boundary conditions 
- `workspace`: LESWorkspace for gradient computations
"""
function compute_strain_rate_tensor!(S, u, v, w, transform_plans, grid, bc, workspace::LESWorkspace{T}) where T
    # Input validation following PencilFlows patterns
    Nx, Ny, Nz = size(u)
    if size(v) != (Nx, Ny, Nz) || size(w) != (Nx, Ny, Nz)
        throw(DimensionMismatch("Velocity components must have same dimensions"))
    end
    
    if size(S) != (Nx, Ny, Nz, 6)
        throw(DimensionMismatch("Strain rate tensor S must have dimensions [Nx, Ny, Nz, 6], got $(size(S))"))
    end
    
    # Extract grid parameters
    if !hasproperty(grid, :Lx) || !hasproperty(grid, :Ly)
        throw(ArgumentError("Grid must have Lx and Ly properties for domain size"))
    end
    Lx, Ly = grid.Lx, grid.Ly
    
    # Compute all velocity gradients using spectral derivatives
    compute_velocity_gradients_spectral!(workspace.grad_u, workspace.grad_v, workspace.grad_w,
                                        u, v, w, transform_plans, grid, bc, workspace)
    
    # Extract gradients for clarity
    grad_u = workspace.grad_u  # [Nx, Ny, Nz, 3] -> [∂u/∂x, ∂u/∂y, ∂u/∂z]
    grad_v = workspace.grad_v  # [Nx, Ny, Nz, 3] -> [∂v/∂x, ∂v/∂y, ∂v/∂z]
    grad_w = workspace.grad_w  # [Nx, Ny, Nz, 3] -> [∂w/∂x, ∂w/∂y, ∂w/∂z]
    
    # Compute strain rate tensor components with optimal performance annotations
    @inbounds @fastmath @simd ivdep for k in 1:Nz
        for j in 1:Ny
            for i in 1:Nx
                # Diagonal components (no factor of 1/2 needed)
                S[i,j,k,1] = grad_u[i,j,k,1]  # S₁₁ = ∂u/∂x
                S[i,j,k,2] = grad_v[i,j,k,2]  # S₂₂ = ∂v/∂y
                S[i,j,k,3] = grad_w[i,j,k,3]  # S₃₃ = ∂w/∂z
                
                # Off-diagonal components (with factor of 1/2)
                S[i,j,k,4] = 0.5 * (grad_u[i,j,k,2] + grad_v[i,j,k,1])  # S₁₂ = ½(∂u/∂y + ∂v/∂x)
                S[i,j,k,5] = 0.5 * (grad_u[i,j,k,3] + grad_w[i,j,k,1])  # S₁₃ = ½(∂u/∂z + ∂w/∂x)
                S[i,j,k,6] = 0.5 * (grad_v[i,j,k,3] + grad_w[i,j,k,2])  # S₂₃ = ½(∂v/∂z + ∂w/∂y)
            end
        end
    end
    
    return nothing
end

"""
    compute_rotation_rate_tensor!(Ω, u, v, w, transform_plans, grid, bc, workspace)

Compute the rotation rate tensor Ωᵢⱼ = ½(∂uᵢ/∂xⱼ - ∂uⱼ/∂xᵢ) using spectral derivatives.

Uses Fourier spectral derivatives for horizontal (x,y) directions and PencilFlows 
finite differences for the vertical (z) direction.

Follows PencilFlows convention: (output, inputs, transforms, grid, bc, workspace)

The rotation rate tensor is anti-symmetric, so we store only 3 components:
- Ω[:,:,:,1] = Ω₁₂ = ½(∂u/∂y - ∂v/∂x)  
- Ω[:,:,:,2] = Ω₁₃ = ½(∂u/∂z - ∂w/∂x)
- Ω[:,:,:,3] = Ω₂₃ = ½(∂v/∂z - ∂w/∂y)

# Arguments
- `Ω`: Output rotation rate tensor [Nx, Ny, Nz, 3] in Z-pencil orientation
- `u, v, w`: Velocity components [Nx, Ny, Nz] in Z-pencil orientation
- `transform_plans`: PencilFlows transform plans for spectral derivatives
- `grid`: PencilFlows grid object with domain size and z-grid
- `bc`: PencilFlows boundary conditions
- `workspace`: LESWorkspace for gradient computations
"""
function compute_rotation_rate_tensor!(Ω, u, v, w, transform_plans, grid, bc, workspace::LESWorkspace{T}) where T
    Nx, Ny, Nz = size(u)
    
    # Reuse gradients computed in strain rate tensor (if already computed)
    # Otherwise compute them using spectral derivatives
    if !any(workspace.grad_u .!= 0)  # Check if gradients need to be computed
        compute_velocity_gradients_spectral!(workspace.grad_u, workspace.grad_v, workspace.grad_w,
                                           u, v, w, transform_plans, grid, bc, workspace)
    end
    
    grad_u = workspace.grad_u
    grad_v = workspace.grad_v  
    grad_w = workspace.grad_w
    
    # Compute rotation rate tensor components with optimal performance annotations
    @inbounds @fastmath @simd ivdep for k in 1:Nz
        for j in 1:Ny
            for i in 1:Nx
                # Anti-symmetric components
                Ω[i,j,k,1] = 0.5 * (grad_u[i,j,k,2] - grad_v[i,j,k,1])  # Ω₁₂ = ½(∂u/∂y - ∂v/∂x)
                Ω[i,j,k,2] = 0.5 * (grad_u[i,j,k,3] - grad_w[i,j,k,1])  # Ω₁₃ = ½(∂u/∂z - ∂w/∂x)
                Ω[i,j,k,3] = 0.5 * (grad_v[i,j,k,3] - grad_w[i,j,k,2])  # Ω₂₃ = ½(∂v/∂z - ∂w/∂y)
            end
        end
    end
    
    return nothing
end

"""
    compute_velocity_gradients_spectral!(grad_u, grad_v, grad_w, u, v, w, transform_plans, grid, bc, workspace)

Compute velocity gradients using spectral derivatives for horizontal directions (x,y) 
and PencilFlows finite differences for vertical direction (z).

Uses the PencilFlows transform system for efficient spectral derivatives and
PencilFlows' boundary-condition-aware finite differences for vertical derivatives.

Follows PencilFlows convention: (output_arrays, input_arrays, transforms, grid, bc, workspace)

# Arguments
- `grad_u, grad_v, grad_w`: Output gradient arrays [Nx, Ny, Nz, 3] in Z-pencil orientation
- `u, v, w`: Velocity components [Nx, Ny, Nz] in Z-pencil orientation
- `transform_plans`: PencilFlows transform plans for spectral operations
- `grid`: PencilFlows grid object with domain size (Lx, Ly) and z-grid  
- `bc`: PencilFlows boundary conditions
- `workspace`: LESWorkspace containing working arrays
"""
function compute_velocity_gradients_spectral!(grad_u, grad_v, grad_w, u, v, w, 
                                            transform_plans, grid, bc, workspace::LESWorkspace{T}) where T
    
    # Extract domain parameters
    Lx, Ly = grid.Lx, grid.Ly
    
    # Get working fields for spectral operations
    working_fields = get_working_arrays(transform_plans)
    
    # Store references to gradient arrays using views for optimal performance (all in Z-pencil orientation)
    dudx = @view workspace.grad_u[:,:,:,1]  # ∂u/∂x
    dudy = @view workspace.grad_u[:,:,:,2]  # ∂u/∂y  
    dudz = @view workspace.grad_u[:,:,:,3]  # ∂u/∂z
    
    dvdx = @view workspace.grad_v[:,:,:,1]  # ∂v/∂x
    dvdy = @view workspace.grad_v[:,:,:,2]  # ∂v/∂y
    dvdz = @view workspace.grad_v[:,:,:,3]  # ∂v/∂z
    
    dwdx = @view workspace.grad_w[:,:,:,1]  # ∂w/∂x
    dwdy = @view workspace.grad_w[:,:,:,2]  # ∂w/∂y
    dwdz = @view workspace.grad_w[:,:,:,3]  # ∂w/∂z
    
    # Compute horizontal derivatives using spectral methods
    # This uses the PencilFlows horizontal derivative infrastructure
    compute_horizontal_derivatives_fd!(
        dudx, dudy,     # u derivatives
        dvdx, dvdy,     # v derivatives  
        dwdx, dwdy,     # w derivatives
        workspace.temp_field, workspace.temp_field,  # dummy p derivatives (not used)
        u, v, w, workspace.temp_field,  # velocity and dummy pressure fields
        transform_plans, working_fields, 
        Lx, Ly
    )
    
    # Compute vertical derivatives using PencilFlows finite difference scheme
    compute_vertical_derivatives_pencilflows!(dudz, dvdz, dwdz, u, v, w, grid, bc)
    
    return nothing
end

"""
    compute_vertical_derivatives_pencilflows!(dudz, dvdz, dwdz, u, v, w, grid, bc)

Compute vertical derivatives using PencilFlows' native finite difference schemes.
Handles non-uniform grids and boundary conditions properly.

Uses PencilFlows' `dz_derivative_nonuniform_with_bcs!` for consistent derivatives.

# Arguments
- `dudz, dvdz, dwdz`: Output vertical derivatives [Nx, Ny, Nz]
- `u, v, w`: Velocity components [Nx, Ny, Nz] in Z-pencil orientation
- `grid`: PencilFlows grid object with z-spacing information
- `bc`: Boundary conditions (optional, defaults to appropriate LES BCs)
"""
function compute_vertical_derivatives_pencilflows!(dudz, dvdz, dwdz, u, v, w, grid, bc=nothing)
    # Create default boundary conditions for LES if not provided
    if bc === nothing
        # For LES, we typically want no-slip at walls (u,v=0) and free-slip for w
        # This can be adjusted based on the specific problem
        bc = create_default_les_bcs()
    end
    
    # Use PencilFlows' native vertical derivative function
    # This handles non-uniform grids and boundary conditions properly
    dz_derivative_nonuniform_with_bcs!(dudz, u, grid, bc, :u)
    dz_derivative_nonuniform_with_bcs!(dvdz, v, grid, bc, :v)
    dz_derivative_nonuniform_with_bcs!(dwdz, w, grid, bc, :w)
    
    return nothing
end

"""
    create_default_les_bcs()

Create default boundary conditions suitable for LES computations.
Assumes no-slip walls for horizontal velocities and appropriate conditions for vertical velocity.
"""
function create_default_les_bcs()
    # For LES applications, we typically have:
    # - No-slip boundary conditions at walls (u,v=0)  
    # - Appropriate vertical velocity boundary conditions
    # This is a simplified version - in practice, this should match
    # the boundary conditions used in the main solver
    
    # We'll use a generic boundary condition that works with PencilFlows
    # In a full implementation, this should be passed from the main solver
    try
        return BoundaryCondition(NO_SLIP, NO_SLIP)
    catch
        # Fallback if BoundaryCondition type is not available in scope
        # In this case, we'll need the BC to be passed from the calling function
        error("Boundary conditions must be provided for LES vertical derivatives")
    end
end

"""
    compute_strain_magnitude(S) -> Array

Compute the strain rate magnitude |S| = √(2SᵢⱼSᵢⱼ).

# Arguments
- `S`: Strain rate tensor [Nx, Ny, Nz, 6]

# Returns
- Array [Nx, Ny, Nz] containing strain rate magnitude
"""
function compute_strain_magnitude(S)
    Nx, Ny, Nz = size(S)[1:3]
    T = eltype(S)
    s_mag = Array{T,3}(undef, Nx, Ny, Nz)
    
    @inbounds @fastmath @simd for k in 1:Nz
        for j in 1:Ny
            for i in 1:Nx
                # |S|² = 2(S₁₁² + S₂₂² + S₃₃²) + 4(S₁₂² + S₁₃² + S₂₃²)
                s_mag_sq = 2 * (S[i,j,k,1]^2 + S[i,j,k,2]^2 + S[i,j,k,3]^2) +
                          4 * (S[i,j,k,4]^2 + S[i,j,k,5]^2 + S[i,j,k,6]^2)
                s_mag[i,j,k] = sqrt(s_mag_sq)
            end
        end
    end
    
    return s_mag
end

"""
    compute_vorticity_magnitude(Ω) -> Array

Compute the vorticity magnitude |Ω| = √(2ΩᵢⱼΩᵢⱼ).

# Arguments
- `Ω`: Rotation rate tensor [Nx, Ny, Nz, 3]

# Returns  
- Array [Nx, Ny, Nz] containing vorticity magnitude
"""
function compute_vorticity_magnitude(Ω)
    Nx, Ny, Nz = size(Ω)[1:3]
    T = eltype(Ω)
    omega_mag = Array{T,3}(undef, Nx, Ny, Nz)
    
    @inbounds @fastmath @simd for k in 1:Nz
        for j in 1:Ny
            for i in 1:Nx
                # |Ω|² = 2(Ω₁₂² + Ω₁₃² + Ω₂₃²)
                omega_mag_sq = 2 * (Ω[i,j,k,1]^2 + Ω[i,j,k,2]^2 + Ω[i,j,k,3]^2)
                omega_mag[i,j,k] = sqrt(omega_mag_sq)
            end
        end
    end
    
    return omega_mag
end

"""
    compute_q_criterion(S, Ω) -> Array

Compute the Q-criterion: Q = ½(|Ω|² - |S|²).

Positive Q identifies vortex cores where rotation dominates strain.

# Arguments
- `S`: Strain rate tensor [Nx, Ny, Nz, 6]
- `Ω`: Rotation rate tensor [Nx, Ny, Nz, 3]

# Returns
- Array [Nx, Ny, Nz] containing Q-criterion values
"""
function compute_q_criterion(S, Ω)
    Nx, Ny, Nz = size(S)[1:3]
    T = eltype(S)
    Q = Array{T,3}(undef, Nx, Ny, Nz)
    
    @inbounds @fastmath @simd for k in 1:Nz
        for j in 1:Ny
            for i in 1:Nx
                # |S|² = 2(S₁₁² + S₂₂² + S₃₃²) + 4(S₁₂² + S₁₃² + S₂₃²)
                s_mag_sq = 2 * (S[i,j,k,1]^2 + S[i,j,k,2]^2 + S[i,j,k,3]^2) +
                          4 * (S[i,j,k,4]^2 + S[i,j,k,5]^2 + S[i,j,k,6]^2)
                
                # |Ω|² = 2(Ω₁₂² + Ω₁₃² + Ω₂₃²)
                omega_mag_sq = 2 * (Ω[i,j,k,1]^2 + Ω[i,j,k,2]^2 + Ω[i,j,k,3]^2)
                
                Q[i,j,k] = 0.5 * (omega_mag_sq - s_mag_sq)
            end
        end
    end
    
    return Q
end