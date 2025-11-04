using PencilFFTs
using PencilArrays
using FFTW
using LinearAlgebra
using MPI

# SIMD is optional for performance but not required
const HAS_SIMD = try
    using SIMD
    true
catch
    false
end

macro maybe_simd(expr)
    if HAS_SIMD
        :(@simd $expr)
    else
        expr
    end
end

# ============================================================================
# COMPATIBILITY HELPERS
# ============================================================================

"""Get local range for a pencil - compatibility wrapper"""
function get_local_range(pencil)
    try
        return range_local(pencil)
    catch
        # Fallback implementation
        try
            return localindices(pencil)
        catch
            error("Cannot determine local range for pencil")
        end
    end
end

"""Get local size for a pencil - compatibility wrapper"""
function get_local_size(pencil)
    try
        return size_local(pencil)
    catch
        # Fallback implementation
        try
            return local_size(pencil)
        catch
            # Last resort - create temp array to get size
            temp = PencilArrays.PencilArray{Float64}(undef, pencil)
            return size(temp)
        end
    end
end

# ============================================================================
# UNIFIED TRANSFORM PLANS STRUCTURE
# ============================================================================

"""
Unified transform plans for rotating stratified flows solver with finite differences in z.
Compatible with 2D domain decomposition that keeps z-direction local.
Handles horizontal FFTs only - vertical direction uses finite differences.

The `optimized` field controls whether to use enhanced features:
- optimized=false: Basic functionality, minimal memory usage
- optimized=true: Precomputed wavenumbers, memory pools, enhanced performance
"""
struct TransformPlans{T<:AbstractFloat}
    # Pencil configurations (compatible with 2D decomposition)
    pencil_x::Pencil{3}           # X-pencils (for x-direction transforms)
    pencil_y::Pencil{3}           # Y-pencils (for y-direction transforms) 
    pencil_z::Pencil{3}           # Z-pencils (optimal for FD operations in z)
    
    # Transform plans
    plan_fft_x::PencilFFTPlan     # FFT in x-direction
    plan_fft_y::PencilFFTPlan     # FFT in y-direction
    
    # Transform objects for pencil orientation changes
    # Note: Using Any since PencilArrays.Transpose API may vary by version
    transform_z_to_x::Any
    transform_x_to_y::Any
    transform_y_to_z::Any
    transform_x_to_z::Any
    transform_y_to_x::Any
    transform_z_to_y::Any
    
    # Grid information
    Nx::Int
    Ny::Int
    Nz::Int
    
    # Process topology
    P1::Int
    P2::Int
    
    # Performance optimization flag
    optimized::Bool
    
    # Optional optimization features (only used if optimized=true)
    kx_local::Union{Vector{T}, Nothing}          # Local x wavenumbers for this process
    ky_local::Union{Vector{T}, Nothing}          # Local y wavenumbers for this process
    kx2_local::Union{Vector{T}, Nothing}         # Local x wavenumbers squared
    ky2_local::Union{Vector{T}, Nothing}         # Local y wavenumbers squared
    
    # Local size information for fast access
    local_size_x::Union{NTuple{3,Int}, Nothing}
    local_size_y::Union{NTuple{3,Int}, Nothing}
    local_size_z::Union{NTuple{3,Int}, Nothing}
    
    # Memory pools for working arrays to avoid allocations (only if optimized=true)
    work_x_real::Union{PencilArrays.PencilArray{T, 3}, Nothing}
    work_x_complex::Union{PencilArrays.PencilArray{Complex{T}, 3}, Nothing}
    work_y_real::Union{PencilArrays.PencilArray{T, 3}, Nothing}
    work_y_complex::Union{PencilArrays.PencilArray{Complex{T}, 3}, Nothing}
    work_z_real::Union{PencilArrays.PencilArray{T, 3}, Nothing}
    work_z_complex::Union{PencilArrays.PencilArray{Complex{T}, 3}, Nothing}
end

# Constructor for basic (non-optimized) plans
function TransformPlans(pencil_x, pencil_y, pencil_z, plan_fft_x, plan_fft_y,
                       transform_z_to_x, transform_x_to_y, transform_y_to_z,
                       transform_x_to_z, transform_y_to_x, transform_z_to_y,
                       Nx, Ny, Nz, P1, P2)
    T = Float64  # Default type
    return TransformPlans{T}(
        pencil_x, pencil_y, pencil_z,
        plan_fft_x, plan_fft_y,
        transform_z_to_x, transform_x_to_y, transform_y_to_z,
        transform_x_to_z, transform_y_to_x, transform_z_to_y,
        Nx, Ny, Nz, P1, P2,
        false,  # optimized = false
        nothing, nothing, nothing, nothing,  # wavenumbers
        nothing, nothing, nothing,           # local sizes
        nothing, nothing, nothing, nothing, nothing, nothing  # work arrays
    )
end

# Constructor for optimized plans  
function TransformPlans(pencil_x, pencil_y, pencil_z, plan_fft_x, plan_fft_y,
                       transform_z_to_x, transform_x_to_y, transform_y_to_z,
                       transform_x_to_z, transform_y_to_x, transform_z_to_y,
                       Nx::Int, Ny::Int, Nz::Int, P1::Int, P2::Int,
                       kx_local::Vector{T}, ky_local::Vector{T}, 
                       kx2_local::Vector{T}, ky2_local::Vector{T},
                       local_size_x, local_size_y, local_size_z,
                       work_x_real, work_x_complex, work_y_real, work_y_complex,
                       work_z_real, work_z_complex) where T
    return TransformPlans{T}(
        pencil_x, pencil_y, pencil_z,
        plan_fft_x, plan_fft_y,
        transform_z_to_x, transform_x_to_y, transform_y_to_z,
        transform_x_to_z, transform_y_to_x, transform_z_to_y,
        Nx, Ny, Nz, P1, P2,
        true,  # optimized = true
        kx_local, ky_local, kx2_local, ky2_local,
        local_size_x, local_size_y, local_size_z,
        work_x_real, work_x_complex, work_y_real, work_y_complex,
        work_z_real, work_z_complex
    )
end

# ============================================================================
# UNIFIED CONSTRUCTOR FUNCTIONS
# ============================================================================

"""
    create_transform_plans_2d(Nx, Ny, Nz; T=Float64, P1=nothing, P2=nothing, comm=MPI.COMM_WORLD, optimized=:auto)

Create transform plans compatible with 2D domain decomposition.
Uses the same decomposition strategy as your pencil_decomposition_2d.jl file.

# Arguments
- `Nx, Ny, Nz`: Grid dimensions
- `T`: Floating point type (default: Float64)
- `P1, P2`: Process grid dimensions (auto-determined if not provided)
- `comm`: MPI communicator
- `optimized`: Use optimized features (true/false/:auto for automatic detection)
"""
function create_transform_plans_2d(Nx::Int, Ny::Int, Nz::Int; 
                                  T::Type{<:AbstractFloat}=Float64,
                                  P1::Union{Int,Nothing}=nothing,
                                  P2::Union{Int,Nothing}=nothing,
                                  comm=MPI.COMM_WORLD,
                                  optimized::Union{Bool,Symbol}=:auto)
    
    nprocs = MPI.Comm_size(comm)
    
    # Determine optimal process grid if not specified
    if P1 === nothing || P2 === nothing
        P1, P2 = find_optimal_process_grid(nprocs)
    end
    
    @assert P1 * P2 == nprocs "Process grid P1x*P2 must equal total number of processes"
    @assert P1 <= Nx "P1 cannot exceed Nx"
    @assert P2 <= Ny "P2 cannot exceed Ny"
    
    # Auto-detect optimization setting
    if optimized == :auto
        total_points = Nx * Ny * Nz
        optimized = HAS_SIMD && (nprocs > 1 && total_points > 64^3 || total_points > 32^3)
    end
    
    # Create process topology for 2D decomposition
    topology = PencilArrays.Topology(comm, (P1, P2))
    dims = (Nx, Ny, Nz)
    
    # Create pencil configurations
    pencil_x = Pencil(topology, dims, (1,))
    pencil_y = Pencil(topology, dims, (2,))
    pencil_z = Pencil(topology, dims, (3,))
    
    # Create transform objects for switching between pencils
    transform_z_to_x = PencilArrays.Transpose(pencil_z => pencil_x)
    transform_x_to_y = PencilArrays.Transpose(pencil_x => pencil_y)
    transform_y_to_z = PencilArrays.Transpose(pencil_y => pencil_z)
    transform_x_to_z = PencilArrays.Transpose(pencil_x => pencil_z)
    transform_y_to_x = PencilArrays.Transpose(pencil_y => pencil_x)
    transform_z_to_y = PencilArrays.Transpose(pencil_z => pencil_y)
    
    # Create transform plans
    plan_fft_x = PencilFFTPlan(pencil_x, 1, flags=FFTW.MEASURE)
    plan_fft_y = PencilFFTPlan(pencil_y, 2, flags=FFTW.MEASURE)
    
    if !optimized
        # Create basic plans
        return TransformPlans(
            pencil_x, pencil_y, pencil_z,
            plan_fft_x, plan_fft_y,
            transform_z_to_x, transform_x_to_y, transform_y_to_z,
            transform_x_to_z, transform_y_to_x, transform_z_to_y,
            Nx, Ny, Nz, P1, P2
        )
    else
        # Create optimized plans with precomputed data
        
        # Get local sizes for fast access
        local_size_x = get_local_size(pencil_x)
        local_size_y = get_local_size(pencil_y)
        local_size_z = get_local_size(pencil_z)
        
        # Precompute local wavenumbers for efficiency
        # X wavenumbers (for X-pencils, accounting for real-to-complex FFT)
        kx_global = T(2π) * [0:Nx÷2; -Nx÷2+1:-1][1:Nx÷2+1]
        local_indices_x = get_local_range(pencil_x)
        kx_local = kx_global[local_indices_x[1]]
        kx2_local = kx_local.^2
        
        # Y wavenumbers (for Y-pencils)
        ky_global = T(2π) * [0:Ny÷2; -Ny÷2+1:-1]
        local_indices_y = get_local_range(pencil_y)
        ky_local = ky_global[local_indices_y[2]]
        ky2_local = ky_local.^2
        
        # Create memory pools for working arrays
        work_x_real = PencilArrays.PencilArray{T}(undef, pencil_x)
        work_x_complex = PencilArrays.PencilArray{Complex{T}}(undef, pencil_x)
        work_y_real = PencilArrays.PencilArray{T}(undef, pencil_y)
        work_y_complex = PencilArrays.PencilArray{Complex{T}}(undef, pencil_y)
        work_z_real = PencilArrays.PencilArray{T}(undef, pencil_z)
        work_z_complex = PencilArrays.PencilArray{Complex{T}}(undef, pencil_z)
        
        return TransformPlans(
            pencil_x, pencil_y, pencil_z,
            plan_fft_x, plan_fft_y,
            transform_z_to_x, transform_x_to_y, transform_y_to_z,
            transform_x_to_z, transform_y_to_x, transform_z_to_y,
            Nx, Ny, Nz, P1, P2,
            kx_local, ky_local, kx2_local, ky2_local,
            local_size_x, local_size_y, local_size_z,
            work_x_real, work_x_complex, work_y_real, work_y_complex,
            work_z_real, work_z_complex
        )
    end
end

function find_optimal_process_grid(nprocs::Int)
    best_P1 = 1
    best_P2 = nprocs
    best_ratio = Inf
    max_p1 = Int(floor(sqrt(nprocs))) + 1
    for p1 in 1:max_p1
        if nprocs % p1 == 0
            p2 = div(nprocs, p1)
            r = max(p1, p2) / min(p1, p2)
            if r < best_ratio
                best_ratio = r
                best_P1 = p1
                best_P2 = p2
            end
        end
    end
    return best_P1, best_P2
end

# ============================================================================
# WORKING ARRAYS AND FIELD CREATION
# ============================================================================

"""
    create_transform_fields(plans::TransformPlans, T=Float64)

Create working arrays for transforms, compatible with your field structure.
"""
function create_transform_fields(plans::TransformPlans{T}) where T
    # Real space arrays for each pencil orientation
    u_x = PencilArrays.PencilArray{T}(undef, plans.pencil_x)
    u_y = PencilArrays.PencilArray{T}(undef, plans.pencil_y)
    u_z = PencilArrays.PencilArray{T}(undef, plans.pencil_z)
    
    # Spectral arrays (complex)
    u_hat_x = PencilArrays.PencilArray{Complex{T}}(undef, plans.pencil_x)
    u_hat_y = PencilArrays.PencilArray{Complex{T}}(undef, plans.pencil_y)
    u_hat_z = PencilArrays.PencilArray{Complex{T}}(undef, plans.pencil_z)
    
    return (
        # Real space working arrays
        u_x=u_x, u_y=u_y, u_z=u_z,
        # Spectral space arrays
        u_hat_x=u_hat_x, u_hat_y=u_hat_y, u_hat_z=u_hat_z
    )
end

"""Get working arrays that work with both optimized and basic plans"""
function get_working_arrays(plans::TransformPlans{T}) where T
    if plans.optimized
        # For optimized plans, use preallocated working arrays
        return (
            work_x_real = plans.work_x_real,
            work_x_complex = plans.work_x_complex,
            work_y_real = plans.work_y_real,
            work_y_complex = plans.work_y_complex,
            work_z_real = plans.work_z_real,
            work_z_complex = plans.work_z_complex
        )
    else
        # For basic plans, create temporary working arrays
        return (
            work_x_real = PencilArrays.PencilArray{T}(undef, plans.pencil_x),
            work_x_complex = PencilArrays.PencilArray{Complex{T}}(undef, plans.pencil_x),
            work_y_real = PencilArrays.PencilArray{T}(undef, plans.pencil_y),
            work_y_complex = PencilArrays.PencilArray{Complex{T}}(undef, plans.pencil_y),
            work_z_real = PencilArrays.PencilArray{T}(undef, plans.pencil_z),
            work_z_complex = PencilArrays.PencilArray{Complex{T}}(undef, plans.pencil_z)
        )
    end
end

# ============================================================================
# WAVENUMBER AND SIZE UTILITIES
# ============================================================================

"""Get wavenumber arrays for both optimized and basic plans"""
function get_wavenumbers(plans::TransformPlans{T}, Lx::T, Ly::T) where T
    if plans.optimized && plans.kx_local !== nothing && plans.ky_local !== nothing
        # For optimized plans, wavenumbers are already scaled and precomputed  
        return plans.kx_local / Lx, plans.ky_local / Ly
    else
        # For basic plans, compute wavenumbers dynamically
        # X wavenumbers (for X-pencils, accounting for real-to-complex FFT)
        kx_global = T(2π/Lx) * [0:plans.Nx÷2; -plans.Nx÷2+1:-1][1:plans.Nx÷2+1]
        local_indices_x = get_local_range(plans.pencil_x)
        kx_local = kx_global[local_indices_x[1]]
        
        # Y wavenumbers (for Y-pencils)
        ky_global = T(2π/Ly) * [0:plans.Ny÷2; -plans.Ny÷2+1:-1]
        local_indices_y = get_local_range(plans.pencil_y)
        ky_local = ky_global[local_indices_y[2]]
        
        return kx_local, ky_local
    end
end

"""
    get_horizontal_wavenumbers(plans::TransformPlans, Lx, Ly)

Get horizontal wavenumber arrays for the spectral space.
Returns kx, ky arrays compatible with local pencil layouts.
"""
function get_horizontal_wavenumbers(plans::TransformPlans{T}, Lx::T, Ly::T) where T
    kx_local, ky_local = get_wavenumbers(plans, Lx, Ly)
    
    # Reshape for broadcasting
    kx = reshape(kx_local, length(kx_local), 1, 1)
    ky = reshape(ky_local, 1, length(ky_local), 1)
    
    return kx, ky
end

"""Get local size that works with both plan types"""
function get_plan_local_size(plans::TransformPlans{T}, pencil_type::Symbol) where T
    if plans.optimized && plans.local_size_x !== nothing
        if pencil_type == :x
            return plans.local_size_x
        elseif pencil_type == :y
            return plans.local_size_y
        elseif pencil_type == :z
            return plans.local_size_z
        else
            error("Unknown pencil type: $pencil_type")
        end
    else
        if pencil_type == :x
            return get_local_size(plans.pencil_x)
        elseif pencil_type == :y
            return get_local_size(plans.pencil_y)
        elseif pencil_type == :z
            return get_local_size(plans.pencil_z)
        else
            error("Unknown pencil type: $pencil_type")
        end
    end
end

# ============================================================================
# HORIZONTAL DERIVATIVES
# ============================================================================

"""
    compute_horizontal_derivatives_fd!(
        dudx, dudy, dvdx, dvdy, dwdx, dwdy, dpdx, dpdy,
        u_z, v_z, w_z, p_z, plans, working_fields, Lx, Ly)

Compute horizontal derivatives using FFTs, compatible with your decomposition approach.
This integrates seamlessly with your existing compute_horizontal_derivatives_2d! function.
Works with both optimized and basic plans.
"""
function compute_horizontal_derivatives_fd!(
    dudx, dudy, dvdx, dvdy, dwdx, dwdy, dpdx, dpdy,
    u_z, v_z, w_z, p_z, plans::TransformPlans{T}, working_fields, 
    Lx::T, Ly::T) where T
    
    # Get wavenumbers (automatically handles optimized vs basic plans)
    kx, ky = get_horizontal_wavenumbers(plans, Lx, Ly)
    
    # Process each field (u, v, w, p)
    fields = [(u_z, dudx, dudy), (v_z, dvdx, dvdy), (w_z, dwdx, dwdy), (p_z, dpdx, dpdy)]
    
    for (field_z, dfdx, dfdy) in fields
        # Step 1: Transform from Z-pencils to X-pencils for x-derivatives
        transpose!(working_fields.work_x_real, plans.transform_z_to_x, field_z)
        
        # Step 2: FFT in x-direction
        mul!(working_fields.work_x_complex, plans.plan_fft_x, working_fields.work_x_real)
        
        # Step 3: Compute x-derivative in spectral space
        ddx!(working_fields.work_x_complex, kx, plans)
        
        # Step 4: Inverse FFT to get x-derivative
        ldiv!(working_fields.work_x_real, plans.plan_fft_x, working_fields.work_x_complex)
        
        # Step 5: Transform back to Z-pencils for x-derivative
        transpose!(dfdx, plans.transform_x_to_z, working_fields.work_x_real)
        
        # Step 6: Transform original field to Y-pencils for y-derivatives
        transpose!(working_fields.work_y_real, plans.transform_z_to_y, field_z)
        
        # Step 7: FFT in y-direction
        mul!(working_fields.work_y_complex, plans.plan_fft_y, working_fields.work_y_real)
        
        # Step 8: Compute y-derivative in spectral space
        ddy!(working_fields.work_y_complex, ky, plans)
        
        # Step 9: Inverse FFT to get y-derivative
        ldiv!(working_fields.work_y_real, plans.plan_fft_y, working_fields.work_y_complex)
        
        # Step 10: Transform back to Z-pencils for y-derivative
        transpose!(dfdy, plans.transform_y_to_z, working_fields.work_y_real)
    end
end

# ============================================================================
# SPECTRAL DERIVATIVE OPERATORS
# ============================================================================

"""
Apply spectral derivative in x-direction - Optimized version
"""
function ddx!(u_hat::PencilArrays.PencilArray{Complex{T}, 3}, 
              kx::AbstractArray{T}, 
              plans::TransformPlans{T}) where T
    
    # Extract wavenumber vector for vectorized operations
    if ndims(kx) == 3
        kx_vec = vec(kx[:, 1, 1])
    elseif ndims(kx) == 1
        kx_vec = kx
    else
        kx_vec = vec(kx)
    end
    
    # Use vectorized implementation
    data = parent(u_hat)
    local_size = size(u_hat)
    
    # Vectorized spectral differentiation: d/dx -> i*kx in Fourier space
    @inbounds for k in 1:local_size[3], j in 1:local_size[2]
        @maybe_simd for i in 1:local_size[1]
            data[i, j, k] *= im * kx_vec[i]
        end
    end
end

"""
Apply spectral derivative in y-direction - Optimized version
"""
function ddy!(u_hat::PencilArrays.PencilArray{Complex{T}, 3}, 
              ky::AbstractArray{T}, 
              plans::TransformPlans{T}) where T
    
    # Extract wavenumber vector for vectorized operations
    if ndims(ky) == 3
        ky_vec = vec(ky[1, :, 1])
    elseif ndims(ky) == 1
        ky_vec = ky
    else
        ky_vec = vec(ky)
    end
    
    # Use vectorized implementation
    data = parent(u_hat)
    local_size = size(u_hat)
    
    # Vectorized spectral differentiation: d/dy -> i*ky in Fourier space
    @inbounds for k in 1:local_size[3], i in 1:local_size[1]
        @maybe_simd for j in 1:local_size[2]
            data[i, j, k] *= im * ky_vec[j]
        end
    end
end

"""
Apply spectral second derivative in x-direction - Optimized version
"""
function d2dx2!(u_hat::PencilArrays.PencilArray{Complex{T}, 3}, 
                kx::AbstractArray{T}, 
                plans::TransformPlans{T}) where T
    
    # Extract wavenumber vector and precompute squares
    if ndims(kx) == 3
        kx_vec = vec(kx[:, 1, 1])
    elseif ndims(kx) == 1
        kx_vec = kx
    else
        kx_vec = vec(kx)
    end
    
    # Use precomputed squares if available
    if plans.optimized && plans.kx2_local !== nothing
        kx2_vec = plans.kx2_local
    else
        kx2_vec = kx_vec.^2
    end
    
    # Use vectorized implementation
    data = parent(u_hat)
    local_size = size(u_hat)
    
    # Vectorized spectral second differentiation: d2/dx2 -> -kx^2 in Fourier space
    @inbounds for k in 1:local_size[3], j in 1:local_size[2]
        @maybe_simd for i in 1:local_size[1]
            data[i, j, k] *= -kx2_vec[i]
        end
    end
end

"""
Apply spectral second derivative in y-direction - Optimized version
"""
function d2dy2!(u_hat::PencilArrays.PencilArray{Complex{T}, 3}, 
                ky::AbstractArray{T}, 
                plans::TransformPlans{T}) where T
    
    # Extract wavenumber vector and precompute squares
    if ndims(ky) == 3
        ky_vec = vec(ky[1, :, 1])
    elseif ndims(ky) == 1
        ky_vec = ky
    else
        ky_vec = vec(ky)
    end
    
    # Use precomputed squares if available
    if plans.optimized && plans.ky2_local !== nothing
        ky2_vec = plans.ky2_local
    else
        ky2_vec = ky_vec.^2
    end
    
    # Use vectorized implementation
    data = parent(u_hat)
    local_size = size(u_hat)
    
    # Vectorized spectral second differentiation: d2/dy2 -> -ky^2 in Fourier space
    @inbounds for k in 1:local_size[3], i in 1:local_size[1]
        @maybe_simd for j in 1:local_size[2]
            data[i, j, k] *= -ky2_vec[j]
        end
    end
end

# ============================================================================
# DEALIASING AND FILTERING
# ============================================================================

"""
    apply_dealias!(plans::TransformPlans, u_hat::PencilArrays.PencilArray; dealias_factor=2/3)

Apply 2/3 dealiasing rule to horizontal spectral coefficients with vectorized operations.
Works with both optimized and basic plans.
"""
function apply_dealias!(plans::TransformPlans{T}, u_hat::PencilArrays.PencilArray{Complex{T}}; 
                       dealias_factor::T=T(2/3)) where T
    
    data = parent(u_hat)
    
    if size(u_hat) == get_plan_local_size(plans, :x)
        # X-pencil orientation - vectorized
        kx_max = dealias_factor * plans.Nx / 2
        local_indices = get_local_range(plans.pencil_x)
        
        @inbounds for k in 1:size(u_hat, 3), j in 1:size(u_hat, 2)
            @maybe_simd for i in 1:size(u_hat, 1)
                global_i = local_indices[1][i]
                if global_i > plans.Nx÷2 + 1 + Int(floor(kx_max))
                    data[i, j, k] = zero(Complex{T})
                end
            end
        end
        
    elseif size(u_hat) == get_plan_local_size(plans, :y)
        # Y-pencil orientation - vectorized
        ky_max = dealias_factor * plans.Ny / 2
        local_indices = get_local_range(plans.pencil_y)
        
        @inbounds for k in 1:size(u_hat, 3), i in 1:size(u_hat, 1)
            @maybe_simd for j in 1:size(u_hat, 2)
                global_j = local_indices[2][j]
                ky_val = global_j <= plans.Ny÷2 ? global_j - 1 : global_j - plans.Ny - 1
                if abs(ky_val) > ky_max
                    data[i, j, k] = zero(Complex{T})
                end
            end
        end
    end
    
    return u_hat
end

# ============================================================================
# VERTICAL GRID AND FINITE DIFFERENCES
# ============================================================================

"""
    get_vertical_grid(plans::TransformPlans, Lz; grid_type=:uniform, kwargs...)

Get vertical grid points for finite difference operations.
Returns z coordinates for the Z-pencil orientation.

# Grid Types
- `:uniform`: Uniform spacing
- `:chebyshev`: Chebyshev-Gauss-Lobatto points (clustered at boundaries)
- `:stretched`: Hyperbolic tangent stretching
- `:stratified`: Boundary layer type grid for stratified flows
- `:analytical`: User-defined analytical function via `grid_function` keyword
- `:custom`: Provide custom grid points via `z_points` keyword
"""
function get_vertical_grid(plans::TransformPlans{T}, Lz::T; 
                          grid_type::Symbol=:uniform,
                          beta::T=T(2.5),           # stretching parameter
                          z_points::Union{Vector{T}, Nothing}=nothing,
                          boundary_clustering::T=T(1.5),  # for stratified grids
                          grid_function::Union{Function, Nothing}=nothing,  # analytical function
                          kwargs...) where T
    
    if grid_type == :uniform
        # Uniform grid from 0 to Lz
        z_1d = LinRange(T(0), Lz, plans.Nz)
        
    elseif grid_type == :chebyshev
        # Chebyshev-Gauss-Lobatto points mapped to [0, Lz]
        # Provides spectral accuracy and clusters points at boundaries
        theta = pi * (0:plans.Nz-1) / (plans.Nz - 1)
        z_1d = Lz/2 * (1 .- cos.(theta))
        
    elseif grid_type == :stretched
        # Hyperbolic tangent stretching
        # Clusters points near boundaries, good for boundary layers
        xi = LinRange(T(-1), T(1), plans.Nz)
        z_1d = Lz/2 * (1 .+ tanh.(beta * xi) / tanh(beta))
        
    elseif grid_type == :stratified
        # Grid designed for stratified flows with enhanced resolution near boundaries
        # Uses a combination of linear and exponential stretching
        N = plans.Nz
        xi = LinRange(T(0), T(1), N)
        
        # Stretch function that clusters points near both boundaries
        stretch_func(xi_val, a) = (exp(a * xi_val) - 1) / (exp(a) - 1)
        
        # Apply stretching to bottom half
        Ndiv2 = div(N, 2)
        z_bottom = stretch_func.(xi[1:Ndiv2], boundary_clustering) * Lz/2
        
        # Apply reverse stretching to top half  
        xi_top = LinRange(T(0), T(1), N - Ndiv2)
        z_top = Lz/2 .+ (Lz/2 - stretch_func.(reverse(xi_top), boundary_clustering) * Lz/2)
        
        z_1d = vcat(z_bottom, z_top)
        
    elseif grid_type == :analytical
        # User-defined analytical function
        if grid_function === nothing
            error("Analytical grid requires grid_function to be provided")
        end
        
        # Create uniform parameter xi from 0 to 1
        xi = LinRange(T(0), T(1), plans.Nz)
        
        # Apply user function to get z coordinates
        try
            z_1d = [grid_function(xi_t, Lz) for xi_t in xi]
            z_1d = convert(Vector{T}, z_1d)
        catch e
            error("Error evaluating grid_function: $e")
        end
        
        # Verify the function produces valid output
        if !issorted(z_1d)
            error("grid_function must produce monotonically increasing values")
        end
        if !(z_1d[1] == T(0)) || !(z_1d[end] == Lz)
            @warn "grid_function should map xi=0 to z=0 and xi=1 to z=Lz. Got z[1]=$(z_1d[1]), z[end]=$(z_1d[end])"
            # Auto-correct the endpoints
            z_1d[1] = T(0)
            z_1d[end] = Lz
        end
        
    elseif grid_type == :custom
        # User-provided grid points
        if z_points === nothing
            error("Custom grid requires z_points to be provided")
        end
        if length(z_points) != plans.Nz
            error("z_points must have length $(plans.Nz)")
        end
        z_1d = sort(z_points)  # Ensure monotonic
        
    else
        error("Unknown grid_type: $grid_type. Available types: :uniform, :chebyshev, :stretched, :stratified, :analytical, :custom")
    end
    
    # Verify grid is monotonic and within bounds
    @assert issorted(z_1d) "Grid points must be monotonically increasing"
    @assert z_1d[1] == T(0) "First grid point should be at z=0"
    @assert z_1d[end] == Lz "Last grid point should be at z=Lz"
    
    # Since z-direction is not decomposed in 2D decomposition, all processes have full z
    return reshape(z_1d, 1, 1, length(z_1d))
end

"""
    create_fd_operators(plans::TransformPlans, z_grid; order=4, boundary_order=nothing)

Create finite difference operators for derivatives in z-direction.
Automatically handles non-uniform grids using Fornberg's algorithm.
"""
function create_fd_operators(plans::TransformPlans{T}, z_grid::AbstractArray{T}; 
                            order::Int=4, boundary_order::Union{Int,Nothing}=nothing) where T
    
    if boundary_order === nothing
        boundary_order = order
    end
    
    Nz = plans.Nz
    z_1d = vec(z_grid)
    
    # Verify grid spacing for stability
    min_spacing = minimum(diff(z_1d))
    max_spacing = maximum(diff(z_1d))
    grid_ratio = max_spacing / min_spacing
    
    if grid_ratio > 100
        @warn "Large grid spacing ratio ($grid_ratio). Consider using lower order near transitions."
    end
    
    # Create derivative matrices
    D1 = zeros(T, Nz, Nz)  # First derivative
    D2 = zeros(T, Nz, Nz)  # Second derivative
    
    half_stencil = div(order, 2)
    
    for i = 1:Nz
        # Determine if we're near a boundary
        near_bottom = i <= half_stencil + 1
        near_top = i >= Nz - half_stencil
        
        # Choose stencil size and centering
        if near_bottom || near_top
            # Use boundary stencil
            current_order = boundary_order
            
            if near_bottom
                # Forward-biased stencil at bottom boundary
                i_start = 1
                i_end = min(current_order + 1, Nz)
            else
                # Backward-biased stencil at top boundary
                i_end = Nz
                i_start = max(1, Nz - current_order)
            end
        else
            # Use centered interior stencil
            current_order = order
            i_start = max(1, i - half_stencil)
            i_end = min(Nz, i + half_stencil)
            
            # Ensure we have enough points
            if i_end - i_start + 1 < current_order + 1
                # Adjust stencil if needed
                needed_points = current_order + 1
                center_adjust = div(needed_points - (i_end - i_start + 1), 2)
                i_start = max(1, i_start - center_adjust)
                i_end = min(Nz, i_end + center_adjust)
            end
        end
        
        # Extract stencil points
        stencil_points = z_1d[i_start:i_end]
        center_point = z_1d[i]
        
        # Compute weights for first derivative using Fornberg's algorithm
        # This automatically handles non-uniform spacing optimally
        weights_1 = finite_difference_weights(stencil_points .- center_point, 
                                            length(stencil_points) - 1, 1)
        D1[i, i_start:i_end] = weights_1
        
        # Compute weights for second derivative
        weights_2 = finite_difference_weights(stencil_points .- center_point, 
                                            length(stencil_points) - 1, 2)
        D2[i, i_start:i_end] = weights_2
    end
    
    return D1, D2
end

"""
Fornberg's algorithm for finite difference weights
"""
function finite_difference_weights(grid_points::Vector{T}, stencil_order::Int, 
                                 derivative_order::Int) where T
    N = length(grid_points)
    weights = zeros(T, N)
    c1, c4 = one(T), grid_points[1]
    
    weights[1] = one(T)
    
    for i = 2:N
        c2 = one(T)
        c5 = c4
        c4 = grid_points[i]
        
        for j = 1:i-1
            c3 = grid_points[i] - grid_points[j]
            c2 *= c3
            
            for k = min(i-1, derivative_order):-1:0
                weights[i] = ((grid_points[i] - c5) * weights[i] - k * weights[i-1]) / c3
                if j < i-1
                    weights[j] = ((grid_points[i] - grid_points[j]) * weights[j] - k * weights[j-1]) / c3
                end
            end
        end
        
        for k = min(i-1, derivative_order):-1:0
            weights[1] = c1/c2 * (k * weights[1] - (c5 - grid_points[1]) * weights[1])
        end
        
        c1 = c2
    end
    
    return weights
end

"""
    vertical_derivative!(result, u, D_matrix, plans)

Apply finite difference derivative in z-direction with optimized memory access.
Operates on Z-pencil arrays (z-direction is local).
"""
function vertical_derivative!(result::PencilArrays.PencilArray{Complex{T}, 3}, 
                            u::PencilArrays.PencilArray{Complex{T}, 3},
                            D_matrix::Matrix{T}, 
                            plans::TransformPlans{T}) where T
    
    local_size = get_plan_local_size(plans, :z)
    Nx_local, Ny_local, Nz_local = local_size
    
    # Use BLAS for matrix-vector multiplication when profitable
    if Nz_local > 32  # Threshold for BLAS efficiency
        u_data = parent(u)
        result_data = parent(result)
        
        # Reshape to 2D for BLAS gemm: [Nz, Nx*Ny]
        u_reshaped = reshape(u_data, Nx_local * Ny_local, Nz_local)'
        result_reshaped = reshape(result_data, Nx_local * Ny_local, Nz_local)'
        
        # Batched matrix multiplication: D * u for all (x,y) at once
        mul!(result_reshaped, D_matrix, u_reshaped)
    else
        # Use optimized loops for smaller arrays
        @inbounds for j = 1:Ny_local, i = 1:Nx_local
            u_profile = @view u[i, j, :]
            result_profile = @view result[i, j, :]
            mul!(result_profile, D_matrix, u_profile)
        end
    end
    
    return result
end

# ============================================================================
# OPTIMIZED MULTI-OPERATION FUNCTIONS (for optimized plans only)
# ============================================================================

"""
Apply multiple derivative operations in a single pass to minimize memory transfers
Only works with optimized plans that have precomputed wavenumbers.
"""
function apply_multiple_derivatives!(results::Vector{PencilArray{Complex{T}, 3}}, 
                                   u_hat::PencilArray{Complex{T}, 3},
                                   derivative_ops::Vector{Symbol},
                                   plans::TransformPlans{T}) where T
    
    if !plans.optimized
        error("apply_multiple_derivatives! requires optimized=true plans")
    end
    
    # Check that u_hat is in correct pencil orientation for requested operations
    is_x_pencil = size(u_hat) == plans.local_size_x
    is_y_pencil = size(u_hat) == plans.local_size_y
    
    data = parent(u_hat)
    
    @inbounds for (i, op) in enumerate(derivative_ops)
        result_data = parent(results[i])
        copyto!(result_data, data)  # Copy input to result
        
        if op == :ddx && is_x_pencil
            # Apply x-derivative
            for k in 1:plans.local_size_x[3], j in 1:plans.local_size_x[2]
                @maybe_simd for ix in 1:plans.local_size_x[1]
                    result_data[ix, j, k] *= im * plans.kx_local[ix]
                end
            end
            
        elseif op == :ddy && is_y_pencil
            # Apply y-derivative
            for k in 1:plans.local_size_y[3], i in 1:plans.local_size_y[1]
                @maybe_simd for jy in 1:plans.local_size_y[2]
                    result_data[i, jy, k] *= im * plans.ky_local[jy]
                end
            end
            
        elseif op == :d2dx2 && is_x_pencil
            # Apply x-second derivative
            for k in 1:plans.local_size_x[3], j in 1:plans.local_size_x[2]
                @maybe_simd for ix in 1:plans.local_size_x[1]
                    result_data[ix, j, k] *= -plans.kx2_local[ix]
                end
            end
            
        elseif op == :d2dy2 && is_y_pencil
            # Apply y-second derivative
            for k in 1:plans.local_size_y[3], i in 1:plans.local_size_y[1]
                @maybe_simd for jy in 1:plans.local_size_y[2]
                    result_data[i, jy, k] *= -plans.ky2_local[jy]
                end
            end
            
        else
            error("Incompatible derivative operation $op for current pencil orientation or basic plans")
        end
    end
end

"""
Optimized Laplacian computation combining both second derivatives
Only works with optimized plans.
"""
function laplacian_2d!(result::PencilArray{Complex{T}, 3},
                      u_hat_x::PencilArray{Complex{T}, 3},  # Input in X-pencil
                      u_hat_y::PencilArray{Complex{T}, 3},  # Input in Y-pencil  
                      plans::TransformPlans{T}) where T
    
    if !plans.optimized
        error("laplacian_2d! requires optimized=true plans")
    end
    
    # Apply both second derivatives and sum them
    # ∇² = ∂²/∂x² + ∂²/∂y² -> -(k_x² + k_y²) in spectral space
    
    # X-component: -k_x² term
    data_x = parent(u_hat_x)
    @inbounds for k in 1:plans.local_size_x[3], j in 1:plans.local_size_x[2]
        @maybe_simd for i in 1:plans.local_size_x[1]
            data_x[i, j, k] *= -plans.kx2_local[i]
        end
    end
    
    # Y-component: -k_y² term
    data_y = parent(u_hat_y)
    @inbounds for k in 1:plans.local_size_y[3], i in 1:plans.local_size_y[1]
        @maybe_simd for j in 1:plans.local_size_y[2]
            data_y[i, j, k] *= -plans.ky2_local[j]
        end
    end
    
    # Transform both to same pencil orientation and sum
    transpose!(plans.work_z_complex, plans.transform_x_to_z, u_hat_x)
    result_data = parent(result)
    copyto!(result_data, parent(plans.work_z_complex))
    
    transpose!(plans.work_z_complex, plans.transform_y_to_z, u_hat_y) 
    @inbounds @maybe_simd for i in eachindex(result_data)
        result_data[i] += parent(plans.work_z_complex)[i]
    end
end

# ============================================================================
# BATCH PROCESSING FUNCTIONS
# ============================================================================

"""
Optimized batch transform for multiple fields simultaneously
"""
function batch_horizontal_fft!(fields_real::Vector{PencilArray{T, 3}},
                              fields_complex::Vector{PencilArray{Complex{T}, 3}},
                              plan::PencilFFTPlan) where T
    
    # Batch process multiple fields with same FFT plan
    for i in eachindex(fields_real)
        mul!(fields_complex[i], plan, fields_real[i])
    end
end

function batch_horizontal_ifft!(fields_real::Vector{PencilArray{T, 3}},
                               fields_complex::Vector{PencilArray{Complex{T}, 3}},
                               plan::PencilFFTPlan) where T
    
    # Batch process multiple fields with same IFFT plan
    for i in eachindex(fields_real)
        ldiv!(fields_real[i], plan, fields_complex[i])
    end
end

# ============================================================================
# COMPATIBILITY AND VALIDATION
# ============================================================================

"""
Validate that all required dependencies are available and compatible
"""
function validate_transform_compatibility()
    issues = String[]
    
    # Check if SIMD is available
    if !HAS_SIMD
        push!(issues, "SIMD package not available - performance may be reduced")
    end
    
    # Check if MPI is initialized
    try
        if !MPI.Initialized()
            push!(issues, "MPI not initialized - parallel operations may fail")
        end
    catch
        push!(issues, "MPI not available - parallel operations disabled")
    end
    
    # Check if PencilArrays functions are available
    try
        # Create a dummy topology to test functions
        topology = PencilArrays.Topology(MPI.COMM_WORLD, (2, 1))
        pencil = Pencil(topology, (4, 4, 4), (1,))
        
        # Test the compatibility wrappers
        _ = get_local_size(pencil)
        _ = get_local_range(pencil)
    catch e
        push!(issues, "PencilArrays compatibility issue: $e")
    end
    
    if isempty(issues)
        @info "Transform compatibility check passed"
    else
        @warn "Transform compatibility issues found:" issues
    end
    
    return isempty(issues)
end

"""
Get recommended plan type based on system capabilities
"""
function get_recommended_optimization(Nx::Int, Ny::Int, Nz::Int)
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    total_points = Nx * Ny * Nz
    
    # Recommend optimized plans for larger problems and parallel execution
    if HAS_SIMD && nprocs > 1 && total_points > 64^3
        return true
    elseif total_points > 32^3
        return true
    else
        return false
    end
end

# Simple linear interpolation utility
struct LinearInterpolation{T}
    x::Vector{T}
    y::Vector{T}
end

function (interp::LinearInterpolation)(xi)
    # Simple linear interpolation
    result = similar(xi)
    for (i, x_val) in enumerate(xi)
        if x_val <= interp.x[1]
            result[i] = interp.y[1]
        elseif x_val >= interp.x[end]
            result[i] = interp.y[end]
        else
            # Find bracketing indices
            j = searchsortedlast(interp.x, x_val)
            if j == length(interp.x)
                result[i] = interp.y[end]
            else
                # Linear interpolation
                t = (x_val - interp.x[j]) / (interp.x[j+1] - interp.x[j])
                result[i] = (1 - t) * interp.y[j] + t * interp.y[j+1]
            end
        end
    end
    return result
end
