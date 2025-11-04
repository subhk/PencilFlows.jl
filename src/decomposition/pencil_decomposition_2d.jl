"""
2D Domain Decomposition for 3D Navier-Stokes using PencilFFTs and PencilArrays
Decomposes the horizontal (x,y) directions while keeping z-direction local
"""

using FFTW
using MPI
using PencilFFTs
using PencilArrays

"""
Structure to hold the decomposition configuration
"""
struct PencilDecomposition
    # MPI setup
    comm::MPI.Comm
    rank::Int
    nprocs::Int
    
    # Grid dimensions
    Nx_global::Int
    Ny_global::Int
    Nz_global::Int
    
    # Process topology (2D decomposition in x,y)
    P1::Int  # Number of processes in x-direction
    P2::Int  # Number of processes in y-direction
    
    # Pencil configurations
    pencil_x::Pencil{3}  # X-pencils (data contiguous in x)
    pencil_y::Pencil{3}  # Y-pencils (data contiguous in y)
    pencil_z::Pencil{3}  # Z-pencils (data contiguous in z)
    
    # FFT plans for each pencil orientation
    fft_x::PencilFFTPlan  # FFT in x-direction (using X-pencils)
    fft_y::PencilFFTPlan  # FFT in y-direction (using Y-pencils)
    fft_xy::PencilFFTPlan # FFT in both x,y directions
    
    # Transform objects for switching between pencil orientations
    # Note: Using Any since PencilArrays.Transpose API may vary by version
    transform_x_to_y::Any  
    transform_y_to_z::Any
    transform_z_to_x::Any
    transform_y_to_x::Any
    transform_z_to_y::Any
    transform_x_to_z::Any
end

"""
Initialize 2D domain decomposition
"""
function init_pencil_decomposition(Nx::Int, Ny::Int, Nz::Int; 
                                 P1::Union{Int,Nothing}=nothing,
                                 P2::Union{Int,Nothing}=nothing)
    
    # Initialize MPI if not already done
    if !MPI.Initialized()
        MPI.Init()
    end
    
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    # Determine optimal process grid if not specified
    if P1 === nothing || P2 === nothing
        P1, P2 = find_optimal_process_grid(nprocs)
        if rank == 0
            println("Auto-selected process grid: $P1 x* $P2 = $(P1*P2) processes")
        end
    end
    
    @assert P1 * P2 == nprocs "Process grid P1x*P2 must equal total number of processes"
    @assert P1 <= Nx "P1 cannot exceed Nx"
    @assert P2 <= Ny "P2 cannot exceed Ny"
    
    # Create process topology for 2D decomposition
    topology = PencilArrays.Topology(comm, (P1, P2))
    
    # Define global array dimensions
    dims = (Nx, Ny, Nz)
    
    # Create pencil configurations
    # X-pencils: contiguous in x-direction, distributed in y,z
    pencil_x = Pencil(topology, dims, (1,))
    
    # Y-pencils: contiguous in y-direction, distributed in x,z  
    pencil_y = Pencil(topology, dims, (2,))
    
    # Z-pencils: contiguous in z-direction, distributed in x,y
    pencil_z = Pencil(topology, dims, (3,))
    
    if rank == 0
        println("Pencil configurations:")
        println("  X-pencils: $(size_local(pencil_x)) local, contiguous in x")
        println("  Y-pencils: $(size_local(pencil_y)) local, contiguous in y") 
        println("  Z-pencils: $(size_local(pencil_z)) local, contiguous in z")
    end
    
    # Create transform objects for switching between pencils
    transform_x_to_y = PencilArrays.Transpose(pencil_x => pencil_y)
    transform_y_to_z = PencilArrays.Transpose(pencil_y => pencil_z)
    transform_z_to_x = PencilArrays.Transpose(pencil_z => pencil_x)

    transform_y_to_x = PencilArrays.Transpose(pencil_y => pencil_x)
    transform_z_to_y = PencilArrays.Transpose(pencil_z => pencil_y)
    transform_x_to_z = PencilArrays.Transpose(pencil_x => pencil_z)
    
    # Create FFT plans
    # Note: We plan FFTs on the pencil orientations where the transform direction is contiguous
    
    # FFT in x-direction using X-pencils (x is contiguous)
    fft_x = PencilFFTPlan(pencil_x, 1, flags=FFTW.MEASURE)  # Transform along first dimension (x)

    # FFT in y-direction using Y-pencils (y is contiguous)
    fft_y = PencilFFTPlan(pencil_y, 2, flags=FFTW.MEASURE)  # Transform along second dimension (y)

    # 2D FFT in x,y directions requires careful orchestration
    # We'll do this in stages: x-direction first, then transpose, then y-direction
    fft_xy = PencilFFTPlan(pencil_x, (1, 2), flags=FFTW.MEASURE)  # This may not work directly
    
    return PencilDecomposition(
        comm, rank, nprocs, Nx, Ny, Nz, P1, P2,
        pencil_x, pencil_y, pencil_z,
        fft_x, fft_y, fft_xy,
        transform_x_to_y, transform_y_to_z, transform_z_to_x,
        transform_y_to_x, transform_z_to_y, transform_x_to_z
    )
end

# find_optimal_process_grid is provided in transforms/transforms.jl

"""
Create distributed arrays for velocity and pressure fields
"""
function create_distributed_fields(decomp::PencilDecomposition)
    # Create arrays in different pencil orientations
    
    # Primary storage in Z-pencils (good for z-derivatives)
    u_z = PencilArrays.PencilArray{Float64}(undef, decomp.pencil_z)
    v_z = PencilArrays.PencilArray{Float64}(undef, decomp.pencil_z)
    w_z = PencilArrays.PencilArray{Float64}(undef, decomp.pencil_z)
    p_z = PencilArrays.PencilArray{Float64}(undef, decomp.pencil_z)
    
    # Working arrays in X-pencils (for x-derivatives and x-FFTs)
    u_x = PencilArrays.PencilArray{Float64}(undef, decomp.pencil_x)
    v_x = PencilArrays.PencilArray{Float64}(undef, decomp.pencil_x)
    w_x = PencilArrays.PencilArray{Float64}(undef, decomp.pencil_x)
    p_x = PencilArrays.PencilArray{Float64}(undef, decomp.pencil_x)
    
    # Working arrays in Y-pencils (for y-derivatives and y-FFTs)
    u_y = PencilArrays.PencilArray{Float64}(undef, decomp.pencil_y)
    v_y = PencilArrays.PencilArray{Float64}(undef, decomp.pencil_y)
    w_y = PencilArrays.PencilArray{Float64}(undef, decomp.pencil_y)
    p_y = PencilArrays.PencilArray{Float64}(undef, decomp.pencil_y)
    
    # Spectral arrays (complex)
    u_hat_x = PencilArrays.PencilArray{ComplexF64}(undef, decomp.pencil_x)
    v_hat_x = PencilArrays.PencilArray{ComplexF64}(undef, decomp.pencil_x)
    w_hat_x = PencilArrays.PencilArray{ComplexF64}(undef, decomp.pencil_x)
    p_hat_x = PencilArrays.PencilArray{ComplexF64}(undef, decomp.pencil_x)
    
    u_hat_y = PencilArrays.PencilArray{ComplexF64}(undef, decomp.pencil_y)
    v_hat_y = PencilArrays.PencilArray{ComplexF64}(undef, decomp.pencil_y)
    w_hat_y = PencilArrays.PencilArray{ComplexF64}(undef, decomp.pencil_y)
    p_hat_y = PencilArrays.PencilArray{ComplexF64}(undef, decomp.pencil_y)
    
    return (
        # Physical space arrays
        u_z=u_z, v_z=v_z, w_z=w_z, p_z=p_z,
        u_x=u_x, v_x=v_x, w_x=w_x, p_x=p_x,
        u_y=u_y, v_y=v_y, w_y=w_y, p_y=p_y,
        
        # Spectral space arrays
        u_hat_x=u_hat_x, v_hat_x=v_hat_x, w_hat_x=w_hat_x, p_hat_x=p_hat_x,
        u_hat_y=u_hat_y, v_hat_y=v_hat_y, w_hat_y=w_hat_y, p_hat_y=p_hat_y
    )
end

"""
Compute horizontal derivatives using 2D decomposition
This is the key function that orchestrates the FFT-based derivatives
"""
function compute_horizontal_derivatives_2d!(
    dudx, dudy, dvdx, dvdy, dwdx, dwdy, dpdx, dpdy,
    u_z, v_z, w_z, p_z, fields, decomp::PencilDecomposition, grid)
    
    Nx, Ny, Nz = decomp.Nx_global, decomp.Ny_global, decomp.Nz_global
    
    # Pre-compute wavenumbers (cache these for repeated use)
    kx = fftfreq(Nx, 2π / grid.dx)
    ky = fftfreq(Ny, 2π / grid.dy)
    
    # Optimized pipeline: compute both x and y derivatives efficiently
    compute_x_derivatives_optimized!(dudx, dvdx, dwdx, dpdx, u_z, v_z, w_z, p_z, fields, decomp, kx)
    compute_y_derivatives_optimized!(dudy, dvdy, dwdy, dpdy, u_z, v_z, w_z, p_z, fields, decomp, ky)
end

"""
Compute x-derivatives in spectral space
"""
function compute_x_derivatives_spectral!(u_hat, v_hat, w_hat, p_hat, kx, pencil_x)
    # Get local array size and indices
    local_size = size_local(pencil_x)
    local_indices = range_local(pencil_x)
    
    # Pre-compute kx values for local indices
    kx_local = [1im * kx[local_indices[1][i]] for i in 1:local_size[1]]
    
    # Apply spectral differentiation with better memory access pattern
    @inbounds for k in 1:local_size[3]  # z-direction
        for j in 1:local_size[2]  # y-direction (local)
            for i in 1:local_size[1]  # x-direction (local)
                kx_val = kx_local[i]
                u_hat[i, j, k] *= kx_val
                v_hat[i, j, k] *= kx_val
                w_hat[i, j, k] *= kx_val
                p_hat[i, j, k] *= kx_val
            end
        end
    end
end

"""
Compute y-derivatives in spectral space
"""
function compute_y_derivatives_spectral!(u_hat, v_hat, w_hat, p_hat, ky, pencil_y)
    local_size = size_local(pencil_y)
    local_indices = range_local(pencil_y)
    
    # Pre-compute ky values for local indices
    ky_local = [1im * ky[local_indices[2][j]] for j in 1:local_size[2]]
    
    @inbounds for k in 1:local_size[3]  # z-direction
        for j in 1:local_size[2]  # y-direction (local)
            ky_val = ky_local[j]
            for i in 1:local_size[1]  # x-direction (local)
                u_hat[i, j, k] *= ky_val
                v_hat[i, j, k] *= ky_val
                w_hat[i, j, k] *= ky_val
                p_hat[i, j, k] *= ky_val
            end
        end
    end
end

"""
Recompute x-derivatives (helper function)
"""
# Optimized x-derivative computation with minimal data movement
function compute_x_derivatives_optimized!(dudx, dvdx, dwdx, dpdx, u_z, v_z, w_z, p_z, fields, decomp, kx)
    # Transform to X-pencils
    transpose!(fields.u_x, decomp.transform_z_to_x, u_z)
    transpose!(fields.v_x, decomp.transform_z_to_x, v_z)
    transpose!(fields.w_x, decomp.transform_z_to_x, w_z)
    transpose!(fields.p_x, decomp.transform_z_to_x, p_z)
    
    # FFT in x-direction
    mul!(fields.u_hat_x, decomp.fft_x, fields.u_x)
    mul!(fields.v_hat_x, decomp.fft_x, fields.v_x)
    mul!(fields.w_hat_x, decomp.fft_x, fields.w_x)
    mul!(fields.p_hat_x, decomp.fft_x, fields.p_x)
    
    # Compute x-derivatives
    compute_x_derivatives_spectral!(fields.u_hat_x, fields.v_hat_x, fields.w_hat_x, fields.p_hat_x, kx, decomp.pencil_x)
    
    # Inverse FFT
    ldiv!(fields.u_x, decomp.fft_x, fields.u_hat_x)
    ldiv!(fields.v_x, decomp.fft_x, fields.v_hat_x)
    ldiv!(fields.w_x, decomp.fft_x, fields.w_hat_x)
    ldiv!(fields.p_x, decomp.fft_x, fields.p_hat_x)
    
    # Transform back to Z-pencils
    transpose!(dudx, decomp.transform_x_to_z, fields.u_x)
    transpose!(dvdx, decomp.transform_x_to_z, fields.v_x)
    transpose!(dwdx, decomp.transform_x_to_z, fields.w_x)
    transpose!(dpdx, decomp.transform_x_to_z, fields.p_x)
end

# Optimized y-derivative computation with minimal data movement
function compute_y_derivatives_optimized!(dudy, dvdy, dwdy, dpdy, u_z, v_z, w_z, p_z, fields, decomp, ky)
    # Transform to Y-pencils
    transpose!(fields.u_y, decomp.transform_z_to_y, u_z)
    transpose!(fields.v_y, decomp.transform_z_to_y, v_z)
    transpose!(fields.w_y, decomp.transform_z_to_y, w_z)
    transpose!(fields.p_y, decomp.transform_z_to_y, p_z)
    
    # FFT in y-direction
    mul!(fields.u_hat_y, decomp.fft_y, fields.u_y)
    mul!(fields.v_hat_y, decomp.fft_y, fields.v_y)
    mul!(fields.w_hat_y, decomp.fft_y, fields.w_y)
    mul!(fields.p_hat_y, decomp.fft_y, fields.p_y)
    
    # Compute y-derivatives
    compute_y_derivatives_spectral!(fields.u_hat_y, fields.v_hat_y, fields.w_hat_y, fields.p_hat_y, ky, decomp.pencil_y)
    
    # Inverse FFT
    ldiv!(fields.u_y, decomp.fft_y, fields.u_hat_y)
    ldiv!(fields.v_y, decomp.fft_y, fields.v_hat_y)
    ldiv!(fields.w_y, decomp.fft_y, fields.w_hat_y)
    ldiv!(fields.p_y, decomp.fft_y, fields.p_hat_y)
    
    # Transform back to Z-pencils
    transpose!(dudy, decomp.transform_y_to_z, fields.u_y)
    transpose!(dvdy, decomp.transform_y_to_z, fields.v_y)
    transpose!(dwdy, decomp.transform_y_to_z, fields.w_y)
    transpose!(dpdy, decomp.transform_y_to_z, fields.p_y)
end

"""
Compute z-derivatives using finite differences (local operation in Z-pencils)
"""
function compute_z_derivatives_2d!(dudz, dvdz, dwdz, dpdz, u_z, v_z, w_z, p_z, decomp, grid, bc)
    # Since we're in Z-pencils, z-direction is contiguous and local
    # We can use the existing finite difference routines
    
    # Apply boundary-condition-aware derivatives
    dz_derivative_nonuniform_with_bcs!(dudz, u_z, grid, bc, :u)
    dz_derivative_nonuniform_with_bcs!(dvdz, v_z, grid, bc, :v)
    dz_derivative_nonuniform_with_bcs!(dwdz, w_z, grid, bc, :w)
    dz_derivative_nonuniform_with_bcs!(dpdz, p_z, grid, bc, :p)
end

"""
Example setup and usage
"""
function demo_2d_decomposition()
    # Grid parameters
    Nx, Ny, Nz = 128, 128, 64
    Lx, Ly, H = 4pi, 4pi, 2.0
    
    # Initialize decomposition
    decomp = init_pencil_decomposition(Nx, Ny, Nz)
    
    if decomp.rank == 0
        println("Initialized 2D decomposition:")
        println("  Global grid: $Nx x* $Ny x* $Nz")
        println("  Process grid: $(decomp.P1) x* $(decomp.P2)")
        println("  Local Z-pencil size: $(size_local(decomp.pencil_z))")
    end
    
    # Create distributed fields
    fields = create_distributed_fields(decomp)
    
    if decomp.rank == 0
        println("Created distributed arrays successfully")
    end
    
    # Initialize with some test data
    fill!(fields.u_z, 1.0 + decomp.rank)  # Different values per process
    fill!(fields.v_z, 2.0 + decomp.rank)
    fill!(fields.w_z, 0.0)
    fill!(fields.p_z, 0.0)
    
    # Test communication
    transpose!(fields.u_x, decomp.transform_z_to_x, fields.u_z)
    transpose!(fields.u_y, decomp.transform_x_to_y, fields.u_x)
    transpose!(fields.u_z, decomp.transform_y_to_z, fields.u_y)
    
    if decomp.rank == 0
        println("Communication test completed successfully")
    end
    
    return decomp, fields
end

"""
Performance monitoring for the decomposition
"""
function profile_decomposition_performance(decomp::PencilDecomposition, fields, n_iterations::Int=10)
    
    if decomp.rank == 0
        println("Profiling decomposition performance...")
    end
    
    # Warm up with optimized pattern
    for i in 1:3
        transpose!(fields.u_x, decomp.transform_z_to_x, fields.u_z)
        mul!(fields.u_hat_x, decomp.fft_x, fields.u_x)
        ldiv!(fields.u_x, decomp.fft_x, fields.u_hat_x)
        transpose!(fields.u_z, decomp.transform_x_to_z, fields.u_x)
    end
    
    # PERFORMANCE NOTE: This barrier synchronizes timing measurements
    # In production, consider removing if timing is not needed
    MPI.Barrier(decomp.comm)
    
    # Detailed timing breakdown
    t_comm = 0.0
    t_fft = 0.0
    t_total_start = time()
    
    for i in 1:n_iterations
        # Time communication
        t1 = time()
        transpose!(fields.u_x, decomp.transform_z_to_x, fields.u_z)
        transpose!(fields.u_z, decomp.transform_x_to_z, fields.u_x)
        t2 = time()
        t_comm += t2 - t1
        
        # Time FFT operations
        t1 = time()
        mul!(fields.u_hat_x, decomp.fft_x, fields.u_x)
        ldiv!(fields.u_x, decomp.fft_x, fields.u_hat_x)
        t2 = time()
        t_fft += t2 - t1
    end
    
    # PERFORMANCE NOTE: Final timing synchronization barrier
    # Can be removed in production if detailed timing not required
    MPI.Barrier(decomp.comm)
    t_total_end = time()
    
    total_elapsed = t_total_end - t_total_start
    
    # Gather detailed timing statistics
    comm_times = MPI.Allgather(t_comm, decomp.comm)
    fft_times = MPI.Allgather(t_fft, decomp.comm)
    total_times = MPI.Allgather(total_elapsed, decomp.comm)
    
    if decomp.rank == 0
        avg_comm = sum(comm_times) / length(comm_times)
        avg_fft = sum(fft_times) / length(fft_times)
        avg_total = sum(total_times) / length(total_times)
        max_total = maximum(total_times)
        min_total = minimum(total_times)
        
        println("Performance results ($n_iterations iterations):")
        println("  Communication time: $(round(avg_comm/n_iterations*1000, digits=2)) ms")
        println("  FFT time: $(round(avg_fft/n_iterations*1000, digits=2)) ms")
        println("  Total time: $(round(avg_total/n_iterations*1000, digits=2)) ms")
        println("  Load balance (max/min): $(round(max_total/min_total, digits=2))")
        println("  Communication efficiency: $(round(avg_comm/avg_total*100, digits=1))%")
        data_size = sizeof(Float64)*decomp.Nx_global*decomp.Ny_global*decomp.Nz_global
        println("  Bandwidth per iteration: $(round(data_size*4/avg_comm/1024^3, digits=2)) GB/s")
    end
    
    return (comm=comm_times, fft=fft_times, total=total_times)
end

# Export main functions
# export PencilDecomposition, init_pencil_decomposition
# export create_distributed_fields, compute_horizontal_derivatives_2d!
# export compute_z_derivatives_2d!, demo_2d_decomposition
# export profile_decomposition_performance
