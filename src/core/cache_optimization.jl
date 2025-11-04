# Cache-Aware Data Structures and Operations for PencilFlows.jl
# Optimizes memory access patterns for modern CPU architectures

using Base.Threads: @threads
using PencilArrays
if !isdefined(@__MODULE__, :DEFAULT_POOL_F64)
    include("memory_pools.jl")
end
export VectorFieldSoA, create_vector_field_soa, cache_blocked_operation!
export TiledArray, create_tiled_array, tiled_copy!, tiled_transpose!
export prefetch_hint, cache_friendly_loop, optimize_loop_order

# Cache and memory hierarchy constants
const L1_CACHE_SIZE = 32 * 1024      # 32 KB typical L1 cache
const L2_CACHE_SIZE = 256 * 1024     # 256 KB typical L2 cache  
const L3_CACHE_SIZE = 8 * 1024 * 1024  # 8 MB typical L3 cache
const CACHE_LINE_SIZE = 64           # 64 bytes typical cache line
const PAGE_SIZE = 4096              # 4 KB typical page size

# Optimal tile sizes for different operations
const TILE_SIZE_L1 = 32    # For L1 cache-resident operations
const TILE_SIZE_L2 = 128   # For L2 cache-resident operations
const TILE_SIZE_L3 = 512   # For L3 cache-resident operations

"""
    VectorFieldSoA{T,N}

Structure of Arrays (SoA) layout for vector fields.
Better vectorization and cache performance than Array of Structures.
"""
struct VectorFieldSoA{T,N}
    u::Array{T,N}     # u-component field
    v::Array{T,N}     # v-component field  
    w::Array{T,N}     # w-component field
    dims::NTuple{N,Int}
    aligned::Bool
    
    function VectorFieldSoA{T,N}(dims::NTuple{N,Int}; aligned::Bool=true) where {T,N}
        if aligned && (HAS_AVX512 || HAS_AVX2)
            u = aligned_array(T, dims...)
            v = aligned_array(T, dims...)
            w = aligned_array(T, dims...)
        else
            u = Array{T,N}(undef, dims...)
            v = Array{T,N}(undef, dims...)
            w = Array{T,N}(undef, dims...)
        end
        new{T,N}(u, v, w, dims, aligned)
    end
end

"""
    create_vector_field_soa(::Type{T}, dims::NTuple{N,Int}; aligned=true) where {T,N}

Create a Structure of Arrays vector field with optimal memory alignment.
"""
function create_vector_field_soa(::Type{T}, dims::NTuple{N,Int}; aligned::Bool=true) where {T,N}
    return VectorFieldSoA{T,N}(dims; aligned=aligned)
end

# Convenience methods for accessing components
@inline Base.getindex(vf::VectorFieldSoA, component::Symbol) = getfield(vf, component)
@inline u_component(vf::VectorFieldSoA) = vf.u
@inline v_component(vf::VectorFieldSoA) = vf.v  
@inline w_component(vf::VectorFieldSoA) = vf.w

"""
    TiledArray{T,N}

Cache-friendly tiled array layout that stores data in small blocks
to improve spatial locality.
"""
struct TiledArray{T,N}
    data::Array{T,N}
    tile_sizes::NTuple{N,Int}
    logical_dims::NTuple{N,Int}
    physical_dims::NTuple{N,Int}
    
    function TiledArray{T,N}(logical_dims::NTuple{N,Int}, 
                            tile_sizes::NTuple{N,Int}) where {T,N}
        # Calculate physical dimensions to accommodate tiling
        physical_dims = ntuple(i -> 
            div(logical_dims[i] + tile_sizes[i] - 1, tile_sizes[i]) * tile_sizes[i], N)
        
        data = Array{T,N}(undef, physical_dims...)
        new{T,N}(data, tile_sizes, logical_dims, physical_dims)
    end
end

"""
    create_tiled_array(::Type{T}, dims::NTuple{N,Int}; tile_sizes=nothing) where {T,N}

Create a tiled array with optimal tile sizes based on cache hierarchy.
"""
function create_tiled_array(::Type{T}, dims::NTuple{N,Int}; 
                           tile_sizes::Union{Nothing,NTuple{N,Int}}=nothing) where {T,N}
    if tile_sizes === nothing
        # Auto-select tile sizes based on array size and cache levels
        element_size = sizeof(T)
        if N == 3
            # For 3D arrays, optimize for L2 cache
            target_tile_bytes = L2_CACHE_SIZE ÷ 4  # Use 1/4 of L2 cache per tile
            target_tile_elements = target_tile_bytes ÷ element_size
            tile_size_1d = Int(round(target_tile_elements^(1/3)))
            tile_sizes = (min(tile_size_1d, dims[1]), 
                         min(tile_size_1d, dims[2]), 
                         min(tile_size_1d, dims[3]))
        elseif N == 2
            # For 2D arrays, optimize for L1 cache
            target_tile_bytes = L1_CACHE_SIZE ÷ 2
            target_tile_elements = target_tile_bytes ÷ element_size
            tile_size_1d = Int(round(sqrt(target_tile_elements)))
            tile_sizes = (min(tile_size_1d, dims[1]), min(tile_size_1d, dims[2]))
        else
            # Fallback: use moderate tile sizes
            tile_sizes = ntuple(i -> min(TILE_SIZE_L2, dims[i]), N)
        end
    end
    
    return TiledArray{T,N}(dims, tile_sizes)
end

"""
    linear_to_tiled_index(linear_idx::Int, tile_sizes::NTuple{N,Int}, 
                         logical_dims::NTuple{N,Int}) where N

Convert linear index to tiled index for cache-friendly access.
"""
function linear_to_tiled_index(linear_idx::Int, tile_sizes::NTuple{N,Int}, 
                              logical_dims::NTuple{N,Int}) where N
    # Convert linear index to logical coordinates
    coords = CartesianIndices(logical_dims)[linear_idx].I
    
    # Convert to tiled coordinates
    tiled_coords = ntuple(N) do i
        tile_idx = div(coords[i] - 1, tile_sizes[i])
        local_idx = mod(coords[i] - 1, tile_sizes[i])
        tile_idx * tile_sizes[i] + local_idx + 1
    end
    
    return tiled_coords
end

"""
    prefetch_hint(ptr::Ptr, offset::Int=CACHE_LINE_SIZE; 
                 locality::Int=3, read_write::Int=0)

Insert prefetch hint for better cache utilization.
"""
@inline function prefetch_hint(ptr::Ptr, offset::Int=CACHE_LINE_SIZE; 
                              locality::Int=3, read_write::Int=0)
    # Use LLVM prefetch intrinsic if available
    ccall(:llvm.prefetch, Cvoid, (Ptr{Cvoid}, Int32, Int32, Int32), 
          ptr + offset, read_write, locality, 1)
end

"""
    cache_blocked_operation!(output, input, operation, block_size=TILE_SIZE_L2)

Apply an operation using cache-friendly blocking.
"""
function cache_blocked_operation!(output::Array{T,3}, input::Array{T,3}, 
                                operation::Function, 
                                block_size::Int=TILE_SIZE_L2) where T
    Nx, Ny, Nz = size(input)
    
    # Process in cache-friendly blocks
    for k_start in 1:block_size:Nz
        k_end = min(k_start + block_size - 1, Nz)
        
        for j_start in 1:block_size:Ny  
            j_end = min(j_start + block_size - 1, Ny)
            
            for i_start in 1:block_size:Nx
                i_end = min(i_start + block_size - 1, Nx)
                
                # Process block with optimal memory access pattern
                process_block!(output, input, operation, 
                             i_start, i_end, j_start, j_end, k_start, k_end)
            end
        end
    end
end

"""
    process_block!(output, input, operation, i_start, i_end, j_start, j_end, k_start, k_end)

Process a cache-sized block with optimal loop ordering.
"""
@inline function process_block!(output, input, operation, 
                               i_start, i_end, j_start, j_end, k_start, k_end)
    # Use k-j-i loop order for better cache locality (column-major arrays)
    for i in i_start:i_end
        for j in j_start:j_end
            # Prefetch next cache line
            if j < j_end
                prefetch_hint(pointer(input, LinearIndices(input)[i, j+1, k_start]))
            end
            
            @inbounds @simd ivdep for k in k_start:k_end
                output[i, j, k] = operation(input[i, j, k])
            end
        end
    end
end

"""
    cache_friendly_loop(f::Function, dims::NTuple{3,Int}; 
                       tile_size::Int=TILE_SIZE_L2)

Execute a function with cache-friendly loop ordering and blocking.
"""
function cache_friendly_loop(f::Function, dims::NTuple{3,Int}; 
                            tile_size::Int=TILE_SIZE_L2)
    Nx, Ny, Nz = dims
    
    for i_block in 1:tile_size:Nx
        i_end = min(i_block + tile_size - 1, Nx)
        
        for j_block in 1:tile_size:Ny
            j_end = min(j_block + tile_size - 1, Ny)
            
            for k_block in 1:tile_size:Nz
                k_end = min(k_block + tile_size - 1, Nz)
                
                # Execute function on cache-friendly block
                f(i_block:i_end, j_block:j_end, k_block:k_end)
            end
        end
    end
end

"""
    optimize_loop_order(operation::Symbol, dims::NTuple{3,Int})

Return optimal loop ordering for different operation types.
"""
function optimize_loop_order(operation::Symbol, dims::NTuple{3,Int})
    Nx, Ny, Nz = dims
    
    if operation == :x_derivative
        # For x-derivatives, x should be innermost
        return (:k, :j, :i), (Nz, Ny, Nx)
    elseif operation == :y_derivative  
        # For y-derivatives, y should be innermost
        return (:k, :i, :j), (Nz, Nx, Ny)
    elseif operation == :z_derivative
        # For z-derivatives, z should be innermost (natural column-major order)
        return (:i, :j, :k), (Nx, Ny, Nz)
    elseif operation == :laplacian
        # For Laplacian, use standard order with blocking
        return (:i, :j, :k), (Nx, Ny, Nz)
    else
        # Default: column-major order
        return (:i, :j, :k), (Nx, Ny, Nz)
    end
end

"""
    tiled_copy!(dest::TiledArray{T,N}, src::Array{T,N}) where {T,N}

Copy from regular array to tiled array with optimal access pattern.
"""
function tiled_copy!(dest::TiledArray{T,N}, src::Array{T,N}) where {T,N}
    @assert size(src) == dest.logical_dims "Size mismatch"
    
    if N == 3
        tiled_copy_3d!(dest, src)
    elseif N == 2
        tiled_copy_2d!(dest, src)  
    else
        # Fallback for other dimensions
        copyto!(dest.data, src)
    end
end

"""
    tiled_copy_3d!(dest::TiledArray{T,3}, src::Array{T,3}) where T

Optimized 3D copy with tiling for cache efficiency.
"""
function tiled_copy_3d!(dest::TiledArray{T,3}, src::Array{T,3}) where T
    Nx, Ny, Nz = dest.logical_dims
    tile_x, tile_y, tile_z = dest.tile_sizes
    
    @threads for tile_k in 0:tile_z:(Nz-1)
        k_end = min(tile_k + tile_z, Nz)
        
        for tile_j in 0:tile_y:(Ny-1)
            j_end = min(tile_j + tile_y, Ny)
            
            for tile_i in 0:tile_x:(Nx-1) 
                i_end = min(tile_i + tile_x, Nx)
                
                # Copy tile with optimal access pattern
                for i in (tile_i+1):i_end
                    for j in (tile_j+1):j_end
                        @inbounds @simd ivdep for k in (tile_k+1):k_end
                            dest.data[i, j, k] = src[i, j, k]
                        end
                    end
                end
            end
        end
    end
end

"""
    tiled_copy_2d!(dest::TiledArray{T,2}, src::Array{T,2}) where T

Optimized 2D copy with tiling.
"""
function tiled_copy_2d!(dest::TiledArray{T,2}, src::Array{T,2}) where T
    Nx, Ny = dest.logical_dims
    tile_x, tile_y = dest.tile_sizes
    
    @threads for tile_j in 0:tile_y:(Ny-1)
        j_end = min(tile_j + tile_y, Ny)
        
        for tile_i in 0:tile_x:(Nx-1)
            i_end = min(tile_i + tile_x, Nx)
            
            # Copy tile
            for i in (tile_i+1):i_end
                @inbounds @simd ivdep for j in (tile_j+1):j_end
                    dest.data[i, j] = src[i, j]
                end
            end
        end
    end
end

"""
    tiled_transpose!(dest::Array{T,2}, src::Array{T,2}, 
                    tile_size::Int=TILE_SIZE_L1) where T

Cache-friendly matrix transpose using blocking.
"""
function tiled_transpose!(dest::Array{T,2}, src::Array{T,2}, 
                         tile_size::Int=TILE_SIZE_L1) where T
    M, N = size(src)
    @assert size(dest) == (N, M) "Destination must be transposed size"
    
    @threads for j_start in 1:tile_size:N
        j_end = min(j_start + tile_size - 1, N)
        
        for i_start in 1:tile_size:M
            i_end = min(i_start + tile_size - 1, M)
            
            # Transpose tile
            for j in j_start:j_end
                @inbounds @simd ivdep for i in i_start:i_end
                    dest[j, i] = src[i, j]
                end
            end
        end
    end
end

"""
    memory_bandwidth_benchmark(array_size::Int=10^6, iterations::Int=100)

Benchmark memory bandwidth to tune cache parameters.
"""
function memory_bandwidth_benchmark(array_size::Int=10^6, iterations::Int=100)
    # Create test arrays
    a = rand(Float64, array_size)
    b = rand(Float64, array_size)
    c = zeros(Float64, array_size)
    
    # Warm up
    for _ in 1:10
        @inbounds @simd for i in 1:array_size
            c[i] = a[i] + b[i]
        end
    end
    
    # Benchmark
    start_time = time()
    for _ in 1:iterations
        @inbounds @simd for i in 1:array_size
            c[i] = a[i] + b[i]
        end
    end
    end_time = time()
    
    # Calculate bandwidth
    bytes_per_iteration = 3 * array_size * sizeof(Float64)  # 2 reads + 1 write
    total_bytes = bytes_per_iteration * iterations
    elapsed_time = end_time - start_time
    bandwidth_gb_s = (total_bytes / elapsed_time) / (1024^3)
    
    return bandwidth_gb_s
end

"""
    cache_hierarchy_info()

Detect and return information about the cache hierarchy.
"""
function cache_hierarchy_info()
    info = Dict{String,Any}()
    
    try
        # Try to read cache info from /proc/cpuinfo (Linux)
        cpuinfo = read(`cat /proc/cpuinfo`, String)
        info["detected"] = true
        
        # Try to detect cache sizes (this is system-specific)
        if isfile("/sys/devices/system/cpu/cpu0/cache/index1/size")
            l1_size_str = read(`cat /sys/devices/system/cpu/cpu0/cache/index1/size`, String)
            info["L1_cache"] = strip(l1_size_str)
        end
        
        if isfile("/sys/devices/system/cpu/cpu0/cache/index2/size")
            l2_size_str = read(`cat /sys/devices/system/cpu/cpu0/cache/index2/size`, String)
            info["L2_cache"] = strip(l2_size_str)
        end
        
        if isfile("/sys/devices/system/cpu/cpu0/cache/index3/size")
            l3_size_str = read(`cat /sys/devices/system/cpu/cpu0/cache/index3/size`, String)
            info["L3_cache"] = strip(l3_size_str)
        end
        
    catch
        info["detected"] = false
        info["note"] = "Using default cache size assumptions"
    end
    
    # Add constants used in optimization
    info["constants"] = Dict(
        "L1_CACHE_SIZE" => L1_CACHE_SIZE,
        "L2_CACHE_SIZE" => L2_CACHE_SIZE,
        "L3_CACHE_SIZE" => L3_CACHE_SIZE,
        "CACHE_LINE_SIZE" => CACHE_LINE_SIZE,
        "TILE_SIZE_L1" => TILE_SIZE_L1,
        "TILE_SIZE_L2" => TILE_SIZE_L2
    )
    
    return info
end

# Benchmark and auto-tune cache parameters on module load
function __init_cache_optimization__()
    # Measure memory bandwidth to validate cache assumptions
    bandwidth = memory_bandwidth_benchmark(10^5, 50)  # Quick benchmark
    
    if bandwidth < 5.0  # Less than 5 GB/s suggests memory-bound system
        # Adjust tile sizes for memory-bound systems
        global TILE_SIZE_L2 = 64
        global TILE_SIZE_L1 = 16
    elseif bandwidth > 50.0  # High-bandwidth system
        # Use larger tiles for high-bandwidth systems
        global TILE_SIZE_L2 = 256
        global TILE_SIZE_L1 = 64
    end
    
    @info "Cache optimization initialized" bandwidth_gb_s=bandwidth
end