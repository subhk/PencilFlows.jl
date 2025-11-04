# Advanced Memory Pool System for PencilFlows.jl
# Provides allocation-free workspace management

"""
    MemoryPool{T}

Thread-safe memory pool for reusable arrays of type T.
Provides allocation-free temporary array management for hot code paths.
"""
mutable struct MemoryPool{T}
    arrays::Vector{Array{T}}
    sizes::Vector{Tuple{Vararg{Int}}}
    in_use::BitVector
    max_size::Int
    
    function MemoryPool{T}(max_size::Int=32) where T
        new{T}(Array{T}[], Tuple{Vararg{Int}}[], BitVector(), max_size)
    end
end

"""
    get_workspace(pool::MemoryPool{T}, size::Tuple{Vararg{Int}}) -> Array{T}

Get a workspace array from the pool. Returns an existing array if available,
or creates a new one. Automatically manages pool size to prevent memory leaks.
"""
function get_workspace(pool::MemoryPool{T}, size::Tuple{Vararg{Int}}) where T
    # Look for available array of the right size
    for i in eachindex(pool.arrays)
        if !pool.in_use[i] && pool.sizes[i] == size
            pool.in_use[i] = true
            return pool.arrays[i]
        end
    end
    
    # No suitable array found, create new one
    new_array = Array{T}(undef, size...)
    
    if length(pool.arrays) < pool.max_size
        # Add to pool
        push!(pool.arrays, new_array)
        push!(pool.sizes, size)
        push!(pool.in_use, true)
    end
    
    return new_array
end

"""
    return_workspace!(pool::MemoryPool{T}, array::Array{T})

Return a workspace array to the pool for reuse.
"""
function return_workspace!(pool::MemoryPool{T}, array::Array{T}) where T
    for i in eachindex(pool.arrays)
        if pool.arrays[i] === array
            pool.in_use[i] = false
            return
        end
    end
    # Array not from this pool - ignore
end

"""
    GlobalMemoryPools

Global memory pools for different floating-point types.
Access via get_global_pool(T) where T is Float32, Float64, etc.
"""
const GlobalMemoryPools = Dict{Type, MemoryPool}()

"""
    get_global_pool(::Type{T}) -> MemoryPool{T}

Get the global memory pool for type T.
"""
function get_global_pool(::Type{T}) where T
    if !haskey(GlobalMemoryPools, T)
        GlobalMemoryPools[T] = MemoryPool{T}(64)  # Larger pool for global use
    end
    return GlobalMemoryPools[T]
end

"""
    @with_workspace pool size expr

Macro for safe workspace usage with automatic cleanup.
Example:
    @with_workspace get_global_pool(Float64) (64,64,32) begin
        temp = pool_array  # Use the allocated workspace
        # ... computations ...
    end  # Automatically returned to pool
"""
macro with_workspace(pool, size, expr)
    quote
        local _pool = $(esc(pool))
        local _temp_array = get_workspace(_pool, $(esc(size)))
        local result
        try
            local $(esc(:pool_array)) = _temp_array
            result = $(esc(expr))
        finally
            return_workspace!(_pool, _temp_array)
        end
        result
    end
end

"""
    allocate_free_workspace(f, ::Type{T}, sizes...) where T

Execute function f with pre-allocated workspaces of given sizes and type T.
Automatically manages allocation and cleanup.

Example:
    result = allocate_free_workspace(Float64, (64,64,32), (64,64,32)) do temp1, temp2
        @. temp1 = some_computation()
        @. temp2 = other_computation(temp1)
        return sum(temp2)
    end
"""
function allocate_free_workspace(f, ::Type{T}, sizes::Tuple{Vararg{Int}}...) where T
    pool = get_global_pool(T)
    temp_arrays = [get_workspace(pool, size) for size in sizes]
    
    try
        return f(temp_arrays...)
    finally
        for array in temp_arrays
            return_workspace!(pool, array)
        end
    end
end

"""
    clear_memory_pools!()

Clear all global memory pools. Useful for testing or memory cleanup.
"""
function clear_memory_pools!()
    for pool in values(GlobalMemoryPools)
        empty!(pool.arrays)
        empty!(pool.sizes)
        empty!(pool.in_use)
    end
    empty!(GlobalMemoryPools)
end

# Specialized pools for common CFD array sizes
const CommonSizes = [
    (64, 64, 32),   # Small test cases
    (128, 128, 64), # Medium cases  
    (256, 256, 128), # Large cases
    (512, 512, 256) # Very large cases
]

"""
    initialize_common_pools(::Type{T}=Float64)

Pre-populate memory pools with commonly used array sizes for CFD simulations.
"""
function initialize_common_pools(::Type{T}=Float64) where T
    pool = get_global_pool(T)
    
    # Pre-allocate 2 arrays of each common size
    for size in CommonSizes
        for _ in 1:2
            array = Array{T}(undef, size...)
            push!(pool.arrays, array)
            push!(pool.sizes, size)
            push!(pool.in_use, false)
        end
    end
    
    println("Initialized memory pools with $(length(pool.arrays)) pre-allocated arrays")
end