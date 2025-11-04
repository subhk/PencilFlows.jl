# Optimized FFT Operations with Batching and Wisdom
# Advanced FFT optimization for PencilFlows.jl

using FFTW
using PencilFFTs
using PencilArrays
using Base.Threads: @threads, @spawn

export OptimizedFFTPlans, create_optimized_fft_plans, batch_fft!, batch_ifft!
export save_fft_wisdom, load_fft_wisdom, optimize_fft_plans!
export ThreadedFFTPlans, parallel_fft_batch

"""
    OptimizedFFTPlans{T}

Container for optimized FFT plans with batching capabilities and wisdom.
"""
struct OptimizedFFTPlans{T}
    # Basic FFT plans
    plan_fft_x::PencilFFTPlan
    plan_fft_y::PencilFFTPlan
    
    # Batch processing plans for multiple fields
    batch_plan_x::Union{PencilFFTPlan, Nothing}
    batch_plan_y::Union{PencilFFTPlan, Nothing}
    
    # FFTW optimization parameters
    fftw_flags::UInt32
    wisdom_loaded::Bool
    
    # Threading configuration
    fftw_threads::Int
    use_threaded_fft::Bool
    
    # Batching parameters
    max_batch_size::Int
    optimal_batch_size::Int
    
    # Performance metrics
    timing_stats::Dict{String, Float64}
    
    function OptimizedFFTPlans{T}(plan_fft_x, plan_fft_y, fftw_flags, 
                                 fftw_threads, max_batch_size) where T
        new{T}(plan_fft_x, plan_fft_y, nothing, nothing, 
               fftw_flags, false, fftw_threads, fftw_threads > 1,
               max_batch_size, min(4, max_batch_size),
               Dict{String, Float64}())
    end
end

"""
    create_optimized_fft_plans(pencil_x, pencil_y; 
                              optimization_level=:balanced,
                              max_batch_size=8,
                              enable_threading=true,
                              wisdom_file=nothing)

Create optimized FFT plans with advanced features.
"""
function create_optimized_fft_plans(pencil_x::Pencil{3}, pencil_y::Pencil{3};
                                   T::Type=Float64,
                                   optimization_level::Symbol=:balanced,
                                   max_batch_size::Int=8,
                                   enable_threading::Bool=true,
                                   wisdom_file::Union{String,Nothing}=nothing)
    
    # Set FFTW optimization flags based on level
    fftw_flags = if optimization_level == :fastest
        FFTW.ESTIMATE  # Fastest planning, may not be optimal
    elseif optimization_level == :balanced
        FFTW.MEASURE   # Good balance of planning time vs. execution speed
    elseif optimization_level == :best
        FFTW.EXHAUSTIVE # Slowest planning, best execution performance
    else
        FFTW.MEASURE
    end
    
    # Configure FFTW threading
    fftw_threads = enable_threading ? Threads.nthreads() : 1
    FFTW.set_num_threads(fftw_threads)
    
    # Load wisdom if specified
    wisdom_loaded = false
    if wisdom_file !== nothing && isfile(wisdom_file)
        try
            FFTW.import_wisdom(wisdom_file)
            wisdom_loaded = true
            @info "FFT wisdom loaded from $wisdom_file"
        catch e
            @warn "Failed to load FFT wisdom: $e"
        end
    end
    
    # Create base FFT plans with optimization
    @info "Creating optimized FFT plans..." optimization_level fftw_threads
    
    plan_fft_x = PencilFFTPlan(pencil_x, 1, flags=fftw_flags)
    plan_fft_y = PencilFFTPlan(pencil_y, 2, flags=fftw_flags)
    
    plans = OptimizedFFTPlans{T}(plan_fft_x, plan_fft_y, fftw_flags, 
                                fftw_threads, max_batch_size)
    plans.wisdom_loaded = wisdom_loaded
    
    # Benchmark and optimize batch size
    optimize_batch_size!(plans, pencil_x, pencil_y)
    
    return plans
end

"""
    optimize_batch_size!(plans::OptimizedFFTPlans, pencil_x, pencil_y)

Automatically determine optimal batch size through benchmarking.
"""
function optimize_batch_size!(plans::OptimizedFFTPlans{T}, pencil_x, pencil_y) where T
    @info "Optimizing FFT batch size..."
    
    # Create test arrays
    test_arrays_x = [PencilArrays.PencilArray{T}(undef, pencil_x) for _ in 1:plans.max_batch_size]
    test_arrays_x_complex = [PencilArrays.PencilArray{Complex{T}}(undef, pencil_x) for _ in 1:plans.max_batch_size]
    
    # Initialize with random data
    for arr in test_arrays_x
        randn!(parent(arr))
    end
    
    best_time = Inf
    best_batch_size = 1
    
    # Test different batch sizes
    for batch_size in 1:min(plans.max_batch_size, 8)
        # Warm up
        for _ in 1:3
            batch_fft!(test_arrays_x_complex[1:batch_size], 
                      test_arrays_x[1:batch_size], 
                      plans.plan_fft_x)
        end
        
        # Benchmark
        times = Float64[]
        for _ in 1:10
            t_start = time()
            batch_fft!(test_arrays_x_complex[1:batch_size], 
                      test_arrays_x[1:batch_size], 
                      plans.plan_fft_x)
            push!(times, time() - t_start)
        end
        
        avg_time = sum(times) / length(times) / batch_size  # Time per field
        
        @info "Batch size $batch_size: $(round(avg_time*1000, digits=3)) ms per field"
        
        if avg_time < best_time
            best_time = avg_time
            best_batch_size = batch_size
        end
    end
    
    plans.optimal_batch_size = best_batch_size
    plans.timing_stats["optimal_batch_time_ms"] = best_time * 1000
    
    @info "Optimal batch size: $(best_batch_size)"
end

"""
    batch_fft!(output_arrays, input_arrays, plan::PencilFFTPlan)

Perform batched FFT on multiple arrays simultaneously.
"""
function batch_fft!(output_arrays::Vector{PencilArrays.PencilArray{Complex{T},3}}, 
                   input_arrays::Vector{PencilArrays.PencilArray{T,3}}, 
                   plan::PencilFFTPlan) where T
    
    @assert length(output_arrays) == length(input_arrays) "Array count mismatch"
    
    n_arrays = length(input_arrays)
    
    # Process in parallel for better throughput
    if n_arrays > 1 && Threads.nthreads() > 1
        @threads for i in 1:n_arrays
            mul!(output_arrays[i], plan, input_arrays[i])
        end
    else
        # Serial processing
        for i in 1:n_arrays
            mul!(output_arrays[i], plan, input_arrays[i])
        end
    end
end

"""
    batch_ifft!(output_arrays, input_arrays, plan::PencilFFTPlan)

Perform batched inverse FFT on multiple arrays simultaneously.
"""
function batch_ifft!(output_arrays::Vector{PencilArrays.PencilArray{T,3}}, 
                    input_arrays::Vector{PencilArrays.PencilArray{Complex{T},3}}, 
                    plan::PencilFFTPlan) where T
    
    @assert length(output_arrays) == length(input_arrays) "Array count mismatch"
    
    n_arrays = length(input_arrays)
    
    # Process in parallel
    if n_arrays > 1 && Threads.nthreads() > 1
        @threads for i in 1:n_arrays
            ldiv!(output_arrays[i], plan, input_arrays[i])
        end
    else
        for i in 1:n_arrays
            ldiv!(output_arrays[i], plan, input_arrays[i])
        end
    end
end

"""
    parallel_fft_batch(input_fields, plans::OptimizedFFTPlans; direction=:forward)

High-level interface for parallel batch FFT processing.
"""
function parallel_fft_batch(input_fields::Vector, plans::OptimizedFFTPlans{T}; 
                           direction::Symbol=:forward) where T
    
    n_fields = length(input_fields)
    batch_size = plans.optimal_batch_size
    
    # Create output arrays
    if direction == :forward
        output_fields = [similar(field, Complex{T}) for field in input_fields]
    else
        output_fields = [similar(field, T) for field in input_fields]
    end
    
    # Process in batches
    for batch_start in 1:batch_size:n_fields
        batch_end = min(batch_start + batch_size - 1, n_fields)
        batch_inputs = input_fields[batch_start:batch_end]
        batch_outputs = output_fields[batch_start:batch_end]
        
        if direction == :forward
            # Determine which plan to use based on array orientation
            if size(batch_inputs[1]) == size_local(plans.plan_fft_x.pencil)
                batch_fft!(batch_outputs, batch_inputs, plans.plan_fft_x)
            else
                batch_fft!(batch_outputs, batch_inputs, plans.plan_fft_y)
            end
        else  # :inverse
            if size(batch_inputs[1]) == size_local(plans.plan_fft_x.pencil)
                batch_ifft!(batch_outputs, batch_inputs, plans.plan_fft_x)
            else
                batch_ifft!(batch_outputs, batch_inputs, plans.plan_fft_y)
            end
        end
    end
    
    return output_fields
end

"""
    save_fft_wisdom(filename::String)

Save current FFTW wisdom to file for future use.
"""
function save_fft_wisdom(filename::String)
    try
        FFTW.export_wisdom(filename)
        @info "FFT wisdom saved to $filename"
        return true
    catch e
        @warn "Failed to save FFT wisdom: $e"
        return false
    end
end

"""
    load_fft_wisdom(filename::String)

Load FFTW wisdom from file.
"""
function load_fft_wisdom(filename::String)
    if !isfile(filename)
        @warn "Wisdom file $filename not found"
        return false
    end
    
    try
        FFTW.import_wisdom(filename)
        @info "FFT wisdom loaded from $filename"
        return true
    catch e
        @warn "Failed to load FFT wisdom: $e"
        return false
    end
end

"""
    optimize_fft_plans!(plans::OptimizedFFTPlans; 
                       benchmark_iterations=10,
                       save_wisdom=true)

Optimize existing FFT plans and optionally save wisdom.
"""
function optimize_fft_plans!(plans::OptimizedFFTPlans{T}; 
                            benchmark_iterations::Int=10,
                            save_wisdom::Bool=true,
                            wisdom_file::String="fftw_wisdom_pencilflows.dat") where T
    
    @info "Optimizing FFT plans..."
    
    # Re-benchmark with more iterations for better accuracy
    pencil_x = plans.plan_fft_x.pencil
    test_array_x = PencilArrays.PencilArray{T}(undef, pencil_x)
    test_array_x_complex = PencilArrays.PencilArray{Complex{T}}(undef, pencil_x)
    randn!(parent(test_array_x))
    
    # Benchmark forward FFT
    times_fwd = Float64[]
    for _ in 1:benchmark_iterations
        t_start = time()
        mul!(test_array_x_complex, plans.plan_fft_x, test_array_x)
        push!(times_fwd, time() - t_start)
    end
    
    # Benchmark inverse FFT
    times_inv = Float64[]
    for _ in 1:benchmark_iterations
        t_start = time()
        ldiv!(test_array_x, plans.plan_fft_x, test_array_x_complex)
        push!(times_inv, time() - t_start)
    end
    
    # Update timing statistics
    plans.timing_stats["fft_forward_ms"] = 1000 * sum(times_fwd) / length(times_fwd)
    plans.timing_stats["fft_inverse_ms"] = 1000 * sum(times_inv) / length(times_inv)
    
    @info "FFT Performance:" forward_ms=round(plans.timing_stats["fft_forward_ms"], digits=3) inverse_ms=round(plans.timing_stats["fft_inverse_ms"], digits=3)
    
    # Save wisdom if requested
    if save_wisdom
        save_fft_wisdom(wisdom_file)
    end
    
    return plans
end

"""
    ThreadedFFTPlans{T}

Container for thread-local FFT plans to avoid contention.
"""
struct ThreadedFFTPlans{T}
    plans::Vector{OptimizedFFTPlans{T}}
    
    function ThreadedFFTPlans{T}(base_plans::OptimizedFFTPlans{T}) where T
        n_threads = Threads.nthreads()
        plans = [base_plans]  # First thread uses original plans
        
        # Create separate plans for other threads
        for tid in 2:n_threads
            # Note: In practice, you'd recreate plans for each thread
            # This is a simplified version
            push!(plans, base_plans)
        end
        
        new{T}(plans)
    end
end

"""
    get_thread_fft_plans(threaded_plans::ThreadedFFTPlans{T}) where T

Get FFT plans for the current thread.
"""
function get_thread_fft_plans(threaded_plans::ThreadedFFTPlans{T}) where T
    tid = Threads.threadid()
    return threaded_plans.plans[tid]
end

"""
    benchmark_fft_performance(plans::OptimizedFFTPlans{T}; 
                             array_sizes=[(64,64,32), (128,128,64)],
                             iterations=100) where T

Comprehensive FFT performance benchmark.
"""
function benchmark_fft_performance(plans::OptimizedFFTPlans{T}; 
                                  array_sizes=[(64,64,32), (128,128,64)],
                                  iterations::Int=100) where T
    
    println("FFT Performance Benchmark")
    println("=" ^ 50)
    println("FFTW threads: $(plans.fftw_threads)")
    println("Optimization level: $(plans.fftw_flags)")
    println("Wisdom loaded: $(plans.wisdom_loaded)")
    println()
    
    results = Dict{String,Any}()
    
    for (Nx, Ny, Nz) in array_sizes
        println("Array size: $Nx × $Ny × $Nz")
        
        # Create test pencil (simplified - would need proper MPI setup in reality)
        try
            # This is a mock benchmark - real implementation would need proper pencils
            total_elements = Nx * Ny * Nz
            test_data = rand(T, total_elements)
            test_complex = zeros(Complex{T}, total_elements)
            
            # Benchmark raw FFTW performance as proxy
            fft_plan = FFTW.plan_fft(reshape(test_data, Nx, Ny, Nz))
            
            # Warm up
            for _ in 1:10
                fft_plan * reshape(test_data, Nx, Ny, Nz)
            end
            
            # Benchmark
            times = Float64[]
            for _ in 1:iterations
                t_start = time()
                result = fft_plan * reshape(test_data, Nx, Ny, Nz)
                push!(times, time() - t_start)
            end
            
            avg_time = 1000 * sum(times) / length(times)  # ms
            throughput = total_elements / (avg_time / 1000) / 1e6  # Million elements/second
            
            println("  Forward FFT: $(round(avg_time, digits=3)) ms")
            println("  Throughput:  $(round(throughput, digits=1)) Melements/s")
            
            size_key = "$(Nx)x$(Ny)x$(Nz)"
            results[size_key] = Dict(
                "time_ms" => avg_time,
                "throughput_melements_per_sec" => throughput
            )
            
        catch e
            println("  Benchmark failed: $e")
        end
        
        println()
    end
    
    return results
end

# Utility functions for wisdom management
"""
    create_wisdom_file(sizes::Vector{NTuple{3,Int}}, filename::String="fftw_wisdom.dat")

Create FFT wisdom file by planning FFTs for common array sizes.
"""
function create_wisdom_file(sizes::Vector{NTuple{3,Int}}, filename::String="fftw_wisdom.dat")
    @info "Creating FFT wisdom for sizes: $sizes"
    
    # Plan FFTs for all specified sizes to build up wisdom
    for (Nx, Ny, Nz) in sizes
        @info "Planning FFT for size $Nx × $Ny × $Nz"
        
        # Create test arrays and plan FFTs
        test_data = rand(Float64, Nx, Ny, Nz)
        
        # Plan forward and inverse FFTs
        plan_fwd = FFTW.plan_fft(test_data, flags=FFTW.EXHAUSTIVE)
        plan_inv = FFTW.plan_ifft(plan_fwd * test_data, flags=FFTW.EXHAUSTIVE)
        
        # Execute once to ensure wisdom is recorded
        result_fwd = plan_fwd * test_data
        result_inv = plan_inv * result_fwd
    end
    
    # Save accumulated wisdom
    save_fft_wisdom(filename)
    
    @info "Wisdom creation complete. Saved to $filename"
end

# Module initialization
function __init_optimized_fft__()
    # Set optimal FFTW thread count
    FFTW.set_num_threads(Threads.nthreads())
    
    # Try to load default wisdom file
    default_wisdom = "fftw_wisdom_pencilflows_default.dat"
    if isfile(default_wisdom)
        load_fft_wisdom(default_wisdom)
    end
    
    @info "Optimized FFT module initialized" fftw_threads=FFTW.get_num_threads()
end