# Performance Benchmarking Utilities for PencilFlows.jl

using BenchmarkTools
using Statistics
using Printf

export BenchmarkSuite, benchmark_tensor_ops, benchmark_derivatives, benchmark_multigrid
export memory_profile, performance_report

"""
    BenchmarkSuite

Comprehensive benchmarking suite for PencilFlows.jl performance analysis.
"""
struct BenchmarkSuite
    name::String
    results::Dict{String, Any}
    metadata::Dict{String, Any}
end

BenchmarkSuite(name::String) = BenchmarkSuite(name, Dict{String, Any}(), Dict{String, Any}())

"""
    benchmark_tensor_ops(; Nx=64, Ny=64, Nz=32, warmup=true)

Benchmark tensor operations from tensor_helpers.jl
"""
function benchmark_tensor_ops(; Nx=64, Ny=64, Nz=32, warmup=true)
    println("Benchmarking tensor operations...")
    
    # Create test data
    U = (randn(Nx, Ny, Nz), randn(Nx, Ny, Nz), randn(Nx, Ny, Nz))
    q_grad = (randn(Nx, Ny, Nz), randn(Nx, Ny, Nz), randn(Nx, Ny, Nz))
    U_grad = ((randn(Nx, Ny, Nz), randn(Nx, Ny, Nz), randn(Nx, Ny, Nz)),
              (randn(Nx, Ny, Nz), randn(Nx, Ny, Nz), randn(Nx, Ny, Nz)),
              (randn(Nx, Ny, Nz), randn(Nx, Ny, Nz), randn(Nx, Ny, Nz)))
    R = (zeros(Nx, Ny, Nz), zeros(Nx, Ny, Nz), zeros(Nx, Ny, Nz))
    Rq = zeros(Nx, Ny, Nz)
    flux_div = (zeros(Nx, Ny, Nz), zeros(Nx, Ny, Nz), zeros(Nx, Ny, Nz))
    div_U = zeros(Nx, Ny, Nz)
    lapU = (randn(Nx, Ny, Nz), randn(Nx, Ny, Nz), randn(Nx, Ny, Nz))
    
    results = Dict{String, BenchmarkTools.Trial}()
    
    if warmup
        println("  Warming up...")
        advect_scalar!(Rq, U, q_grad)
        advect_vector!(R, U, U_grad)
        add_coriolis_fplane!(R, U, 1e-4)
        add_diffusion3!(R, lapU, 1e-6)
    end
    
    # Benchmark individual operations
    println("  Scalar advection...")
    results["scalar_advection"] = @benchmark advect_scalar!($Rq, $U, $q_grad)
    
    println("  Vector advection...")
    results["vector_advection"] = @benchmark advect_vector!($R, $U, $U_grad)
    
    println("  Conservative advection...")
    compute_momentum_flux_divergence!(flux_div, U, U_grad)
    compute_velocity_divergence!(div_U, U_grad)
    results["conservative_advection"] = @benchmark advect_vector_conservative!($R, $U, $flux_div, $div_U)
    
    println("  Momentum flux divergence...")
    results["momentum_flux_div"] = @benchmark compute_momentum_flux_divergence!($flux_div, $U, $U_grad)
    
    println("  Coriolis force...")
    results["coriolis"] = @benchmark add_coriolis_fplane!($R, $U, 1e-4)
    
    println("  Viscous diffusion...")
    results["diffusion"] = @benchmark add_diffusion3!($R, $lapU, 1e-6)
    
    return results
end

"""
    benchmark_derivatives(decomp, fields, grid; warmup=true)

Benchmark derivative computations in pencil decomposition
"""
function benchmark_derivatives(decomp, fields, grid; warmup=true)
    println("Benchmarking derivative operations...")
    
    # Create output arrays
    dudx = similar(fields.u_z)
    dudy = similar(fields.u_z)
    dvdx = similar(fields.u_z)
    dvdy = similar(fields.u_z)
    dwdx = similar(fields.u_z)
    dwdy = similar(fields.u_z)
    dpdx = similar(fields.u_z)
    dpdy = similar(fields.u_z)
    
    # Initialize input fields with test data
    fill!(fields.u_z, 1.0)
    fill!(fields.v_z, 1.0)
    fill!(fields.w_z, 0.0)
    fill!(fields.p_z, 0.0)
    
    results = Dict{String, BenchmarkTools.Trial}()
    
    if warmup
        println("  Warming up...")
        compute_horizontal_derivatives_2d!(
            dudx, dudy, dvdx, dvdy, dwdx, dwdy, dpdx, dpdy,
            fields.u_z, fields.v_z, fields.w_z, fields.p_z, 
            fields, decomp, grid
        )
    end
    
    println("  Horizontal derivatives...")
    results["horizontal_derivatives"] = @benchmark compute_horizontal_derivatives_2d!(
        $dudx, $dudy, $dvdx, $dvdy, $dwdx, $dwdy, $dpdx, $dpdy,
        $(fields.u_z), $(fields.v_z), $(fields.w_z), $(fields.p_z), 
        $fields, $decomp, $grid
    )
    
    # Benchmark individual components
    kx = fftfreq(decomp.Nx_global, 2π / grid.dx)
    ky = fftfreq(decomp.Ny_global, 2π / grid.dy)
    
    println("  X-derivatives...")
    results["x_derivatives"] = @benchmark compute_x_derivatives_optimized!(
        $dudx, $dvdx, $dwdx, $dpdx, 
        $(fields.u_z), $(fields.v_z), $(fields.w_z), $(fields.p_z),
        $fields, $decomp, $kx
    )
    
    println("  Y-derivatives...")
    results["y_derivatives"] = @benchmark compute_y_derivatives_optimized!(
        $dudy, $dvdy, $dwdy, $dpdy,
        $(fields.u_z), $(fields.v_z), $(fields.w_z), $(fields.p_z),
        $fields, $decomp, $ky
    )
    
    return results
end

"""
    benchmark_multigrid(plan, u, rhs; warmup=true)

Benchmark multigrid solver performance
"""
function benchmark_multigrid(plan, u, rhs; warmup=true)
    println("Benchmarking multigrid solver...")
    
    u_test = copy(u)
    results = Dict{String, BenchmarkTools.Trial}()
    
    if warmup
        println("  Warming up...")
        mg_solve!(u_test, rhs, plan; cycles=1)
        u_test .= 0  # Reset
    end
    
    println("  Single V-cycle...")
    results["single_vcycle"] = @benchmark mg_solve!($u_test, $rhs, $plan; cycles=1) setup=(fill!($u_test, 0))
    
    println("  Two V-cycles...")
    results["two_vcycles"] = @benchmark mg_solve!($u_test, $rhs, $plan; cycles=2) setup=(fill!($u_test, 0))
    
    # Benchmark individual MG operations if possible
    if length(plan.levels) > 1
        lev = plan.levels[1]
        u_fine = zeros(eltype(u), lev.Nx, lev.Ny, lev.Nz)
        f_fine = copy(u_fine)
        r_fine = copy(u_fine)
        
        lev_coarse = plan.levels[2]
        u_coarse = zeros(eltype(u), lev_coarse.Nx, lev_coarse.Ny, lev.Nz)
        f_coarse = copy(u_coarse)
        
        println("  Restriction...")
        results["restriction"] = @benchmark _restrict!($f_coarse, $r_fine)
        
        println("  Prolongation...")
        results["prolongation"] = @benchmark _prolongate_add!($u_fine, $u_coarse)
        
        println("  Smoothing (3 iterations)...")
        results["smoothing"] = @benchmark _smooth_jacobi!($u_fine, $f_fine, $lev; iters=3) setup=(fill!($u_fine, 0))
    end
    
    return results
end

"""
    memory_profile(func, args...; samples=5)

Profile memory allocations for a function call
"""
function memory_profile(func, args...; samples=5)
    # Initial run to compile
    func(args...)
    
    # Collect allocation statistics
    alloc_samples = Float64[]
    time_samples = Float64[]
    
    for _ in 1:samples
        stats = @timed func(args...)
        push!(alloc_samples, stats.bytes / 1024^2)  # MB
        push!(time_samples, stats.time * 1000)      # ms
    end
    
    return (
        mean_alloc_mb = mean(alloc_samples),
        std_alloc_mb = std(alloc_samples),
        mean_time_ms = mean(time_samples),
        std_time_ms = std(time_samples),
        samples = samples
    )
end

"""
    performance_report(results::Dict)

Generate a formatted performance report
"""
function performance_report(results::Dict)
    println("\n" * "="^80)
    println("PERFORMANCE BENCHMARK REPORT")
    println("="^80)
    
    for (category, benchmarks) in results
        println("\n$category:")
        println("-" * "^" * string(length(category)+1))
        
        if isa(benchmarks, Dict)
            for (name, trial) in benchmarks
                if isa(trial, BenchmarkTools.Trial)
                    t_min = minimum(trial).time / 1e6  # ms
                    t_med = median(trial).time / 1e6   # ms
                    allocs = trial.allocs
                    bytes_mb = trial.memory / 1024^2   # MB
                    
                    @printf "  %-25s: %8.3f ms (min) | %8.3f ms (med) | %6d allocs | %8.2f MB\n" name t_min t_med allocs bytes_mb
                elseif isa(trial, NamedTuple) # memory profile result
                    @printf "  %-25s: %8.3f ± %6.3f ms | %8.2f ± %6.2f MB\n" name trial.mean_time_ms trial.std_time_ms trial.mean_alloc_mb trial.std_alloc_mb
                end
            end
        end
    end
    
    println("\n" * "="^80)
end

"""
    run_comprehensive_benchmark(; problem_sizes=[(32,32,16), (64,64,32), (128,128,64)])

Run comprehensive benchmarks across different problem sizes
"""
function run_comprehensive_benchmark(; problem_sizes=[(32,32,16), (64,64,32), (128,128,64)])
    all_results = Dict{String, Any}()
    
    println("Running comprehensive PencilFlows.jl performance benchmarks")
    println("Problem sizes: $problem_sizes")
    
    for (i, (Nx, Ny, Nz)) in enumerate(problem_sizes)
        println("\n" * "="^60)
        println("Problem size $i: $Nx × $Ny × $Nz")
        println("="^60)
        
        size_key = "$(Nx)x$(Ny)x$(Nz)"
        all_results[size_key] = Dict{String, Any}()
        
        # Tensor operations benchmark
        all_results[size_key]["tensor_ops"] = benchmark_tensor_ops(Nx=Nx, Ny=Ny, Nz=Nz)
        
        # Memory profiling for key operations
        U = (randn(Nx, Ny, Nz), randn(Nx, Ny, Nz), randn(Nx, Ny, Nz))
        q_grad = (randn(Nx, Ny, Nz), randn(Nx, Ny, Nz), randn(Nx, Ny, Nz))
        Rq = zeros(Nx, Ny, Nz)
        
        all_results[size_key]["memory_profiles"] = Dict(
            "scalar_advection" => memory_profile(advect_scalar!, Rq, U, q_grad)
        )
        
        println("Completed benchmarks for size $Nx × $Ny × $Nz")
    end
    
    # Generate comprehensive report
    performance_report(all_results)
    
    return all_results
end