"""
Smoke test - Basic sanity checks that the package loads and key components work
This is a lightweight test that should run quickly in CI
"""

println("Testing: Smoke test (basic functionality)")

using Test

@testset "Package Loading" begin
    println("  Loading PencilFlows...")

    loaded = false
    try
        if !isdefined(Main, :PencilFlows)
            include(joinpath(@__DIR__, "..", "src", "PencilFlows.jl"))
            using .PencilFlows
        end
        loaded = true
        println("    ✓ Package loaded successfully")
    catch e
        println("    ✗ Failed to load PencilFlows: $e")
        loaded = false
    end

    @test loaded
end

@testset "Core Dependencies" begin
    println("  Checking core dependencies...")

    # Check that key packages are available
    @test begin
        try
            using MPI
            using FFTW
            using LinearAlgebra
            using PencilFFTs
            using PencilArrays
            true
        catch e
            println("Dependency check failed: $e")
            false
        end
    end

    println("    ✓ All core dependencies present")
end

@testset "MPI Initialization" begin
    println("  Initializing MPI...")
    using MPI

    if !MPI.Initialized()
        MPI.Init()
    end

    @test MPI.Initialized()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    @test nprocs >= 1
    @test rank >= 0
    @test rank < nprocs

    println("    ✓ MPI initialized (rank $rank of $nprocs)")
end

@testset "Key Exports Available" begin
    println("  Checking key exports...")

    # Core functionality
    @test isdefined(PencilFlows, :PencilDecomposition)
    @test isdefined(PencilFlows, :init_pencil_decomposition)

    # Transform operations
    @test isdefined(PencilFlows, :create_transform_plans_2d)
    @test isdefined(PencilFlows, :ddx!)
    @test isdefined(PencilFlows, :ddy!)

    # Poisson solver
    @test isdefined(PencilFlows, :make_poisson_plan)
    @test isdefined(PencilFlows, :solve_poisson!)

    # Boundary conditions
    @test isdefined(PencilFlows, :BoundaryCondition)
    @test isdefined(PencilFlows, :NO_SLIP)
    @test isdefined(PencilFlows, :FREE_SLIP)

    # Symbolic interface
    @test isdefined(PencilFlows, :SymbolicProblem)
    @test isdefined(PencilFlows, :build_problem!)

    println("    ✓ All key exports available")
end

@testset "Type Definitions" begin
    println("  Checking type definitions...")

    # Check that key types are properly defined
    @test isdefined(PencilFlows, :BoundaryCondition)
    @test isdefined(PencilFlows, :PencilDecomposition)

    if isdefined(PencilFlows, :BoundaryCondition)
        @test PencilFlows.BoundaryCondition isa Type
    end
    if isdefined(PencilFlows, :PencilDecomposition)
        @test PencilFlows.PencilDecomposition isa Type
    end

    println("    ✓ Key types are properly defined")
end

@testset "Simple Math Operations" begin
    println("  Testing basic array operations...")

    # Test simple array operations that don't require MPI decomposition
    arr = rand(8, 8, 8)

    @test size(arr) == (8, 8, 8)
    @test eltype(arr) == Float64

    # Test FFTW works
    using FFTW
    fft_arr = fft(arr)
    @test size(fft_arr) == (8, 8, 8)
    @test eltype(fft_arr) == ComplexF64

    # Test inverse
    ifft_arr = ifft(fft_arr)
    @test isapprox(real.(ifft_arr), arr, rtol=1e-10)

    println("    ✓ Basic math operations work")
end

@testset "Grid and Domain Setup" begin
    println("  Testing grid setup...")

    # Test creating grid parameters
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 2π, 2π, 1.0
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz

    @test dx > 0
    @test dy > 0
    @test dz > 0

    println("    ✓ Grid setup works")
end

println("\n✓ Smoke test passed - basic functionality verified!")
println("  Ready for more comprehensive testing")
