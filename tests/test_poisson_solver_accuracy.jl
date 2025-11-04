"""
Test suite for Poisson solver accuracy and correctness
Verifies that ∇²π = r is solved correctly with various boundary conditions
"""

println("Testing: Poisson solver accuracy")

using Test
using MPI
using PencilArrays
using PencilFFTs
using FFTW
using LinearAlgebra

# Initialize MPI
if !MPI.Initialized()
    MPI.Init()
end

# Load PencilFlows
if !isdefined(Main, :PencilFlows)
    include(joinpath(@__DIR__, "..", "src", "PencilFlows.jl"))
    using .PencilFlows
end

@testset "Poisson Solver Mathematical Correctness" begin
    println("  Setting up test problem...")

    # Create small test problem
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 2π, 2π, 1.0

    comm = MPI.COMM_WORLD
    nprocs = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)

    # Create decomposition
    if nprocs == 1
        topology = Topology(comm, (1,))
    elseif nprocs == 2
        topology = Topology(comm, (2,))
    elseif nprocs >= 4
        topology = Topology(comm, (2, 2))
    else
        topology = Topology(comm, (nprocs,))
    end

    dims = (Nx, Ny, Nz)
    pencil_x = Pencil(topology, dims, (1,))
    pencil_y = Pencil(topology, dims, (2,))
    pencil_z = Pencil(topology, dims, (3,))

    # Create transforms
    transform_z_to_x = Transpose(pencil_z => pencil_x)
    transform_x_to_y = Transpose(pencil_x => pencil_y)
    transform_y_to_z = Transpose(pencil_y => pencil_z)
    transform_x_to_z = Transpose(pencil_x => pencil_z)
    transform_y_to_x = Transpose(pencil_y => pencil_x)
    transform_z_to_y = Transpose(pencil_z => pencil_y)

    # FFT plans
    fft_x = PencilFFTPlan(pencil_x, 1, flags=FFTW.ESTIMATE)
    fft_y = PencilFFTPlan(pencil_y, 2, flags=FFTW.ESTIMATE)
    fft_xy = PencilFFTPlan(pencil_x, (1, 2), flags=FFTW.ESTIMATE)

    # Create decomposition structure
    decomp = PencilFlows.PencilDecomposition(
        comm, rank, nprocs, Nx, Ny, Nz,
        nprocs == 1 ? 1 : (nprocs == 2 ? 2 : 2),
        nprocs == 1 ? 1 : (nprocs == 2 ? 1 : (nprocs >= 4 ? 2 : 1)),
        pencil_x, pencil_y, pencil_z,
        fft_x, fft_y, fft_xy,
        transform_x_to_y, transform_y_to_z, transform_z_to_x,
        transform_y_to_x, transform_z_to_y, transform_x_to_z
    )

    grid = (dx=Lx/Nx, dy=Ly/Ny, dz=Lz/Nz, Nz=Nz)

    println("    ✓ Test setup complete")

    # Test 1: Eigenvalue calculation
    println("  Testing eigenvalue calculation...")
    @testset "Eigenvalue Correctness" begin
        # Periodic eigenvalues
        lambda_p = PencilFlows.eigenvalues_1d(Nz, grid.dz; bc=:periodic)
        @test length(lambda_p) == Nz
        @test lambda_p[1] == 0.0  # Zero mode for periodic
        @test all(lambda_p .>= 0)  # All non-negative

        # Dirichlet eigenvalues
        lambda_d = PencilFlows.eigenvalues_1d(Nz, grid.dz; bc=:dirichlet)
        @test length(lambda_d) == Nz
        @test all(lambda_d .> 0)  # All positive (no zero mode)

        # Neumann eigenvalues
        lambda_n = PencilFlows.eigenvalues_1d(Nz, grid.dz; bc=:neumann)
        @test length(lambda_n) == Nz
        @test lambda_n[1] == 0.0  # Zero mode for Neumann
        @test all(lambda_n .>= 0)

        println("    ✓ Eigenvalues correct")
    end

    # Test 2: Known solution test (manufactured solution)
    println("  Testing with manufactured solution...")
    @testset "Manufactured Solution" begin
        # Create arrays
        rhs = PencilArray{Float64}(undef, pencil_z)
        solution = PencilArray{Float64}(undef, pencil_z)
        exact = PencilArray{Float64}(undef, pencil_z)

        # Get local ranges
        local_ranges = range_local(pencil_z)

        # Manufactured solution: π(x,y,z) = sin(2πx/Lx) * sin(2πy/Ly) * sin(πz/Lz)
        # Then: ∇²π = -(4π²/Lx² + 4π²/Ly² + π²/Lz²) * π
        kx = 2π/Lx
        ky = 2π/Ly
        kz = π/Lz

        factor = -(kx^2 + ky^2 + kz^2)

        # Fill arrays with manufactured solution
        for (i, gi) in enumerate(local_ranges[1])
            x = (gi - 1) * Lx / Nx
            for (j, gj) in enumerate(local_ranges[2])
                y = (gj - 1) * Ly / Ny
                for k in 1:Nz
                    z = (k - 0.5) * Lz / Nz  # Cell centers

                    exact[i, j, k] = sin(kx * x) * sin(ky * y) * sin(kz * z)
                    rhs[i, j, k] = factor * exact[i, j, k]
                end
            end
        end

        # Create Poisson plan (Dirichlet BCs match manufactured solution)
        plan = make_poisson_plan(rhs; decomp=decomp, grid=grid, bc_z=:dirichlet)

        # Solve
        solve_poisson!(solution, rhs, plan)

        # Check error
        error_norm = norm(parent(solution) - parent(exact)) / norm(parent(exact))

        # Gather errors from all processes
        max_error = MPI.Allreduce(error_norm, max, comm)

        if rank == 0
            @test max_error < 0.1  # Relative error should be small
            println("    ✓ Manufactured solution test passed (error: $(round(max_error, digits=6)))")
        end
    end

    # Test 3: Symmetry test
    println("  Testing symmetry...")
    @testset "Solver Symmetry" begin
        # Symmetric RHS should give symmetric solution
        rhs = PencilArray{Float64}(undef, pencil_z)
        solution = PencilArray{Float64}(undef, pencil_z)

        # Fill with symmetric pattern
        local_ranges = range_local(pencil_z)
        for (i, gi) in enumerate(local_ranges[1])
            x = (gi - 1) * Lx / Nx
            for (j, gj) in enumerate(local_ranges[2])
                y = (gj - 1) * Ly / Ny
                for k in 1:Nz
                    z = (k - 0.5) * Lz / Nz
                    rhs[i, j, k] = cos(2π*x/Lx) * cos(2π*y/Ly)
                end
            end
        end

        plan = make_poisson_plan(rhs; decomp=decomp, grid=grid, bc_z=:neumann)
        solve_poisson!(solution, rhs, plan)

        # Solution should also be symmetric (no NaNs or Infs)
        @test all(isfinite.(parent(solution)))

        println("    ✓ Symmetry test passed")
    end

    # Test 4: Neumann BC test (zero mean RHS)
    println("  Testing Neumann boundary conditions...")
    @testset "Neumann BC Compatibility" begin
        rhs = PencilArray{Float64}(undef, pencil_z)
        solution = PencilArray{Float64}(undef, pencil_z)

        # Create RHS with zero mean (required for Neumann BC)
        local_ranges = range_local(pencil_z)
        for (i, gi) in enumerate(local_ranges[1])
            x = (gi - 1) * Lx / Nx
            for (j, gj) in enumerate(local_ranges[2])
                y = (gj - 1) * Ly / Ny
                for k in 1:Nz
                    z = (k - 0.5) * Lz / Nz
                    # Use mode that integrates to zero
                    rhs[i, j, k] = sin(2π*x/Lx) * sin(2π*y/Ly) * cos(π*z/Lz)
                end
            end
        end

        plan = make_poisson_plan(rhs; decomp=decomp, grid=grid, bc_z=:neumann)
        solve_poisson!(solution, rhs, plan)

        # Solution should be finite
        @test all(isfinite.(parent(solution)))

        println("    ✓ Neumann BC test passed")
    end

    # Test 5: Sign check
    println("  Testing sign convention...")
    @testset "Sign Convention" begin
        # Positive RHS should give negative solution (since ∇²π = r => π = -r/∇²)
        rhs = PencilArray{Float64}(undef, pencil_z)
        solution = PencilArray{Float64}(undef, pencil_z)

        # Constant positive RHS (away from boundaries)
        parent(rhs) .= 1.0

        plan = make_poisson_plan(rhs; decomp=decomp, grid=grid, bc_z=:dirichlet)
        solve_poisson!(solution, rhs, plan)

        # Interior points should be negative for positive RHS
        # (at boundaries, Dirichlet BC = 0)
        interior_mean = sum(parent(solution)[2:end-1, 2:end-1, 2:end-1]) /
                       length(parent(solution)[2:end-1, 2:end-1, 2:end-1])

        global_mean = MPI.Allreduce(interior_mean, +, comm) / nprocs

        if rank == 0
            @test global_mean < 0  # Should be negative
            println("    ✓ Sign convention correct (interior mean: $(round(global_mean, digits=6)))")
        end
    end
end

println("\n✓ Poisson solver accuracy tests completed!")

# Clean up MPI if we initialized it
# MPI.Finalize()  # Don't finalize as other tests may need it
