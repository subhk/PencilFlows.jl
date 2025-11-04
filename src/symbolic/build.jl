# Building the discretized problem and discretization helpers

function build_problem!(prob::SymbolicProblem; Nx::Int=64, Ny::Int=64, Nz::Int=64)
    pencilflow_header()
    BANNER_SHOWN[] = false
    equation_summary(prob)
    if prob.domain !== nothing
        show_build_progress("Domain setup", "Using domain-based grid specification")
    else
        if isempty(prob.grid_points)
            prob.grid_points[:x] = Nx; prob.grid_points[:y] = Ny; prob.grid_points[:z] = Nz
        end
        x_basis = Fourier(:x, (0.0, 2π))
        y_basis = Fourier(:y, (0.0, 2π))
        z_basis = FiniteDifference(:z, (0.0, 1.0))
        prob.domain = Domain((x_basis, Nx), (y_basis, Ny), (z_basis, Nz))
        show_build_progress("Domain setup", "Created default domain from grid specification")
    end
    build_discretization!(prob)
    disc = prob.discretization
    Nx_final = length(disc.grid_x); Ny_final = length(disc.grid_y); Nz_final = length(disc.grid_z)
    show_build_progress("Problem built successfully!", "Grid: $Nx_final x $Ny_final x $Nz_final")
    domain_summary(prob)
    println()
    printstyled("                   BUILD COMPLETE                           ", color=:green, bold=true)
    println()
    printstyled(" Fields: ", color=:blue, bold=true); printstyled("$(length(prob.fields))", color=:green)
    printstyled("  Equations: ", color=:blue, bold=true); printstyled("$(length(prob.equations))", color=:green)
    printstyled("  BCs: ", color=:blue, bold=true); printstyled("$(length(prob.boundary_conditions))", color=:green)
    println()
    return prob
end

function build_discretization!(prob::SymbolicProblem)
    show_build_progress("Building discretization", "Using PencilFlow infrastructure")
    prob.discretization = DiscretizationInfo()
    disc = prob.discretization
    if prob.domain === nothing
        error("Domain must be specified before building discretization")
    end
    build_computational_grids!(prob)
    initialize_pencil_infrastructure!(prob)
    setup_pencil_fft_plans!(prob)
    initialize_pencil_workspaces!(prob)
    initialize_extended_components!(prob)
    build_differential_operators!(prob)
    separate_equation_terms!(prob)
    validate_imex_separation!(prob)
    apply_boundary_conditions!(prob)
    grid_str = "$(length(disc.grid_x)) x $(length(disc.grid_y)) x $(length(disc.grid_z))"
    show_build_progress("Discretization complete", "Grid: $grid_str")
    if disc.pencil_decomposition !== nothing
        proc_str = "$(disc.pencil_decomposition.P1) x $(disc.pencil_decomposition.P2) process grid"
        show_build_progress("Parallel setup", "$proc_str ($(disc.pencil_decomposition.nprocs) processes)")
    end
    show_build_progress("Infrastructure ready", "FFT plans, Poisson solver, workspaces initialized")
    return nothing
end

