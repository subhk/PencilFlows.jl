#module PoissonFFTSolver

# 
#  FFT-based Poisson solver using discrete Laplacian eigenvalues.  
#  Periodic in x,y (Fourier), Dirichlet/Neumann/
#  Periodic options in z via real-to-real sine/cosine transforms.  Works with
#  2-D (x,y) domain decomposition from PencilArrays/PencilFFTs; z is local.
# 

using MPI
using PencilFFTs
using PencilArrays
using FFTW

# export PoissonPlan, make_poisson_plan, solve_poisson!, eigenvalues_1d

#  Types 

"""
    PoissonPlan

Holds FFT plans, eigenvalues and scratch arrays for repeatedly solving
∇²π = r on a pencil-decomposed grid.

Fields
------
* `decomp`       PencilDecomposition (x,y distributed; z local)
* `bc_z`         Symbol (:dirichlet, :neumann, :periodic)
* `kx2_local`    local slice of kx^2 (Float64)
* `ky2_local`    local slice of k_y2 (Float64)
* `lambda_z`      vector of vertical eigenvalues (length Nz)
* `fft_x, fft_y` PencilFFTPlans for x and y
* `Tz_fwd, Tz_inv`  FFTW plans for z (DST/DCT/Fourier) on local columns
* `rhs_x, rhs_y`    scratch physical arrays in X/Y pencils
* `rhs_hat_x, rhs_hat_y`  scratch complex arrays in spectral space
* `pi_x, pi_y`      work arrays used during inverse transforms
"""
mutable struct PoissonPlan{D, G, PX, PY, TZF, TZI, TA, TC}
    decomp      :: D
    grid        :: G
    bc_z        :: Symbol
    bc_spec     :: Union{Nothing, Dict{Symbol, Any}}
    kx2_local   :: Vector{Float64}
    ky2_local   :: Vector{Float64}
    lambda_z     :: Vector{Float64}
    fft_x       :: PX
    fft_y       :: PY
    Tz_fwd      :: TZF
    Tz_inv      :: TZI
    rhs_x       :: TA
    rhs_y       :: TA
    rhs_hat_x   :: TC
    rhs_hat_y   :: TC
    pi_x         :: TA
    pi_y         :: TA
end

#  Eigenvalues utilities 

"""
    eigenvalues_1d(N, h; bc=:periodic)

Return vector of discrete Laplacian eigenvalues for a second-order stencil on a
uniform grid with spacing h and boundary condition `bc`.

* `:periodic` : λ_j = 4/h^2 * sin(pi j / N)^2,  j = 0:(N-1)
* `:dirichlet`: λ_j = 4/h^2 * sin(pi (j+1) / (2N+2))^2, j = 0:(N-1)
* `:neumann`  : λ_j = 4/h^2 * sin(pi j / (2N))^2,       j = 0:(N-1)

These are positive definite. We divide RHS by (kx2 + ky2 + lambda_z[m]).
"""
function eigenvalues_1d(N::Int, h::Real; bc::Symbol=:periodic)
    lambda = zeros(Float64, N)
    if bc === :periodic
        for j in 0:N-1
            lambda[j+1] = 4/h^2 * sin(pi * j / N)^2
        end
    elseif bc === :dirichlet
        for j in 0:N-1
            lambda[j+1] = 4/h^2 * sin(pi * (j+1) / (2*(N+1)))^2
        end
    elseif bc === :neumann
        for j in 0:N-1
            lambda[j+1] = 4/h^2 * sin(pi * j / (2*N))^2
        end
    else
        error("Unsupported bc: $bc")
    end
    return lambda
end

# Map bc symbol to FFTW real-to-real kinds
const _R2R_KIND = Dict(:dirichlet => (FFTW.RODFT10, FFTW.RODFT01),  # DST-I pair
                       :neumann   => (FFTW.REDFT10, FFTW.REDFT01),  # DCT-I pair
                       :periodic  => nothing)                      # use complex FFT instead

#  Plan builder 

function make_poisson_plan(rhs_z::PencilArrays.PencilArray;
                           decomp,
                           grid,
                           bc_z::Symbol = :dirichlet,
                           bc_spec::Union{Nothing, Dict{Symbol,Any}}=nothing)
    # MPI
    pencil = PencilArrays.pencil(rhs_z)
    comm = PencilArrays.get_comm(pencil)

    # FFT plans for x and y
    fft_x = PencilFFTPlan(decomp.pencil_x, 1, flags=FFTW.MEASURE)
    fft_y = PencilFFTPlan(decomp.pencil_y, 2, flags=FFTW.MEASURE)

    # Allocate scratch arrays
    TA = eltype(rhs_z)
    TC = Complex{TA}

    rhs_x     = PencilArrays.PencilArray{TA}(undef, decomp.pencil_x)
    rhs_y     = PencilArrays.PencilArray{TA}(undef, decomp.pencil_y)
    pi_x       = PencilArrays.PencilArray{TA}(undef, decomp.pencil_x)
    pi_y       = PencilArrays.PencilArray{TA}(undef, decomp.pencil_y)
    rhs_hat_x = PencilArrays.PencilArray{TC}(undef, decomp.pencil_x)
    rhs_hat_y = PencilArrays.PencilArray{TC}(undef, decomp.pencil_y)

    # Local wavenumber squares for x & y
    kx = FFTW.fftfreq(decomp.Nx_global, 2π / grid.dx)
    ky = FFTW.fftfreq(decomp.Ny_global, 2π / grid.dy)
    # Slice to local ranges - using compatibility functions
    try
        kx_loc = kx[PencilArrays.range_local(decomp.pencil_x, 1)]
        ky_loc = ky[PencilArrays.range_local(decomp.pencil_y, 2)]
    catch
        # Fallback for different PencilArrays versions
        kx_loc = kx[1:size(rhs_x, 1)]
        ky_loc = ky[1:size(rhs_y, 2)]
    end
    kx2_local = kx_loc.^2
    ky2_local = ky_loc.^2

    # Vertical eigenvalues and R2R plans (z local => standard FFTW)
    # Check for non-uniform grid - spectral methods require uniform spacing
    is_nonuniform = isa(grid.dz, AbstractVector)

    if is_nonuniform && bc_spec === nothing
        # Non-uniform grid detected but no bc_spec provided
        # Automatically create bc_spec to force tridiagonal solver path
        @warn "Non-uniform grid detected. Automatically using tridiagonal solver for exact solution. " *
              "Spectral methods (FFT/DST/DCT) require uniform spacing."

        # Create default bc_spec based on bc_z
        bc_spec = Dict{Symbol, Any}()
        if bc_z === :dirichlet
            bc_spec[:bottom_type] = :dirichlet
            bc_spec[:top_type] = :dirichlet
            bc_spec[:bottom_value] = 0.0
            bc_spec[:top_value] = 0.0
        elseif bc_z === :neumann
            bc_spec[:bottom_type] = :neumann
            bc_spec[:top_type] = :neumann
            bc_spec[:bottom_value] = 0.0  # Zero flux
            bc_spec[:top_value] = 0.0
        elseif bc_z === :periodic
            error("Non-uniform grids with periodic BC in z not supported. " *
                  "Periodic BC requires uniform spacing for spectral methods.")
        end
    end

    # For uniform grids or when using tridiagonal solver, proceed normally
    dz_eff = if isa(grid.dz, AbstractVector)
        # Use first spacing as representative (only used if bc_spec path is taken)
        grid.dz[1]
    else
        grid.dz
    end

    lambda_z = eigenvalues_1d(grid.Nz, dz_eff; bc=bc_z)

    if bc_z === :periodic
        Tz_fwd = nothing
        Tz_inv = nothing
    else
        kind_fwd, kind_inv = _R2R_KIND[bc_z]
        # Plan for a single column; we'll reuse it across all (i,j)
        tmp  = zeros(Float64, grid.Nz)
        Tz_fwd = FFTW.plan_r2r!(tmp, kind_fwd)
        Tz_inv = FFTW.plan_r2r!(tmp, kind_inv)
    end

    return PoissonPlan(decomp, grid, bc_z, bc_spec, kx2_local, ky2_local, lambda_z,
                       fft_x, fft_y, Tz_fwd, Tz_inv,
                       rhs_x, rhs_y, rhs_hat_x, rhs_hat_y,
                       pi_x, pi_y)
end

#  Solver core 

"""
    solve_poisson!(pipi, r, plan)

Solve ∇²π = r using cached plans/eigenvalues.  `π` and `r` are in Z-pencil
orientation (same as `rhs_z` used to build the plan). The solution overwrites `pipi`.

Steps
1. Z->X transpose, FFT in x
2. X->Y transpose, FFT in y
3. (Optional) R2R transform in z
4. Divide by eigenvalue sum
5. Inverse transforms back to Z pencils
"""
function solve_poisson!(pipi::PencilArrays.PencilArray, r::PencilArrays.PencilArray, plan::PoissonPlan;
                       t::Real=0.0, xnodes=nothing, ynodes=nothing)
    decomp = plan.decomp

    # 1) Z -> X and forward FFT in x
    transpose!(plan.rhs_x,  decomp.transform_z_to_x, r)
    mul!(plan.rhs_hat_x, plan.fft_x, plan.rhs_x)

    # 2) X -> Y and forward FFT in y
    transpose!(plan.rhs_y,  decomp.transform_x_to_y, plan.rhs_hat_x)  # still complex
    mul!(plan.rhs_hat_y, plan.fft_y, plan.rhs_y)

    # At this point we have R_hat(kx, ky, z_phys)
    # 3) Solve along z for each (kx,ky). If bc_spec provided, use 1D Helmholtz solver
    #    which supports Dirichlet/Neumann/Robin. Otherwise, fall back to R2R eigen solve.
    # Get local size using compatibility approach
    try
        local_size = size_local(decomp.pencil_y)   # (Nx_loc, Ny_loc, Nz)
    catch
        local_size = size(plan.rhs_hat_y)
    end
    Nx_l, Ny_l, Nz = local_size

    # iterate over local spectral (kx, ky) and vertical index
    if plan.bc_spec === nothing
        @inbounds for j in 1:Ny_l, i in 1:Nx_l
            col = view(plan.rhs_hat_y, i, j, :)
            if plan.bc_z === :periodic
                nothing
            else
                FFTW.r2r!(col, plan.Tz_fwd)
            end
            @simd ivdep for m in 1:Nz  # SIMD optimization for division loop
                denom = plan.kx2_local[i] + plan.ky2_local[j] + plan.lambda_z[m]
                if denom == 0
                    col[m] = 0
                else
                    col[m] /= -denom
                end
            end
            if plan.bc_z !== :periodic
                FFTW.r2r!(col, plan.Tz_inv)
                if plan.bc_z === :dirichlet
                    col ./= (2*(Nz+1))
                elseif plan.bc_z === :neumann
                    col ./= (2*Nz)
                end
            end
        end
    else
        # Use 1D tridiagonal Helmholtz solver along z for each (kx,ky)
        @inbounds for j in 1:Ny_l, i in 1:Nx_l
            col = view(plan.rhs_hat_y, i, j, :)
            lambda_xy = plan.kx2_local[i] + plan.ky2_local[j]
            _solve_helmholtz_tridiag!(col, lambda_xy, plan, t, i, j, xnodes, ynodes)
        end
    end

    # 4) Inverse FFTs back
    # Y -> X inverse FFT in y
    ldiv!(plan.pi_y, plan.fft_y, plan.rhs_hat_y)
    transpose!(plan.rhs_hat_x, decomp.transform_y_to_x, plan.pi_y)

    # X inverse FFT
    ldiv!(plan.pi_x, plan.fft_x, plan.rhs_hat_x)
    transpose!(pipi, decomp.transform_x_to_z, plan.pi_x)

    return pipi
end

# -------------------------- Time-dependent BC evaluation -----------------------

"""
    _eval_poisson_bc_value(bc_spec, key, t, i, j, xnodes, ynodes)

Evaluate a boundary condition value that may be time/spatial dependent.
Similar to _eval_velocity_bc_values but for Poisson solver BCs.
"""
function _eval_poisson_bc_value(bc_spec::Dict{Symbol,Any}, key::Symbol, t::Real, 
                               i::Int, j::Int, xnodes, ynodes)
    val = get(bc_spec, key, 0.0)
    
    if isa(val, Function)
        try
            # Try time and spatial coordinates
            if xnodes !== nothing && ynodes !== nothing
                x = xnodes[i]
                y = ynodes[j]
                return val(x, y, t)
            else
                # Try time only
                return val(t)
            end
        catch MethodError
            try
                # Try grid indices
                return val(i, j, t)
            catch MethodError
                # Fall back to time only
                return val(t)
            end
        end
    else
        return val
    end
end

# -------------------------- 1D Helmholtz solver (z) ---------------------------

"""
    _solve_helmholtz_tridiag!(col, lambda_xy, plan)

Solve (∂²/∂z² - λ)π = r for a single (kx,ky) column using a tridiagonal
second-order discretization in z with boundary conditions from `plan.bc_spec`.

Supports bottom/top `:dirichlet`, `:neumann`, or `:robin` with constants
provided in `bc_spec`:
- Dirichlet: `:bottom_value`, `:top_value`
- Neumann:   flux q via `:bottom_value`, `:top_value`
- Robin:     alpha, beta, gamma via `:bottom_alpha`, `:bottom_beta`, `:bottom_gamma`, and `:top_*`
"""
function _solve_helmholtz_tridiag!(col, lambda_xy::Real, plan::PoissonPlan, t::Real=0.0, i::Int=1, j::Int=1,
                                  xnodes=nothing, ynodes=nothing)
    Nz = size(col, 1)
    rhs = collect(col)  # work copy
    a = zeros(eltype(rhs), Nz)
    b = zeros(eltype(rhs), Nz)
    c = zeros(eltype(rhs), Nz)

    # Get spacing vector h
    # For non-uniform grids: grid.dz is a vector of cell spacings
    # For uniform grids: grid.dz is a scalar
    grid = plan.grid

    if isa(grid.dz, AbstractVector)
        # Non-uniform grid: use actual spacings
        h = grid.dz
        if length(h) != Nz - 1
            error("Non-uniform grid.dz must have length Nz-1 = $(Nz-1), got $(length(h))")
        end
    else
        # Uniform grid: create constant spacing vector
        h = fill(Float64(grid.dz), Nz - 1)
    end

    h1 = h[1]
    hN = h[end]

    # Interior coefficients for non-uniform grid
    # Second-order centered finite difference:
    # d²π/dz² ≈ 2/(h[k-1]*(h[k-1]+h[k])) * π[k-1]
    #         - 2/(h[k-1]*h[k]) * π[k]
    #         + 2/(h[k]*(h[k-1]+h[k])) * π[k+1]
    for k in 2:Nz-1
        hm = h[k-1]  # spacing below point k
        hp = h[k]    # spacing above point k
        a[k] =  2 / (hm * (hm + hp))
        b[k] = -2 / (hm * hp) - lambda_xy
        c[k] =  2 / (hp * (hm + hp))
    end

    # Boundary rows from bc_spec
    btm = get(plan.bc_spec, :bottom_type, :dirichlet)
    top = get(plan.bc_spec, :top_type, :dirichlet)
    if btm === :dirichlet
        a[1] = 0; b[1] = 1; c[1] = 0
        rhs[1] = _eval_poisson_bc_value(plan.bc_spec, :bottom_value, t, i, j, xnodes, ynodes)
    elseif btm === :neumann
        q = _eval_poisson_bc_value(plan.bc_spec, :bottom_value, t, i, j, xnodes, ynodes)
        a[1] = 0; b[1] = -1/h1; c[1] = 1/h1
        rhs[1] = q
    elseif btm === :robin
        alpha = _eval_poisson_bc_value(plan.bc_spec, :bottom_alpha, t, i, j, xnodes, ynodes)
        beta = _eval_poisson_bc_value(plan.bc_spec, :bottom_beta, t, i, j, xnodes, ynodes)
        gamma = _eval_poisson_bc_value(plan.bc_spec, :bottom_gamma, t, i, j, xnodes, ynodes)
        a[1] = 0; b[1] = alpha - beta/h1; c[1] = beta/h1
        rhs[1] = gamma
    else
        error("Unknown bottom BC type: $btm")
    end

    if top === :dirichlet
        a[Nz] = 0; b[Nz] = 1; c[Nz] = 0
        rhs[Nz] = _eval_poisson_bc_value(plan.bc_spec, :top_value, t, i, j, xnodes, ynodes)
    elseif top === :neumann
        q = _eval_poisson_bc_value(plan.bc_spec, :top_value, t, i, j, xnodes, ynodes)
        a[Nz] = -1/hN; b[Nz] = 1/hN; c[Nz] = 0
        rhs[Nz] = q
    elseif top === :robin
        alpha = _eval_poisson_bc_value(plan.bc_spec, :top_alpha, t, i, j, xnodes, ynodes)
        beta = _eval_poisson_bc_value(plan.bc_spec, :top_beta, t, i, j, xnodes, ynodes)
        gamma = _eval_poisson_bc_value(plan.bc_spec, :top_gamma, t, i, j, xnodes, ynodes)
        a[Nz] = -beta/hN; b[Nz] = alpha + beta/hN; c[Nz] = 0
        rhs[Nz] = gamma
    else
        error("Unknown top BC type: $top")
    end

    # Thomas algorithm (in-place on rhs)
    # Forward sweep
    for k in 2:Nz
        m = a[k] / b[k-1]
        b[k] -= m * c[k-1]
        rhs[k] -= m * rhs[k-1]
    end
    # Back substitution
    sol = rhs
    sol[Nz] /= b[Nz]
    for k in Nz-1:-1:1
        sol[k] = (sol[k] - c[k] * sol[k+1]) / b[k]
    end
    col .= sol
    return col
end


#  Pretty printing 
# textual summary (no Unicode boxes, clearly indented)

# Compact one-liner (used in arrays, logging, etc.)
Base.show(io::IO, plan::PoissonPlan) = begin
    Nx = plan.decomp.Nx_global; Ny = plan.decomp.Ny_global; Nz = length(plan.lambda_z)
    print(io, "FFTPoissonSolver(grid=($(Nx), $(Ny), $(Nz)), bc_z=$(plan.bc_z))")
end

# Helper to pretty-print byte counts
_fmt_bytes(b) = b < 1024        ? string(b, " B") :
                b < 1024^2      ? string(round(b/1024; digits=1), " KiB") :
                b < 1024^3      ? string(round(b/1024^2; digits=2), " MiB") :
                                   string(round(b/1024^3; digits=2), " GiB")

function Base.show(io::IO, ::MIME"text/plain", plan::PoissonPlan)
    Nx = plan.decomp.Nx_global
    Ny = plan.decomp.Ny_global
    Nz = length(plan.lambda_z)
    bc = plan.bc_z

    # Scratch memory estimate
    bytes = 0
    for fld in fieldnames(typeof(plan))
        v = getfield(plan, fld)
        v isa AbstractArray && (bytes += sizeof(eltype(v)) * length(v))
    end

    ztr = bc === :periodic  ? "Fourier (complex)" :
          bc === :dirichlet ? "DST-I (RODFT10/01)" :
          bc === :neumann   ? "DCT-I (REDFT10/01)" : string(bc)

    comm_sz = try
        pencil = PencilArrays.pencil(plan.rhs_x)
        comm = PencilArrays.get_comm(pencil)
        MPI.Comm_size(comm)
    catch
        missing
    end

    println(io, "FFTPoissonSolver:")
    println(io, "  grid size            : (", Nx, ", ", Ny, ", ", Nz, ")")
    println(io, "  bc_z                 : ", bc)
    println(io, "  transforms           :")
    println(io, "    x, y  -> Complex FFT (PencilFFTs)")
    println(io, "    z     -> ", ztr)
    println(io, "  scratch arrays       : rhs_x, rhs_y, rhs_hat_x, rhs_hat_y, pi_x, pi_y")
    println(io, "  scratch memory       : ", _fmt_bytes(bytes), " (approx.)")
    comm_sz !== missing && println(io, "  MPI ranks            : ", comm_sz)
    println(io, "  call                  : solve_poisson!(phi, rhs, plan)")
end


#end # module PoissonFFTSolver
