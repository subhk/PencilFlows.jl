# Horizontal-coarsening multigrid Poisson solver (serial fallback)

using LinearAlgebra

export MGPoissonPlan, make_mg_poisson, mg_solve!
export MGDistPlan, make_mg_poisson_distributed, mg_solve_distributed!
export auto_mg_levels, make_mg_poisson_distributed_auto

"""
    MGPoissonPlan

Geometric multigrid plan with horizontal coarsening (x,y) and fixed vertical
resolution (z). Designed to handle non-periodic/complex BCs horizontally while
reusing existing non-uniform z stencils.

Notes
- This implementation operates on regular `Array`s. For `PencilArrays`, gather to
  a serial array, solve, and scatter back (future work: fully distributed MG).
- Coarsening: Nx, Ny halved each level; Nz unchanged.
"""
struct MGLevel{T}
    Nx::Int; Ny::Int; Nz::Int
    dx::T; dy::T
    dz::Vector{T}     # length Nz-1 spacings between cell centers
    Π::T              # Jacobi relaxation weight (Unicode Pi to avoid conflict with Base.pi)
end

struct MGPoissonPlan{T}
    levels::Vector{MGLevel{T}}
    bc_z::Union{Nothing, Dict{Symbol, Any}}
end

"""
    MGWorkspace{T}

Reusable workspace arrays for multigrid V-cycle to minimize allocations.
"""
struct MGWorkspace{T}
    residuals::Dict{Int, Array{T,3}}     # Residual arrays for each level
    coarse_rhs::Dict{Int, Array{T,3}}    # Right-hand side arrays for coarse levels
    corrections::Dict{Int, Array{T,3}}   # Correction arrays for each level
    
    function MGWorkspace{T}(levels::Vector{MGLevel{T}}) where T
        residuals = Dict{Int, Array{T,3}}()
        coarse_rhs = Dict{Int, Array{T,3}}()
        corrections = Dict{Int, Array{T,3}}()
        
        # Pre-allocate workspace arrays for each level
        for (i, lev) in enumerate(levels)
            residuals[i] = Array{T,3}(undef, lev.Nx, lev.Ny, lev.Nz)
            if i < length(levels)
                coarse_rhs[i+1] = Array{T,3}(undef, levels[i+1].Nx, levels[i+1].Ny, lev.Nz)
                corrections[i+1] = Array{T,3}(undef, levels[i+1].Nx, levels[i+1].Ny, lev.Nz)
            end
        end
        
        new{T}(residuals, coarse_rhs, corrections)
    end
end

# Convenience constructor
MGWorkspace(levels::Vector{MGLevel{T}}) where T = MGWorkspace{T}(levels)

"""
    make_mg_poisson(grid; levels=3, Π=0.8, bc=Dict(...))

Build a horizontal-coarsening multigrid hierarchy from a grid-like object
providing fields: Nx, Ny, Nz, dx, dy, dz (Vector).

bc keys (Dirichlet/Neumann/Robin on horizontal faces):
- :left_type, :right_type  (:dirichlet, :neumann, :robin)
- :bottom_type, :top_type for y-faces
- Values for each type:
  - Dirichlet: :left_value, :right_value, :bottom_value, :top_value
  - Neumann:   same keys, interpret as flux q
  - Robin:     use triplets: :left_alpha, :left_beta, :left_gamma, etc.
Z-boundary handling is not changed here; vertical BCs should be enforced in the
operator application via the existing non-uniform stencil logic if needed.
"""
function make_mg_poisson(grid; levels::Int=3, Π::Real=0.8, 
                        bc_z::Union{Nothing,Dict{Symbol,Any}}=nothing)
    _elt(x) = x isa AbstractArray ? eltype(x) : typeof(x)
    T = promote_type(_elt(grid.dx), _elt(grid.dy), eltype(grid.dz))
    levs = MGLevel{T}[]
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    dx, dy = grid.dx, grid.dy
    dz = grid.dz
    for level in 1:levels
        push!(levs, MGLevel{T}(Nx, Ny, Nz, T(dx), T(dy), Vector{T}(dz), T(Π)))
        # coarsen x,y only (>= 3)
        Nx = max(3, Nx ÷ 2)
        Ny = max(3, Ny ÷ 2)
        dx *= 2
        dy *= 2
        # keep Nz, dz unchanged
    end
    return MGPoissonPlan{T}(levs, bc_z)
end

# ----------------------- MG core operations ------------------------

# Apply 3D Laplacian (7-point) with non-uniform z second-derivative (periodic x,y).
# At vertical boundaries (k=1,Nz), enforce BC equations into the operator rows:
#   Dirichlet: out = u - value
#   Neumann:   out = (u - u)/h - q   and  out = (u_N - u_{N-1})/h_N - q
#   Robin:     out = alpha u * beta (du/dz)_*  gamma   using first-order one-sided du/dz
function _apply_A!(out, u, lev::MGLevel{T}, bc_z::Union{Nothing,Dict{Symbol,Any}}=nothing) where T
    Nx, Ny, Nz = lev.Nx, lev.Ny, lev.Nz
    dx, dy = lev.dx, lev.dy
    hx2 = one(T)/dx^2
    hy2 = one(T)/dy^2
    
    # Pre-compute z-direction coefficients for better cache usage
    z_coeffs = Vector{NTuple{3,T}}(undef, Nz-2)
    @inbounds for k in 2:Nz-1
        hm = lev.dz[k-1]
        hp = lev.dz[k]
        a = 2/(hm*(hm+hp))
        b = -2/(hm*hp)
        c = 2/(hp*(hm+hp))
        z_coeffs[k-1] = (a, b, c)
    end
    
    # Cache-optimized loop order: k-j-i for better memory access
    # Added @fastmath for aggressive floating-point optimizations
    @inbounds @fastmath for k in 2:Nz-1
        a, b, c = z_coeffs[k-1]
        for j in 1:Ny
            jp = j == Ny ? 1 : j+1
            jm = j == 1 ? Ny : j-1
            @simd ivdep for i in 1:Nx  # SIMD vectorization for innermost loop
                ip = i == Nx ? 1 : i+1
                im = i == 1 ? Nx : i-1
                
                u_c = u[i,j,k]
                # Compute Laplacian
                lapxy = hx2*(u[im,j,k] - 2u_c + u[ip,j,k]) +
                        hy2*(u[i,jm,k] - 2u_c + u[i,jp,k])
                lapz = a*u[i,j,k-1] + b*u_c + c*u[i,j,k+1]
                out[i,j,k] = lapxy + lapz
            end
        end
    end
    
    # Handle boundary conditions efficiently
    _apply_boundary_conditions!(out, u, lev, bc_z)
    
    return out
end

# Helper function for boundary conditions
function _apply_boundary_conditions!(out, u, lev::MGLevel{T}, bc_z::Union{Nothing,Dict{Symbol,Any}}) where T
    Nx, Ny, Nz = lev.Nx, lev.Ny, lev.Nz
    
    @inbounds for j in 1:Ny, i in 1:Nx
        if bc_z === nothing
            # fallback: copy interior (legacy behavior)
            out[i,j,1]  = out[i,j,2]
            out[i,j,Nz] = out[i,j,Nz-1]
        else
            h1 = lev.dz[1]; hN = lev.dz[end]
            btype = get(bc_z, :bottom_type, :dirichlet)
            ttype = get(bc_z, :top_type, :dirichlet)
            
            # Bottom boundary
            if btype === :dirichlet
                v = T(get(bc_z, :bottom_value, 0.0))
                out[i,j,1] = u[i,j,1] - v
            elseif btype === :neumann
                q = T(get(bc_z, :bottom_value, 0.0))
                out[i,j,1] = (u[i,j,2] - u[i,j,1]) / h1 - q
            elseif btype === :robin
                alpha = T(get(bc_z, :bottom_alpha, 0.0)) 
                beta = T(get(bc_z, :bottom_beta, 1.0))
                gamma = T(get(bc_z, :bottom_gamma, 0.0))
                out[i,j,1] = alpha*u[i,j,1] - beta*(u[i,j,2] - u[i,j,1])/h1 - gamma
            else
                out[i,j,1] = out[i,j,2]
            end
            
            # Top boundary
            if ttype === :dirichlet
                v = T(get(bc_z, :top_value, 0.0))
                out[i,j,Nz] = u[i,j,Nz] - v
            elseif ttype === :neumann
                q = T(get(bc_z, :top_value, 0.0))
                out[i,j,Nz] = (u[i,j,Nz] - u[i,j,Nz-1]) / hN - q
            elseif ttype === :robin
                alpha = T(get(bc_z, :top_alpha, 0.0))
                beta = T(get(bc_z, :top_beta, 1.0))
                gamma = T(get(bc_z, :top_gamma, 0.0))
                out[i,j,Nz] = alpha*u[i,j,Nz] + beta*(u[i,j,Nz] - u[i,j,Nz-1])/hN - gamma
            else
                out[i,j,Nz] = out[i,j,Nz-1]
            end
        end
    end
end

# Enforce vertical BCs (Dirichlet/Neumann/Robin) on k=1 and k=Nz
function _apply_vertical_bcs!(u, lev::MGLevel{T}, bc_z::Dict{Symbol,Any}) where T
    Nx, Ny, Nz = lev.Nx, lev.Ny, lev.Nz
    h1 = lev.dz[1]; hN = lev.dz[end]
    btype = get(bc_z, :bottom_type, :dirichlet)
    ttype = get(bc_z, :top_type, :dirichlet)
    # Bottom (k=1)
    @inbounds for j in 1:Ny, i in 1:Nx
        if btype === :dirichlet
            u[i,j,1] = T(get(bc_z, :bottom_value, 0.0))
        elseif btype === :neumann
            q = T(get(bc_z, :bottom_value, 0.0))
            u[i,j,1] = u[i,j,2] - q*h1
        elseif btype === :robin
            alpha = T(get(bc_z, :bottom_alpha, 0.0)); 
            beta = T(get(bc_z, :bottom_beta, 1.0)); 
            gamma = T(get(bc_z, :bottom_gamma, 0.0))
            denom = (alpha - beta/h1)
            u[i,j,1] = denom == 0 ? u[i,j,1] : (gamma - beta*u[i,j,2]/h1) / denom
        end
    end
    # Top (k=Nz)
    @inbounds for j in 1:Ny, i in 1:Nx
        if ttype === :dirichlet
            u[i,j,Nz] = T(get(bc_z, :top_value, 0.0))
        elseif ttype === :neumann
            q = T(get(bc_z, :top_value, 0.0))
            u[i,j,Nz] = u[i,j,Nz-1] + q*hN
        elseif ttype === :robin
            alpha = T(get(bc_z, :top_alpha, 0.0)); 
            beta = T(get(bc_z, :top_beta, 1.0)); 
            gamma = T(get(bc_z, :top_gamma, 0.0))
            denom = (alpha + beta/hN)
            u[i,j,Nz] = denom == 0 ? u[i,j,Nz] : (gamma + beta*u[i,j,Nz-1]/hN) / denom
        end
    end
    return u
end

function _residual!(r, u, f, lev, bc_z=nothing)
    _apply_A!(r, u, lev, bc_z)
    @. r = f - r
    return r
end

function _smooth_jacobi!(u, f, lev; iters::Int=3, bc_z::Union{Nothing,Dict{Symbol,Any}}=nothing)
    Nx, Ny, Nz = lev.Nx, lev.Ny, lev.Nz
    dx, dy = lev.dx, lev.dy
    hx2 = 1/dx^2
    hy2 = 1/dy^2
    Π = lev.Π
    tmp = similar(u)
    
    # Pre-compute diagonal coefficients
    T = eltype(u)
    diag_coeffs = Vector{T}(undef, Nz-2)
    @inbounds for k in 2:Nz-1
        hm = lev.dz[k-1]
        hp = lev.dz[k]
        b = -2/(hm*hp)
        diag_coeffs[k-1] = Π / (-2*hx2 - 2*hy2 + b)
    end
    
    for _ in 1:iters
        # Enforce vertical BCs if provided
        if bc_z !== nothing
            _apply_vertical_bcs!(u, lev, bc_z)
        end
        
        _residual!(tmp, u, f, lev, bc_z)
        
        # Optimized Jacobi update with pre-computed coefficients
        @inbounds for k in 2:Nz-1
            coeff = diag_coeffs[k-1]
            for j in 1:Ny, i in 1:Nx
                u[i,j,k] = muladd(coeff, tmp[i,j,k], u[i,j,k])
            end
        end
        
        # Re-impose vertical BCs
        if bc_z !== nothing
            _apply_vertical_bcs!(u, lev, bc_z)
        end
    end
    return u
end

# Full-weighting restriction (x,y, periodic wrap), injection in z
function _restrict!(f_coarse, f_fine)
    Nx_c, Ny_c, Nz = size(f_coarse)
    Nx_f, Ny_f, _ = size(f_fine)
    @inline wrap(i, N) = i < 1 ? i + N * ((1 - i) ÷ N + 1) : (i > N ? ((i - 1) % N) + 1 : i)
    
    # Optimized restriction with better memory access pattern
    @inbounds for k in 1:Nz
        for j in 1:Ny_c
            jj = 2j - 1
            j0 = wrap(jj-1, Ny_f)
            j1 = wrap(jj, Ny_f)
            j2 = wrap(jj+1, Ny_f)
            for i in 1:Nx_c
                ii = 2i - 1
                i0 = wrap(ii-1, Nx_f)
                i1 = wrap(ii, Nx_f)
                i2 = wrap(ii+1, Nx_f)
                
                # Unrolled full-weighting restriction
                f_coarse[i,j,k] = (f_fine[i0,j0,k] + 2*f_fine[i1,j0,k] + f_fine[i2,j0,k] +
                                  2*f_fine[i0,j1,k] + 4*f_fine[i1,j1,k] + 2*f_fine[i2,j1,k] +
                                  f_fine[i0,j2,k] + 2*f_fine[i1,j2,k] + f_fine[i2,j2,k]) * 0.0625  # /16
            end
        end
    end
    return f_coarse
end

# Piecewise-constant prolongation (x,y), injection in z (periodic wrap)
function _prolongate_add!(u_fine, u_coarse)
    Nx_c, Ny_c, Nz = size(u_coarse)
    Nx_f, Ny_f, _ = size(u_fine)
    @inline wrap(i, N) = i < 1 ? i + N * ((1 - i) ÷ N + 1) : (i > N ? ((i - 1) % N) + 1 : i)
    
    # Optimized prolongation with better cache usage
    @inbounds for k in 1:Nz
        for j in 1:Ny_c
            jj = 2j - 1
            j1 = wrap(jj, Ny_f)
            j2 = wrap(jj+1, Ny_f)
            for i in 1:Nx_c
                ii = 2i - 1
                val = u_coarse[i,j,k]
                i1 = wrap(ii, Nx_f)
                i2 = wrap(ii+1, Nx_f)
                
                u_fine[i1, j1, k] += val
                u_fine[i2, j1, k] += val
                u_fine[i1, j2, k] += val
                u_fine[i2, j2, k] += val
            end
        end
    end
    return u_fine
end

function _vcycle!(u, f, levs::Vector{MGLevel{T}}, level_idx::Int; pre=3, 
                post=3, bc_z::Union{Nothing,Dict{Symbol,Any}}=nothing,
                workspace::Union{Nothing,MGWorkspace{T}}=nothing) where T
    lev = levs[level_idx]
    # pre-smooth
    _smooth_jacobi!(u, f, lev; iters=pre, bc_z=bc_z)
    
    # residual - use workspace if available
    if workspace !== nothing && haskey(workspace.residuals, level_idx)
        r = workspace.residuals[level_idx]
        _residual!(r, u, f, lev, bc_z)
    else
        r = similar(u); _residual!(r, u, f, lev, bc_z)
    end

    if level_idx == length(levs)
        # coarsest grid: solve approximately by extra smoothing
        _smooth_jacobi!(u, f, lev; iters=20, bc_z=bc_z)
        return u
    end

    # restrict residual to coarse grid - use workspace if available
    next_lev = levs[level_idx+1]
    if workspace !== nothing && haskey(workspace.coarse_rhs, level_idx+1)
        f_c = workspace.coarse_rhs[level_idx+1]
        fill!(f_c, zero(T))
    else
        f_c = zeros(eltype(f), next_lev.Nx, next_lev.Ny, lev.Nz)
    end
    _restrict!(f_c, r)

    # coarse-grid correction - use workspace if available
    if workspace !== nothing && haskey(workspace.corrections, level_idx+1)
        e_c = workspace.corrections[level_idx+1] 
        fill!(e_c, zero(T))
    else
        e_c = zeros(eltype(f), size(f_c))
    end
    _vcycle!(e_c, f_c, levs, level_idx+1; pre=pre, post=post, bc_z=bc_z, workspace=workspace)

    # prolongate and add
    _prolongate_add!(u, e_c)

    # post-smooth
    _smooth_jacobi!(u, f, lev; iters=post, bc_z=bc_z)

    return u
end

"""
    mg_solve!(Π, r, plan; cycles=2, pre=3, post=3, workspace=nothing)

Solve nabla²Π = r using horizontal-coarsening multigrid. Operates on Arrays of size
`(Nx, Ny, Nz)` as provided in plan.levels[1].

For better performance, provide a pre-allocated workspace via `MGWorkspace(plan.levels)`.
"""
function mg_solve!(Π::Array, r::Array, plan::MGPoissonPlan; 
                cycles::Int=2, pre::Int=3, post::Int=3,
                workspace::Union{Nothing,MGWorkspace}=nothing)

    @assert size(Π) == size(r) "Π and r must have same size"

    for _ in 1:cycles
        _vcycle!(Π, r, plan.levels, 1; pre=pre, post=post, bc_z=plan.bc_z, workspace=workspace)
    end

    return Π
end

# ============================================================================
# Distributed MG on Z-pencils (PencilArrays) with halo exchanges
# ============================================================================

using PencilArrays
using MPI

struct MGDistLevel{T}
    lev::MGLevel{T}
    decomp::PencilDecomposition   # PencilDecomposition for this level
end

struct MGDistPlan{T}
    levels::Vector{MGDistLevel{T}}
    bc_z::Union{Nothing, Dict{Symbol, Any}}
end

"""
    make_mg_poisson_distributed(decomp, grid; levels=3, Π=0.8, bc_z=nothing)

Build a distributed multigrid plan operating on Z-pencils. Coarsens Nx, Ny by 2
per level while keeping Nz fixed. Assumes Nx_global and Ny_global are ÷isible
by 2^(levels-1) and local sizes are even at each level.
"""
function make_mg_poisson_distributed(decomp, grid; levels::Int=3, Π::Real=0.8, bc_z=nothing)

    levs_serial = make_mg_poisson(grid; levels=levels, Π=Π, bc_z=bc_z).levels
    # Build matching PencilDecompositions per level (same P1,P2)

    levs = MGDistLevel{eltype(grid.dz)}[]
    Nx, Ny, Nz = decomp.Nx_global, decomp.Ny_global, decomp.Nz_global
    P1, P2 = decomp.P1, decomp.P2
    
    for level in 1:levels
        d = init_pencil_decomposition(Nx, Ny, Nz; P1=P1, P2=P2)
        push!(levs, MGDistLevel{eltype(grid.dz)}(levs_serial[level], d))
        Nx = max(3, Nx ÷ 2)
        Ny = max(3, Ny ÷ 2)
    end
    
    return MGDistPlan{eltype(grid.dz)}(levs, bc_z)
end

"""
    auto_mg_levels(Nx, Ny, requested)

Clamp the requested number of MG levels so that 2^(levels-1) ÷ides both Nx and Ny.
Returns a positive Int ge 1.
"""
function auto_mg_levels(Nx::Integer, Ny::Integer, requested::Integer)
    requested < 1 && return 1
    L = requested
    while L > 1
        f = 1 << (L - 1)  # 2^(L-1)
        if (Nx % f == 0) && (Ny % f == 0)
            return L
        end
        L -= 1
    end
    return 1
end

"""
    make_mg_poisson_distributed_auto(decomp, grid; levels=3, Π=0.8, bc_z=nothing)

Builds a distributed MG plan after automatically clamping `levels` based on Nx,Ny ÷isibility.
"""
function make_mg_poisson_distributed_auto(decomp, grid; levels::Int=3, Π::Real=0.8, bc_z=nothing)
    L = auto_mg_levels(decomp.Nx_global, decomp.Ny_global, levels)
    return make_mg_poisson_distributed(decomp, grid; levels=L, Π=Π, bc_z=bc_z)
end

"""
    halo_exchange_xy!(u_z::PencilArrays.PencilArray, decomp)

Exchange x- and y-direction halos for periodic domains. Returns a NamedTuple of
left/right (Nyx*Nz) and bottom/top (Nxx*Nz) halos.
"""
# Optimized halo exchange with persistent buffers
struct HaloBuffers{T}
    lhalo::Array{T,2}
    rhalo::Array{T,2}
    bhalo::Array{T,2}
    thalo::Array{T,2}
    send_left::Array{T,2}
    send_right::Array{T,2}
    send_bot::Array{T,2}
    send_top::Array{T,2}
end

function create_halo_buffers(::Type{T}, Nx_l, Ny_l, Nz_l) where T
    return HaloBuffers{T}(
        Array{T}(undef, Ny_l, Nz_l),
        Array{T}(undef, Ny_l, Nz_l),
        Array{T}(undef, Nx_l, Nz_l),
        Array{T}(undef, Nx_l, Nz_l),
        Array{T}(undef, Ny_l, Nz_l),
        Array{T}(undef, Ny_l, Nz_l),
        Array{T}(undef, Nx_l, Nz_l),
        Array{T}(undef, Nx_l, Nz_l)
    )
end

function halo_exchange_xy!(u::PencilArrays.PencilArray{T,3}, decomp, buffers::HaloBuffers{T}) where T
    Nx_l, Ny_l, Nz_l = size(u)
    A = parent(u)

    comm = decomp.comm
    rank = decomp.rank
    P1, P2 = decomp.P1, decomp.P2
    px = rank % P1; py = rank ÷ P1
    left_rank   = ( (px-1+P1) % P1 ) + py * P1
    right_rank  = ( (px+1)    % P1 ) + py * P1
    bottom_rank = px + ( (py-1+P2) % P2 ) * P1
    top_rank    = px + ( (py+1)    % P2 ) * P1

    # Copy data to send buffers (optimized memory access)
    @inbounds for k in 1:Nz_l, j in 1:Ny_l
        buffers.send_left[j,k] = A[1,j,k]
        buffers.send_right[j,k] = A[Nx_l,j,k]
    end

    @inbounds for k in 1:Nz_l, i in 1:Nx_l
        buffers.send_bot[i,k] = A[i,1,k]
        buffers.send_top[i,k] = A[i,Ny_l,k]
    end

    # Tags (local constants not allowed; use local bindings)
    TAG_XL = 101; TAG_XR = 102; TAG_YB = 201; TAG_YT = 202

    # Optimized overlapped communication - start all transfers simultaneously
    req_all = Vector{MPI.Request}(undef, 8)
    
    # Start all non-blocking communications at once for better overlap
    req_all[1] = MPI.Isend(buffers.send_left, left_rank, TAG_XL, comm)
    req_all[2] = MPI.Irecv!(buffers.rhalo, right_rank, TAG_XL, comm)
    req_all[3] = MPI.Isend(buffers.send_right, right_rank, TAG_XR, comm)
    req_all[4] = MPI.Irecv!(buffers.lhalo, left_rank, TAG_XR, comm)
    req_all[5] = MPI.Isend(buffers.send_bot, bottom_rank, TAG_YB, comm)
    req_all[6] = MPI.Irecv!(buffers.thalo, top_rank, TAG_YB, comm)
    req_all[7] = MPI.Isend(buffers.send_top, top_rank, TAG_YT, comm)
    req_all[8] = MPI.Irecv!(buffers.bhalo, bottom_rank, TAG_YT, comm)
    
    # TODO: Could perform interior computations here while communication proceeds
    # Wait for all communications to complete
    MPI.Waitall(req_all)

    return (lhalo=buffers.lhalo, rhalo=buffers.rhalo, bhalo=buffers.bhalo, thalo=buffers.thalo)
end

# Fallback for compatibility
function halo_exchange_xy!(u::PencilArrays.PencilArray{T,3}, decomp) where T
    Nx_l, Ny_l, Nz_l = size(u)
    buffers = create_halo_buffers(T, Nx_l, Ny_l, Nz_l)
    return halo_exchange_xy!(u, decomp, buffers)
end

function _apply_A_distributed!(out::PencilArrays.PencilArray{T,3},
                               u::PencilArrays.PencilArray{T,3},
                               lev::MGLevel{T}, decomp; bc_z=nothing) where T
    Nx_l, Ny_l, Nz = size(u)
    halos = halo_exchange_xy!(u, decomp)
    A = parent(u); 
    B = parent(out)

    hx2 = one(T)/lev.dx^2; 
    hy2 = one(T)/lev.dy^2

    @inbounds for j in 1:Ny_l, i in 1:Nx_l
        ip = i == Nx_l ? i : i+1
        im = i == 1    ? i : i-1
        jp = j == Ny_l ? j : j+1
        jm = j == 1    ? j : j-1
        for k in 2:Nz-1
            u_c = A[i,j,k]
            u_im = i == 1    ? halos.lhalo[j,k] : A[im,j,k]
            u_ip = i == Nx_l ? halos.rhalo[j,k] : A[ip,j,k]
            u_jm = j == 1    ? halos.bhalo[i,k] : A[i,jm,k]
            u_jp = j == Ny_l ? halos.thalo[i,k] : A[i,jp,k]
            lapxy = hx2*(u_im - 2u_c + u_ip) + hy2*(u_jm - 2u_c + u_jp)
            hm = lev.dz[k-1]; hp = lev.dz[k]
            a =  2/(hm*(hm+hp)); b = -2/(hm*hp); c =  2/(hp*(hm+hp))
            lapz = a*A[i,j,k-1] + b*u_c + c*A[i,j,k+1]
            B[i,j,k] = lapxy + lapz
        end
        if bc_z === nothing
            B[i,j,1]   = B[i,j,2]
            B[i,j,end] = B[i,j,end-1]
        else
            h1 = lev.dz[1]; hN = lev.dz[end]
            btype = get(bc_z, :bottom_type, :dirichlet)
            ttype = get(bc_z, :top_type, :dirichlet)
            if btype === :dirichlet
                v = T(get(bc_z, :bottom_value, 0.0))
                B[i,j,1] = A[i,j,1] - v
            elseif btype === :neumann
                q = T(get(bc_z, :bottom_value, 0.0))
                B[i,j,1] = (A[i,j,2] - A[i,j,1]) / h1 - q
            elseif btype === :robin
                alpha = T(get(bc_z, :bottom_alpha, 0.0)); 
                beta = T(get(bc_z, :bottom_beta, 1.0)); 
                gamma = T(get(bc_z, :bottom_gamma, 0.0))
                B[i,j,1] = alpha*A[i,j,1] - beta*(A[i,j,2] - A[i,j,1])/h1 - gamma
            else
                B[i,j,1] = B[i,j,2]
            end
            if ttype === :dirichlet
                v = T(get(bc_z, :top_value, 0.0))
                B[i,j,end] = A[i,j,end] - v
            elseif ttype === :neumann
                q = T(get(bc_z, :top_value, 0.0))
                B[i,j,end] = (A[i,j,end] - A[i,j,end-1]) / hN - q
            elseif ttype === :robin
                alpha = T(get(bc_z, :top_alpha, 0.0)); 
                beta = T(get(bc_z, :top_beta, 1.0)); 
                gamma = T(get(bc_z, :top_gamma, 0.0))
                B[i,j,end] = alpha*A[i,j,end] + beta*(A[i,j,end] - A[i,j,end-1])/hN - gamma
            else
                B[i,j,end] = B[i,j,end-1]
            end
        end
    end
    return out
end

function _residual_distributed!(r, u, f, lev, decomp, bc_z)
    _apply_A_distributed!(r, u, lev, decomp; bc_z=bc_z)
    @. parent(r) = parent(f) - parent(r)
    return r
end

function _smooth_jacobi_distributed!(u, f, lev, decomp; iters::Int=3, bc_z=nothing)
    Nx_l, Ny_l, Nz = size(u)
    Π = lev.Π; 
    hx2 = 1/lev.dx^2; 
    hy2 = 1/lev.dy^2
    tmp = similar(u)
    
    for _ in 1:iters
        if bc_z !== nothing
            _apply_vertical_bcs!(u, lev, bc_z)
        end
        _residual_distributed!(tmp, u, f, lev, decomp, bc_z)
        A = parent(u); R = parent(tmp)
        @inbounds for j in 1:Ny_l, i in 1:Nx_l, k in 2:Nz-1
            hm = lev.dz[k-1]; hp = lev.dz[k]
            b = -2/(hm*hp)
            D = -2*hx2 - 2*hy2 + b
            A[i,j,k] += Π * R[i,j,k] / D
        end
        if bc_z !== nothing
            _apply_vertical_bcs!(u, lev, bc_z)
        end
    end
    return u
end

function _restrict_distributed!(f_c, r_f, lev_c::MGLevel, lev_f::MGLevel)
    Nx_c, Ny_c, Nz = size(f_c)
    RF = parent(r_f); 
    RC = parent(f_c)
    
    @assert size(r_f,3) == Nz

    @inbounds for jc in 1:Ny_c, ic in 1:Nx_c, k in 1:Nz
        i1 = 2*ic - 1; i2 = min(i1+1, size(r_f,1))
        j1 = 2*jc - 1; j2 = min(j1+1, size(r_f,2))
        s = RF[i1,j1,k] + RF[i2,j1,k] + RF[i1,j2,k] + RF[i2,j2,k]
        RC[ic,jc,k] = s / 4
    end

    return f_c
end

function _prolongate_add_distributed!(u_f, e_c)
    Nx_c, Ny_c, Nz = size(e_c)
    UF = parent(u_f); 
    EC = parent(e_c)
    
    @inbounds for jc in 1:Ny_c, ic in 1:Nx_c, k in 1:Nz
        i1 = 2*ic - 1; i2 = min(i1+1, size(u_f,1))
        j1 = 2*jc - 1; j2 = min(j1+1, size(u_f,2))
        val = EC[ic,jc,k]
        UF[i1,j1,k]      += val
        UF[i2,j1,k]      += val
        UF[i1,j2,k]      += val
        UF[i2,j2,k]      += val
    end

    return u_f
end

function _vcycle_distributed!(u, f, plan::MGDistPlan{T}, level_idx::Int; pre=3, post=3) where T
    level = plan.levels[level_idx]
    decomp = level.decomp

    _smooth_jacobi_distributed!(u, f, level.lev, decomp; iters=pre, bc_z=plan.bc_z)
    r = similar(u); _residual_distributed!(r, u, f, level.lev, decomp, plan.bc_z)
    
    if level_idx == length(plan.levels)
        _smooth_jacobi_distributed!(u, f, level.lev, decomp; iters=20, bc_z=plan.bc_z)
        return u
    end
    
    coar = plan.levels[level_idx+1]
    f_c = PencilArrays.PencilArray{T}(undef, coar.decomp.pencil_z)
    e_c = PencilArrays.PencilArray{T}(undef, coar.decomp.pencil_z)
    
    fill!(parent(f_c), zero(T)); fill!(parent(e_c), zero(T))
    _restrict_distributed!(f_c, r, coar.lev, level.lev)
    _vcycle_distributed!(e_c, f_c, plan, level_idx+1; pre=pre, post=post)
    _prolongate_add_distributed!(u, e_c)
    _smooth_jacobi_distributed!(u, f, level.lev, decomp; iters=post, bc_z=plan.bc_z)
    
    return u
end

function mg_solve_distributed!(Π::PencilArrays.PencilArray{T,3}, 
                            r::PencilArrays.PencilArray{T,3}, 
                            plan::MGDistPlan{T}; 
                            cycles::Int=2, 
                            pre::Int=3, 
                            post::Int=3) where T
    @assert size(Π) == size(r) "Π and r must have same size"

    for _ in 1:cycles
        _vcycle_distributed!(Π, r, plan, 1; pre=pre, post=post)
    end
    
    return Π
end
