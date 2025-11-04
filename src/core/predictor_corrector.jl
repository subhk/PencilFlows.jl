# Rotating-Stratified Navier-Stokes: Predictor-Corrector step with projection

using PencilFFTs
using PencilArrays
using FFTW
using Base.Threads: @threads

# Import spatial field types (will be included by main PencilFlow module)

## Tensor-form helpers are provided in tensor_helpers.jl and included via PencilFlow
## Note: unified transforms.jl functions (ddx!, ddy!, etc.) are loaded via main module

"""
    RSNSWorkspace(decomp, proto)

Allocate scratch arrays for the rotating-stratified NS predictor-corrector step.
`proto` is a prototype Z-pencil array (e.g., `u`), used to inherit layout and eltype.
"""
mutable struct RSNSWorkspace{A}
    # RHS accumulators (two evaluations for Heun)
    Ru1::A; Rv1::A; Rw1::A; Rb1::A
    Ru2::A; Rv2::A; Rw2::A; Rb2::A

    # Predictor storage
    ustar::A; vstar::A; wstar::A; bstar::A

    # Projection / pressure correction
    pipi::A; divu::A

    # Scalar gradient scratch and laplacian
    dbdx::A; dbdy::A; dbdz::A
    lap::A

    # Pre-allocated Laplacian arrays (PERFORMANCE OPTIMIZATION)
    lap_u::A; lap_v::A; lap_w::A

    # Flux conservative advection workspace
    div_uu::A; div_vu::A; div_wu::A  # Momentum flux divergences
    div_U::A                         # Velocity divergence

    # Pre-allocated tridiagonal solver arrays (PERFORMANCE OPTIMIZATION)
    tri_a::Vector{Float64}
    tri_b::Vector{Float64}
    tri_c::Vector{Float64}
    tri_temp::Vector{Float64}

    # Cached local wavenumbers for horizontal ops
    kx_loc::Vector{Float64}
    ky_loc::Vector{Float64}
    kx2_loc::Vector{Float64}
    ky2_loc::Vector{Float64}
end

function RSNSWorkspace(decomp, grid, proto::PencilArrays.PencilArray{T,3}) where {T}
    pencil_obj = PencilArrays.pencil(proto)
    newA() = PencilArrays.PencilArray{T}(undef, pencil_obj)
    # local wavenumbers
    kx = FFTW.fftfreq(decomp.Nx_global, 2pi/grid.dx)
    ky = FFTW.fftfreq(decomp.Ny_global, 2pi/grid.dy)

    kx_loc = kx[PencilArrays.range_local(decomp.pencil_x, 1)]
    ky_loc = ky[PencilArrays.range_local(decomp.pencil_y, 2)]
    
    # Pre-allocate tridiagonal solver arrays (PERFORMANCE OPTIMIZATION)
    Nz = size(proto, 3)
    tri_a = zeros(Float64, Nz)
    tri_b = zeros(Float64, Nz) 
    tri_c = zeros(Float64, Nz)
    tri_temp = zeros(Float64, Nz)
    
    return RSNSWorkspace{typeof(proto)}(
        newA(), newA(), newA(), newA(),        # Ru1, Rv1, Rw1, Rb1
        newA(), newA(), newA(), newA(),        # Ru2, Rv2, Rw2, Rb2
        newA(), newA(), newA(), newA(),        # ustar, vstar, wstar, bstar
        newA(), newA(),                        # pipi, divu
        newA(), newA(), newA(),                # dbdx, dbdy, dbdz
        newA(),                                # lap
        newA(), newA(), newA(),                # lap_u, lap_v, lap_w (PERFORMANCE OPTIMIZATION)
        newA(), newA(), newA(),                # div_uu, div_vu, div_wu
        newA(),                                # div_U
        tri_a, tri_b, tri_c, tri_temp,         # Tridiagonal solver arrays (PERFORMANCE OPTIMIZATION)
        collect(kx_loc), 
        collect(ky_loc), 
        collect(kx_loc.^2), 
        collect(ky_loc.^2)
    )
end

"""
    horizontal_laplacian_2d!(Lxy, q, fields, decomp, grid, ws::RSNSWorkspace)

Compute horizontal Laplacian d2q/dx2 + d2q/dy2 in Z-pencil using existing d2dx2!/d2dy2! functions.
Uses precomputed squared wavenumbers from RSNSWorkspace for efficiency.
Requires that `fields` was created with `create_distributed_fields(decomp)`.
"""
function horizontal_laplacian_2d!(Lxy, q, fields, decomp, grid, ws::Union{RSNSWorkspace, NamedTuple})
    # Extract wavenumber arrays from workspace (supports both RSNSWorkspace and NamedTuple)
    kx2_loc = ws.kx2_loc
    ky2_loc = ws.ky2_loc
    
    # d2/dx2 using existing d2dx2! function with precomputed wavenumbers
    transpose!(fields.u_x, decomp.transform_z_to_x, q)
    mul!(fields.u_hat_x, decomp.fft_x, fields.u_x)

    # Use existing d2dx2! function with precomputed kx2_loc from workspace
    _d2dx2_adapter!(fields.u_hat_x, kx2_loc)
    
    ldiv!(fields.u_x, decomp.fft_x, fields.u_hat_x)
    transpose!(Lxy, decomp.transform_x_to_z, fields.u_x)   # Lxy = q_xx

    # d2/dy2 using existing d2dy2! function with precomputed wavenumbers and accumulate
    transpose!(fields.u_y, decomp.transform_z_to_y, q)
    mul!(fields.u_hat_y, decomp.fft_y, fields.u_y)

    # Use existing d2dy2! function with precomputed ky2_loc from workspace
    _d2dy2_adapter!(fields.u_hat_y, ky2_loc)
    
    ldiv!(fields.u_y, decomp.fft_y, fields.u_hat_y)
    transpose!(fields.u_z, decomp.transform_y_to_z, fields.u_y)   # reuse u_z as temp

    Lxy_data = parent(Lxy)
    u_z_data = parent(fields.u_z)
    @. Lxy_data += u_z_data
    
    return nothing
end

# Backward compatibility version - creates workspace and delegates
function horizontal_laplacian_2d!(Lxy, q, fields, decomp, grid, 
                                kx2_loc::Vector{<:Real}, ky2_loc::Vector{<:Real})
    temp_ws = (kx2_loc = kx2_loc, ky2_loc = ky2_loc)
    return horizontal_laplacian_2d!(Lxy, q, fields, decomp, grid, temp_ws)
end

"""
    _d2dx2_adapter!(u_hat, kx2_loc)
    
Adapter to use existing d2dx2! function with current interface.
Creates a minimal TransformPlans-like interface to ensure consistency.
"""
@inline function _d2dx2_adapter!(u_hat::PencilArrays.PencilArray{Complex{T}}, kx2_loc::Vector{T}) where T
    # Create a minimal mock TransformPlans for d2dx2! compatibility
    # This ensures we use exactly the same logic as the existing d2dx2! function
    mock_plans = _create_mock_transform_plans_x(u_hat)
    
    # Call the existing d2dx2! function directly
    d2dx2!(u_hat, kx2_loc, mock_plans)
end

"""
    _d2dy2_adapter!(u_hat, ky2_loc)
    
Adapter to use existing d2dy2! function with current interface.
Creates a minimal TransformPlans-like interface to ensure consistency.
"""
@inline function _d2dy2_adapter!(u_hat::PencilArrays.PencilArray{Complex{T}}, ky2_loc::Vector{T}) where T
    # Create a minimal mock TransformPlans for d2dy2! compatibility
    # This ensures we use exactly the same logic as the existing d2dy2! function
    mock_plans = _create_mock_transform_plans_y(u_hat)
    
    # Call the existing d2dy2! function directly
    d2dy2!(u_hat, ky2_loc, mock_plans)
end

"""
    gradient_xy!(gx, gy, q, fields, decomp, grid, ws::RSNSWorkspace)

Compute horizontal gradient (dq/dx, dq/dy) in Z-pencil using existing ddx!/ddy! functions.
Uses precomputed wavenumbers from RSNSWorkspace for efficiency.
This ensures consistency with the spectral derivative functions throughout the codebase.
"""
function gradient_xy!(gx, gy, q, fields, decomp, grid, ws::Union{RSNSWorkspace, NamedTuple})
    # Extract wavenumber arrays from workspace (supports both RSNSWorkspace and NamedTuple)
    kx_loc = ws.kx_loc
    ky_loc = ws.ky_loc
    
    # d/dx using existing ddx! function with precomputed wavenumbers
    transpose!(fields.u_x, decomp.transform_z_to_x, q)
    mul!(fields.u_hat_x, decomp.fft_x, fields.u_x)
    
    # Use existing ddx! function with precomputed kx_loc from workspace
    _ddx_adapter!(fields.u_hat_x, kx_loc)
    
    ldiv!(fields.u_x, decomp.fft_x, fields.u_hat_x)
    transpose!(gx, decomp.transform_x_to_z, fields.u_x)

    # d/dy using existing ddy! function with precomputed wavenumbers
    transpose!(fields.u_y, decomp.transform_z_to_y, q)
    mul!(fields.u_hat_y, decomp.fft_y, fields.u_y)
    
    # Use existing ddy! function with precomputed ky_loc from workspace
    _ddy_adapter!(fields.u_hat_y, ky_loc)
    
    ldiv!(fields.u_y, decomp.fft_y, fields.u_hat_y)
    transpose!(gy, decomp.transform_y_to_z, fields.u_y)
    return gx, gy
end

# Backward compatibility version - creates workspace and delegates
function gradient_xy!(gx, gy, q, fields, decomp, grid, kx_loc::Vector{<:Real}, ky_loc::Vector{<:Real})
    temp_ws = (kx_loc = kx_loc, ky_loc = ky_loc)
    return gradient_xy!(gx, gy, q, fields, decomp, grid, temp_ws)
end

"""
    _ddx_adapter!(u_hat, kx_loc)
    
Adapter to use existing ddx! function with current interface.
Creates a minimal TransformPlans-like interface to ensure consistency.
"""
@inline function _ddx_adapter!(u_hat::PencilArrays.PencilArray{Complex{T}}, kx_loc::Vector{T}) where T
    # Create a minimal mock TransformPlans for ddx! compatibility
    # This ensures we use exactly the same logic as the existing ddx! function
    mock_plans = _create_mock_transform_plans_x(u_hat)
    
    # Call the existing ddx! function directly
    ddx!(u_hat, kx_loc, mock_plans)
end

"""
    _ddy_adapter!(u_hat, ky_loc)
    
Adapter to use existing ddy! function with current interface.
Creates a minimal TransformPlans-like interface to ensure consistency.
"""
@inline function _ddy_adapter!(u_hat::PencilArrays.PencilArray{Complex{T}}, ky_loc::Vector{T}) where T
    # Create a minimal mock TransformPlans for ddy! compatibility
    # This ensures we use exactly the same logic as the existing ddy! function
    mock_plans = _create_mock_transform_plans_y(u_hat)
    
    # Call the existing ddy! function directly
    ddy!(u_hat, ky_loc, mock_plans)
end

"""
    _create_mock_transform_plans_x(u_hat)
    
Create a minimal mock TransformPlans object for x-derivatives.
Only contains the fields needed by ddx! function.
"""
@inline function _create_mock_transform_plans_x(u_hat::PencilArrays.PencilArray{Complex{T}}) where T
    # Return a NamedTuple that looks like TransformPlans but only has what ddx! needs
    # The ddx! function primarily uses the plans parameter for type checking
    return (Nx = size(u_hat, 1), Ny = size(u_hat, 2), Nz = size(u_hat, 3))
end

"""
    _create_mock_transform_plans_y(u_hat)
    
Create a minimal mock TransformPlans object for y-derivatives.
Only contains the fields needed by ddy! function.
"""
@inline function _create_mock_transform_plans_y(u_hat::PencilArrays.PencilArray{Complex{T}}) where T
    # Return a NamedTuple that looks like TransformPlans but only has what ddy! needs
    # The ddy! function primarily uses the plans parameter for type checking
    return (Nx = size(u_hat, 1), Ny = size(u_hat, 2), Nz = size(u_hat, 3))
end

"""
    laplacian_3d!(L, q, fields, decomp, grid, bc, var, ws::RSNSWorkspace)

Compute full Laplacian nabla2q = q_xx + q_yy + q_zz in Z-pencil, honoring vertical
BCs via `d2z_derivative_nonuniform_with_bcs!` for component `var in (:u,:v,:w,:b)`.
Uses precomputed wavenumbers from RSNSWorkspace for efficiency.
"""
function laplacian_3d!(L, q, fields, decomp, grid, bc, var::Symbol, ws::Union{RSNSWorkspace, NamedTuple})
    horizontal_laplacian_2d!(L, q, fields, decomp, grid, ws)
    d2z_derivative_nonuniform_with_bcs!(fields.u_z, q, grid, bc, var)
    L_data = parent(L)
    u_z_data = parent(fields.u_z)
    @. L_data += u_z_data
    return L
end

# Backward compatibility version - creates workspace and delegates
function laplacian_3d!(L, q, fields, decomp, grid, bc, var::Symbol, kx2_loc::Vector{<:Real}, ky2_loc::Vector{<:Real})
    temp_ws = (kx2_loc = kx2_loc, ky2_loc = ky2_loc)
    return laplacian_3d!(L, q, fields, decomp, grid, bc, var, temp_ws)
end

"""Compute nablaU for U=(u,v,w) using NonlinearWorkspace; returns tuples."""
function compute_gradients_U!(u, v, w, fields, decomp, grid, bc, ws::NonlinearWorkspace)
    compute_horizontal_derivatives_2d!(ws.dudx, ws.dudy,
                                       ws.dvdx, ws.dvdy,
                                       ws.dwdx, ws.dwdy,
                                       ws.tmpz, ws.tmpz,
                                       u, v, w, ws.tmpz,
                                       fields, decomp, grid)

    compute_z_derivatives_2d!(ws.dudz, ws.dvdz, ws.dwdz, ws.tmpz,
                              u, v, w, ws.tmpz,
                              decomp, grid, bc)

    nablau = (ws.dudx, ws.dudy, ws.dudz)
    nablav = (ws.dvdx, ws.dvdy, ws.dvdz)
    nablaw = (ws.dwdx, ws.dwdy, ws.dwdz)

    return (nablau, nablav, nablaw)
end

"""
    coriolis_implicit_solve!(u, v; f::Real, dt::Real, theta::Real=1.0)

Pointwise implicit solve for the linear Coriolis term.
"""
function coriolis_implicit_solve!(u::PencilArrays.PencilArray, v::PencilArrays.PencilArray;
                                  f::Real, dt::Real, theta::Real=1.0)
    if f == 0
        return u, v
    end
    A = parent(u); B = parent(v)
    alpha = theta * dt * f
    denom = 1 + alpha^2
    @inbounds for k in axes(A,3), j in axes(A,2), i in axes(A,1)
        urhs = A[i,j,k]
        vrhs = B[i,j,k]
        A[i,j,k] = (  urhs + alpha * vrhs) / denom
        B[i,j,k] = ( -alpha * urhs + vrhs) / denom
    end
    return u, v
end

"""
    helmholtz_z_solve!(pipi, rhs; alpha, grid, bc, theta=1.0)

Solve (I - alpha∂²/∂z²)Π = rhs along vertical columns with non-uniform spacings
using a tridiagonal Thomas algorithm.
"""
function helmholtz_z_solve!(pipi::PencilArrays.PencilArray, 
                            rhs::PencilArrays.PencilArray;
                            alpha::Real, 
                            grid, 
                            bc, 
                            theta::Real=1.0)
    if alpha == 0
        copy!(parent(pipi), parent(rhs))
        return pipi
    end
    Nz = size(pipi, 3)
    h = grid.dz
    Apipi = parent(pipi); R = parent(rhs)
    # Use pre-allocated tridiagonal arrays (PERFORMANCE OPTIMIZATION)
    a, b, c = ws.tri_a, ws.tri_b, ws.tri_c
    for k in 2:Nz-1
        hm = h[k-1]; hp = h[k]
        L =  2/(hm*(hm+hp))
        D = -2/(hm*hp)
        U =  2/(hp*(hm+hp))
        a[k] = -alpha*theta*L
        b[k] = 1 - alpha*theta*D
        c[k] = -alpha*theta*U
    end
    h1 = h[1]; hN = h[end]
    # Approximate BC rows (Dirichlet or Neumann-like based on velocity BC)
    if bc.bottom in (NO_SLIP, PRESCRIBED_VELOCITY)
        a[1] = 0; b[1] = 1; c[1] = 0
    else
        a[1] = 0; b[1] = -1/h1; c[1] = 1/h1
    end
    if bc.top in (NO_SLIP, PRESCRIBED_VELOCITY)
        a[Nz] = 0; b[Nz] = 1; c[Nz] = 0
    else
        a[Nz] = -1/hN; b[Nz] = 1/hN; c[Nz] = 0
    end
    # Use pre-allocated temporary array (PERFORMANCE OPTIMIZATION)
    tmp = ws.tri_temp
    @inbounds for j in axes(Apipi,2), i in axes(Apipi,1)
        for k in 1:Nz; tmp[k] = R[i,j,k]; end
        for k in 2:Nz
            m = a[k]/b[k-1]
            b[k] -= m*c[k-1]
            tmp[k] -= m*tmp[k-1]
        end
        Apipi[i,j,Nz] = tmp[Nz]/b[Nz]
        for k in Nz-1:-1:1
            Apipi[i,j,k] = (tmp[k] - c[k]*Apipi[i,j,k+1]) / b[k]
        end
    end
    return pipi
end

"""
    imex_step!(u,v,w,b,p, t, dt; decomp, grid, fields, bc, nu, fplane, 
               poisson_method=:fft, poisson_plan=nothing, mg_plan=nothing,
               theta::Real=1.0, nlin_ws=nothing, ws=nothing)

Advance one semi-implicit step: implicit linear (Coriolis, vertical viscosity), explicit nonlinear.
"""
function imex_step!(u, v, w, b, p, t, dt; decomp, grid, fields, bc,
                    nu::Real, 
                    fplane=FPlane(),
                    poisson_method::Symbol=:fft, 
                    poisson_plan=nothing, 
                    mg_plan=nothing,
                    theta::Real=1.0, 
                    nlin_ws=nothing, 
                    ws=nothing)

    nlin_ws === nothing && (nlin_ws = NonlinearWorkspace(decomp))
    ws === nothing && (ws = RSNSWorkspace(decomp, grid, u))

    # Nonlinear RHS (explicit): advection and buoyancy
    fill!(ws.Ru1, zero(eltype(ws.Ru1)))
    fill!(ws.Rv1, zero(eltype(ws.Rv1)))
    fill!(ws.Rw1, zero(eltype(ws.Rw1)))
    fill!(ws.Rb1, zero(eltype(ws.Rb1)))

    compute_nonlinear_terms!(ws.Ru1, ws.Rv1, ws.Rw1, u, v, w, fields, decomp, grid, bc, nlin_ws)

    compute_scalar_advection!(ws.Rb1, u, v, w, b, fields, decomp, grid, bc, nlin_ws)

    # Optionally include a small explicit horizontal diffusion for stability
    if nu != 0
        # Use pre-allocated Laplacian arrays (PERFORMANCE OPTIMIZATION)
        horizontal_laplacian_2d!(ws.lap_u, u, fields, decomp, grid, ws)
        horizontal_laplacian_2d!(ws.lap_v, v, fields, decomp, grid, ws)
        horizontal_laplacian_2d!(ws.lap_w, w, fields, decomp, grid, ws)

        parent(ws.Ru1) .+= nu .* parent(ws.lap_u)
        parent(ws.Rv1) .+= nu .* parent(ws.lap_v)
        parent(ws.Rw1) .+= nu .* parent(ws.lap_w)
    end

    # Explicit update for RHS*
    parent(u) .+= dt .* parent(ws.Ru1)
    parent(v) .+= dt .* parent(ws.Rv1)
    parent(w) .+= dt .* parent(ws.Rw1)

    # Implicit Coriolis
    fval = fplane === nothing ? 0.0 : (hasproperty(fplane, :f) ? fplane.f : 0.0)
    coriolis_implicit_solve!(u, v; f=fval, dt=dt, theta=theta)

    # Implicit vertical viscosity via Helmholtz in z
    alpha = nu * dt
    if alpha != 0
        helmholtz_z_solve!(u, u; alpha=alpha, grid=grid, bc=bc, theta=theta)
        helmholtz_z_solve!(v, v; alpha=alpha, grid=grid, bc=bc, theta=theta)
        helmholtz_z_solve!(w, w; alpha=alpha, grid=grid, bc=bc, theta=theta)
    end

    # Projection
    project_velocity!(u, v, w, p, ws.divu, dt;
                      fields=fields, 
                      decomp=decomp, 
                      grid=grid, 
                      bc=bc,
                      poisson_plan=poisson_plan, 
                      ws=ws,
                      poisson_method=poisson_method, 
                      mg_plan=mg_plan)

    return u, v, w, b, p
end

"""
    momentum_rhs!(Ru,Rv,Rw, u,v,w, b; nu, fplane, fields, decomp, grid, bc, nlin_ws, ws)

Assemble momentum RHS for Boussinesq NSE in Z-pencil:
Ru = (u·∇)u + Cu + nu∇²u
Rv = (u·∇)v + Cv + nu∇²v
Rw = (u·∇)w + b + nu∇²w

Uses precomputed wavenumbers from RSNSWorkspace for efficiency.
Pressure gradient is excluded (handled by projection).
"""
function momentum_rhs!(Ru, Rv, Rw, u, v, w, b;
                       nu::Real, fplane,
                       fields, decomp, grid, bc,
                       nlin_ws::NonlinearWorkspace,
                       ws::Union{RSNSWorkspace, NamedTuple, Nothing}=nothing,
                       velocity_forcing::Union{VelocityField, Nothing}=nothing,
                       forcing_strength::Real=0.0,
                       kx2_loc::Vector{<:Real}=Float64[],
                       ky2_loc::Vector{<:Real}=Float64[])

    nablaU = compute_gradients_U!(u, v, w, fields, decomp, grid, bc, nlin_ws)
    lap_u = nlin_ws.tmpx; 
    lap_v = nlin_ws.tmpy; 
    lap_w = nlin_ws.tmpz

    # Use workspace or separate wavenumber arrays
    if ws !== nothing
        laplacian_3d!(lap_u, u, fields, decomp, grid, bc, :u, ws)
        laplacian_3d!(lap_v, v, fields, decomp, grid, bc, :v, ws)
        laplacian_3d!(lap_w, w, fields, decomp, grid, bc, :w, ws)
    else
        laplacian_3d!(lap_u, u, fields, decomp, grid, bc, :u, kx2_loc, ky2_loc)
        laplacian_3d!(lap_v, v, fields, decomp, grid, bc, :v, kx2_loc, ky2_loc)
        laplacian_3d!(lap_w, w, fields, decomp, grid, bc, :w, kx2_loc, ky2_loc)
    end
    lapU = (lap_u, lap_v, lap_w)

    U = (u,v,w); 
    R = (Ru,Rv,Rw)
    advect_vector!(R, U, nablaU)

    if fplane !== nothing && getfield(fplane, :f) != 0
        add_coriolis_fplane!(R, U, fplane.f)
    end
    Rw_data = parent(Rw)
    b_data = parent(b)
    @. Rw_data += b_data
    add_diffusion3!(R, lapU, nu)
    
    # Add spatial velocity forcing if provided
    if velocity_forcing !== nothing && forcing_strength != 0.0
        add_spatial_forcing!(Ru, Rv, Rw, velocity_forcing, grid; 
                        strength=forcing_strength)
    end
    
    return Ru, Rv, Rw
end

# The unified function above handles both workspace and separate wavenumber arguments

"""
    momentum_rhs_conservative!(Ru,Rv,Rw, u,v,w, b; nu, fplane, fields, decomp, grid, bc, nlin_ws, ws, kx2_loc, ky2_loc)

Assemble momentum RHS for Boussinesq NSE using flux conservative advection:
Ru = -∇·(uu) + u(∇·u) + Cu + nu∇²u
Rv = -∇·(vu) + v(∇·u) + Cv + nu∇²v
Rw = -∇·(wu) + w(∇·u) + b + nu∇²w

Pressure gradient is excluded (handled by projection).
Requires RSNSWorkspace for flux conservative arrays and wavenumber arrays.

# Benefits of Flux Conservative Form
- Exact momentum conservation in discrete form
- Better numerical stability for high Reynolds numbers  
- Preserves energy cascade properties in turbulent flows
- Maintains Galilean invariance
"""
function momentum_rhs_conservative!(Ru, Rv, Rw, u, v, w, b;
                                   nu::Real, fplane,
                                   fields, decomp, grid, bc,
                                   nlin_ws::NonlinearWorkspace,
                                   ws::RSNSWorkspace,
                                   kx2_loc::Vector{<:Real}=Float64[],
                                   ky2_loc::Vector{<:Real}=Float64[])
    # Compute velocity gradients
    nablaU = compute_gradients_U!(u, v, w, fields, decomp, grid, bc, nlin_ws)
    
    # Compute Laplacians for diffusion using workspace or separate wavenumber arrays
    lap_u = nlin_ws.tmpx; 
    lap_v = nlin_ws.tmpy; 
    lap_w = nlin_ws.tmpz

    # Use workspace wavenumbers if separate arrays not provided
    if isempty(kx2_loc) || isempty(ky2_loc)
        laplacian_3d!(lap_u, u, fields, decomp, grid, bc, :u, ws)
        laplacian_3d!(lap_v, v, fields, decomp, grid, bc, :v, ws)
        laplacian_3d!(lap_w, w, fields, decomp, grid, bc, :w, ws)
    else
        laplacian_3d!(lap_u, u, fields, decomp, grid, bc, :u, kx2_loc, ky2_loc)
        laplacian_3d!(lap_v, v, fields, decomp, grid, bc, :v, kx2_loc, ky2_loc)
        laplacian_3d!(lap_w, w, fields, decomp, grid, bc, :w, kx2_loc, ky2_loc)
    end
    lapU = (lap_u, lap_v, lap_w)

    # Compute flux conservative quantities using workspace
    U = (u, v, w)
    R = (Ru, Rv, Rw)
    flux_div = (ws.div_uu, ws.div_vu, ws.div_wu)
    
    # Compute momentum flux divergence and velocity divergence
    compute_momentum_flux_divergence!(flux_div, U, nablaU)
    compute_velocity_divergence!(ws.div_U, nablaU)
    
    # Apply flux conservative advection
    advect_vector_conservative!(R, U, flux_div, ws.div_U)
    
    # Add other physics
    if fplane !== nothing && getfield(fplane, :f) != 0
        add_coriolis_fplane!(R, U, fplane.f)
    end
    Rw_data = parent(Rw)
    b_data = parent(b)
    @. Rw_data += b_data
    add_diffusion3!(R, lapU, nu)
    
    return Ru, Rv, Rw
end

# The unified function above handles both workspace and separate wavenumber arguments

"""
    buoyancy_rhs!(Rb, u,v,w, b; kappa, N2, fields, decomp, grid, bc, ws)

Assemble buoyancy RHS (linear stratification with constant N2):
Rb = (u·∇)b + N²w + κ∇²b
"""
function buoyancy_rhs!(Rb, u, v, w, b;
                       kappa::Real, N2::Real,
                       fields, decomp, grid, bc,
                       ws::RSNSWorkspace,
                       bc_b=nothing, 
                       t=0.0, 
                       xnodes=nothing, 
                       ynodes=nothing,
                       stratification_forcing::Union{StratificationField, Nothing}=nothing,
                       stratification_strength::Real=0.0)

    # Gradients using workspace wavenumbers
    gradient_xy!(ws.dbdx, ws.dbdy, b, fields, decomp, grid, ws)
    if bc_b === nothing
        dz_derivative_nonuniform_with_bcs!(ws.dbdz, b, grid, bc, :b)
    else
        dz_derivative_buoyancy_with_bcs!(ws.dbdz, b, grid, bc_b, t; xnodes=xnodes, ynodes=ynodes)
    end
    nablab = (ws.dbdx, ws.dbdy, ws.dbdz)
    
    # Laplacian using workspace wavenumbers
    laplacian_3d!(ws.lap, b, fields, decomp, grid, bc, :b, ws)

    # Tensor-form assembly
    advect_scalar!(Rb, (u,v,w), nablab)
    Rb_data = parent(Rb)
    w_data = parent(w)
    @. Rb_data += N2 * w_data
    
    lap_data = parent(ws.lap)
    @. Rb_data += kappa * lap_data
    
    # Add spatial stratification forcing if provided
    if stratification_forcing !== nothing && stratification_strength != 0.0
        add_stratification_forcing!(Rb, stratification_forcing, grid; strength=stratification_strength)
    end
    
    return Rb
end

"""
    project_velocity!(u,v,w, pipi, divu, dt; fields, decomp, grid, bc, poisson_plan, ws::RSNSWorkspace)

Pressure projection: solve ∇²Π = (1/dt)∇·u and correct u = u - dt∇Π.
Uses precomputed wavenumbers from RSNSWorkspace for efficiency.
Assumes Z-pencil arrays and vertical BCs represented in the Poisson plan.
"""
function project_velocity!(u, v, w, pipi, divu, dt; 
                           fields, decomp, grid, bc,
                           poisson_plan=nothing,
                           ws::Union{RSNSWorkspace, NamedTuple, Nothing}=nothing,
                           poisson_method::Symbol=:fft,
                           mg_plan=nothing,
                           mg_cycles::Int=6, 
                           mg_pre::Int=3, 
                           mg_post::Int=3,
                           kx_loc::Vector{<:Real}=Float64[],
                           ky_loc::Vector{<:Real}=Float64[])

    # Compute divergence: dudx + dvdy + dwdz using workspace or separate wavenumbers
    if ws !== nothing
        gradient_xy!(divu, pipi, u, fields, decomp, grid, ws)   # divu:=dudx, pipi:=dudy (we will overwrite pipi)
        dudx = divu
        gradient_xy!(divu, pipi, v, fields, decomp, grid, ws)   # divu:=dvdx (unused), pipi:=dvdy
        dvdy = pipi
    else
        gradient_xy!(divu, pipi, u, fields, decomp, grid, kx_loc, ky_loc)   # divu:=dudx, pipi:=dudy (we will overwrite pipi)
        dudx = divu
        gradient_xy!(divu, pipi, v, fields, decomp, grid, kx_loc, ky_loc)   # divu:=dvdx (unused), pipi:=dvdy
        dvdy = pipi
    end
    dz_derivative_nonuniform_with_bcs!(pipi, w, grid, bc, :w)     # pipi:=dwdz
    @. divu = dudx + dvdy + pipi

    # Solve ∇²Π = (1/dt)∇·u
    @. divu /= dt
    if poisson_method === :fft
        if poisson_plan === nothing
            poisson_plan = make_poisson_plan(pipi; decomp=decomp, grid=grid, bc_z=:neumann)
        end
        solve_poisson!(pipi, divu, poisson_plan)
    elseif poisson_method === :mg
        if mg_plan === nothing
            # auto-adjust levels and build plan
            mg_plan = make_mg_poisson_distributed_auto(decomp, grid; levels=4, Π=0.8, 
                                            bc_z=Dict(:bottom_type=>:neumann, :top_type=>:neumann))
        end
        mg_solve_distributed!(pipi, divu, mg_plan; cycles=mg_cycles, pre=mg_pre, post=mg_post)
    else
        error("Unknown poisson_method=$(poisson_method). Use :fft or :mg.")
    end

    # Compute ∇Π using workspace or separate wavenumbers
    if ws !== nothing
        gradient_xy!(divu, pipi, pipi, fields, decomp, grid, ws)  # divu:=dxpipi, pipi:=dypipi
    else
        gradient_xy!(divu, pipi, pipi, fields, decomp, grid, kx_loc, ky_loc)  # divu:=dxpipi, pipi:=dypipi
    end
    dz_derivative_nonuniform_with_bcs!(fields.u_z, pipi, grid, bc, :p)  # reuse fields.u_z for dzpipi

    # Correct velocities: u -= dt∇Π
    @. u -= dt * divu
    @. v -= dt * pipi
    @. w -= dt * fields.u_z
    return u, v, w
end

# The unified function above handles both workspace and separate wavenumber arguments

"""
    predictor_corrector_step!(u,v,w,b,p, t, dt; kwargs...)

Advance one time step using a Heun predictorcorrector with pressure projection
for rotating, stratified Boussinesq flow. Arrays must be Z-pencil `PencilArray`s.

Keyword arguments (required)
- `decomp`: 2-D pencil decomposition (see `pencil_decomposition_2d.jl`)
- `grid`: grid object used by derivative helpers (must provide `dx, dy, Nz, dz`)
- `fields`: working arrays from `create_distributed_fields(decomp)`
- `bc`: `BoundaryCondition` for vertical operators
- `nu`: viscosity
- `kappa`: scalar diffusivity

Optional
- `fplane=FPlane()`: Coriolis f-plane (or `nothing` for non-rotating)
- `N2=0.0`: constant buoyancy frequency squared
- `poisson_plan=nothing`: prebuilt Poisson plan (created on first call if `nothing`)
- `nlin_ws=nothing`: `NonlinearWorkspace` cache (auto-alloc if `nothing`)
- `ws=nothing`: `RSNSWorkspace` cache (auto-alloc if `nothing`)

Returns updated `u,v,w,b,p` in place and the workspace/plan for reuse.
"""
function predictor_corrector_step!(u, v, w, b, p, t, dt;
                                decomp, grid, fields, bc,
                                nu::Real, kappa::Real,
                                fplane=FPlane(), N2::Real=0.0,
                                poisson_plan=nothing,
                                poisson_method::Symbol=:fft,
                                mg_plan=nothing,
                                mg_cycles::Int=6, mg_pre::Int=3, mg_post::Int=3,
                                nlin_ws=nothing,
                                ws=nothing,
                                bc_b=nothing,
                                xnodes=nothing,
                                ynodes=nothing)

    nlin_ws === nothing && (nlin_ws = NonlinearWorkspace(decomp))
    ws === nothing && (ws = RSNSWorkspace(decomp, grid, u))
    if poisson_method === :fft
        poisson_plan === nothing && (poisson_plan = make_poisson_plan(u; decomp=decomp, grid=grid, bc_z=:neumann))
    end

    # Enforce BCs before stepping
    apply_velocity_bcs_nonuniform!(u, v, w, grid, bc, t; xnodes=xnodes, ynodes=ynodes)
    if bc_b !== nothing
        apply_buoyancy_bcs_nonuniform!(b, grid, bc_b, t; xnodes=xnodes, ynodes=ynodes)
    end

    # RHS at t^n
    fill!(ws.Ru1, zero(eltype(ws.Ru1)))
    fill!(ws.Rv1, zero(eltype(ws.Rv1)))
    fill!(ws.Rw1, zero(eltype(ws.Rw1)))
    fill!(ws.Rb1, zero(eltype(ws.Rb1)))

    momentum_rhs_conservative!(ws.Ru1, ws.Rv1, ws.Rw1, u, v, w, b;
                            nu=nu, 
                            fplane=fplane,     
                            fields=fields, 
                            decomp=decomp, 
                            grid=grid, 
                            bc=bc, 
                            nlin_ws=nlin_ws, 
                            ws=ws)

    buoyancy_rhs!(ws.Rb1, u, v, w, b; 
                kappa=kappa, 
                N2=N2, 
                fields=fields, 
                decomp=decomp, 
                grid=grid, 
                bc=bc, 
                ws=ws,
                bc_b=bc_b, 
                t=t, 
                xnodes=xnodes, 
                ynodes=ynodes)

    # Predictor
    @. ws.ustar = u + dt * ws.Ru1
    @. ws.vstar = v + dt * ws.Rv1
    @. ws.wstar = w + dt * ws.Rw1
    @. ws.bstar = b + dt * ws.Rb1

    if bc_b !== nothing
        apply_buoyancy_bcs_nonuniform!(ws.bstar, grid, bc_b, t + dt; xnodes=xnodes, ynodes=ynodes)
    end

    # Project predictor to divergence-free
    project_velocity!(ws.ustar, ws.vstar, ws.wstar, ws.pipi, ws.divu, dt;
                    fields=fields, 
                    decomp=decomp, 
                    grid=grid, bc=bc,
                    poisson_plan=poisson_plan, 
                    ws=ws,
                    poisson_method=poisson_method, 
                    mg_plan=mg_plan,
                    mg_cycles=mg_cycles, 
                    mg_pre=mg_pre, 
                    mg_post=mg_post)

    # RHS at t^{n+1} using projected predictor state
    fill!(ws.Ru2, zero(eltype(ws.Ru2)))
    fill!(ws.Rv2, zero(eltype(ws.Rv2)))
    fill!(ws.Rw2, zero(eltype(ws.Rw2)))
    fill!(ws.Rb2, zero(eltype(ws.Rb2)))

    momentum_rhs_conservative!(ws.Ru2, ws.Rv2, ws.Rw2, ws.ustar, ws.vstar, ws.wstar, ws.bstar;
                            nu=nu, 
                            fplane=fplane, 
                            fields=fields, 
                            decomp=decomp, 
                            grid=grid, 
                            bc=bc, 
                            nlin_ws=nlin_ws, 
                            ws=ws)

    buoyancy_rhs!(ws.Rb2, ws.ustar, ws.vstar, ws.wstar, ws.bstar; kappa=kappa, N2=N2, 
                fields=fields, 
                decomp=decomp, 
                grid=grid, 
                bc=bc, 
                ws=ws,
                bc_b=bc_b, 
                t=t+dt, 
                xnodes=xnodes, 
                ynodes=ynodes)

    # Heun corrector
    @. u += 0.5 * dt * (ws.Ru1 + ws.Ru2)
    @. v += 0.5 * dt * (ws.Rv1 + ws.Rv2)
    @. w += 0.5 * dt * (ws.Rw1 + ws.Rw2)
    @. b += 0.5 * dt * (ws.Rb1 + ws.Rb2)

    if bc_b !== nothing
        apply_buoyancy_bcs_nonuniform!(b, grid, bc_b, t + dt; 
                                xnodes=xnodes, ynodes=ynodes)
    end

    # Final projection
    project_velocity!(u, v, w, ws.pipi, ws.divu, dt;
                    fields=fields, 
                    decomp=decomp, 
                    grid=grid, 
                    bc=bc,
                    poisson_plan=poisson_plan, 
                    ws=ws,
                    poisson_method=poisson_method, 
                    mg_plan=mg_plan,
                    mg_cycles=mg_cycles, 
                    mg_pre=mg_pre, 
                    mg_post=mg_post)

    # Pressure update (optional: accumulate correction)
    @. p += ws.pipi

    return u, v, w, b, p, (ws=ws, nlin_ws=nlin_ws, poisson_plan=poisson_plan)
end
