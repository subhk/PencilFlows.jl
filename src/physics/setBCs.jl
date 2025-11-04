module BoundaryConditionsNonUniform

"""
BoundaryConditionsNonUniform *robust boundary-condition utilities for stretched grids.

This lightweight module complements the *PencilDecomp2D* solver.  It
implements first- and second-order finite-difference operators that respect
no-slip, free-slip, stress-free, Robin, and prescribed-velocity conditions on
non-uniform vertical meshes.

Usage example
-------------
```julia
using BoundaryConditionsNonUniform, PencilDecomp2D

# build a boundary-layer friendly mesh
gr = Grid(Nx, Ny, Nz, Lx, Ly, H; dz = create_boundary_layer_grid(Nz, H, delta_bl))

# choose BCs
bc = BoundaryCondition(NO_SLIP, FREE_SLIP)

validate_bc_grid_compatibility(gr, bc)

# allocate arrays and apply BCs
a = rand(Float64, Nx, Ny, Nz)
apply_velocity_bcs_nonuniform!(a, a, a, gr, bc)  # quick test
```
"""

using LinearAlgebra

# ---------------------------
# Public API
# ---------------------------
export BoundaryType, BoundaryCondition,
       NO_SLIP, 
       FREE_SLIP, 
       PRESCRIBED_VELOCITY, 
       STRESS_FREE, 
       PERIODIC_Z, 
       ROBIN,
       compute_bc_coefficients,
       apply_velocity_bcs_nonuniform!,
       dz_derivative_nonuniform_with_bcs!,
       d2z_derivative_nonuniform_with_bcs!,
       validate_bc_grid_compatibility,
       create_boundary_layer_grid,
       BuoyancyBCType, BuoyancyBC,
       apply_buoyancy_bcs_nonuniform!,
       dz_derivative_buoyancy_with_bcs!

# ---------------------------
# Enums & data containers
# ---------------------------
@enum BoundaryType begin
    NO_SLIP
    FREE_SLIP
    PRESCRIBED_VELOCITY
    STRESS_FREE
    PERIODIC_Z
    ROBIN  # α u + β du/dz = γ
end

"""Composite type storing all boundary information for top & bottom walls."""
struct BoundaryCondition
    bottom::BoundaryType
    top::BoundaryType

    # prescribed velocities (if relevant) - can be Number or Function
    bottom_u::Union{Real, Function}
    bottom_v::Union{Real, Function}
    bottom_w::Union{Real, Function}
    top_u::Union{Real, Function}
    top_v::Union{Real, Function}
    top_w::Union{Real, Function}

    # Robin coefficients  α u + beta du/dz = gamma - can be Number or Function
    bottom_alpha_u::Union{Real, Function}
    bottom_beta_u::Union{Real, Function}
    bottom_gamma_u::Union{Real, Function}
    top_alpha_u::Union{Real, Function}
    top_beta_u::Union{Real, Function}
    top_gamma_u::Union{Real, Function}
end

"""Convenience constructor for the typical case when only the BC types are needed."""
function BoundaryCondition(bottom::BoundaryType, top::BoundaryType)
    BoundaryCondition(bottom, top,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

"""
Enhanced constructor for time/spatial-dependent velocity boundary conditions.

# Arguments
- `bottom`, `top`: BoundaryType for bottom and top walls
- `bottom_vel`, `top_vel`: Velocity tuples (u,v,w) - can be Numbers, Functions, or mixed
- `bottom_robin`, `top_robin`: Robin coefficient tuples (α,beta,gamma) for ROBIN BCs (optional)

# Function Signatures Supported
- `f(t)`: Time-dependent only
- `f(i,j,t)`: Grid index dependent  
- `f(x,y,t)`: Coordinate dependent (requires xnodes/ynodes in apply call)

# Example Usage
```julia
# Time-dependent velocity BC
bc = BoundaryCondition(PRESCRIBED_VELOCITY, NO_SLIP,
                      (t -> sin(t), 0.0, 0.0),  # u oscillates at bottom
                      (0.0, 0.0, 0.0))          # no-slip at top

# Spatial-dependent BC  
bc = BoundaryCondition(PRESCRIBED_VELOCITY, FREE_SLIP,
                      ((x,y,t) -> x*sin(t), 0.0, 0.0),  # u varies with x
                      (0.0, 0.0, 0.0))
```
"""
function BoundaryCondition(bottom::BoundaryType, top::BoundaryType,
                          bottom_vel::Tuple{Any,Any,Any}, top_vel::Tuple{Any,Any,Any};
                          bottom_robin::Tuple{Any,Any,Any}=(0.0, 0.0, 0.0),
                          top_robin::Tuple{Any,Any,Any}=(0.0, 0.0, 0.0))
    BoundaryCondition(bottom, top,
                      bottom_vel[1], bottom_vel[2], bottom_vel[3],
                      top_vel[1], top_vel[2], top_vel[3],
                      bottom_robin[1], bottom_robin[2], bottom_robin[3],
                      top_robin[1], top_robin[2], top_robin[3])
end

# ---------------------------
# Coefficient helpers
# ---------------------------
"""Return modified first- & second-derivative stencil coefficients that respect the
specified boundary conditions on a non-uniform grid."""
function compute_bc_coefficients(grid, bc::BoundaryCondition)
    Nz   = grid.Nz
    dz   = grid.dz

    c1L = copy(grid.c1_lower)
    c1D = copy(grid.c1_diag)
    c1U = copy(grid.c1_upper)
    c2L = copy(grid.c2_lower)
    c2D = copy(grid.c2_diag)
    c2U = copy(grid.c2_upper)

    # ---------- bottom wall ----------
    if bc.bottom == NO_SLIP
        c1D[1] = c1U[1] = 0.0
        c2D[1] = c2U[1] = 0.0

    elseif bc.bottom == FREE_SLIP
        h1, h2 = dz[1], (length(dz) > 1 ? dz[2] : dz[1])
        c1L[1] = c1D[1] = c1U[1] = 0.0
        c2D[1] = -2 / (h1 * h2)
        c2U[1] =  2 / (h1 * h2)

    elseif bc.bottom == ROBIN
        h1 = dz[1]
        alpha, beta = bc.bottom_alpha_u, bc.bottom_beta_u
        c1D[1] = alpha - beta / h1
        c1U[1] = beta / h1
        c1L[1] = 0.0
    end

    # ---------- top wall ----------
    if bc.top == NO_SLIP
        c1D[Nz] = c1L[Nz] = 0.0
        c2D[Nz] = c2L[Nz] = 0.0

    elseif bc.top == FREE_SLIP
        h1, h2 = dz[Nz-1], (Nz > 2 ? dz[Nz-2] : dz[Nz-1])
        c1L[Nz] = c1D[Nz] = c1U[Nz] = 0.0
        c2D[Nz] = -2 / (h1 * h2)
        c2L[Nz] =  2 / (h1 * h2)

    elseif bc.top == ROBIN
        h1 = dz[Nz-1]
        alpha, beta = bc.top_alpha_u, bc.top_beta_u
        c1D[Nz] = alpha + beta / h1
        c1L[Nz] = -beta / h1
        c1U[Nz] = 0.0
    end

    return c1L, c1D, c1U, c2L, c2D, c2U
end

# ---------------------------
# Boundary-aware finite differences
# ---------------------------
"""First-order derivative d/dz on stretched meshes with BC enforcement."""
function dz_derivative_nonuniform_with_bcs!(out, input, grid, bc::BoundaryCondition, var::Symbol)
    c1L, c1D, c1U, _, _, _ = compute_bc_coefficients(grid, bc)
    Nz = grid.Nz
    for k in 1:size(input,1), j in 1:size(input,2)
        # interior
        for i in 2:Nz-1
            out[k,j,i] = c1L[i]*input[k,j,i-1] + c1D[i]*input[k,j,i] + c1U[i]*input[k,j,i+1]
        end
        # boundaries
        out[k,j,1]  = c1D[1]*input[k,j,1]   + c1U[1]*input[k,j,2]
        out[k,j,Nz] = c1L[Nz]*input[k,j,Nz-1] + c1D[Nz]*input[k,j,Nz]
    end
    if bc.bottom == FREE_SLIP && var in (:u,:v)
        out[:,:,1] .= 0.0
    end
    if bc.top == FREE_SLIP && var in (:u,:v)
        out[:,:,Nz] .= 0.0
    end
    return out
end

"""Second-order derivative d2/dz2 on stretched meshes with BC enforcement."""
function d2z_derivative_nonuniform_with_bcs!(out, input, grid, bc::BoundaryCondition, var::Symbol)
    _, _, _, c2L, c2D, c2U = compute_bc_coefficients(grid, bc)
    Nz = grid.Nz

    # Interior points
    for k in 1:size(input,1), j in 1:size(input,2)
        for i in 2:Nz-1
            out[k,j,i] = c2L[i]*input[k,j,i-1] + c2D[i]*input[k,j,i] + c2U[i]*input[k,j,i+1]
        end
    end

    # Boundary points - handle differently based on BC type
    # Bottom boundary
    if bc.bottom == NO_SLIP || bc.bottom == PRESCRIBED_VELOCITY
        # For no-slip/prescribed: d2u/dz2 at wall computed from interior points
        out[:,:,1] .= c2D[1]*input[:,:,1] + c2U[1]*input[:,:,2]
    elseif bc.bottom == FREE_SLIP
        # For free-slip on u,v: du/dz = 0, so d2u/dz2 is computed normally
        if var in (:u, :v)
            out[:,:,1] .= c2D[1]*input[:,:,1] + c2U[1]*input[:,:,2]
        else
            # For w: d2w/dz2 at wall
            out[:,:,1] .= c2D[1]*input[:,:,1] + c2U[1]*input[:,:,2]
        end
    elseif bc.bottom == STRESS_FREE
        # For stress-free: d2u/dz2 = 0 at wall
        out[:,:,1] .= 0.0
    else
        # Robin or other: compute from coefficients
        out[:,:,1] .= c2D[1]*input[:,:,1] + c2U[1]*input[:,:,2]
    end

    # Top boundary
    if bc.top == NO_SLIP || bc.top == PRESCRIBED_VELOCITY
        # For no-slip/prescribed: d2u/dz2 at wall computed from interior points
        out[:,:,Nz] .= c2L[Nz]*input[:,:,Nz-1] + c2D[Nz]*input[:,:,Nz]
    elseif bc.top == FREE_SLIP
        # For free-slip on u,v: du/dz = 0, so d2u/dz2 is computed normally
        if var in (:u, :v)
            out[:,:,Nz] .= c2L[Nz]*input[:,:,Nz-1] + c2D[Nz]*input[:,:,Nz]
        else
            # For w: d2w/dz2 at wall
            out[:,:,Nz] .= c2L[Nz]*input[:,:,Nz-1] + c2D[Nz]*input[:,:,Nz]
        end
    elseif bc.top == STRESS_FREE
        # For stress-free: d2u/dz2 = 0 at wall
        out[:,:,Nz] .= 0.0
    else
        # Robin or other: compute from coefficients
        out[:,:,Nz] .= c2L[Nz]*input[:,:,Nz-1] + c2D[Nz]*input[:,:,Nz]
    end

    return out
end

# ---------------------------
# Velocity array BC setter
# ---------------------------
function apply_velocity_bcs_nonuniform!(u, v, w, grid, bc::BoundaryCondition, t::Real;
                                       xnodes=nothing, ynodes=nothing)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    dz = grid.dz

    # --- helper closures ---
    extrap_bottom!(A) = (A[:,:,1] .= Nz>=3 ? A[:,:,2] .- (dz[1]/(Nz>1 ? dz[2] : dz[1])) .* (A[:,:,3] .- A[:,:,2]) : A[:,:,2])
    extrap_top!(A)    = (A[:,:,Nz] .= Nz>=3 ? A[:,:,Nz-1] .+ (dz[Nz-1]/(Nz>2 ? dz[Nz-2] : dz[Nz-1])) .* (A[:,:,Nz-1] .- A[:,:,Nz-2]) : A[:,:,Nz-1])

    # bottom wall
    if bc.bottom == NO_SLIP || bc.bottom == PRESCRIBED_VELOCITY
        # Handle time/spatial dependent velocity values
        @inbounds @simd for j in 1:Ny
            for i in 1:Nx
                u_val, _, _, _ = _eval_velocity_bc_values(bc, :u, :bottom, t; i=i, j=j, xnodes=xnodes, ynodes=ynodes)
                v_val, _, _, _ = _eval_velocity_bc_values(bc, :v, :bottom, t; i=i, j=j, xnodes=xnodes, ynodes=ynodes)
                w_val, _, _, _ = _eval_velocity_bc_values(bc, :w, :bottom, t; i=i, j=j, xnodes=xnodes, ynodes=ynodes)
                u[i,j,1] = u_val
                v[i,j,1] = v_val
                w[i,j,1] = w_val
            end
        end
    elseif bc.bottom == FREE_SLIP
        w[:,:,1] .= 0.0; 
        extrap_bottom!(u); 
        extrap_bottom!(v)
    elseif bc.bottom == STRESS_FREE
        extrap_bottom!(u); 
        extrap_bottom!(v); 
        extrap_bottom!(w)
    elseif bc.bottom == ROBIN
        # Handle Robin BC: α u + beta du/dz = gamma with time/spatial dependence
        h1 = dz[1]
        @inbounds @simd for j in 1:Ny
            for i in 1:Nx
                # For each velocity component
                for (comp, vel_array) in [(:u, u), (:v, v), (:w, w)]
                    _, alpha, beta, gamma = _eval_velocity_bc_values(bc, comp, :bottom, t; i=i, j=j, xnodes=xnodes, ynodes=ynodes)
                    if beta != 0
                        # Solve α u + beta (u[2] - u[1])/h1 = gamma for u[1]
                        denom = alpha - beta/h1
                        if abs(denom) > 1e-12
                            vel_array[i,j,1] = (gamma - beta*vel_array[i,j,2]/h1) / denom
                        else
                            # Fallback to extrapolation if ill-conditioned
                            vel_array[i,j,1] = vel_array[i,j,2]
                        end
                    else
                        # Dirichlet-like: α u = gamma
                        vel_array[i,j,1] = alpha != 0 ? gamma/alpha : 0.0
                    end
                end
            end
        end
    end

    # top wall
    if bc.top == NO_SLIP || bc.top == PRESCRIBED_VELOCITY
        # Handle time/spatial dependent velocity values
        @inbounds for j in 1:Ny, i in 1:Nx
            u_val, _, _, _ = _eval_velocity_bc_values(bc, :u, :top, t; i=i, j=j, xnodes=xnodes, ynodes=ynodes)
            v_val, _, _, _ = _eval_velocity_bc_values(bc, :v, :top, t; i=i, j=j, xnodes=xnodes, ynodes=ynodes)
            w_val, _, _, _ = _eval_velocity_bc_values(bc, :w, :top, t; i=i, j=j, xnodes=xnodes, ynodes=ynodes)
            u[i,j,Nz] = u_val
            v[i,j,Nz] = v_val
            w[i,j,Nz] = w_val
        end
    elseif bc.top == FREE_SLIP
        w[:,:,Nz] .= 0.0; 
        extrap_top!(u); 
        extrap_top!(v)
    elseif bc.top == STRESS_FREE
        extrap_top!(u); 
        extrap_top!(v); 
        extrap_top!(w)
    elseif bc.top == ROBIN
        # Handle Robin BC: α u + beta du/dz = gamma with time/spatial dependence
        hN = dz[end]
        @inbounds for j in 1:Ny, i in 1:Nx
            # For each velocity component
            for (comp, vel_array) in [(:u, u), (:v, v), (:w, w)]
                _, alpha, beta, gamma = _eval_velocity_bc_values(bc, comp, :top, t; i=i, j=j, xnodes=xnodes, ynodes=ynodes)
                if beta != 0
                    # Solve α u + beta (u[Nz] - u[Nz-1])/hN = gamma for u[Nz]
                    denom = alpha + beta/hN
                    if abs(denom) > 1e-12
                        vel_array[i,j,Nz] = (gamma + beta*vel_array[i,j,Nz-1]/hN) / denom
                    else
                        # Fallback to extrapolation if ill-conditioned
                        vel_array[i,j,Nz] = vel_array[i,j,Nz-1]
                    end
                else
                    # Dirichlet-like: α u = gamma
                    vel_array[i,j,Nz] = alpha != 0 ? gamma/alpha : 0.0
                end
            end
        end
    end

    return nothing
end

# Backward compatibility method - maintains old signature for existing code
function apply_velocity_bcs_nonuniform!(u, v, w, grid, bc::BoundaryCondition)
    apply_velocity_bcs_nonuniform!(u, v, w, grid, bc, 0.0; xnodes=nothing, ynodes=nothing)
end

# ---------------------------
# Utility / diagnostics
# ---------------------------
function validate_bc_grid_compatibility(grid, bc::BoundaryCondition)
    ar = maximum(grid.dz) / minimum(grid.dz)
    if ar > 10
        @warn "High grid aspect ratio ($(round(ar,digits=1))) *BC accuracy may suffer."
    end
    if (bc.bottom == NO_SLIP && grid.dz[1] > 0.1) || (bc.top == NO_SLIP && grid.dz[end] > 0.1)
        @warn "Near-wall cell too coarse for no-slip BC; consider stronger stretching."
    end
    return true
end

# ---------------------------
# Boundary-layer mesh generator
# ---------------------------
"""Return a 1-D mesh (cell centres & faces) concentrated in layers of thickness `delta_bl`."""
function create_boundary_layer_grid(Nz::Int, H::Float64, delta_bl::Float64)
    lambda = 3.0                      # stretching strength
    eta = range(-1, 1; length=Nz+1)
    zf = (H/2) .* (1 .+ tanh.(lambda .* eta) ./ tanh(lambda))

    # tweak to guarantee delta_bl resolution
    zf .= map(zf) do zz
        if zz < delta_bl
            zz * (delta_bl / (2*delta_bl))
        elseif zz > H - delta_bl
            delta_bl + (zz - (H - delta_bl)) * (delta_bl / (H - (H - delta_bl)))
        else
            zz
        end
    end

    z  = 0.5 .* (zf[1:end-1] .+ zf[2:end])
    dz = diff(zf)
    return z, zf, dz, dz[1:end-1]
end

# ---------------------------
# Buoyancy boundary conditions (Dirichlet / Neumann; constant / time-/space-varying)
# ---------------------------

@enum BuoyancyBCType begin
    B_CONSTANT        # Dirichlet: b = constant (optionally time-dependent via function)
    B_FLUX            # Neumann:   db/dz = q (flux), value or function
    B_FUNCTION        # Dirichlet: b = f(...)
    B_FLUX_FUNCTION   # Neumann:   db/dz = q(...)
    B_ROBIN           # Robin:     α b + β db/dz = γ (constants or functions)
end

"""BuoyancyBC  boundary condition specification for buoyancy/temperature.

Fields `bottom_value` and `top_value` may be `Number` or `Function`.
Function signatures supported (auto-detected):
  ¢ f(t)
  ¢ f(i, j, t)
  ¢ f(x, y, t)   when `xnodes`, `ynodes` are provided in the call
"""
struct BuoyancyBC
    bottom_type :: BuoyancyBCType
    top_type    :: BuoyancyBCType
    bottom_value::Union{Real, Function}
    top_value   :: Union{Real, Function}
    bottom_alpha::Union{Real, Function}
    bottom_beta ::Union{Real, Function}
    bottom_gamma::Union{Real, Function}
    top_alpha   ::Union{Real, Function}
    top_beta    ::Union{Real, Function}
    top_gamma   ::Union{Real, Function}
end

BuoyancyBC(bottom::BuoyancyBCType, top::BuoyancyBCType; bottom_value=0.0, top_value=0.0,
           bottom_alpha=0.0, bottom_beta=1.0, bottom_gamma=0.0,
           top_alpha=0.0, top_beta=1.0, top_gamma=0.0) =
    BuoyancyBC(bottom, top, bottom_value, top_value,
               bottom_alpha, bottom_beta, bottom_gamma,
               top_alpha, top_beta, top_gamma)

"""
    apply_buoyancy_bcs_nonuniform!(b, grid, bc_b::BuoyancyBC, t; xnodes=nothing, ynodes=nothing)

Apply buoyancy boundary conditions on non-uniform vertical meshes by setting the
boundary planes `b[:,:,1]` and `b[:,:,Nz]` to satisfy Dirichlet/Neumann specs.

Supported types:
- `B_CONSTANT`/`B_FUNCTION`: sets `b` directly at the boundary.
- `B_FLUX`/`B_FLUX_FUNCTION`: enforces flux `q = db/dz` via first-order one-sided update:
    bottom: b = b -  q alphaz,  top: b_N = b_{N- 1} + q alphaz_{N- 1}

For function values, accepted signatures: `f(t)`, `f(i,j,t)`, or `f(x,y,t)` if
`xnodes`, `ynodes` are passed.
"""
function apply_buoyancy_bcs_nonuniform!(b, grid, bc_b::BuoyancyBC, t; 
                                    xnodes=nothing, ynodes=nothing)
    Nz = grid.Nz
    h1 = grid.dz[1]
    hN = grid.dz[end]

    # helper to evaluate a possibly functional BC
    _eval(val, i, j) = val isa Number ? val : (
        val isa Function ? (
            try
                val(t)
            catch
                try
                    val(i, j, t)
                catch
                    if xnodes === nothing || ynodes === nothing
                        error("Buoyancy BC function requires xnodes/ynodes or (i,j,t) signature")
                    end
                    val(xnodes[i], ynodes[j], t)
                end
            end
        ) : error("Unsupported BC value type: $(typeof(val))")
    )

    _eval3(a, bb, c, i, j) = (_eval(a, i, j), _eval(bb, i, j), _eval(c, i, j))

    @inbounds for j in axes(b,2), i in axes(b,1)
        # bottom boundary
        if bc_b.bottom_type in (B_CONSTANT, B_FUNCTION)
            b[i,j,1] = _eval(bc_b.bottom_value, i, j)

        elseif bc_b.bottom_type in (B_FLUX, B_FLUX_FUNCTION)
            q = _eval(bc_b.bottom_value, i, j)
            b[i,j,1] = b[i,j,2] - q*h1

        elseif bc_b.bottom_type == B_ROBIN
            alpha, beta, gamma = _eval3(bc_b.bottom_alpha, bc_b.bottom_beta, bc_b.bottom_gamma, i, j)
            denom = (alpha - beta/h1)
            b[i,j,1] = denom == 0 ? b[i,j,1] : (gamma - beta*b[i,j,2]/h1) / denom
            
        else
            error("Unknown bottom buoyancy BC type: $(bc_b.bottom_type)")
        end

        # top boundary
        if bc_b.top_type in (B_CONSTANT, B_FUNCTION)
            b[i,j,Nz] = _eval(bc_b.top_value, i, j)
        elseif bc_b.top_type in (B_FLUX, B_FLUX_FUNCTION)
            q = _eval(bc_b.top_value, i, j)
            b[i,j,Nz] = b[i,j,Nz-1] + q*hN
        elseif bc_b.top_type == B_ROBIN
            alpha, beta, gamma = _eval3(bc_b.top_alpha, bc_b.top_beta, bc_b.top_gamma, i, j)
            denom = (alpha + beta/hN)
            b[i,j,Nz] = denom == 0 ? b[i,j,Nz] : (gamma + beta*b[i,j,Nz-1]/hN) / denom
        else
            error("Unknown top buoyancy BC type: $(bc_b.top_type)")
        end
    end
    return nothing
end

"""
    dz_derivative_buoyancy_with_bcs!(dbdz, b, grid, bc_b::BuoyancyBC, t; xnodes=nothing, ynodes=nothing)

Compute db/dz with BC enforcement for buoyancy on non-uniform z.
Interior: 3-point non-uniform stencil. Boundaries: Dirichlet   one-sided
stencil; Neumann   set prescribed flux; Robin   from α·b + β·∂b/∂z = γ.
"""
function dz_derivative_buoyancy_with_bcs!(dbdz, b, grid, bc_b::BuoyancyBC, t; 
                                        xnodes=nothing, ynodes=nothing)
    Nz = grid.Nz
    h = grid.dz

    _eval(val, i, j) = val isa Number ? val : (
        val isa Function ? (
            try
                val(t)
            catch
                try
                    val(i, j, t)
                catch
                    if xnodes === nothing || ynodes === nothing
                        error("Buoyancy BC function requires xnodes/ynodes or (i,j,t) signature")
                    end
                    val(xnodes[i], ynodes[j], t)
                end
            end
        ) : error("Unsupported BC value type: $(typeof(val))")
    )
    _eval3(a, bb, c, i, j) = (_eval(a, i, j), _eval(bb, i, j), _eval(c, i, j))

    @inbounds for j in axes(b,2), i in axes(b,1)
        # interior
        for k in 2:Nz-1
            hb = h[k-1]; hf = h[k]
            a = -hf/(hb*(hb+hf))
            c =  hb/(hf*(hb+hf))
            d = (hf - hb)/(hb*hf)

            dbdz[i,j,k] = a*b[i,j,k-1] + d*b[i,j,k] + c*b[i,j,k+1]
        end
        # bottom
        if bc_b.bottom_type in (B_CONSTANT, B_FUNCTION)
            h1 = h[1]; h2 = h[2]
            a1 = -(2h1 + h2)/(h1*(h1 + h2))
            a2 =  (h1 + h2)/(h1*h2)
            a3 = -h1/(h2*(h1 + h2))

            dbdz[i,j,1] = a1*b[i,j,1] + a2*b[i,j,2] + a3*b[i,j,3]
        elseif bc_b.bottom_type in (B_FLUX,)
            dbdz[i,j,1] = _eval(bc_b.bottom_value, i, j)
        elseif bc_b.bottom_type == B_ROBIN
            alpha, beta, gamma = _eval3(bc_b.bottom_alpha, bc_b.bottom_beta, bc_b.bottom_gamma, i, j)
            dbdz[i,j,1] = beta == 0 ? dbdz[i,j,1] : (gamma - alpha*b[i,j,1]) / beta
        end
        # top
        if bc_b.top_type in (B_CONSTANT, B_FUNCTION)
            hN1 = h[end-1]; hN = h[end]
            # backward 3-point at top using b[N], b[N-1], b[N-2]
            c1 =  (2hN1 + hN)/(hN*(hN1 + hN))
            c2 = -(hN1 + hN)/(hN1*hN)
            c3 =  hN1/(hN*(hN1 + hN))

            dbdz[i,j,Nz] = c1*b[i,j,Nz-2] + c3*b[i,j,Nz-1] + c2*b[i,j,Nz]
        elseif bc_b.top_type in (B_FLUX,)
            dbdz[i,j,Nz] = _eval(bc_b.top_value, i, j)
        elseif bc_b.top_type == B_ROBIN
            alpha, beta, gamma = _eval3(bc_b.top_alpha, bc_b.top_beta, bc_b.top_gamma, i, j)
            dbdz[i,j,Nz] = beta == 0 ? dbdz[i,j,Nz] : (gamma - alpha*b[i,j,Nz]) / beta
        end
    end
    return dbdz
end

# ---------------------------
# Centralized time-/space-dependent BC utilities (moved from symbolic_interface)
# ---------------------------

"""
    substitute_time_variable(expr, t_val)

Replace all instances of symbol `t` in an expression with `t_val`.
"""
# substitute_time_variable is now defined in parse_utils.jl to avoid duplication

"""
    _eval_bc_value(val, t; i=nothing, j=nothing, xnodes=nothing, ynodes=nothing)

Evaluate a BC value that may be a Number, Function, or Expr. Supports signatures:
- f(t)
- f(i,j,t)
- f(x,y,t) if xnodes/ynodes provided
"""
function _eval_bc_value(val, t; i=nothing, j=nothing, xnodes=nothing, ynodes=nothing)
    if val isa Number
        return val
    elseif val isa Function
        try
            # Try f(t) signature first
            return val(t)
        catch
            try
                # Try f(i,j,t) signature
                if i !== nothing && j !== nothing
                    return val(i, j, t)
                else
                    error("Function requires i,j parameters but they were not provided")
                end
            catch
                # Try f(x,y,t) signature
                if xnodes !== nothing && ynodes !== nothing && i !== nothing && j !== nothing
                    return val(xnodes[i], ynodes[j], t)
                else
                    error("Velocity BC function requires xnodes/ynodes or (i,j,t) signature")
                end
            end
        end
    else
        error("Unsupported velocity BC value type: $(typeof(val))")
    end
end

"""
    BCCache

Cached boundary condition values to avoid repeated evaluations during time stepping.
"""
mutable struct BCCache{T}
    time::T
    values::Dict{Tuple{Symbol,Symbol,Int,Int}, NTuple{4,T}}  # (component, boundary, i, j) -> (val, α, β, γ)
    valid::Bool
    
    BCCache{T}() where T = new{T}(zero(T), Dict{Tuple{Symbol,Symbol,Int,Int}, NTuple{4,T}}(), false)
end

"""
    _eval_velocity_bc_values(bc::BoundaryCondition, component::Symbol, boundary::Symbol, t; i=nothing, j=nothing, xnodes=nothing, ynodes=nothing, cache=nothing)

Evaluate velocity BC values for a specific component (:u, :v, :w) and boundary (:bottom, :top).
Returns (value, alpha, beta, gamma) tuple for the given boundary condition.

Uses caching to avoid repeated evaluations at the same time step.
"""
function _eval_velocity_bc_values(bc::BoundaryCondition, component::Symbol, boundary::Symbol, t; 
                                 i=nothing, j=nothing, xnodes=nothing, ynodes=nothing, 
                                 cache::Union{Nothing,BCCache}=nothing)
    
    # Check cache if available
    if cache !== nothing && cache.valid && cache.time == t && i !== nothing && j !== nothing
        key = (component, boundary, i, j)
        if haskey(cache.values, key)
            return cache.values[key]
        end
    end
    
    # Select the appropriate fields based on boundary and component
    if boundary === :bottom
        vel_val = component === :u ? bc.bottom_u : 
                  component === :v ? bc.bottom_v : bc.bottom_w
        alpha_val = bc.bottom_alpha_u  # Note: only u-component Robin coeffs are stored
        beta_val = bc.bottom_beta_u
        gamma_val = bc.bottom_gamma_u
    elseif boundary === :top
        vel_val = component === :u ? bc.top_u : 
                  component === :v ? bc.top_v : bc.top_w
        alpha_val = bc.top_alpha_u
        beta_val = bc.top_beta_u
        gamma_val = bc.top_gamma_u
    else
        error("Unknown boundary: $boundary")
    end
    
    # Evaluate each value
    vel = _eval_bc_value(vel_val, t; i=i, j=j, xnodes=xnodes, ynodes=ynodes)
    alpha = _eval_bc_value(alpha_val, t; i=i, j=j, xnodes=xnodes, ynodes=ynodes)
    beta = _eval_bc_value(beta_val, t; i=i, j=j, xnodes=xnodes, ynodes=ynodes)
    gamma = _eval_bc_value(gamma_val, t; i=i, j=j, xnodes=xnodes, ynodes=ynodes)
    
    result = (vel, alpha, beta, gamma)
    
    # Cache result if cache is available
    if cache !== nothing && i !== nothing && j !== nothing
        if !cache.valid || cache.time != t
            # New time step, clear cache
            empty!(cache.values)
            cache.time = t
            cache.valid = true
        end
        key = (component, boundary, i, j)
        cache.values[key] = result
    end
    
    return result
end


"""
    update_bc_value(bc::BoundaryCondition, field_name::Symbol, new_value::Real)

Helper to create a new BoundaryCondition with an updated numeric value.
"""
function update_bc_value(bc::BoundaryCondition, field_name::Symbol, new_value::Real)
    current_values = Dict(
        :bottom_u => bc.bottom_u, :bottom_v => bc.bottom_v, :bottom_w => bc.bottom_w,
        :top_u => bc.top_u, :top_v => bc.top_v, :top_w => bc.top_w,
        :bottom_alpha_u => bc.bottom_alpha_u, :bottom_beta_u => bc.bottom_beta_u, 
        :bottom_gamma_u => bc.bottom_gamma_u,
        :top_alpha_u => bc.top_alpha_u, :top_beta_u => bc.top_beta_u, 
        :top_gamma_u => bc.top_gamma_u
    )
    current_values[field_name] = new_value
    return BoundaryCondition(bc.bottom, bc.top,
                             current_values[:bottom_u], current_values[:bottom_v], current_values[:bottom_w],
                             current_values[:top_u], current_values[:top_v], current_values[:top_w],
                             current_values[:bottom_alpha_u], current_values[:bottom_beta_u], current_values[:bottom_gamma_u],
                             current_values[:top_alpha_u], current_values[:top_beta_u], current_values[:top_gamma_u])
end

"""
    update_velocity_bcs!(disc, velocity_bcs, t, solution)

Update time-/space-dependent velocity BCs. If BC values are time-only, updates
the numeric BoundaryCondition and applies via `apply_velocity_bcs_nonuniform!`.
If BC values depend on space, directly writes boundary planes using the
provided functions when `solution` is supplied.
"""
function update_velocity_bcs!(disc, velocity_bcs::Vector, t::Real, solution)
    bc = disc.boundary_conditions  # numeric BoundaryCondition (this file)
    grid = disc.grid

    Nx     = hasproperty(disc, :grid_x) ? length(disc.grid_x) : size(solution[:u], 1)
    Ny     = hasproperty(disc, :grid_y) ? length(disc.grid_y) : size(solution[:u], 2)
    xnodes = hasproperty(disc, :grid_x) ? disc.grid_x : nothing
    ynodes = hasproperty(disc, :grid_y) ? disc.grid_y : nothing

    # Collect specs that must be applied directly to boundary planes
    # Each item: (field::Symbol, location::Symbol, type::Symbol, value)
    apply_specs = Vector{Tuple{Symbol,Symbol,Symbol,Any}}()

    for bc_td in velocity_bcs
        val = getfield(bc_td, :value)
        bctype = getfield(bc_td, :type)
        is_spatial = false
        vtime = nothing
        if bctype == :robin && (val isa Tuple || val isa NamedTuple)
            # Tuple of (alpha, beta, gamma) coefficients (each Number/Expr/Function). Defer to direct apply.
            is_spatial = true
        elseif val isa Function
            # Try time-only first
            local tmp
            try
                tmp = val(t)
                vtime = tmp
            catch
                is_spatial = true
            end
        elseif val isa Expr || val isa Number
            vtime = _eval_bc_value(val, t)
        else
            @warn "Unsupported velocity BC value type $(typeof(val)); skipping"
            continue
        end

        fld = getfield(bc_td, :field)
        loc = getfield(bc_td, :location)
        # Dirichlet with time-only value can be stored in numeric BC.
        # Neumann/Robin (time-only or spatial) are applied directly below.
        if !is_spatial && bctype == :dirichlet
            if fld == :u
                if loc in (:left, :bottom); 
                    bc = update_bc_value(bc, :bottom_u, vtime)
                elseif loc in (:right, :top); 
                    bc = update_bc_value(bc, :top_u, vtime) 
                end
            elseif fld == :v
                if loc in (:left, :bottom); 
                    bc = update_bc_value(bc, :bottom_v, vtime)
                elseif loc in (:right, :top); 
                    bc = update_bc_value(bc, :top_v, vtime) 
                end
            elseif fld == :w
                if loc in (:left, :bottom); 
                    bc = update_bc_value(bc, :bottom_w, vtime)
                elseif loc in (:right, :top); 
                    bc = update_bc_value(bc, :top_w, vtime) 
                end
            end
        else
            # Defer to direct application; wrap time-only constants as constant functions
            wrapped_val = (is_spatial || val isa Function) ? val : (_ -> vtime)
            push!(apply_specs, (fld, loc, bctype, wrapped_val))
        end
    end

    # Store updated numeric BC on discretization
    disc.boundary_conditions = bc

    # Apply to solution fields if provided
    if solution !== nothing && haskey(solution, :u) && haskey(solution, :v) && haskey(solution, :w)
        u, v, w = solution[:u], solution[:v], solution[:w]

        # First apply numeric parts (time-only)
        apply_velocity_bcs_nonuniform!(u, v, w, grid, bc)

        # Then apply direct specifications (spatial or non-Dirichlet time-only)
        if !isempty(apply_specs)
            Nz = grid.Nz
            h1 = grid.dz[1]; hN = grid.dz[end]
            for (fld, loc, bctype, valf) in apply_specs
                k = (loc in (:left, :bottom)) ? 1 : Nz
                is_bottom = k == 1
                @inbounds for j in 1:Ny, i in 1:Nx
                    if bctype == :dirichlet
                        val = _eval_bc_value(valf, t; i=i, j=j, xnodes=xnodes, ynodes=ynodes)
                        if fld == :u
                            u[i,j,k] = val
                        elseif fld == :v
                            v[i,j,k] = val
                        else
                            w[i,j,k] = val 
                        end
                    elseif bctype == :neumann
                        q = _eval_bc_value(valf, t; i=i, j=j, xnodes=xnodes, ynodes=ynodes)
                        if fld == :u
                            if is_bottom
                                u[i,j,1] = u[i,j,2] - q*h1 
                            else 
                                u[i,j,Nz] = u[i,j,Nz-1] + q*hN 
                            end
                        elseif fld == :v
                            if is_bottom
                                v[i,j,1] = v[i,j,2] - q*h1 
                            else 
                                v[i,j,Nz] = v[i,j,Nz-1] + q*hN 
                            end
                        else
                            if is_bottom
                                w[i,j,1] = w[i,j,2] - q*h1 
                            else 
                                w[i,j,Nz] = w[i,j,Nz-1] + q*hN 
                            end
                        end
                    elseif bctype == :robin
                        # Expect valf to evaluate to (alpha, beta, gamma) or NamedTuple/tuple with 3 entries
                        raw = valf
                        # Support Function returning tuple, or tuple of functions/numbers
                        alpha = beta = gamma = nothing
                        if raw isa Function
                            tval = raw(t)
                            alpha, beta, gamma = tval isa NamedTuple ? (tval[1], tval[2], tval[3]) : tval
                        else
                            alphabetagamma = raw
                            alpha, beta, gamma = alphabetagamma
                        end
                        alphav  = _eval_bc_value(alpha,  t; i=i, j=j, xnodes=xnodes, ynodes=ynodes)
                        betav = _eval_bc_value(beta, t; i=i, j=j, xnodes=xnodes, ynodes=ynodes)
                        gammav = _eval_bc_value(gamma, t; i=i, j=j, xnodes=xnodes, ynodes=ynodes)
                        if fld == :u
                            if is_bottom
                                denom = (alphav - betav/h1); u[i,j,1] = denom == 0 ? u[i,j,1] : (gammav - betav*u[i,j,2]/h1) / denom
                            else
                                denom = (alphav + betav/hN); u[i,j,Nz] = denom == 0 ? u[i,j,Nz] : (gammav + betav*u[i,j,Nz-1]/hN) / denom
                            end
                        elseif fld == :v
                            if is_bottom
                                denom = (alphav - betav/h1); v[i,j,1] = denom == 0 ? v[i,j,1] : (gammav - betav*v[i,j,2]/h1) / denom
                            else
                                denom = (alphav + betav/hN); v[i,j,Nz] = denom == 0 ? v[i,j,Nz] : (gammav + betav*v[i,j,Nz-1]/hN) / denom
                            end
                        else
                            if is_bottom
                                denom = (alphav - betav/h1); w[i,j,1] = denom == 0 ? w[i,j,1] : (gammav - betav*w[i,j,2]/h1) / denom
                            else
                                denom = (alphav + betav/hN); w[i,j,Nz] = denom == 0 ? w[i,j,Nz] : (gammav + betav*w[i,j,Nz-1]/hN) / denom
                            end
                        end
                    else
                        @warn "Unknown velocity BC type $(bctype); skipping direct application"
                    end
                end
            end
        end

        # Update solution dict
        solution[:u] = u
        solution[:v] = v
        solution[:w] = w
    end
    return nothing
end

"""
    update_scalar_field_bcs!(disc, field_name, field_bcs, t, solution)

Update time-/space-dependent scalar field BCs using BuoyancyBC infrastructure.
"""
function update_scalar_field_bcs!(disc, field_name::Symbol, field_bcs::Vector, t::Real, solution)
    grid = disc.grid

    bottom_type = B_CONSTANT
    top_type = B_CONSTANT
    bottom_value = 0.0
    top_value = 0.0

    for bc in field_bcs
        current_value = _eval_bc_value(getfield(bc, :value), t)
        loc = getfield(bc, :location)
        bctype = getfield(bc, :type)
        if loc in (:left, :bottom)
            bottom_value = current_value
            bottom_type = bctype == :dirichlet ? B_CONSTANT : bctype == :neumann ? B_FLUX : B_CONSTANT
        elseif loc in (:right, :top)
            top_value = current_value
            top_type = bctype == :dirichlet ? B_CONSTANT : bctype == :neumann ? B_FLUX : B_CONSTANT
        end
    end

    buoyancy_bc = BuoyancyBC(bottom_type, top_type; bottom_value=bottom_value, top_value=top_value)

    if !haskey(disc.time_dependent_bcs, field_name)
        disc.time_dependent_bcs[field_name] = Dict{String, Any}()
    end
    disc.time_dependent_bcs[field_name]["buoyancy_bc"] = buoyancy_bc
    disc.time_dependent_bcs[field_name]["last_update_time"] = t

    if solution !== nothing && haskey(solution, field_name)
        field_data = solution[field_name]
        xnodes = hasproperty(disc, :grid_x) ? disc.grid_x : nothing
        ynodes = hasproperty(disc, :grid_y) ? disc.grid_y : nothing
        apply_buoyancy_bcs_nonuniform!(field_data, grid, buoyancy_bc, t; 
                                    xnodes=xnodes, ynodes=ynodes)
        solution[field_name] = field_data
    end
    return nothing
end

"""
    get_time_dependent_bc(prob, field_name, t)

Retrieve the current BuoyancyBC for a field at time t.
"""
function get_time_dependent_bc(prob, field_name::Symbol, t::Real)
    if prob.discretization === nothing || 
       !haskey(prob.discretization.time_dependent_bcs, field_name)
        return nothing
    end
    bc_info = prob.discretization.time_dependent_bcs[field_name]
    if !haskey(bc_info, "last_update_time") || abs(t - bc_info["last_update_time"]) > 1e-12
        update_boundary_conditions!(prob, t)
    end
    return get(bc_info, "buoyancy_bc", nothing)
end

end # module BoundaryConditionsNonUniform
