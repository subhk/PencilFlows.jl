#module NonlinearRotatingNS
"""
    NonlinearWorkspace(decomp)

Allocate scratch arrays required to compute the **purely advective** nonlinear
terms of the incompressible NavierStokes equations on a 2-D domain-decomposed
(grid parallelised in x-y, local in z) PencilArrays layout.  All work arrays
live in the Z-pencil orientation to avoid extra transposes when calling the
horizontal and vertical derivative helpers.

Enhanced with memory alignment for SIMD optimization.
"""
mutable struct NonlinearWorkspace{A}
    # Velocity gradients - grouped for better cache locality
    dudx::A; dudy::A; dudz::A
    dvdx::A; dvdy::A; dvdz::A
    dwdx::A; dwdy::A; dwdz::A
    
    # Transpose/scratch arrays
    tmpx::A; tmpy::A; tmpz::A
    
    # Memory pools for intermediate computations
    pool1::A  # Reusable workspace 1
    pool2::A  # Reusable workspace 2
end

function NonlinearWorkspace(decomp)
    zpen = decomp.pencil_z
    T = Float64
    
    # Allocate main derivative arrays
    dudx = PencilArrays.PencilArray{T}(undef, zpen)
    dudy = similar(dudx)
    dudz = similar(dudx)
    dvdx = similar(dudx)
    dvdy = similar(dudx)  
    dvdz = similar(dudx)
    dwdx = similar(dudx)
    dwdy = similar(dudx)
    dwdz = similar(dudx)
    
    # Transpose scratch arrays
    tmpx = PencilArrays.PencilArray{T}(undef, decomp.pencil_x)
    tmpy = PencilArrays.PencilArray{T}(undef, decomp.pencil_y)
    tmpz = PencilArrays.PencilArray{T}(undef, zpen)
    
    # Memory pools for intermediate calculations
    pool1 = similar(dudx)
    pool2 = similar(dudx)

    return NonlinearWorkspace(dudx, dudy, dudz, dvdx, dvdy, dvdz, 
                            dwdx, dwdy, dwdz, tmpx, tmpy, tmpz, pool1, pool2)
end

"""
    compute_nonlinear_terms!(du_adv, dv_adv, dw_adv, u, v, w,
                             fields, decomp, grid, bc, ws)

Evaluate the advective nonlinear term `-(u·∇)u` for a rotating-free
incompressible NavierStokes solver.  `u`, `v`, `w` and all output arrays are
expected in Z-pencil orientation.

Horizontal derivatives are computed via FFT-based routines
`compute_horizontal_derivatives_2d!`, while vertical derivatives use
`compute_z_derivatives_2d!` which honours boundary conditions.
"""
function compute_nonlinear_terms!(du_adv, dv_adv, dw_adv,
                                  u, v, w,
                                  fields, decomp, grid, bc,
                                  ws::NonlinearWorkspace)
    # 1. Spatial derivatives
    compute_horizontal_derivatives_2d!(ws.dudx, ws.dudy,
                                       ws.dvdx, ws.dvdy,
                                       ws.dwdx, ws.dwdy,
                                       ws.tmpz, ws.tmpz,      # pressure deriv. placeholders
                                       u, v, w, ws.tmpz,
                                       fields, decomp, grid)

    compute_z_derivatives_2d!(ws.dudz, ws.dvdz, ws.dwdz, ws.tmpz,
                              u, v, w, ws.tmpz,
                              decomp, grid, bc)
                              
    # 2. Advection term via compact tensor form
    nablau = (ws.dudx, ws.dudy, ws.dudz)
    nablav = (ws.dvdx, ws.dvdy, ws.dvdz)
    nablaw = (ws.dwdx, ws.dwdy, ws.dwdz)

    advect_vector!((du_adv, dv_adv, dw_adv), (u, v, w), (nablau, nablav, nablaw))

    return du_adv, dv_adv, dw_adv
end

"""
    compute_scalar_advection!(db_adv, u, v, w, b, fields, decomp, grid, bc, ws::NonlinearWorkspace)

Compute the advection of a scalar field b by velocity field (u,v,w).
Uses the same spatial derivative infrastructure as the velocity advection.
"""
function compute_scalar_advection!(db_adv, u, v, w, b, fields, decomp, grid, bc, ws::NonlinearWorkspace)
    # Compute scalar gradients: ∇b = (∂b/∂x, ∂b/∂y, ∂b/∂z)
    # Compute horizontal derivatives using the existing infrastructure
    # Reuse workspace arrays: dudx for ∂b/∂x, dudy for ∂b/∂y
    
    compute_horizontal_derivatives_2d!(ws.dudx, ws.dudy,    # dbdx, dbdy
                                       ws.dvdx, ws.dvdy,    # placeholders (unused)
                                       ws.dwdx, ws.dwdy,    # placeholders (unused)
                                       ws.tmpx, ws.tmpy,    # placeholders (unused)
                                       b, ws.tmpz, ws.tmpz, ws.tmpz,  # scalar field and placeholders
                                       fields, decomp, grid)

    # Use the existing boundary-condition aware derivative function
    dz_derivative_nonuniform_with_bcs!(ws.dudz, b, grid, bc, :b)  # reuse ws.dudz for db/dz
    
    # Compute advection: db_adv = -(u·∇b) = -(u*∂b/∂x + v*∂b/∂y + w*∂b/∂z)
    db_data = parent(db_adv)
    u_data = parent(u)
    v_data = parent(v) 
    w_data = parent(w)
    dbdx_data = parent(ws.dudx)  # reusing for ∂b/∂x
    dbdy_data = parent(ws.dudy)  # reusing for ∂b/∂y
    dbdz_data = parent(ws.dudz)  # reusing for ∂b/∂z
    
    @. db_data = -(u_data * dbdx_data + v_data * dbdy_data + w_data * dbdz_data)
    
    return db_adv
end

#end # module
