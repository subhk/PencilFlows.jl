# Tensor Helpers for Compact PDE Assembly
# High-performance broadcast helpers for tensor-form PDE operations.
# Optimized for both regular arrays and distributed PencilArrays.

using PencilArrays
using Base.Threads: @threads, @spawn, nthreads, threadid

# Optional SIMD support
const HAS_SIMD_PKG = try
    using SIMD
    true
catch
    false
end

# Load optimization modules (with error handling)
# Note: cache_optimization.jl includes memory_pools.jl, so we only need to include it once
try
    include("cache_optimization.jl")
catch e
    @warn "Cache optimization not available: $e"
end

if HAS_SIMD_PKG
    try
        include("simd_optimization.jl")
    catch e
        @warn "SIMD optimization not available: $e"
    end
end

# Threading and optimization configuration
const MIN_PARALLEL_SIZE = 1000  # Minimum array size for parallel processing
const CACHE_BLOCK_SIZE = 64     # Cache-friendly block size
const USE_MEMORY_POOLS = Ref(true)  # Global toggle for memory pool usage
const OPTIMIZATION_LEVEL = Ref(:high)  # :low, :medium, :high

# 
# Local Data Accessor
# 

"""
    lp(x) -> Array

Extract local part from PencilArray for cache-efficient operations.
For regular arrays, returns the array unchanged.

# Performance Benefits
- Reduces memory access overhead for distributed arrays
- Enables SIMD vectorization on local data chunks
- Maintains compatibility with both array types
"""
@inline lp(x) = x isa PencilArrays.PencilArray ? parent(x) : x

# 
# Scalar Advection
# 

"""
    advect_scalar!(Rq, U, nablaq) -> Rq

Compute scalar advection term: -(u·∇)q

# Arguments
- `Rq`: Result array for advection term
- `U`:  Velocity vector (u, v, w)
- `∇q`: Scalar gradient vector (∂q/∂x, ∂q/∂y, ∂q/∂z)

# Mathematical Form
```
Rq = -(u*dq/dx + v*dq/dy + w*dq/dz)
```
"""
@inline function advect_scalar!(Rq, U, q)
    # Unpack velocity and gradient components
    u, v, w = U
    dqdx, dqdy, dqdz = q
    
    # Extract local data for optimal performance
    Rp = lp(Rq)
    up, vp, wp = lp(u), lp(v), lp(w)
    qx, qy, qz = lp(dqdx), lp(dqdy), lp(dqdz)
    
    N = length(Rp)
    
    # Use parallel processing for large arrays
    if N >= MIN_PARALLEL_SIZE && nthreads() > 1
        advect_scalar_parallel!(Rp, up, vp, wp, qx, qy, qz, N)
    else
        advect_scalar_serial!(Rp, up, vp, wp, qx, qy, qz, N)
    end
    
    return Rq
end

@inline function advect_scalar_serial!(Rp, up, vp, wp, qx, qy, qz, N)
    @inbounds @simd ivdep for i in 1:N
        Rp[i] = -(up[i] * qx[i] + vp[i] * qy[i] + wp[i] * qz[i])
    end
end

@inline function advect_scalar_parallel!(Rp, up, vp, wp, qx, qy, qz, N)
    nt = nthreads()
    chunk_size = div(N, nt)
    
    @threads for tid in 1:nt
        start_idx = (tid - 1) * chunk_size + 1
        end_idx = tid == nt ? N : tid * chunk_size
        
        @inbounds @simd ivdep for i in start_idx:end_idx
            Rp[i] = -(up[i] * qx[i] + vp[i] * qy[i] + wp[i] * qz[i])
        end
    end
end

# 
# Vector Advection
# 

"""
    advect_vector!(R, U, nablaU) -> R

Compute vector advection term: -(u·∇)u (Advective form - NON-CONSERVATIVE)

  **DEPRECATED**: Use `advect_vector_conservative!` for flux conservative form.

# Arguments
- `R`:   Result vector (Ru, Rv, Rw) for advection terms
- `U`:   Velocity vector (u, v, w)
- `∇U`: Velocity gradient tensor [(∇u), (∇v), (∇w)]

# Mathematical Form
```
Ru = -(u·∇u) = -(u∂u/∂x + v∂u/∂y + w∂u/∂z)
Rv = -(u·∇v) = -(u∂v/∂x + v∂v/∂y + w∂v/∂z)
Rw = -(u·∇w) = -(u∂w/∂x + v∂w/∂y + w∂w/∂z)
```

# Note
This form is NOT flux conservative. For better numerical stability and 
physical accuracy, use the flux conservative form instead.
"""
@inline function advect_vector!(R, U, nablaU)
    # Unpack velocity and gradient components
    u, v, w = U
    nablau, nablav, nablaw = nablaU
    Ru, Rv, Rw = R
    
    # Extract local velocity data once
    up, vp, wp = lp(u), lp(v), lp(w)
    
    # Extract local gradient data efficiently
    dudx, dudy, dudz = lp(nablau[1]), lp(nablau[2]), lp(nablau[3])
    dvdx, dvdy, dvdz = lp(nablav[1]), lp(nablav[2]), lp(nablav[3])
    dwdx, dwdy, dwdz = lp(nablaw[1]), lp(nablaw[2]), lp(nablaw[3])
    
    # Extract local result arrays
    Rup, Rvp, Rwp = lp(Ru), lp(Rv), lp(Rw)
    
    N = length(Rup)
    
    # Use parallel processing for large arrays
    if N >= MIN_PARALLEL_SIZE && nthreads() > 1
        advect_vector_parallel!(Rup, Rvp, Rwp, up, vp, wp, 
                               dudx, dudy, dudz, dvdx, dvdy, dvdz, 
                               dwdx, dwdy, dwdz, N)
    else
        advect_vector_serial!(Rup, Rvp, Rwp, up, vp, wp, 
                             dudx, dudy, dudz, dvdx, dvdy, dvdz, 
                             dwdx, dwdy, dwdz, N)
    end
    
    return R
end

@inline function advect_vector_serial!(Rup, Rvp, Rwp, up, vp, wp, 
                                      dudx, dudy, dudz, dvdx, dvdy, dvdz, 
                                      dwdx, dwdy, dwdz, N)
    @inbounds @simd ivdep for i in 1:N
        u_val, v_val, w_val = up[i], vp[i], wp[i]
        Rup[i] = -(u_val * dudx[i] + v_val * dudy[i] + w_val * dudz[i])
        Rvp[i] = -(u_val * dvdx[i] + v_val * dvdy[i] + w_val * dvdz[i])
        Rwp[i] = -(u_val * dwdx[i] + v_val * dwdy[i] + w_val * dwdz[i])
    end
end

@inline function advect_vector_parallel!(Rup, Rvp, Rwp, up, vp, wp, 
                                        dudx, dudy, dudz, dvdx, dvdy, dvdz, 
                                        dwdx, dwdy, dwdz, N)
    nt = nthreads()
    chunk_size = div(N, nt)
    
    @threads for tid in 1:nt
        start_idx = (tid - 1) * chunk_size + 1
        end_idx = tid == nt ? N : tid * chunk_size
        
        @inbounds @simd ivdep for i in start_idx:end_idx
            u_val, v_val, w_val = up[i], vp[i], wp[i]
            Rup[i] = -(u_val * dudx[i] + v_val * dudy[i] + w_val * dudz[i])
            Rvp[i] = -(u_val * dvdx[i] + v_val * dvdy[i] + w_val * dvdz[i])
            Rwp[i] = -(u_val * dwdx[i] + v_val * dwdy[i] + w_val * dwdz[i])
        end
    end
end

"""
    advect_vector_conservative!(R, U, flux_div, div_U) -> R

Compute flux conservative vector advection: -(∇·(u⊗u) - u(∇·u))

# Arguments
- `R`:        Result vector (Ru, Rv, Rw) for advection terms
- `U`:        Velocity vector (u, v, w)
- `flux_div`: Divergence of momentum flux tensor ∇·(u⊗u) = (∇·(uu), ∇·(vu), ∇·(wu))
- `div_U`:    Velocity divergence ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z

# Mathematical Form (Flux Conservative)
```
(u·∇)u = ∇·(u⊗u) - u(∇·u)

where u⊗u = [u²   uv  uw ]     ∇·(u⊗u) = [∂(u²)/∂x + ∂(uv)/∂y + ∂(uw)/∂z]
              [uv   v^2  vw ]                 [d(uv)/dx + d(v^2)/dy + d(vw)/dz]
              [uw   vw  w^2 ]                 [d(uw)/dx + d(vw)/dy + d(w^2)/dz]

Ru = -(∇·(uu) - u∇·u)
Rv = -(∇·(vu) - v∇·u)  
Rw = -(∇·(wu) - w∇·u)
```

# Benefits
- Conserves momentum exactly in discrete form
- Better numerical stability for high Reynolds numbers
- Preserves energy cascade properties in turbulence
- Maintains Galilean invariance
"""
@inline function advect_vector_conservative!(R, U, flux_div, div_U)
    # Unpack velocity and result components
    u, v, w = U
    Ru, Rv, Rw = R
    div_uu, div_vu, div_wu = flux_div
    
    # Extract local data for optimal performance
    up, vp, wp = lp(u), lp(v), lp(w)
    Rup, Rvp, Rwp = lp(Ru), lp(Rv), lp(Rw)
    div_uup, div_vup, div_wup = lp(div_uu), lp(div_vu), lp(div_wu)
    div_Up = lp(div_U)
    
    # Compute flux conservative advection with better vectorization
    N = length(Rup)
    @inbounds @simd ivdep for i in 1:N
        div_U_val = div_Up[i]
        Rup[i] = -(div_uup[i] - up[i] * div_U_val)
        Rvp[i] = -(div_vup[i] - vp[i] * div_U_val)
        Rwp[i] = -(div_wup[i] - wp[i] * div_U_val)
    end
    
    return R
end

# 
# Flux Conservative Helper Functions
# 

"""
    compute_momentum_flux_divergence!(flux_div, U, nablaU) -> flux_div

Compute divergence of momentum flux tensor: nabla*(u*u)

# Arguments
- `flux_div`: Result vector (nabla*(uu), nabla*(vu), nabla*(wu))
- `U`:        Velocity vector (u, v, w) 
- `nablaU`:       Velocity gradient tensor [(nablau), (nablav), (nablaw)]

# Mathematical Form
```
nabla*(uu) = d(u^2)/dx + d(uv)/dy + d(uw)/dz = u*du/dx + v*du/dy + w*du/dz + u(du/dx + dv/dy + dw/dz)
nabla*(vu) = d(uv)/dx + d(v^2)/dy + d(vw)/dz = u*dv/dx + v*dv/dy + w*dv/dz + v(du/dx + dv/dy + dw/dz)
nabla*(wu) = d(uw)/dx + d(vw)/dy + d(w^2)/dz = u*dw/dx + v*dw/dy + w*dw/dz + w(du/dx + dv/dy + dw/dz)
```

# Note
This computes the flux divergence directly without explicitly forming the tensor.
More efficient than computing u*u then taking divergence.
"""
@inline function compute_momentum_flux_divergence!(flux_div, U, nablaU)
    # Unpack velocity and gradient components
    u, v, w = U
    nablau, nablav, nablaw = nablaU
    div_uu, div_vu, div_wu = flux_div
    
    # Extract velocity gradients
    dudx, dudy, dudz = nablau[1], nablau[2], nablau[3]
    dvdx, dvdy, dvdz = nablav[1], nablav[2], nablav[3]
    dwdx, dwdy, dwdz = nablaw[1], nablaw[2], nablaw[3]
    
    # Extract local data for optimal performance
    up, vp, wp = lp(u), lp(v), lp(w)
    dudxp, dudyp, dudzp = lp(dudx), lp(dudy), lp(dudz)
    dvdxp, dvdyp, dvdzp = lp(dvdx), lp(dvdy), lp(dvdz)
    dwdxp, dwdyp, dwdzp = lp(dwdx), lp(dwdy), lp(dwdz)
    div_uup, div_vup, div_wup = lp(div_uu), lp(div_vu), lp(div_wu)
    
    N = length(div_uup)
    
    # Use cache-blocked parallel computation for large arrays
    if N >= MIN_PARALLEL_SIZE && nthreads() > 1
        flux_divergence_blocked_parallel!(div_uup, div_vup, div_wup, 
                                        up, vp, wp, dudxp, dudyp, dudzp, 
                                        dvdxp, dvdyp, dvdzp, dwdxp, dwdyp, dwdzp, N)
    else
        flux_divergence_serial!(div_uup, div_vup, div_wup, 
                               up, vp, wp, dudxp, dudyp, dudzp, 
                               dvdxp, dvdyp, dvdzp, dwdxp, dwdyp, dwdzp, N)
    end
    
    return flux_div
end

@inline function flux_divergence_serial!(div_uup, div_vup, div_wup, 
                                        up, vp, wp, dudxp, dudyp, dudzp, 
                                        dvdxp, dvdyp, dvdzp, dwdxp, dwdyp, dwdzp, N)
    @inbounds @simd ivdep for i in 1:N
        u_val, v_val, w_val = up[i], vp[i], wp[i]
        dudx_val, dudy_val, dudz_val = dudxp[i], dudyp[i], dudzp[i]
        dvdx_val, dvdy_val, dvdz_val = dvdxp[i], dvdyp[i], dvdzp[i]
        dwdx_val, dwdy_val, dwdz_val = dwdxp[i], dwdyp[i], dwdzp[i]
        
        # Use explicit FMA operations for better performance
        div_uup[i] = muladd(2*u_val, dudx_val, 
                    muladd(v_val, dudy_val, 
                    muladd(u_val, dvdy_val, 
                    muladd(w_val, dudz_val, u_val * dwdz_val))))
        
        div_vup[i] = muladd(v_val, dudx_val, 
                    muladd(u_val, dvdx_val, 
                    muladd(2*v_val, dvdy_val, 
                    muladd(w_val, dvdz_val, v_val * dwdz_val))))
        
        div_wup[i] = muladd(w_val, dudx_val, 
                    muladd(u_val, dwdx_val, 
                    muladd(w_val, dvdy_val, 
                    muladd(v_val, dwdy_val, 2*w_val * dwdz_val))))
    end
end

@inline function flux_divergence_blocked_parallel!(div_uup, div_vup, div_wup, 
                                                  up, vp, wp, dudxp, dudyp, dudzp, 
                                                  dvdxp, dvdyp, dvdzp, dwdxp, dwdyp, dwdzp, N)
    nt = nthreads()
    chunk_size = div(N, nt)
    
    @threads for tid in 1:nt
        start_idx = (tid - 1) * chunk_size + 1
        end_idx = tid == nt ? N : tid * chunk_size
        
        # Process in cache-friendly blocks
        for block_start in start_idx:CACHE_BLOCK_SIZE:end_idx
            block_end = min(block_start + CACHE_BLOCK_SIZE - 1, end_idx)
            
            @inbounds @simd ivdep for i in block_start:block_end
                u_val, v_val, w_val = up[i], vp[i], wp[i]
                dudx_val, dudy_val, dudz_val = dudxp[i], dudyp[i], dudzp[i]
                dvdx_val, dvdy_val, dvdz_val = dvdxp[i], dvdyp[i], dvdzp[i]
                dwdx_val, dwdy_val, dwdz_val = dwdxp[i], dwdyp[i], dwdzp[i]
                
                div_uup[i] = muladd(2*u_val, dudx_val, 
                            muladd(v_val, dudy_val, 
                            muladd(u_val, dvdy_val, 
                            muladd(w_val, dudz_val, u_val * dwdz_val))))
                
                div_vup[i] = muladd(v_val, dudx_val, 
                            muladd(u_val, dvdx_val, 
                            muladd(2*v_val, dvdy_val, 
                            muladd(w_val, dvdz_val, v_val * dwdz_val))))
                
                div_wup[i] = muladd(w_val, dudx_val, 
                            muladd(u_val, dwdx_val, 
                            muladd(w_val, dvdy_val, 
                            muladd(v_val, dwdy_val, 2*w_val * dwdz_val))))
            end
        end
    end
end

"""
    compute_velocity_divergence!(div_U, nablaU) -> div_U

Compute velocity divergence: nabla*u = du/dx + dv/dy + dw/dz

# Arguments
- `div_U`: Result array for velocity divergence
- `nablaU`:    Velocity gradient tensor [(nablau), (nablav), (nablaw)]
"""
@inline function compute_velocity_divergence!(div_U, nablaU)
    # Extract diagonal gradient components
    nablau, nablav, nablaw = nablaU
    dudx, dvdy, dwdz = nablau[1], nablav[2], nablaw[3]  # Extract diagonal terms
    
    # Extract local data
    dudxp, dvdyp, dwdzp = lp(dudx), lp(dvdy), lp(dwdz)
    div_Up = lp(div_U)
    
    # Compute divergence
    N = length(div_Up)
    @inbounds @simd ivdep for i in 1:N
        div_Up[i] = dudxp[i] + dvdyp[i] + dwdzp[i]
    end
    
    return div_U
end

# 
# Coriolis Force (f-plane)
# 

"""
    add_coriolis_fplane!(R, U, f) -> R

Add f-plane Coriolis force to momentum equations.

# Arguments
- `R`: Momentum tendency vector (Ru, Rv, Rw)
- `U`: Velocity vector (u, v, w)
- `f`: Coriolis parameter (constant)

# Mathematical Form
```
Ru += f * v    (adds eastward acceleration from northward velocity)
Rv += -f * u   (adds southward acceleration from eastward velocity)
Rw += 0        (no vertical Coriolis effect in f-plane approximation)
```
"""
@inline function add_coriolis_fplane!(R, U, f::Real)
    # Unpack horizontal velocity and momentum tendencies
    u, v, _ = U
    Ru, Rv, _ = R
    
    # Extract local data for SIMD optimization
    up, vp = lp(u), lp(v)
    Rup, Rvp = lp(Ru), lp(Rv)
    
    # Add Coriolis acceleration with better vectorization
    N = length(Rup)
    @inbounds @simd ivdep for i in 1:N
        u_val, v_val = up[i], vp[i]
        Rup[i] += f * v_val
        Rvp[i] -= f * u_val
    end
    
    return R
end

# 
# Viscous Diffusion
# 

"""
    add_diffusion3!(R, lapU, nu) -> R

Add viscous diffusion terms to momentum equations.

# Arguments
- `R`:    Momentum tendency vector (Ru, Rv, Rw)
- `lapU`: Laplacian of velocity vector (nabla^2u, nabla^2v, nabla^2w)
- `nu`:    Kinematic viscosity (positive scalar)

# Mathematical Form
```
Ru += ν∇²u   (viscous acceleration in x-direction)
Rv += ν∇²v   (viscous acceleration in y-direction)
Rw += ν∇²w   (viscous acceleration in z-direction)
```
"""
@inline function add_diffusion3!(R, lapU, nu::Real)
    # Unpack momentum tendencies and Laplacians
    Ru, Rv, Rw = R
    lap_u, lap_v, lap_w = lapU
    
    # Extract local data for efficient computation
    Rup, Rvp, Rwp = lp(Ru), lp(Rv), lp(Rw)
    lap_up, lap_vp, lap_wp = lp(lap_u), lp(lap_v), lp(lap_w)
    
    # Add viscous diffusion with improved vectorization
    N = length(Rup)
    @inbounds @simd ivdep for i in 1:N
        Rup[i] = muladd(nu, lap_up[i], Rup[i])
        Rvp[i] = muladd(nu, lap_vp[i], Rvp[i])
        Rwp[i] = muladd(nu, lap_wp[i], Rwp[i])
    end
    
    return R
end

# End of Tensor Helpers