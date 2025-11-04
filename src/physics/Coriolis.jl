#module CoriolisFPlane

# 
#  fplane Coriolis helper wrapped in a small `FPlane` struct.
#  If `plane === nothing` we produce **zero** Coriolis accelerations, giving a
#  convenient switch for nonrotating test cases.
# 

export FPlane, coriolis_terms!, coriolis_terms

using PencilArrays

"""
    FPlane(; f=1.0)

Lightweight container for the Coriolis parameter *f* on an fplane.  Pass an
`FPlane` instance to `coriolis_terms!` / `coriolis_terms` to apply the Coriolis
acceleration. If you pass `nothing` instead, the functions return **zero**
accelerations (nonrotating).
"""
struct FPlane{T<:Real}
    f::T
end
FPlane(; f::Real=1.0) = FPlane(f)

# 
# Inplace version
# 

"""
    coriolis_terms!(Cu, Cv, u, v[, plane])

Compute horizontal fplane Coriolis accelerations **in place**:

```
Cu =  f * v
Cv = -f * u
```

* All arrays may be `Array` or `PencilArray`.
* `plane` can be an `FPlane` or `nothing`. When `nothing`, `Cu` and `Cv` are
  set to zero (no rotation).
* Returns `Cu, Cv`.
"""
function coriolis_terms!(Cu, Cv, u, v, plane::Union{FPlane,Nothing}=FPlane())
    if plane === nothing
        fill!(Cu, zero(eltype(Cu)))
        fill!(Cv, zero(eltype(Cv)))
        return Cu, Cv
    end
    f = plane.f
    Cu_loc = isa(Cu, PencilArrays.PencilArray) ? parent(Cu) : Cu
    Cv_loc = isa(Cv, PencilArrays.PencilArray) ? parent(Cv) : Cv
    u_loc  = isa(u , PencilArrays.PencilArray) ? parent(u ) : u
    v_loc  = isa(v , PencilArrays.PencilArray) ? parent(v ) : v

    @inbounds for k in axes(u_loc,3), j in axes(u_loc,2), i in axes(u_loc,1)
        Cu_loc[i,j,k] =  f * v_loc[i,j,k]
        Cv_loc[i,j,k] = -f * u_loc[i,j,k]
    end
    return Cu, Cv
end

# 
# Outofplace convenience wrapper
# 

"""
    coriolis_terms(u, v; plane = FPlane()) -> Cu, Cv

Return new arrays with the horizontal Coriolis accelerations. If `plane ===
nothing`, the returned arrays are filled with zeros.
"""
function coriolis_terms(u, v; plane::Union{FPlane,Nothing}=FPlane())
    T = promote_type(eltype(u), eltype(v))
    Cu = isa(u, PencilArrays.PencilArray) ? PencilArrays.PencilArray{T}(undef, PencilArrays.pencil(u)) : Array{T}(undef, size(u))
    Cv = isa(v, PencilArrays.PencilArray) ? PencilArrays.PencilArray{T}(undef, PencilArrays.pencil(v)) : Array{T}(undef, size(v))
    coriolis_terms!(Cu, Cv, u, v, plane)
    return Cu, Cv
end

#  Example 

# if !isinteractive()
#     using Random; Random.seed!(123)
#     Nx, Ny, Nz = 8, 8, 4
#     u = randn(Float32, Nx, Ny, Nz)
#     v = randn(Float32, Nx, Ny, Nz)

#     # Rotating test (f  0)
#     Cu1, Cv1 = coriolis_terms(u, v; plane = FPlane(f=1e-4f0))
#     @info "Rotating test" sample_Cu=Cu1[1,1,1] sample_expected=1e-4f0*v[1,1,1]

#     # Nonrotating test (plane = nothing)
#     Cu2, Cv2 = coriolis_terms(u, v; plane = nothing)
#     @info "Nonrotating test" Cu_zero = maximum(abs, Cu2) Cv_zero = maximum(abs, Cv2)
# end

#end # module
