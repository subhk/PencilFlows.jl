#module Timestep2DPencil

# 
# Lowstorage timesteppers tailored for 2D (xy) pencildecomposed fields
# produced with PencilArrays/PencilFFTs.  All helpers automatically allocate
# work arrays on the same MPI communicator and pencil layout as the prototype
# field `u`, making them safe for largescale parallel runs.
#
# Exports
# -------
#     ¢ create_stepper, supported_schemes
#     ¢ advance!, step!, TimestepProblem
#     ¢ memory_usage
#
# Two families of **lowstorage schemes** are provided:
#     1. LSRK4    Williamson 4stage, 4thorder (2N storage)
#     2. LSAB(14)  AdamsBashforth multistep with an internal LSRK4 bootstrap
#
# A classical fullstorage RK4 is kept for reference / debugging.
# 

# using LinearAlgebra, MPI
# using PencilArrays, PencilFFTs

# Ensure MPI is initialised (harmless if it already is)
MPI.Initialized() || MPI.Init()

# 
# Utility helpers that understand PencilArrays
# 

"Lazy allocation of an array with the same distribution as `u`."
function _similar_pencil(u::PencilArrays.PencilArray{T,N}) where {T,N}
    pencil = PencilArrays.pencil(u)  # keeps comm & layout
    return PencilArrays.PencilArray{T}(undef, pencil)
end

# Fallbacks for regular Julia Arrays
_similar_pencil(u::AbstractArray) = similar(u)

# Copy / axpy helpers  broadcast works on PencilArrays but we route through
# `localpart` for cache friendliness when available.
localview(a::PencilArrays.PencilArray) = parent(a)
localview(a) = a

copy_array!(dest, src) = copyto!(localview(dest), localview(src))
zero_array!(arr) = fill!(localview(arr), zero(eltype(arr)))
function axpy_array!(y, a, x)
    lv_y = localview(y)
    lv_x = localview(x)
    @. lv_y += a * lv_x
    return y
end

function axpby_array!(z, a, x, b, y)
    lv_z = localview(z)
    lv_x = localview(x)
    lv_y = localview(y)
    @. lv_z = a * lv_x + b * lv_y
    return z
end

# Parallelaware maximum
function parallel_maximum(arr)
    local_max = maximum(abs, localview(arr))
    if arr isa PencilArrays.PencilArray
        pencil = PencilArrays.pencil(arr)
        comm = PencilArrays.get_comm(pencil)
        return MPI.Allreduce(local_max, max, comm)
    else
        return local_max
    end
end

# 
# Abstract interface
# 

abstract type AbstractStepper end
order(::AbstractStepper) = error("order not implemented")
is_low_storage(::AbstractStepper) = false
advance!(::AbstractStepper, args...; kwargs...) = error("advance! not implemented")

# 
# Lowstorage RungeKutta (2N)  LSRK family
# 

struct LSRKCoeffs{T}
    a::Vector{T}
    b::Vector{T}
    c::Vector{T}
    order::Int
    name::Symbol
end

mutable struct LSRKStepper{A<:AbstractArray} <: AbstractStepper
    coeffs::LSRKCoeffs{Float64}
    w::A  # accumulation register
    k::A  # stage derivative
end

order(s::LSRKStepper) = s.coeffs.order
is_low_storage(::LSRKStepper) = true

function _build_lsrk_coeffs(scheme::Symbol)
    if scheme == :LSRK4
        # Williamson 4(5)  5stage, but 4thorder accuracy with 2N storage
        a = [0.0,
             -567301805773.0/1357537059087.0,
             -2404267990393.0/2016746695238.0,
             -3550918686646.0/2091501179385.0,
             -1275806237668.0/842570457699.0]
        b = [1432997174477.0/9575080441755.0,
             5161836677717.0/13612068292357.0,
             1720146321549.0/2090206949498.0,
             3134564353537.0/4481467310338.0,
             2277821191437.0/14882151754819.0]
        c = cumsum([0.0; b[1:4]])
        return LSRKCoeffs(a, b, c, 4, :LSRK4)
    elseif scheme == :LSRK3SSP
        a = [0.0, -2/3, -1/3]
        b = [1.0, 1/4, 2/3]
        c = [0.0, 1.0, 0.5]
        return LSRKCoeffs(a, b, c, 3, :LSRK3SSP)
    elseif scheme == :LSRK2Heun
        a = [0.0, -1.0]
        b = [1.0, 1/2]
        c = [0.0, 1.0]
        return LSRKCoeffs(a, b, c, 2, :LSRK2Heun)
    else
        error("Unknown LSRK scheme $scheme")
    end
end

function LSRKStepper(scheme::Symbol, u::AbstractArray)
    coeffs = _build_lsrk_coeffs(scheme)
    return LSRKStepper(coeffs, _similar_pencil(u), _similar_pencil(u))
end

function advance!(s::LSRKStepper, u, t, dt, rhs!; context=nothing)
    cfs = s.coeffs
    copy_array!(s.w, u)
    for i in eachindex(cfs.a)
        stage_t = t + (i == 1 ? 0.0 : cfs.c[i]) * dt
        rhs!(s.k, u, stage_t, context)                 # k <- F(u)
        axpy_array!(s.w, cfs.b[i] * dt, s.k)           # w += b_i dt k
        if i < length(cfs.a)
            axpy_array!(u, cfs.a[i] * dt, s.k)         # u  += a_i dt k (prep next stage)
        end
    end
    copy_array!(u, s.w)                                # final solution in w
    return u
end

# 
# Lowstorage AdamsBashforth (1-4) with LSRK4 bootstrap
# 

const _AB_COEFFS = Dict(1 => [1.0],
                        2 => [ 3/2, -1/2],
                        3 => [23/12, -16/12, 5/12],
                        4 => [55/24, -59/24, 37/24, -9/24])

mutable struct LSABStepper{A<:AbstractArray} <: AbstractStepper
    order::Int
    hist::Vector{A}         # RHS history (most recent first)
    tmp::A                  # current RHS scratch
    bootstrap::LSRKStepper{A}
    filled::Int             # how many history slots are valid
end

order(s::LSABStepper) = s.order
is_low_storage(::LSABStepper) = true

function LSABStepper(order::Int, u::AbstractArray)
    order in (1:4) || error("AB order must be 1-4")
    hist  = [_similar_pencil(u) for _ in 1:order]
    tmp   = _similar_pencil(u)
    boot  = LSRKStepper(:LSRK4, u)      # always safe bootstrap
    return LSABStepper(order, hist, tmp, boot, 0)
end

function _shift_history!(s::LSABStepper)
    for k = s.order:-1:2
        copy_array!(s.hist[k], s.hist[k-1])
    end
end

function advance!(s::LSABStepper, u, t, dt, rhs!; context=nothing)
    if s.filled < s.order        # Bootstrap phase
        advance!(s.bootstrap, u, t, dt, rhs!; context=context)
        rhs!(s.tmp, u, t + dt, context)            # RHS at new time
        _shift_history!(s)
        copy_array!(s.hist[1], s.tmp)
        s.filled += 1
        return u
    end

    rhs!(s.tmp, u, t, context)                    # current RHS
    _shift_history!(s)
    copy_array!(s.hist[1], s.tmp)

    coeffs = _AB_COEFFS[s.order]
    for i = 1:s.order
        axpy_array!(u, dt * coeffs[i], s.hist[i])
    end
    return u
end

# 
# Classical RK4 (fullstorage)
# 

mutable struct RK4Stepper{A<:AbstractArray} <: AbstractStepper
    k1::A; k2::A; k3::A; k4::A; tmp::A
end

order(::RK4Stepper) = 4

function RK4Stepper(u::AbstractArray)
    return RK4Stepper(_similar_pencil(u), _similar_pencil(u), _similar_pencil(u), _similar_pencil(u), _similar_pencil(u))
end

function advance!(s::RK4Stepper, u, t, dt, rhs!; context=nothing)
    rhs!(s.k1, u, t, context)
    axpby_array!(s.tmp, 1.0, u, 0.5*dt, s.k1)
    rhs!(s.k2, s.tmp, t + 0.5*dt, context)
    axpby_array!(s.tmp, 1.0, u, 0.5*dt, s.k2)
    rhs!(s.k3, s.tmp, t + 0.5*dt, context)
    axpby_array!(s.tmp, 1.0, u, dt, s.k3)
    rhs!(s.k4, s.tmp, t + dt, context)

    lv_u = localview(u)
    lv_k1 = localview(s.k1)
    lv_k2 = localview(s.k2)
    lv_k3 = localview(s.k3)
    lv_k4 = localview(s.k4)
    @. lv_u += dt/6*(lv_k1 + 2*lv_k2 + 2*lv_k3 + lv_k4)
    return u
end

# 
# Stepper factory / public helpers
# 

const _LOW_STORAGE = [:LSRK4, :LSRK3SSP, :LSRK2Heun, :LSAB1, :LSAB2, :LSAB3, :LSAB4]

supported_schemes() = vcat(_LOW_STORAGE, [:RK4])

function create_stepper(scheme::Symbol, u::AbstractArray; low_storage=false)
    scheme in supported_schemes() || error("Unsupported scheme $scheme")

    if scheme in [:LSRK4, :LSRK3SSP, :LSRK2Heun]
        return LSRKStepper(scheme, u)
    elseif scheme in [:LSAB1, :LSAB2, :LSAB3, :LSAB4]
        order = parse(Int, String(scheme)[end])
        return LSABStepper(order, u)
    elseif scheme == :RK4 && low_storage
        return LSRKStepper(:LSRK4, u)
    elseif scheme == :RK4
        return RK4Stepper(u)
    else
        error("Failed to create stepper $scheme")
    end
end

# 
# Lightweight TimestepProblem wrapper (CFL optional)
# 

mutable struct TimestepProblem{S<:AbstractStepper}
    stepper::S
    scheme::Symbol
end

function TimestepProblem(scheme::Symbol, u::AbstractArray; low_storage=false)
    return TimestepProblem(create_stepper(scheme, u; low_storage=low_storage), scheme)
end

function step!(prob::TimestepProblem, u, t, dt, rhs!; kwargs...)
    advance!(prob.stepper, u, t, dt, rhs!; kwargs...)
    return u
end

# 
# Memory usage estimator
# 

function memory_usage(stepper::AbstractStepper, array_size::Int, element_size::Int=8)
    if stepper isa LSRKStepper
        n = 2  # w, k
    elseif stepper isa LSABStepper
        n = stepper.order + 2  # hist + tmp + implicit w,k inside bootstrap
    elseif stepper isa RK4Stepper
        n = 5
    else
        n = 0
    end
    bytes = n * array_size * element_size
    return (total_arrays=n, total_bytes=bytes, storage_factor=n)
end

# 
# Exports
# 

# export supported_schemes, create_stepper, step!, TimestepProblem, memory_usage

#end # module

