############################ utils.jl (FULL) #################################

# Required imports for standalone utils functionality
using MPI
using Dates
using Printf

# # ============================================================================
# # EXPORTS
# # ============================================================================
# export \
#     # MPI / Parallel helpers
#     comm_rank, comm_size, isroot, mpi_inited, root_print, sync_print,
#     mpi_allreduce_sum, mpi_allreduce_max, mpi_allreduce_min,
#     mpi_allreduce_any, mpi_allreduce_all,
#     maybe_barrier, \
#     # Allocation / workspace
#     alloc_like, alloc_zeros_like, ensure_complex, promote_real,
#     WorkspacePool, get_workspace!, release_workspace!, workbuffer!, \
#     # Timing & profiling
#     Timer, tic!, toc!, elapsed, lap!, reset!, ScopedTimer, @time_block, timeit,
#     # Norms / reductions
#     local_sum, global_sum, global_norm2, l2norm, linfnorm, parsevalnorm_factor,
#     # Adaptive / error weighting
#     error_weighted_norm, relative_tolerance_met,
#     # Dealias / filtering utilities
#     dealias_mask_1d, build_two_thirds_mask, apply_two_thirds_mask!,
#     spectral_filter_exp!, build_k2_array, ksq, \
#     # Logging
#     LogLevel, set_log_level!, should_log, init_logger!, log_event!, flush_log!,
#     # Callbacks (diagnostics registration)
#     CallbackHandle, register_callback!, run_callbacks!, clear_callbacks!,
#     # Random / initial conditions
#     seeded_rng, random_phase_field!, random_spectrum_field!,
#     # Formatting / memory / rates
#     mem_usage_MB, bytestr, format_rate,
#     # Safe numeric helpers
#     safediv, clamp_step, stable_pow,
#     # JSON-lite (KV output)
#     kv_encode, write_kv_file,
#     # Hashing / cache
#     simple_hash64, hash_array64,
#     # Date/time
#     iso_timestamp

# ============================================================================
# MPI HELPERS
# ============================================================================
mpi_inited() = MPI.Initialized()

function _ensure_init()
    MPI.Initialized() || MPI.Init()
end

comm_rank() = (mpi_inited() ? MPI.Comm_rank(MPI.COMM_WORLD) : 0)
comm_size() = (mpi_inited() ? MPI.Comm_size(MPI.COMM_WORLD) : 1)
isroot()    = comm_rank() == 0

maybe_barrier() = mpi_inited() ? MPI.Barrier(MPI.COMM_WORLD) : nothing

mpi_allreduce_sum(x) = mpi_inited() ? MPI.Allreduce(x, +, MPI.COMM_WORLD) : x
mpi_allreduce_max(x) = mpi_inited() ? MPI.Allreduce(x, max, MPI.COMM_WORLD) : x
mpi_allreduce_min(x) = mpi_inited() ? MPI.Allreduce(x, min, MPI.COMM_WORLD) : x
mpi_allreduce_any(flag::Bool) = (mpi_inited() ?
    MPI.Allreduce(flag, (a,b)->(a||b), MPI.COMM_WORLD) : flag)
mpi_allreduce_all(flag::Bool) = (mpi_inited() ?
    MPI.Allreduce(flag, (a,b)->(a&&b), MPI.COMM_WORLD) : flag)

function root_print(args...)
    isroot() && println(args...)
end

"""
    sync_print(msg)

Barrier then root prints `msg`. Useful for step-wise debug ordering.
"""
function sync_print(msg)
    maybe_barrier()
    root_print(msg)
    maybe_barrier()
end

# ============================================================================
# ALLOCATION / WORKSPACE
# ============================================================================
alloc_like(A) = similar(A)
alloc_zeros_like(A) = fill!(similar(A), zero(eltype(A)))

"""
    get_workspace(template, key; pool=nothing)

Get a reusable workspace array matching the template. Uses memory pool if available,
otherwise falls back to similar() allocation.
"""
function get_workspace(template, key::String; pool=nothing)
    if pool !== nothing && haskey(pool, key)
        workspace = pool[key]
        if size(workspace) == size(template) && eltype(workspace) == eltype(template)
            return workspace
        end
    end
    # Fallback to allocation
    return similar(template)
end
promote_real(x::AbstractArray{T}) where {T<:Complex} = real.(x)
promote_real(x) = x
ensure_complex(A::AbstractArray{<:Real}) = complex.(A)
ensure_complex(A::AbstractArray{<:Complex}) = A

mutable struct WorkspacePool
    # keyed by eltype symbol => vector of vectors
    store::Dict{DataType, Vector{Vector}}
end
WorkspacePool() = WorkspacePool(Dict{DataType, Vector{Vector}}())

function get_workspace!(pool::WorkspacePool, T::DataType, n::Int)
    vecs = get!(pool.store, T) do; Vector{Vector{T}}(); end
    for (i,v) in pairs(vecs)
        if length(v) == n
            return splice!(vecs, i)  # take it out
        end
    end
    return Vector{T}(undef, n)
end

function release_workspace!(pool::WorkspacePool, v::Vector)
    push!(get!(pool.store, eltype(v)) do; Vector{Vector}(); end, v)
    nothing
end

"""
    workbuffer!(dict, key, template)

Ensure `dict[key]` exists as a workspace array similar to `template`.
"""
function workbuffer!(dict::AbstractDict, key, template)
    haskey(dict, key) || (dict[key] = similar(template))
    return dict[key]
end

# ============================================================================
# TIMING & PROFILING
# ============================================================================
mutable struct Timer
    t0::Float64
    laps::Vector{Float64}
end
Timer() = Timer(time(), Float64[])

tic!(T::Timer) = (T.t0 = time(); nothing)
elapsed(T::Timer) = time() - T.t0
function lap!(T::Timer)
    push!(T.laps, elapsed(T))
    T.laps[end]
end
reset!(T::Timer) = (T.t0 = time(); empty!(T.laps); nothing)

struct ScopedTimer
    name::String
    start::Float64
end
ScopedTimer(name::String) = ScopedTimer(name, time())

macro time_block(name, ex)
    quote
        local __st = time()
        local __val = $(esc(ex))
        local __et = time() - __st
        root_print("[time] $($name): $(round(__et, digits=6)) s")
        __val
    end
end

"""
    timeit(f; reps=5)

Return (median, mean, std) timing (seconds) over `reps` executions of `f()`.
"""
function timeit(f; reps=5)
    ts = Vector{Float64}(undef, reps)
    for i in 1:reps
        t = time(); f(); ts[i] = time() - t
    end
    return median(ts), mean(ts), std(ts)
end

# ============================================================================
# NORMS / REDUCTIONS
# ============================================================================
local_sum(x::AbstractArray) = sum(x)
global_sum(x::AbstractArray) = mpi_allreduce_sum(sum(x))
global_sum(x::Number) = mpi_allreduce_sum(x)

function global_norm2(x::AbstractArray)
    s = mpi_allreduce_sum(sum(abs2, x))
    return sqrt(s)
end

l2norm(x) = global_norm2(x)
linfnorm(x) = mpi_allreduce_max(maximum(abs, x))

"""
    parsevalnorm_factor(N..., L...)

Compute factor relating sum(|x»|^2) to physical L2 norm under chosen normalization.
Assumes conventional FFT with forward unnormalized and inverse 1/N scaling.
"""
function parsevalnorm_factor(Ns::NTuple{M,Int}, Ls::NTuple{M,Real}) where {M}
    # Physical integral approx:  |u|^2 * (lambda L_i / N_i)
    cell = prod(Ls) / prod(Ns)
    return cell
end

parsevalnorm_factor(Nx::Int, Lx::Real) = Lx / Nx

# Weighted error norm (RMS of (err / (atol + rtol*scale)))
function error_weighted_norm(err, ref; atol=1e-8, rtol=1e-5)
    accum = 0.0; n = length(err)
    @inbounds for i in eachindex(err, ref)
        denom = atol + rtol*max(abs(ref[i]), abs(ref[i]+err[i]))
        accum += (err[i]/denom)^2
    end
    s = mpi_allreduce_sum(accum)
    return sqrt(s / n)
end
relative_tolerance_met(errnorm, tol) = errnorm <= tol

# ============================================================================
# DEALIAS / FILTERING
# ============================================================================
dealias_mask_1d(kvec::AbstractVector) = begin
    kmax = maximum(abs.(kvec))
    cutoff = (2/3)*kmax + eps(eltype(kvec))
    map(k->abs(k) <= cutoff, kvec)
end

function build_two_thirds_mask(k::AbstractVector, l::AbstractVector)
    mk = dealias_mask_1d(k); ml = dealias_mask_1d(l)
    mask = Array{Bool}(undef, length(k), length(l))
    for i in eachindex(k), j in eachindex(l)
        mask[i,j] = mk[i] & ml[j]
    end
    mask
end

function apply_two_thirds_mask!(A::AbstractArray{T,2}, mask::AbstractArray{Bool,2}) where {T}
    @inbounds for I in eachindex(A)
        mask[I] || (A[I] = zero(T))
    end
    A
end

"""
    spectral_filter_exp!(A, k, l; lambda=36.0, p=8)

Apply exponential spectral filter exp(-lambda (k/kmax)^p) (radial in max norm).
"""
function spectral_filter_exp!(A, k::AbstractVector, l::AbstractVector; lambda=36.0, p=8)
    kmax = maximum(abs.(k)); lmax = maximum(abs.(l))
    for i in eachindex(k), j in eachindex(l)
        eta = max(abs(k[i])/kmax, abs(l[j])/lmax)
        A[i,j] *= exp(-lambda * eta^p)
    end
    A
end

ksq(k,l) = k.^2 .+ l'.^2

function build_k2_array(k::AbstractVector, l::AbstractVector)
    K2 = Array{eltype(k)}(undef, length(k), length(l))
    for i in eachindex(k), j in eachindex(l)
        K2[i,j] = k[i]^2 + l[j]^2
    end
    K2
end

# ============================================================================
# LOGGING
# ============================================================================
@enum LogLevel begin
    LOG_DEBUG=1
    LOG_INFO=2
    LOG_WARN=3
    LOG_ERROR=4
end

const _LOGGER = Dict{Symbol,Any}(
    :level => LOG_INFO,
    :buffer => Vector{String}(),
    :autoflush => true,
    :file => nothing
)

set_log_level!(lvl::LogLevel) = (_LOGGER[:level]=lvl)
should_log(lvl::LogLevel) = lvl >= _LOGGER[:level]

function init_logger!(; filename=nothing, level=LOG_INFO, autoflush=true)
    _LOGGER[:level] = level
    _LOGGER[:autoflush] = autoflush
    if filename !== nothing && isroot()
        _LOGGER[:file] = open(filename, "w")
    end
end

function log_event!(lvl::LogLevel, msg::AbstractString)
    should_log(lvl) || return
    ts = iso_timestamp()
    line = "[$(string(lvl))][$ts][rank=$(comm_rank())] $msg"
    if isroot()
        push!(_LOGGER[:buffer], line)
        println(line)
        if _LOGGER[:autoflush]
            flush_log!()
        end
    end
end

function flush_log!()
    (_LOGGER[:file] === nothing) && return
    for ln in _LOGGER[:buffer]
        println(_LOGGER[:file], ln)
    end
    flush(_LOGGER[:file])
    empty!(_LOGGER[:buffer])
end

# ============================================================================
# CALLBACK REGISTRATION (Diagnostics Hooks)
# ============================================================================
mutable struct CallbackHandle
    id::Int
end
const _CALLBACKS = Dict{Int, Function}()
const _NEXT_CB_ID = Ref(1)

function register_callback!(f::Function)
    id = _NEXT_CB_ID[]
    _CALLBACKS[id] = f
    _NEXT_CB_ID[] += 1
    CallbackHandle(id)
end

function run_callbacks!(args...)
    for (_,f) in _CALLBACKS
        f(args...)
    end
end

clear_callbacks!() = empty!(_CALLBACKS)

# ============================================================================
# RANDOM / INITIAL CONDITIONS
# ============================================================================
seeded_rng(seed::Integer=0) = MersenneTwister(seed)

"""
    random_phase_field!(A, rng)

Populate `A` with complex numbers of unit magnitude and random phase.
"""
function random_phase_field!(A::AbstractArray{Complex{T}}, rng=seeded_rng()) where {T<:Real}
    @inbounds for I in eachindex(A)
        lambda = 2*pi*rand(rng)
        A[I] = cis(lambda)
    end
    A
end

"""
    random_spectrum_field!(A, k, l; slope = -3)

Construct a random complex field with isotropic power-law amplitude such that
|A|^2 ∝ k^{slope} approximately.
"""
function random_spectrum_field!(A, k::AbstractVector, l::AbstractVector; slope=-3, rng=seeded_rng())
    kmax = sqrt(maximum(k.^2)) # not exact isotropic but heuristic
    for i in eachindex(k), j in eachindex(l)
        k_tot = sqrt(k[i]^2 + l[j]^2) + eps(eltype(k))
        amp = k_tot^((slope)/2)  # amplitude ~ k^{slope/2} so |A|^2 ~ k^{slope}
        lambda = 2*pi*rand(rng)
        A[i,j] = amp * cis(lambda)
    end
    A
end

# ============================================================================
# MEMORY / FORMATTING
# ============================================================================
mem_usage_MB() = (Sys.total_memory() > 0 ? Sys.total_memory()/1024^2 : 0)  # placeholder

function bytestr(n::Integer)
    units = ["B","KB","MB","GB","TB"]
    f = float(n)
    for u in units
        f < 1024 && return string(round(f, digits=2), u)
        f /= 1024
    end
    string(round(f, digits=2), "PB")
end

format_rate(val, unit::AbstractString; per="s") = "$(round(val, digits=3)) $unit/$per"

# ============================================================================
# SAFE NUMERIC HELPERS
# ============================================================================
safediv(a,b; default=0) = (b==0 ? default : a/b)
clamp_step(val, minv, maxv) = min(max(val, minv), maxv)
stable_pow(x,p) = (x==0 ? zero(x) : x^p)

# ============================================================================
# JSON-LITE (Very Simple Key=Value Writer)
# ============================================================================
function kv_encode(dict::AbstractDict)
    io = IOBuffer()
    print(io, "{")
    first = true
    for (k,v) in dict
        first || print(io, ",")
        print(io, "\"", k, "\":")
        if v isa Number
            print(io, v)
        else
            print(io, "\"", v, "\"")
        end
        first = false
    end
    print(io, "}")
    String(take!(io))
end

function write_kv_file(path::AbstractString, dict::AbstractDict)
    open(path, "w") do io
        write(io, kv_encode(dict))
    end
end

# ============================================================================
# HASHING / CACHE KEYS
# ============================================================================
"""
    simple_hash64(x)

Naive 64-bit mixing for small structures (not cryptographic).
"""
function simple_hash64(x)
    h = UInt64(0x9e3779b97f4a7c15)
    h ⊻= hash(x, h)
    # final mix
    h ⊻= (h >> 30)
    h *= 0xbf58476d1ce4e5b9
    h ⊻= (h >> 27)
    h *= 0x94d049bb133111eb
    h ⊻= (h >> 31)
    return h
end

function hash_array64(A)
    h = UInt64(0x123456789abcdef0)
    @inbounds for x in A
        h ⊻= hash(x, h)
        h = (h << 7) | (h >> 57)
        h ⊻= 0x9e3779b97f4a7c15
    end
    return h
end

# ============================================================================
# DATE / TIME
# ============================================================================
iso_timestamp() = Dates.format(Dates.now(), Dates.dateformat"yyyy-mm-ddTHH:MM:SS")

# ============================================================================
# FIELD ANALYSIS UTILITIES
# ============================================================================
"""
    compute_kinetic_energy(solution)

Compute kinetic energy: 0.5 * (u2 + v2 + w2)
"""
function compute_kinetic_energy(solution::Dict{Symbol, Any})
    ke = zeros(size(get(solution, :u, zeros(1,1,1))))
    
    if haskey(solution, :u) && solution[:u] !== nothing
        ke += 0.5 * solution[:u].^2
    end
    if haskey(solution, :v) && solution[:v] !== nothing
        ke += 0.5 * solution[:v].^2
    end
    if haskey(solution, :w) && solution[:w] !== nothing
        ke += 0.5 * solution[:w].^2
    end
    
    return ke
end

"""
    compute_rms_field(solution, field_name)

Compute RMS value of a field.
"""
function compute_rms_field(solution::Dict{Symbol, Any}, field_name::Symbol)
    if haskey(solution, field_name) && solution[field_name] !== nothing
        field_data = solution[field_name]
        return sqrt(mean(field_data.^2))
    else
        return 0.0
    end
end

"""
    compute_mean_field(solution, field_name)

Compute horizontal mean of a field.
"""
function compute_mean_field(solution::Dict{Symbol, Any}, field_name::Symbol)
    if haskey(solution, field_name) && solution[field_name] !== nothing
        field_data = solution[field_name]
        # Compute horizontal average (assuming last dimension is vertical)
        return dropdims(mean(field_data, dims=(1,2)), dims=(1,2))
    else
        return zeros(1)
    end
end

# ============================================================================
# INTERNAL / CONVENIENCE
# ============================================================================
"""
    workbuffer!(dict, key, template)

(Already exported) Guarantee a work array exists in `dict[key]`.
"""
# (implementation above)

#end # module Utils
######################## END utils.jl ########################################
