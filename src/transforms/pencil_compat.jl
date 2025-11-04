# module PencilCompat
# Note: This file is included by the main module, so dependencies are already loaded
# using MPI
# using PencilFFTs
# using PencilArrays
# export PencilPlanRFFT, make_pencil_plan_rfft, forward_rfft!, backward_rfft!,
#        localindices_dim, localrange_dim, pencilize_array, isparallel, ensure_comm,
#        alloc_phys, alloc_spec, global_sizes

# Wrapper structure to mimic existing plan interfaces
struct PencilPlanRFFT{F,B,Meta}
    forward :: F      # forward real->complex (or complex->complex) plan
    backward :: B     # backward complex->real plan
    meta :: Meta      # NamedTuple: dims, T, scaling, layout info
end

# Ensure MPI is initialized once
function ensure_comm(comm::MPI.Comm=MPI.COMM_WORLD)
    MPI.Initialized() || MPI.Init()
    return comm
end

isparallel(pp::PencilPlanRFFT) = get(pp.meta, :parallel, false)

function make_pencil_plan_rfft(::Type{T}, dims::NTuple{N,Int}; comm=MPI.COMM_WORLD, real_transform=true) where {T,N}
    comm = ensure_comm(comm)
    if real_transform
        fplan = PencilFFTs.plan_rfft(T, dims; comm)
        bplan = PencilFFTs.plan_irfft(T, dims; comm)
    else
        fplan = PencilFFTs.plan_fft(T, dims; comm)
        bplan = PencilFFTs.plan_ifft(T, dims; comm)
    end
    meta = (dims=dims, T=T, parallel=true, real_transform=real_transform)
    return PencilPlanRFFT(fplan, bplan, meta)
end

# Forward / backward wrappers (real-complex or complex-complex)
forward_rfft!(plan::PencilPlanRFFT, u_real, u_spec) = PencilFFTs.forward!(plan.forward, u_real, u_spec)
backward_rfft!(plan::PencilPlanRFFT, u_spec, u_real) = PencilFFTs.backward!(plan.backward, u_spec, u_real)

# Local index helpers
localindices_dim(A::PencilArrays.PencilArray, d::Int) = PencilFFTs.localindices(A, d)
localrange_dim(A::PencilArrays.PencilArray, d::Int) = first(localindices_dim(A,d)):last(localindices_dim(A,d))

pencilize_array(A::AbstractArray) = A  # serial fallback convenience

alloc_phys(::Type{T}, plan::PencilPlanRFFT) where {T} = PencilArrays.PencilArray{T}(undef, plan.meta.dims; comm=plan.forward.comm)
alloc_spec(::Type{T}, plan::PencilPlanRFFT) where {T} = PencilArrays.PencilArray{T}(undef, plan.meta.dims; comm=plan.forward.comm)

global_sizes(plan::PencilPlanRFFT) = plan.meta.dims

# end # module PencilCompat
