# Large Eddy Simulation Module for PencilFlows.jl
# Optional module that provides LES capabilities with sub-grid scale modeling

module LES

using LinearAlgebra, Statistics
using ..PencilFlows: AbstractArray, AbstractWorkspace

export 
    # Abstract types
    AbstractSGSModel,
    
    # SGS Models
    SmagorinskyLilly,
    AnisotropicMinimumDissipation,
    
    # LES solver and utilities
    LESConfiguration,
    LESWorkspace,
    create_les_workspace,
    
    # Core LES functions
    compute_sgs_stress!,
    apply_sgs_model!,
    compute_strain_rate_tensor!,
    compute_rotation_rate_tensor!,
    
    # Filtering operations
    apply_les_filter!,
    box_filter!,
    gaussian_filter!,
    
    # Activation and configuration
    activate_les!,
    deactivate_les!,
    is_les_active,
    
    # Memory management (PencilFlows pattern)
    get_les_buffer!,
    clear_les_buffers!,
    estimate_les_memory_usage

# Global LES state - Type stable implementation
const LES_ACTIVE = Ref(false)
const LES_CONFIG_VALID = Ref(false)
const CURRENT_LES_CONFIG = Ref{LESConfiguration}()

"""
    AbstractSGSModel

Abstract base type for all sub-grid scale models.
All SGS models must implement:
- `compute_sgs_stress!(τ, model, S, Ω, Δ, workspace)`
- `model_viscosity(model, S, Ω, Δ)`
"""
abstract type AbstractSGSModel end

"""
    LESConfiguration{T, M<:AbstractSGSModel}

Configuration parameters for LES simulations.

# Fields
- `sgs_model::M`: Sub-grid scale model (Smagorinsky-Lilly, AMD, etc.)
- `filter_width::T`: Grid filter width (typically Δ = (ΔxΔyΔz)^(1/3))
- `filter_type::Symbol`: Type of spatial filter (:box, :gaussian, :spectral, :implicit)
- `cs_coefficient::T`: Smagorinsky coefficient (for Smagorinsky-based models)
- `enable_backscatter::Bool`: Allow negative viscosity for energy backscatter
- `clip_negative_viscosity::Bool`: Prevent negative eddy viscosity

# Examples
```julia
# Smagorinsky-Lilly model with box filter
sgs_model = SmagorinskyLilly(0.17; wall_damping=true)
config = LESConfiguration(sgs_model, 0.1; filter_type=:box)

# AMD model with implicit filtering (no explicit filtering)
amd_model = AnisotropicMinimumDissipation(c_amd=1.2)  
config = LESConfiguration(amd_model, 0.05; filter_type=:implicit)

# Enable energy backscatter for research applications
config = LESConfiguration(sgs_model, 0.1; enable_backscatter=true, 
                         clip_negative_viscosity=false)
```
"""
struct LESConfiguration{T<:AbstractFloat, M<:AbstractSGSModel}
    sgs_model::M
    filter_width::T
    filter_type::Symbol
    cs_coefficient::T
    enable_backscatter::Bool
    clip_negative_viscosity::Bool
    
    function LESConfiguration(sgs_model::M, filter_width::T=0.1;
                             filter_type::Symbol=:box,
                             cs_coefficient::T=0.17,
                             enable_backscatter::Bool=false,
                             clip_negative_viscosity::Bool=true) where {T<:AbstractFloat, M<:AbstractSGSModel}
        new{T,M}(sgs_model, filter_width, filter_type, cs_coefficient,
                 enable_backscatter, clip_negative_viscosity)
    end
end

"""
    LESWorkspace{T}

Workspace for LES computations to minimize allocations.
Uses Structure-of-Arrays layout for optimal cache performance and vectorization.
Compatible with PencilFlows spectral derivative system and memory pooling patterns.

# Fields
## Strain Rate Tensor Components (Structure-of-Arrays for cache efficiency)
- `S11, S22, S33::Array{T,3}`: Diagonal strain rate components [Nx, Ny, Nz]  
- `S12, S13, S23::Array{T,3}`: Off-diagonal strain rate components [Nx, Ny, Nz]

## Rotation Rate Tensor Components  
- `Ω12, Ω13, Ω23::Array{T,3}`: Anti-symmetric rotation rate components [Nx, Ny, Nz]

## SGS Stress Tensor Components
- `τ11, τ22, τ33::Array{T,3}`: Diagonal SGS stress components [Nx, Ny, Nz]
- `τ12, τ13, τ23::Array{T,3}`: Off-diagonal SGS stress components [Nx, Ny, Nz]

## Scalar Fields
- `eddy_viscosity::Array{T,3}`: Eddy viscosity field νt [Nx, Ny, Nz]
- `temp_field::Array{T,3}`: General purpose temporary [Nx, Ny, Nz]
- `filter_buffer::Array{T,3}`: Buffer for filtering operations [Nx, Ny, Nz]

## Velocity Gradient Components (Structure-of-Arrays)
- `dudx, dudy, dudz::Array{T,3}`: u-velocity gradients [Nx, Ny, Nz]
- `dvdx, dvdy, dvdz::Array{T,3}`: v-velocity gradients [Nx, Ny, Nz]  
- `dwdx, dwdy, dwdz::Array{T,3}`: w-velocity gradients [Nx, Ny, Nz]

## Memory Management
- `buffer_pool::Dict{String,Any}`: Reusable buffer pool for allocation-free operations
- `strain_magnitude::Array{T,3}`: Pre-allocated strain magnitude field
- `vorticity_magnitude::Array{T,3}`: Pre-allocated vorticity magnitude field
"""
mutable struct LESWorkspace{T} <: AbstractWorkspace
    # Strain rate tensor components (Structure-of-Arrays for cache efficiency)
    S11::Array{T,3}; S22::Array{T,3}; S33::Array{T,3}  # Diagonal components
    S12::Array{T,3}; S13::Array{T,3}; S23::Array{T,3}  # Off-diagonal components
    
    # Rotation rate tensor components (anti-symmetric)
    Ω12::Array{T,3}; Ω13::Array{T,3}; Ω23::Array{T,3}
    
    # SGS stress tensor components  
    τ11::Array{T,3}; τ22::Array{T,3}; τ33::Array{T,3}  # Diagonal stress components
    τ12::Array{T,3}; τ13::Array{T,3}; τ23::Array{T,3}  # Off-diagonal stress components
    
    # Scalar fields
    eddy_viscosity::Array{T,3}     # Eddy viscosity field νt
    temp_field::Array{T,3}         # General purpose temporary
    filter_buffer::Array{T,3}      # For filtering operations
    
    # Velocity gradient components (Structure-of-Arrays for vectorization)
    dudx::Array{T,3}; dudy::Array{T,3}; dudz::Array{T,3}  # ∇u components
    dvdx::Array{T,3}; dvdy::Array{T,3}; dvdz::Array{T,3}  # ∇v components  
    dwdx::Array{T,3}; dwdy::Array{T,3}; dwdz::Array{T,3}  # ∇w components
    
    # Pre-allocated diagnostic fields (avoid allocation in hot paths)
    strain_magnitude::Array{T,3}      # |S| = √(2SᵢⱼSᵢⱼ)
    vorticity_magnitude::Array{T,3}   # |Ω| = √(2ΩᵢⱼΩᵢⱼ)
    
    # Memory pool for workspace buffers (PencilFlows pattern)
    buffer_pool::Dict{String,Any}  # Reusable buffer pool for allocation-free operations
    
    function LESWorkspace{T}(Nx::Int, Ny::Int, Nz::Int) where T
        new{T}(
            # Strain rate tensor components (Structure-of-Arrays)
            Array{T,3}(undef, Nx, Ny, Nz),    # S11
            Array{T,3}(undef, Nx, Ny, Nz),    # S22
            Array{T,3}(undef, Nx, Ny, Nz),    # S33
            Array{T,3}(undef, Nx, Ny, Nz),    # S12
            Array{T,3}(undef, Nx, Ny, Nz),    # S13
            Array{T,3}(undef, Nx, Ny, Nz),    # S23
            
            # Rotation rate tensor components
            Array{T,3}(undef, Nx, Ny, Nz),    # Ω12
            Array{T,3}(undef, Nx, Ny, Nz),    # Ω13
            Array{T,3}(undef, Nx, Ny, Nz),    # Ω23
            
            # SGS stress tensor components
            Array{T,3}(undef, Nx, Ny, Nz),    # τ11
            Array{T,3}(undef, Nx, Ny, Nz),    # τ22
            Array{T,3}(undef, Nx, Ny, Nz),    # τ33
            Array{T,3}(undef, Nx, Ny, Nz),    # τ12
            Array{T,3}(undef, Nx, Ny, Nz),    # τ13
            Array{T,3}(undef, Nx, Ny, Nz),    # τ23
            
            # Scalar fields
            Array{T,3}(undef, Nx, Ny, Nz),    # eddy_viscosity
            Array{T,3}(undef, Nx, Ny, Nz),    # temp_field
            Array{T,3}(undef, Nx, Ny, Nz),    # filter_buffer
            
            # Velocity gradient components (Structure-of-Arrays)
            Array{T,3}(undef, Nx, Ny, Nz),    # dudx
            Array{T,3}(undef, Nx, Ny, Nz),    # dudy
            Array{T,3}(undef, Nx, Ny, Nz),    # dudz
            Array{T,3}(undef, Nx, Ny, Nz),    # dvdx
            Array{T,3}(undef, Nx, Ny, Nz),    # dvdy
            Array{T,3}(undef, Nx, Ny, Nz),    # dvdz
            Array{T,3}(undef, Nx, Ny, Nz),    # dwdx
            Array{T,3}(undef, Nx, Ny, Nz),    # dwdy
            Array{T,3}(undef, Nx, Ny, Nz),    # dwdz
            
            # Pre-allocated diagnostic fields
            Array{T,3}(undef, Nx, Ny, Nz),    # strain_magnitude
            Array{T,3}(undef, Nx, Ny, Nz),    # vorticity_magnitude
            
            # Memory pool
            Dict{String,Any}()                 # buffer_pool
        )
    end
    
    # Constructor from template array for PencilArray compatibility
    function LESWorkspace{T}(template_array) where T
        Nx, Ny, Nz = size(template_array)
        new{T}(
            # Strain rate tensor components
            similar(template_array, T, (Nx, Ny, Nz)), similar(template_array, T, (Nx, Ny, Nz)), similar(template_array, T, (Nx, Ny, Nz)),  # S11, S22, S33
            similar(template_array, T, (Nx, Ny, Nz)), similar(template_array, T, (Nx, Ny, Nz)), similar(template_array, T, (Nx, Ny, Nz)),  # S12, S13, S23
            
            # Rotation rate tensor components  
            similar(template_array, T, (Nx, Ny, Nz)), similar(template_array, T, (Nx, Ny, Nz)), similar(template_array, T, (Nx, Ny, Nz)),  # Ω12, Ω13, Ω23
            
            # SGS stress tensor components
            similar(template_array, T, (Nx, Ny, Nz)), similar(template_array, T, (Nx, Ny, Nz)), similar(template_array, T, (Nx, Ny, Nz)),  # τ11, τ22, τ33
            similar(template_array, T, (Nx, Ny, Nz)), similar(template_array, T, (Nx, Ny, Nz)), similar(template_array, T, (Nx, Ny, Nz)),  # τ12, τ13, τ23
            
            # Scalar fields
            similar(template_array, T, (Nx, Ny, Nz)),  # eddy_viscosity
            similar(template_array, T, (Nx, Ny, Nz)),  # temp_field
            similar(template_array, T, (Nx, Ny, Nz)),  # filter_buffer
            
            # Velocity gradient components
            similar(template_array, T, (Nx, Ny, Nz)), similar(template_array, T, (Nx, Ny, Nz)), similar(template_array, T, (Nx, Ny, Nz)),  # dudx, dudy, dudz
            similar(template_array, T, (Nx, Ny, Nz)), similar(template_array, T, (Nx, Ny, Nz)), similar(template_array, T, (Nx, Ny, Nz)),  # dvdx, dvdy, dvdz
            similar(template_array, T, (Nx, Ny, Nz)), similar(template_array, T, (Nx, Ny, Nz)), similar(template_array, T, (Nx, Ny, Nz)),  # dwdx, dwdy, dwdz
            
            # Pre-allocated diagnostic fields  
            similar(template_array, T, (Nx, Ny, Nz)),  # strain_magnitude
            similar(template_array, T, (Nx, Ny, Nz)),  # vorticity_magnitude
            
            # Memory pool
            Dict{String,Any}()
        )
    end
end

"""
    create_les_workspace(::Type{T}, Nx::Int, Ny::Int, Nz::Int) -> LESWorkspace{T}

Create a LES workspace with appropriate array dimensions.
Includes memory usage estimation following PencilFlows patterns.
"""
function create_les_workspace(::Type{T}, Nx::Int, Ny::Int, Nz::Int) where T
    # Memory usage estimation
    memory_per_3d_field = sizeof(T) * Nx * Ny * Nz
    memory_per_4d_tensor = sizeof(T) * Nx * Ny * Nz * 6  # For symmetric tensors
    memory_per_4d_vector = sizeof(T) * Nx * Ny * Nz * 3  # For anti-symmetric tensors or gradients
    
    total_memory = (
        2 * memory_per_4d_tensor +    # strain_rate, sgs_stress  
        1 * memory_per_4d_vector +    # rotation_rate
        2 * memory_per_3d_field +     # eddy_viscosity, temp_field, filter_buffer
        3 * memory_per_4d_vector      # grad_u, grad_v, grad_w
    ) + memory_per_3d_field  # filter_buffer
    
    total_mb = total_memory / (1024^2)
    @info "Creating LES workspace" memory_usage_MB=round(total_mb, digits=2) grid_size="$Nx×$Ny×$Nz" precision=T
    
    return LESWorkspace{T}(Nx, Ny, Nz)
end

"""
    activate_les!(config::LESConfiguration)

Activate LES mode with the specified configuration.
Includes validation consistent with PencilFlows patterns.
"""
function activate_les!(config::LESConfiguration)
    # Input validation following PencilFlows patterns
    if config.filter_width <= 0
        throw(ArgumentError("Filter width must be positive, got $(config.filter_width)"))
    end
    
    if config.cs_coefficient < 0
        throw(ArgumentError("Smagorinsky coefficient must be non-negative, got $(config.cs_coefficient)"))
    end
    
    if !(config.filter_type in [:box, :gaussian, :spectral, :implicit])
        throw(ArgumentError("Unknown filter type: $(config.filter_type). Available: :box, :gaussian, :spectral, :implicit"))
    end
    
    # Activate LES with type-stable state management
    CURRENT_LES_CONFIG[] = config
    LES_CONFIG_VALID[] = true
    LES_ACTIVE[] = true
    
    # Provide informative output following PencilFlows style
    @info "LES module activated" SGS_model=typeof(config.sgs_model) filter_width=config.filter_width filter_type=config.filter_type Cs=config.cs_coefficient
    
    return nothing
end

"""
    deactivate_les!()

Deactivate LES mode and return to DNS mode.
"""
function deactivate_les!()
    LES_ACTIVE[] = false
    LES_CONFIG_VALID[] = false
    @info "LES module deactivated - returning to DNS mode"
    return nothing
end

"""
    is_les_active() -> Bool

Check if LES mode is currently active.
"""
is_les_active() = LES_ACTIVE[]

"""
    get_les_config() -> Union{Nothing, LESConfiguration}

Get the current LES configuration, or nothing if LES is not active.
Type-stable implementation using separate validation flag.
"""
get_les_config() = LES_CONFIG_VALID[] ? CURRENT_LES_CONFIG[] : nothing

"""
    get_les_config_unsafe() -> LESConfiguration

Get the current LES configuration without validation check.
Use only when LES activation is guaranteed. Type-stable and faster.
"""
@inline get_les_config_unsafe() = CURRENT_LES_CONFIG[]

"""
    get_les_buffer!(workspace::LESWorkspace, key::String, template)

Get a reusable buffer from the LES workspace pool following PencilFlows patterns.
Creates and caches buffer if not present, returns cached buffer if available.

# Arguments
- `workspace`: LES workspace with buffer pool
- `key`: String identifier for the buffer  
- `template`: Template array for size and type information

# Returns
- Reusable buffer array compatible with template
"""
function get_les_buffer!(workspace::LESWorkspace{T}, key::String, template) where T
    if haskey(workspace.buffer_pool, key)
        buffer = workspace.buffer_pool[key]
        if size(buffer) == size(template) && eltype(buffer) == eltype(template)
            return buffer
        end
    end
    
    # Create new buffer and cache it
    buffer = similar(template)
    workspace.buffer_pool[key] = buffer
    return buffer
end

"""
    clear_les_buffers!(workspace::LESWorkspace)

Clear all cached buffers from the LES workspace pool.
Useful for memory management in long-running simulations.
"""
function clear_les_buffers!(workspace::LESWorkspace)
    empty!(workspace.buffer_pool)
    return nothing
end

"""
    estimate_les_memory_usage(workspace::LESWorkspace{T}) -> Dict

Estimate memory usage of LES workspace following PencilFlows diagnostic patterns.

# Returns
Dictionary with memory usage breakdown in MB
"""
function estimate_les_memory_usage(workspace::LESWorkspace{T}) where T
    usage = Dict{String, Float64}()
    
    # Main arrays
    usage["strain_rate"] = sizeof(workspace.strain_rate) / (1024^2)
    usage["rotation_rate"] = sizeof(workspace.rotation_rate) / (1024^2)  
    usage["sgs_stress"] = sizeof(workspace.sgs_stress) / (1024^2)
    usage["eddy_viscosity"] = sizeof(workspace.eddy_viscosity) / (1024^2)
    usage["temp_field"] = sizeof(workspace.temp_field) / (1024^2)
    usage["filter_buffer"] = sizeof(workspace.filter_buffer) / (1024^2)
    
    # Gradient arrays
    usage["grad_u"] = sizeof(workspace.grad_u) / (1024^2)
    usage["grad_v"] = sizeof(workspace.grad_v) / (1024^2)
    usage["grad_w"] = sizeof(workspace.grad_w) / (1024^2)
    
    # Buffer pool
    buffer_size = 0
    for (key, buffer) in workspace.buffer_pool
        buffer_size += sizeof(buffer)
    end
    usage["buffer_pool"] = buffer_size / (1024^2)
    
    usage["total"] = sum(values(usage))
    
    return usage
end

# Include sub-modules
include("sgs_models.jl")
include("filtering.jl")
include("tensor_operations.jl")
include("les_solver.jl")

end # module LES