# Sub-Grid Scale Models for Large Eddy Simulation
# Implements various SGS models including Smagorinsky-Lilly and AMD

"""
    SmagorinskyLilly{T} <: AbstractSGSModel

Classical Smagorinsky-Lilly sub-grid scale model.

The eddy viscosity is computed as:
νₜ = (Cs Δ)² |S| 

where:
- Cs is the Smagorinsky coefficient (typically 0.17-0.2)
- Δ is the filter width
- |S| = √(2SᵢⱼSᵢⱼ) is the strain rate magnitude

# Fields
- `cs::T`: Smagorinsky coefficient
- `wall_damping::Bool`: Enable van Driest wall damping
- `damping_constant::T`: Wall damping constant (default 25)

# Reference
Smagorinsky, J. (1963). General circulation experiments with the primitive equations.
Monthly Weather Review, 91(3), 99-164.
"""
struct SmagorinskyLilly{T<:AbstractFloat} <: AbstractSGSModel
    cs::T                    # Smagorinsky coefficient  
    wall_damping::Bool       # Enable van Driest wall damping
    damping_constant::T      # Wall damping constant A⁺ ≈ 25
    
    function SmagorinskyLilly{T}(cs::T=0.17; wall_damping::Bool=false, 
                                damping_constant::T=25.0) where T<:AbstractFloat
        # Validation following PencilFlows patterns
        if cs < 0
            throw(ArgumentError("Smagorinsky coefficient must be non-negative, got $cs"))
        end
        if cs > 1.0
            @warn "Large Smagorinsky coefficient (Cs = $cs) - typical values are 0.1-0.2"
        end
        if wall_damping && damping_constant <= 0
            throw(ArgumentError("Wall damping constant must be positive when wall damping is enabled, got $damping_constant"))
        end
        
        new{T}(cs, wall_damping, damping_constant)
    end
end

SmagorinskyLilly(cs::T=0.17; kwargs...) where T = SmagorinskyLilly{T}(cs; kwargs...)

"""
    AnisotropicMinimumDissipation{T} <: AbstractSGSModel

Anisotropic Minimum Dissipation (AMD) model by Rozema et al.

The AMD model computes the eddy viscosity based on the balance between
kinetic energy transfer and dissipation, without requiring a model coefficient.

The eddy viscosity is:
νₜ = -Cᴬᴹᴰ Δ² (∂ᵢūⱼ ∂ⱼ ∂ₖūᵢ) / (∂ₖūᵢ ∂ₖūᵢ)

# Fields  
- `c_amd::T`: AMD coefficient (computed dynamically, typically ~1.0)
- `enable_clipping::Bool`: Clip negative viscosities
- `minimum_viscosity::T`: Minimum allowed viscosity

# Reference
Rozema, W., Bae, H. J., Moin, P., & Verstappen, R. (2015).
Minimum-dissipation models for large-eddy simulation.
Physics of Fluids, 27(8), 085107.
"""
struct AnisotropicMinimumDissipation{T<:AbstractFloat} <: AbstractSGSModel
    c_amd::T                 # AMD coefficient (dynamically computed)
    enable_clipping::Bool    # Clip negative viscosities
    minimum_viscosity::T     # Minimum allowed viscosity
    
    function AnisotropicMinimumDissipation{T}(; c_amd::T=1.0,
                                             enable_clipping::Bool=true,
                                             minimum_viscosity::T=0.0) where T<:AbstractFloat
        # Validation following PencilFlows patterns
        if c_amd <= 0
            throw(ArgumentError("AMD coefficient must be positive, got $c_amd"))
        end
        if c_amd > 5.0
            @warn "Large AMD coefficient (C_AMD = $c_amd) - typical values are 0.5-2.0"
        end
        if minimum_viscosity < 0
            throw(ArgumentError("Minimum viscosity must be non-negative, got $minimum_viscosity"))
        end
        
        new{T}(c_amd, enable_clipping, minimum_viscosity)
    end
end

AnisotropicMinimumDissipation(; kwargs...) = AnisotropicMinimumDissipation{Float64}(; kwargs...)

"""
    compute_sgs_stress!(τ, model::SmagorinskyLilly, S, Ω, Δ, workspace)

Compute sub-grid stress tensor using the Smagorinsky-Lilly model.

# Arguments
- `τ`: Output SGS stress tensor [Nx, Ny, Nz, 6]
- `model`: SmagorinskyLilly model parameters
- `S`: Strain rate tensor [Nx, Ny, Nz, 6] 
- `Ω`: Rotation rate tensor [Nx, Ny, Nz, 3] (unused in this model)
- `Δ`: Filter width (scalar or array)
- `workspace`: LESWorkspace for temporary computations
"""
function compute_sgs_stress!(τ, model::SmagorinskyLilly{T}, S, Ω, Δ, 
                           workspace::LESWorkspace{T}) where T
    Nx, Ny, Nz = size(workspace.eddy_viscosity)
    
    # Compute strain rate magnitude |S| = √(2SᵢⱼSᵢⱼ)
    @inbounds @simd for k in 1:Nz
        for j in 1:Ny
            for i in 1:Nx
                # S magnitude: |S|² = 2(S₁₁² + S₂₂² + S₃₃²) + 4(S₁₂² + S₁₃² + S₂₃²)
                s_mag_sq = 2 * (S[i,j,k,1]^2 + S[i,j,k,2]^2 + S[i,j,k,3]^2) +
                          4 * (S[i,j,k,4]^2 + S[i,j,k,5]^2 + S[i,j,k,6]^2)
                
                s_mag = sqrt(s_mag_sq)
                
                # Compute filter width (can be spatially varying)
                delta = isa(Δ, AbstractArray) ? Δ[i,j,k] : Δ
                
                # Smagorinsky eddy viscosity: νₜ = (Cs Δ)² |S|
                nu_t = (model.cs * delta)^2 * s_mag
                
                # Apply wall damping if enabled
                if model.wall_damping
                    # Simple wall distance approximation (z-direction)
                    wall_dist = min(k-1, Nz-k) * delta  # Approximate wall distance
                    damping = 1 - exp(-wall_dist / (model.damping_constant * delta))
                    nu_t *= damping
                end
                
                workspace.eddy_viscosity[i,j,k] = nu_t
            end
        end
    end
    
    # Compute SGS stress: τᵢⱼ = -2νₜ Sᵢⱼ with optimal performance
    @inbounds @fastmath @simd ivdep for k in 1:Nz
        for j in 1:Ny
            for i in 1:Nx
                nu_t = workspace.eddy_viscosity[i,j,k]
                two_nu_t = -2 * nu_t  # Precompute for efficiency
                
                # Store -2νₜSᵢⱼ in stress tensor
                τ[i,j,k,1] = two_nu_t * S[i,j,k,1]  # τ₁₁
                τ[i,j,k,2] = two_nu_t * S[i,j,k,2]  # τ₂₂  
                τ[i,j,k,3] = two_nu_t * S[i,j,k,3]  # τ₃₃
                τ[i,j,k,4] = two_nu_t * S[i,j,k,4]  # τ₁₂
                τ[i,j,k,5] = two_nu_t * S[i,j,k,5]  # τ₁₃
                τ[i,j,k,6] = two_nu_t * S[i,j,k,6]  # τ₂₃
            end
        end
    end
    
    return nothing
end

"""
    compute_sgs_stress!(τ, model::AnisotropicMinimumDissipation, S, Ω, Δ, workspace)

Compute sub-grid stress tensor using the AMD model.
"""
function compute_sgs_stress!(τ, model::AnisotropicMinimumDissipation{T}, S, Ω, Δ,
                           workspace::LESWorkspace{T}) where T
    Nx, Ny, Nz = size(workspace.eddy_viscosity)
    
    # The AMD model requires second derivatives of velocity for full implementation
    # Here we provide a simplified version using available quantities
    
    @inbounds @simd for k in 1:Nz
        for j in 1:Ny
            for i in 1:Nx
                # Compute local strain and rotation invariants
                # Q₁ = SᵢⱼSᵢⱼ (strain rate squared)
                Q1 = S[i,j,k,1]^2 + S[i,j,k,2]^2 + S[i,j,k,3]^2 +
                     2*(S[i,j,k,4]^2 + S[i,j,k,5]^2 + S[i,j,k,6]^2)
                
                # Q₂ = ΩᵢⱼΩᵢⱼ (rotation rate squared)  
                Q2 = 2*(Ω[i,j,k,1]^2 + Ω[i,j,k,2]^2 + Ω[i,j,k,3]^2)
                
                # Q₃ = SᵢⱼSⱼₖSₖᵢ (third invariant - simplified)
                Q3 = S[i,j,k,1]^3 + S[i,j,k,2]^3 + S[i,j,k,3]^3
                
                # AMD coefficient computation (simplified)
                delta = isa(Δ, AbstractArray) ? Δ[i,j,k] : Δ
                
                if Q1 > 1e-12  # Avoid division by zero
                    # Simplified AMD viscosity
                    nu_t = model.c_amd * delta^2 * sqrt(Q1) * 
                          (Q3 / (Q1^1.5 + 1e-12))
                else
                    nu_t = 0.0
                end
                
                # Apply clipping if enabled
                if model.enable_clipping && nu_t < model.minimum_viscosity
                    nu_t = model.minimum_viscosity
                end
                
                workspace.eddy_viscosity[i,j,k] = nu_t
            end
        end
    end
    
    # Compute SGS stress using computed viscosity
    @inbounds @simd for k in 1:Nz
        for j in 1:Ny
            for i in 1:Nx
                nu_t = workspace.eddy_viscosity[i,j,k]
                
                # τᵢⱼ = -2νₜ Sᵢⱼ
                τ[i,j,k,1] = -2 * nu_t * S[i,j,k,1]  
                τ[i,j,k,2] = -2 * nu_t * S[i,j,k,2]  
                τ[i,j,k,3] = -2 * nu_t * S[i,j,k,3]  
                τ[i,j,k,4] = -2 * nu_t * S[i,j,k,4]  
                τ[i,j,k,5] = -2 * nu_t * S[i,j,k,5]  
                τ[i,j,k,6] = -2 * nu_t * S[i,j,k,6]  
            end
        end
    end
    
    return nothing
end

"""
    model_viscosity(model::AbstractSGSModel, S, Ω, Δ) -> Array

Compute the eddy viscosity field for a given SGS model.
This is a convenience function for diagnostics.
"""
function model_viscosity(model::SmagorinskyLilly{T}, S, Ω, Δ) where T
    Nx, Ny, Nz = size(S)[1:3]
    nu_t = Array{T,3}(undef, Nx, Ny, Nz)
    
    @inbounds @simd for k in 1:Nz
        for j in 1:Ny
            for i in 1:Nx
                s_mag_sq = 2 * (S[i,j,k,1]^2 + S[i,j,k,2]^2 + S[i,j,k,3]^2) +
                          4 * (S[i,j,k,4]^2 + S[i,j,k,5]^2 + S[i,j,k,6]^2)
                s_mag = sqrt(s_mag_sq)
                
                delta = isa(Δ, AbstractArray) ? Δ[i,j,k] : Δ
                nu_t[i,j,k] = (model.cs * delta)^2 * s_mag
            end
        end
    end
    
    return nu_t
end

function model_viscosity(model::AnisotropicMinimumDissipation{T}, S, Ω, Δ) where T
    Nx, Ny, Nz = size(S)[1:3]
    nu_t = Array{T,3}(undef, Nx, Ny, Nz)
    
    @inbounds @simd for k in 1:Nz
        for j in 1:Ny
            for i in 1:Nx
                Q1 = S[i,j,k,1]^2 + S[i,j,k,2]^2 + S[i,j,k,3]^2 +
                     2*(S[i,j,k,4]^2 + S[i,j,k,5]^2 + S[i,j,k,6]^2)
                
                Q3 = S[i,j,k,1]^3 + S[i,j,k,2]^3 + S[i,j,k,3]^3
                
                delta = isa(Δ, AbstractArray) ? Δ[i,j,k] : Δ
                
                if Q1 > 1e-12
                    nu_t[i,j,k] = model.c_amd * delta^2 * sqrt(Q1) * 
                                 (Q3 / (Q1^1.5 + 1e-12))
                else
                    nu_t[i,j,k] = 0.0
                end
                
                if model.enable_clipping && nu_t[i,j,k] < model.minimum_viscosity
                    nu_t[i,j,k] = model.minimum_viscosity
                end
            end
        end
    end
    
    return nu_t
end