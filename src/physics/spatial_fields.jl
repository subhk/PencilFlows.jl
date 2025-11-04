# Spatial Field Functions for PencilFlow.jl
# Provides time-independent, spatially-varying velocity fields and stratification

using LinearAlgebra

export SpatialField, VelocityField, StratificationField
export apply_spatial_field!, evaluate_spatial_field
export linear_shear, quadratic_profile, exponential_profile, sinusoidal_field
export constant_stratification, linear_stratification, exponential_stratification

# ============================================================================
# ABSTRACT TYPES AND STRUCTURES
# ============================================================================

"""
    SpatialField{F}

Abstract type for spatially-varying fields that are time-independent.
The field function F should accept spatial coordinates and return field values.
"""
abstract type SpatialField{F} end

"""
    VelocityField{F} <: SpatialField{F}

Represents a spatially-varying velocity field U(x,y,z) = (u(x,y,z), v(x,y,z), w(x,y,z)).
The function F should return a tuple (u, v, w) given coordinates (x, y, z).
"""
struct VelocityField{F} <: SpatialField{F}
    func::F
    components::Symbol  # :u, :v, :w, :uv, :uw, :vw, :uvw
    description::String
    
    VelocityField(func, components=:uvw, description="Custom velocity field") = 
        new{typeof(func)}(func, components, description)
end

"""
    StratificationField{F} <: SpatialField{F}

Represents a spatially-varying stratification/temperature field T(x,y,z) or pi(x,y,z).
The function F should return a scalar value given coordinates (x, y, z).
"""
struct StratificationField{F} <: SpatialField{F}
    func::F
    field_type::Symbol  # :temperature, :density, :buoyancy
    description::String
    
    StratificationField(func, field_type=:temperature, description="Custom stratification field") = 
        new{typeof(func)}(func, field_type, description)
end

# ============================================================================
# FIELD EVALUATION FUNCTIONS
# ============================================================================

"""
    evaluate_spatial_field(field::SpatialField, x, y, z)

Evaluate a spatial field at coordinates (x, y, z).
"""
function evaluate_spatial_field(field::VelocityField, x, y, z)
    return field.func(x, y, z)
end

function evaluate_spatial_field(field::StratificationField, x, y, z)
    return field.func(x, y, z)
end

"""
    apply_spatial_field!(target_array, field::SpatialField, grid, component=nothing)

Apply a spatial field to a distributed array using the grid coordinates.
For velocity fields, specify component inin [:u, :v, :w] or leave as nothing for full field.
"""
function apply_spatial_field!(target_array, field::VelocityField, grid; component=nothing)
    # Get local array data
    local_data = parent(target_array)
    local_size = size(local_data)
    
    # Get grid coordinates (assuming uniform grid for now)
    if hasfield(typeof(grid), :x) && hasfield(typeof(grid), :y) && hasfield(typeof(grid), :z)
        x_coords = grid.x
        y_coords = grid.y  
        z_coords = grid.z
    else
        error("Grid must have x, y, z coordinate arrays")
    end
    
    # Apply field at each grid point
    if component === nothing
        # Apply full velocity field (assuming target_array stores one component)
        error("Must specify component (:u, :v, or :w) when applying velocity field")
    else
        for k in 1:local_size[3], j in 1:local_size[2], i in 1:local_size[1]
            x, y, z = x_coords[i], y_coords[j], z_coords[k]
            velocity_components = field.func(x, y, z)
            
            if component == :u
                local_data[i, j, k] += velocity_components[1]
            elseif component == :v
                local_data[i, j, k] += velocity_components[2]
            elseif component == :w
                local_data[i, j, k] += velocity_components[3]
            else
                error("Invalid component: $component. Use :u, :v, or :w")
            end
        end
    end
    
    return target_array
end

function apply_spatial_field!(target_array, field::StratificationField, grid)
    # Get local array data
    local_data = parent(target_array)
    local_size = size(local_data)
    
    # Get grid coordinates
    if hasfield(typeof(grid), :x) && hasfield(typeof(grid), :y) && hasfield(typeof(grid), :z)
        x_coords = grid.x
        y_coords = grid.y  
        z_coords = grid.z
    else
        error("Grid must have x, y, z coordinate arrays")
    end
    
    # Apply field at each grid point
    for k in 1:local_size[3], j in 1:local_size[2], i in 1:local_size[1]
        x, y, z = x_coords[i], y_coords[j], z_coords[k]
        field_value = field.func(x, y, z)
        local_data[i, j, k] += field_value
    end
    
    return target_array
end

# ============================================================================
# PREDEFINED VELOCITY FIELD FUNCTIONS
# ============================================================================

"""
    linear_shear(U0, dU_dz; direction=:x)

Create a linear shear velocity field: U(z) = U0 + dU_dz * z
Direction specifies which velocity component (:xu, :yv, :zw)
"""
function linear_shear(U0::Real, dU_dz::Real; direction::Symbol=:x)
    if direction == :x
        return (x, y, z) -> (U0 + dU_dz * z, 0.0, 0.0)
    elseif direction == :y  
        return (x, y, z) -> (0.0, U0 + dU_dz * z, 0.0)
    elseif direction == :z
        return (x, y, z) -> (0.0, 0.0, U0 + dU_dz * z)
    else
        error("Direction must be :x, :y, or :z")
    end
end

"""
    quadratic_profile(U0, a, b; direction=:x)

Create a quadratic velocity profile: U(z) = U0 + a*z + b*z2
"""
function quadratic_profile(U0::Real, a::Real, b::Real; direction::Symbol=:x)
    if direction == :x
        return (x, y, z) -> (U0 + a*z + b*z^2, 0.0, 0.0)
    elseif direction == :y
        return (x, y, z) -> (0.0, U0 + a*z + b*z^2, 0.0)
    elseif direction == :z
        return (x, y, z) -> (0.0, 0.0, U0 + a*z + b*z^2)
    else
        error("Direction must be :x, :y, or :z")
    end
end

"""
    exponential_profile(U0, decay_rate, z0=0.0; direction=:x)

Create an exponential velocity profile: U(z) = U0 * exp(-decay_rate * (z - z0))
"""
function exponential_profile(U0::Real, decay_rate::Real, z0::Real=0.0; direction::Symbol=:x)
    if direction == :x
        return (x, y, z) -> (U0 * exp(-decay_rate * (z - z0)), 0.0, 0.0)
    elseif direction == :y
        return (x, y, z) -> (0.0, U0 * exp(-decay_rate * (z - z0)), 0.0)
    elseif direction == :z
        return (x, y, z) -> (0.0, 0.0, U0 * exp(-decay_rate * (z - z0)))
    else
        error("Direction must be :x, :y, or :z")
    end
end

"""
    sinusoidal_field(A, k; direction=:x, phase=0.0)

Create a sinusoidal velocity field: U(x,y,z) = A * sin(k**r* + pi)
k can be a scalar (for z-direction) or tuple (k0, kky, kkyz)
"""
function sinusoidal_field(A::Real, k; direction::Symbol=:x, phase::Real=0.0)
    if isa(k, Real)
        # 1D sinusoidal in z-direction
        wave_func = (x, y, z) -> A * sin(k * z + phase)
    elseif isa(k, Tuple) && length(k) == 3
        # 3D sinusoidal field
        kx, ky, kz = k
        wave_func = (x, y, z) -> A * sin(kx * x + ky * y + kz * z + phase)
    else
        error("k must be a scalar or 3-tuple (kx, ky, kz)")
    end
    
    if direction == :x
        return (x, y, z) -> (wave_func(x, y, z), 0.0, 0.0)
    elseif direction == :y
        return (x, y, z) -> (0.0, wave_func(x, y, z), 0.0)
    elseif direction == :z
        return (x, y, z) -> (0.0, 0.0, wave_func(x, y, z))
    else
        error("Direction must be :x, :y, or :z")
    end
end

# ============================================================================
# PREDEFINED STRATIFICATION FIELD FUNCTIONS
# ============================================================================

"""
    constant_stratification(N2)

Create a constant stratification field with Brunt-Vxzisxzlxz frequency squared N2.
Returns a function T(z) corresponding to dT/dz = -N2/g (assuming g=1).
"""
function constant_stratification(N2::Real)
    return (x, y, z) -> -N2 * z  # Linear temperature profile
end

"""
    linear_stratification(N20, dN2_dz)

Create a linearly varying stratification: N2(z) = N20 + dN2_dz * z
"""
function linear_stratification(N20::Real, dN2_dz::Real)
    return (x, y, z) -> -N20 * z - 0.5 * dN2_dz * z^2
end

"""
    exponential_stratification(N20, decay_scale, z0=0.0)

Create an exponential stratification: N2(z) = N20 * exp(-(z-z0)/decay_scale)
"""
function exponential_stratification(N20::Real, decay_scale::Real, z0::Real=0.0)
    # Integrate to get temperature profile
    return (x, y, z) -> -N20 * decay_scale * (exp(-(z - z0)/decay_scale) - 1.0)
end

"""
    tanh_stratification(N20, thickness, z_center)

Create a tanh stratification profile (useful for thermoclines):
N2(z) = N20 * (1 + tanh((z - z_center)/thickness)) / 2
"""
function tanh_stratification(N20::Real, thickness::Real, z_center::Real)
    return (x, y, z) -> begin
        xi = (z - z_center) / thickness
        # Analytical integral of tanh stratification  
        -N20 * thickness * (xi/2 + log(cosh(xi)))
    end
end

# ============================================================================
# INTEGRATION WITH EXISTING PENCILFLOW FUNCTIONS
# ============================================================================

"""
    initialize_velocity_field!(u, v, w, velocity_field::VelocityField, grid)

Initialize velocity components with a spatial velocity field.
"""
function initialize_velocity_field!(u, v, w, velocity_field::VelocityField, grid)
    apply_spatial_field!(u, velocity_field, grid; component=:u)
    apply_spatial_field!(v, velocity_field, grid; component=:v)
    apply_spatial_field!(w, velocity_field, grid; component=:w)
    return u, v, w
end

"""
    initialize_stratification_field!(b, strat_field::StratificationField, grid)

Initialize buoyancy/temperature field with spatial stratification.
"""
function initialize_stratification_field!(b, strat_field::StratificationField, grid)
    apply_spatial_field!(b, strat_field, grid)
    return b
end

"""
    add_spatial_forcing!(Ru, Rv, Rw, velocity_field::VelocityField, grid; strength=1.0)

Add spatial velocity field as a forcing term to momentum equations.
This can be used to maintain a background flow or apply body forces.
"""
function add_spatial_forcing!(Ru, Rv, Rw, velocity_field::VelocityField, grid; strength::Real=1.0)
    # Create temporary arrays for the spatial field
    temp_u = similar(Ru)
    temp_v = similar(Rv) 
    temp_w = similar(Rw)
    
    # Initialize with zero and add spatial field
    fill!(temp_u, 0.0)
    fill!(temp_v, 0.0)
    fill!(temp_w, 0.0)
    
    apply_spatial_field!(temp_u, velocity_field, grid; component=:u)
    apply_spatial_field!(temp_v, velocity_field, grid; component=:v)
    apply_spatial_field!(temp_w, velocity_field, grid; component=:w)
    
    # Add to RHS with specified strength
    Ru_data = parent(Ru)
    Rv_data = parent(Rv)
    Rw_data = parent(Rw)
    temp_u_data = parent(temp_u)
    temp_v_data = parent(temp_v)
    temp_w_data = parent(temp_w)
    
    @. Ru_data += strength * temp_u_data
    @. Rv_data += strength * temp_v_data
    @. Rw_data += strength * temp_w_data
    
    return Ru, Rv, Rw
end

"""
    add_stratification_forcing!(Rb, strat_field::StratificationField, grid; strength=1.0)

Add spatial stratification as a forcing/restoring term to buoyancy equation.
"""
function add_stratification_forcing!(Rb, strat_field::StratificationField, grid; strength::Real=1.0)
    temp_b = similar(Rb)
    fill!(temp_b, 0.0)
    
    apply_spatial_field!(temp_b, strat_field, grid)
    
    Rb_data = parent(Rb)
    temp_b_data = parent(temp_b)
    
    @. Rb_data += strength * temp_b_data
    
    return Rb
end

# ============================================================================
# EXAMPLE USAGE AND PREDEFINED CONFIGURATIONS
# ============================================================================

"""
    create_channel_flow(U_max, H; profile=:parabolic)

Create a channel flow velocity profile.
- :parabolic  u(z) = U_max * 4 * (z/H) * (1 - z/H)
- :linear  u(z) = U_max * (2z/H - 1) (linear shear)
"""
function create_channel_flow(U_max::Real, H::Real; profile::Symbol=:parabolic)
    if profile == :parabolic
        func = (x, y, z) -> (U_max * 4 * (z/H) * (1 - z/H), 0.0, 0.0)
    elseif profile == :linear
        func = (x, y, z) -> (U_max * (2*z/H - 1), 0.0, 0.0)
    else
        error("Profile must be :parabolic or :linear")
    end
    
    return VelocityField(func, :u, "Channel flow profile")
end

"""
    create_atmospheric_profile(u_ref, z_ref, z0; profile=:logarithmic)

Create atmospheric boundary layer velocity profile.
- :logarithmic  u(z) = u_ref * ln(z/z0) / ln(z_ref/z0)
- :power_law  u(z) = u_ref * (z/z_ref)^alpha
"""
function create_atmospheric_profile(u_ref::Real, z_ref::Real, z0::Real; 
                                  profile::Symbol=:logarithmic, alpha::Real=0.1)
    if profile == :logarithmic
        func = (x, y, z) -> begin
            z_safe = max(z, z0 + 1e-10)  # Avoid log(0)
            u_val = u_ref * log(z_safe/z0) / log(z_ref/z0)
            return (max(u_val, 0.0), 0.0, 0.0)
        end
    elseif profile == :power_law
        func = (x, y, z) -> (u_ref * (max(z, z0)/z_ref)^alpha, 0.0, 0.0)
    else
        error("Profile must be :logarithmic or :power_law")
    end
    
    return VelocityField(func, :u, "Atmospheric boundary layer profile")
end

"""
    create_ocean_stratification(N2_surface, N2_deep, thermocline_depth, thickness)

Create ocean-like stratification with surface mixed layer and thermocline.
"""
function create_ocean_stratification(N2_surface::Real, N2_deep::Real, 
                                    thermocline_depth::Real, thickness::Real)
    func = (x, y, z) -> begin
        # Smooth transition between surface and deep stratification
        transition = 0.5 * (1 + tanh((z - thermocline_depth) / thickness))
        N2 = N2_surface + (N2_deep - N2_surface) * transition
        return -N2 * z
    end
    
    return StratificationField(func, :buoyancy, "Ocean stratification profile")
end

# ============================================================================
# UTILITY FUNCTIONS FOR DIAGNOSTICS
# ============================================================================

"""
    compute_field_statistics(field::SpatialField, grid)

Compute statistics (min, max, mean, std) of a spatial field over the computational domain.
"""
function compute_field_statistics(field::VelocityField, grid)
    # This would need to be implemented with proper grid traversal
    # For now, return placeholder
    return Dict(
        :description => field.description,
        :components => field.components,
        :type => "VelocityField"
    )
end

function compute_field_statistics(field::StratificationField, grid)
    return Dict(
        :description => field.description,
        :field_type => field.field_type,
        :type => "StratificationField"
    )
end

"""
    plot_field_profile(field::SpatialField, z_range; x0=0.0, y0=0.0)

Generate data for plotting 1D profiles of spatial fields.
Returns (z_coords, field_values) for plotting.
"""
function plot_field_profile(field::StratificationField, z_range; x0::Real=0.0, y0::Real=0.0)
    z_coords = collect(z_range)
    field_values = [field.func(x0, y0, z) for z in z_coords]
    return z_coords, field_values
end

function plot_field_profile(field::VelocityField, z_range; x0::Real=0.0, y0::Real=0.0, component::Symbol=:u)
    z_coords = collect(z_range)
    field_values = []
    
    for z in z_coords
        velocity_tuple = field.func(x0, y0, z)
        if component == :u
            push!(field_values, velocity_tuple[1])
        elseif component == :v
            push!(field_values, velocity_tuple[2])
        elseif component == :w
            push!(field_values, velocity_tuple[3])
        else
            error("Component must be :u, :v, or :w")
        end
    end
    
    return z_coords, field_values
end