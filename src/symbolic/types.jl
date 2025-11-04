# Core Types and Operators for the Symbolic Interface

# Basis types for different coordinate directions
abstract type AbstractBasis end

struct Fourier <: AbstractBasis
    name::Symbol
    interval::Tuple{Float64, Float64}
    dealias::Float64
    Fourier(name, interval, dealias=2/3) = new(name, Float64.(interval), dealias)
end

struct FiniteDifference <: AbstractBasis
    name::Symbol
    interval::Tuple{Float64, Float64}
    grid_type::Symbol
    FiniteDifference(name, interval, grid_type=:chebyshev_points) = new(name, Float64.(interval), grid_type)
end

const Coordinate = AbstractBasis

function Basis(name::Symbol, N::Int, interval::Tuple{<:Real, <:Real}; basis_type::Symbol=:Fourier, dealias=2/3, grid_type::Symbol=:chebyshev_points)
    if basis_type == :Fourier
        return Fourier(name, interval, dealias), N
    elseif basis_type == :FiniteDifference
        return FiniteDifference(name, interval, grid_type), N
    else
        error("Unknown basis type: $basis_type. Available: :Fourier, :FiniteDifference")
    end
end

struct Domain
    bases::Vector{AbstractBasis}
    grid_points::Vector{Int}
    function Domain(bases_and_points...)
        bases = AbstractBasis[]
        points = Int[]
        for item in bases_and_points
            if isa(item, Tuple{<:AbstractBasis, <:Int})
                push!(bases, item[1]); push!(points, item[2])
            elseif isa(item, AbstractBasis)
                push!(bases, item); push!(points, 64)
            else
                error("Domain expects (basis, N) tuples or basis objects")
            end
        end
        new(bases, points)
    end
end

struct Field
    name::Symbol
    domain::Union{Domain, Nothing}
    Field(name::Symbol) = new(name, nothing)
    Field(name::Symbol, domain::Domain) = new(name, domain)
end

struct Parameter
    name::Symbol
    value::Float64
    Parameter(name::Symbol, value::Real) = new(name, Float64(value))
end

# Helper functions will be defined after SymbolicProblem struct

struct SymbolicBoundaryCondition
    field::Symbol
    location::Symbol
    type::Symbol
    value::Union{Real, Function, Expr}
    time_dependent::Bool
    SymbolicBoundaryCondition(field, location, type, value::Real) = new(field, location, type, value, false)
    SymbolicBoundaryCondition(field, location, type, value::Union{Function, Expr}) = new(field, location, type, value, true)
end

# Abstract types for better type stability
abstract type AbstractPlan end
abstract type AbstractWorkspace end
abstract type AbstractTimeStepper end
abstract type AbstractLogger end
abstract type AbstractOutputHandler end

mutable struct DiscretizationInfo{T<:AbstractFloat}
    pencil_decomposition::Union{AbstractPlan, Nothing}
    distributed_fields::Union{NamedTuple, Nothing}
    grid_x::Vector{T}
    grid_y::Vector{T}
    grid_z::Vector{T}
    fft_plan_rfft::Union{AbstractPlan, Nothing}
    poisson_plan::Union{AbstractPlan, Nothing}
    mg_plan::Union{AbstractPlan, Nothing}
    rsns_workspace::Union{AbstractWorkspace, Nothing}
    nonlinear_workspace::Union{AbstractWorkspace, Nothing}
    pencil_grid::Union{AbstractPlan, Nothing}
    transform_plans::Union{AbstractPlan, Nothing}
    time_stepper::Union{AbstractTimeStepper, Nothing}
    domain_grids::Union{AbstractPlan, Nothing}
    coriolis_plane::Union{AbstractPlan, Nothing}
    workspace_pool::Union{Dict{String,Any}, Nothing}
    timer::Union{Any, Nothing}  # Keep as Any - external timer types
    logger::Union{AbstractLogger, Nothing}
    output_handler::Union{AbstractOutputHandler, Nothing}
    file_handlers::Dict{Symbol, Any}  # Keep as Any for file handle flexibility
    dx_matrix::Union{Nothing, Any}
    dy_matrix::Union{Nothing, Any}
    dz_matrix::Union{Nothing, Any}
    d2z_matrix::Union{Nothing, Any}
    bc_matrices::Dict{Symbol, AbstractMatrix}
    linear_operators::Dict{Symbol, AbstractMatrix}
    nonlinear_functions::Dict{Symbol, Function}
    function DiscretizationInfo{T}() where T<:AbstractFloat
        new{T}(nothing, nothing, T[], T[], T[],
            nothing, nothing, nothing, nothing, nothing,
            nothing, nothing, nothing, nothing, nothing,
            nothing, nothing, nothing,
            nothing, Dict{Symbol, Any}(),
            nothing, nothing, nothing, nothing,
            Dict{Symbol, AbstractMatrix}(), Dict{Symbol, AbstractMatrix}(), Dict{Symbol, Function}())
    end
end

# Convenience constructor
DiscretizationInfo() = DiscretizationInfo{Float64}()

mutable struct SymbolicProblem
    domain::Union{Domain, Nothing}
    fields::Vector{Field}
    parameters::Dict{Symbol, Float64}
    equations::Vector{Expr}
    boundary_conditions::Vector{SymbolicBoundaryCondition}
    discretization::Union{DiscretizationInfo, Nothing}
    grid_points::Dict{Symbol, Int}
    vars::Dict{Symbol, Any}
    metadata::Union{Dict{Symbol, Any}, Nothing}
    function SymbolicProblem(domain::Union{Domain, Nothing}=nothing)
        new(domain, Field[], Dict{Symbol, Float64}(), Expr[], SymbolicBoundaryCondition[],
            nothing, Dict{Symbol, Int}(), Dict{Symbol, Any}(), Dict{Symbol, Any}())
    end
end

# Convenience helpers for Domain management
function set_domain!(prob::SymbolicProblem, domain::Domain)
    prob.domain = domain
    return prob
end

function get_basis(prob::SymbolicProblem, name::Symbol)
    prob.domain === nothing && return nothing
    for (i, b) in enumerate(prob.domain.bases)
        b.name == name && return b
    end
    return nothing
end

function get_grid_points(prob::SymbolicProblem)
    if prob.domain === nothing
        return Dict{Symbol,Int}()
    end
    d = Dict{Symbol,Int}()
    for (i, b) in enumerate(prob.domain.bases)
        d[b.name] = prob.domain.grid_points[i]
    end
    return d
end

struct DifferentialOperator
    name::Symbol
    order::Int
    direction::Union{Symbol, Nothing}
end

const dx = DifferentialOperator(:dx, 1, :x)
const dy = DifferentialOperator(:dy, 1, :y)
const dz = DifferentialOperator(:dz, 1, :z)
const dt = DifferentialOperator(:dt, 1, :t)

const t = :t

lap(field) = :(d2x($field) + d2y($field) + d2z($field))
div(u, v, w) = :(dx($u) + dy($v) + dz($w))
grad(field) = :((dx($field), dy($field), dz($field)))

cross(u::Tuple, v::Tuple) = begin
    u1, u2, u3 = u
    v1, v2, v3 = v
    (:(($u2)*($v3) - ($u3)*($v2)), :(($u3)*($v1) - ($u1)*($v3)), :(($u1)*($v2) - ($u2)*($v1)))
end

curl(u, v, w) = begin
    (:(dy($w) - dz($v)), :(dz($u) - dx($w)), :(dx($v) - dy($u)))
end

struct BoundaryLocation
    name::Symbol
end

const left = BoundaryLocation(:left)
const right = BoundaryLocation(:right)
const bottom = BoundaryLocation(:bottom)
const top = BoundaryLocation(:top)
const front = BoundaryLocation(:front)
const back = BoundaryLocation(:back)
