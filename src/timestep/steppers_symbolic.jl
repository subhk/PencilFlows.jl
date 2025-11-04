# Symbolic steppers (IMEX and simple RK4/Euler) for serial fallback workflows

abstract type AbstractTimeStepper end

# Rename to avoid conflict with core RK4Stepper in timestep.jl
struct SymbolicRK4Stepper{D} <: AbstractTimeStepper
    dt::Float64
    discretization::D
end

struct EulerStepper{D} <: AbstractTimeStepper
    dt::Float64
    discretization::D
end

function time_step!(stepper::SymbolicRK4Stepper, solution, prob, t, dt)
    k1 = compute_rhs(solution, prob, t)
    k2 = compute_rhs(add_solution(solution, k1, dt/2), prob, t + dt/2)
    k3 = compute_rhs(add_solution(solution, k2, dt/2), prob, t + dt/2)
    k4 = compute_rhs(add_solution(solution, k3, dt), prob, t + dt)
    new_solution = Dict{Symbol, Array{Float64, 3}}()
    for (field_name, field_data) in solution
        new_solution[field_name] = field_data + dt/6 * (k1[field_name] + 2*k2[field_name] + 2*k3[field_name] + k4[field_name])
    end
    return new_solution, t + dt
end

function time_step!(stepper::EulerStepper, solution, prob, t, dt)
    rhs = compute_rhs(solution, prob, t)
    new_solution = Dict{Symbol, Array{Float64, 3}}()
    for (field_name, field_data) in solution
        new_solution[field_name] = field_data + dt * rhs[field_name]
    end
    return new_solution, t + dt
end

function compute_rhs(solution::Dict{Symbol,<:AbstractArray}, prob, t::Real)
    linear_rhs = apply_linear_operators(solution, prob, t)
    nonlinear_rhs = apply_nonlinear_operators(solution, prob, t)
    rhs = Dict{Symbol, Array{Float64, 3}}()
    for field_name in keys(solution)
        if haskey(linear_rhs, field_name) && haskey(nonlinear_rhs, field_name)
            rhs[field_name] = linear_rhs[field_name] + nonlinear_rhs[field_name]
        elseif haskey(linear_rhs, field_name)
            rhs[field_name] = linear_rhs[field_name]
        elseif haskey(nonlinear_rhs, field_name)
            rhs[field_name] = nonlinear_rhs[field_name]
        else
            rhs[field_name] = zeros(size(solution[field_name]))
        end
    end
    return rhs
end

function apply_linear_operators(solution::Dict{Symbol,<:AbstractArray}, prob, t::Real)
    disc = prob.discretization
    linear_rhs = Dict{Symbol, Array{Float64, 3}}()
    for (field_name, field_data) in solution
        linear_result = zeros(size(field_data))
        if haskey(disc.linear_operators, field_name)
            operators = disc.linear_operators[field_name]
            if :laplacian in keys(operators)
                linear_result += apply_laplacian(field_data, disc, field_name)
            end
            if :gradient in keys(operators)
                linear_result += apply_gradient_operators(field_data, disc)
            end
        end
        linear_rhs[field_name] = linear_result
    end
    return linear_rhs
end

function apply_nonlinear_operators(solution::Dict{Symbol,<:AbstractArray}, prob, t::Real)
    nonlinear_rhs = Dict{Symbol, Array{Float64, 3}}()
    for (field_name, _) in solution
        nonlinear_result = zeros(size(solution[field_name]))
        if haskey(solution, :u) && haskey(solution, :v) && haskey(solution, :w)
            nonlinear_result += compute_advection_terms(solution, field_name, prob.discretization)
        end
        nonlinear_rhs[field_name] = nonlinear_result
    end
    return nonlinear_rhs
end

function imex_solve(solution, linear_rhs, nonlinear_rhs, dt, prob)
    new_solution = Dict{Symbol, Array{Float64, 3}}()
    for (field_name, field_data) in solution
        new_solution[field_name] = field_data + dt * (
            get(linear_rhs, field_name, zeros(size(field_data))) +
            get(nonlinear_rhs, field_name, zeros(size(field_data))))
    end
    return new_solution
end

function add_solution(solution, increment, factor)
    result = Dict{Symbol, Array{Float64, 3}}()
    for (field_name, field_data) in solution
        if haskey(increment, field_name)
            result[field_name] = field_data + factor * increment[field_name]
        else
            result[field_name] = field_data
        end
    end
    return result
end

function create_time_stepper(scheme::Symbol, dt::Float64, prob)
    if scheme == :RK4
        return SymbolicRK4Stepper(dt, prob.discretization)
    elseif scheme == :Euler
        return EulerStepper(dt, prob.discretization)
    elseif scheme in [:IMEX, :IMEXRK2, :IMEXAB, :IMEXRK443, :IMEXBDF2]
        # Use IMEX steppers that enforce implicit linear / explicit nonlinear
        include("imex_stepper.jl")
        actual_scheme = scheme == :IMEX ? :IMEXRK2 : scheme  # Default to 2nd order RK
        return create_imex_stepper(actual_scheme, dt, prob)
    else
        available = [:RK4, :Euler, :IMEX, :IMEXRK2, :IMEXAB, :IMEXRK443, :IMEXBDF2]
        error("Time stepping scheme $scheme not supported. Available: $available")
    end
end

# ------------------------ Serial IMEX convenience ----------------------------

"""
    imex_step_serial!(solution::Dict, prob, t, dt; theta=1.0)

Array-based semi-implicit step for the serial symbolic path. Treats linear
terms implicitly (Coriolis rotation, vertical viscosity) and nonlinear terms
explicitly. This is a convenience implementation and does not perform pressure
projection in the serial mode.
"""
function imex_step_serial!(solution::Dict{Symbol,<:AbstractArray}, prob, t, dt; theta::Real=1.0)
    disc = prob.discretization
    # Pull fields if present
    has = haskey
    u = has(solution, :u) ? solution[:u] : nothing
    v = has(solution, :v) ? solution[:v] : nothing
    w = has(solution, :w) ? solution[:w] : nothing
    b = has(solution, :b) ? solution[:b] : nothing

    Î½ = get(prob.parameters, :nu, get(prob.parameters, :Î½, 0.0))
    f = get(prob.parameters, :f, get(prob.parameters, :Omega, 0.0))

    # 1) Explicit nonlinear advection and buoyancy coupling
    if u !== nothing
        u .+= dt .* compute_advection_terms(solution, :u, disc)
    end
    if v !== nothing
        v .+= dt .* compute_advection_terms(solution, :v, disc)
    end
    if w !== nothing
        w .+= dt .* compute_advection_terms(solution, :w, disc)
        if b !== nothing
            w .+= dt .* b
        end
    end
    if b !== nothing
        b .+= dt .* compute_advection_terms(solution, :b, disc)
    end

    # 2) Implicit Coriolis (pointwise 2x2 rotation)
    if u !== nothing && v !== nothing && f != 0
        Î = theta * dt * f
        denom = 1 + Î^2
        @inbounds for k in axes(u,3), j in axes(u,2), i in axes(u,1)
            ur = u[i,j,k]; vr = v[i,j,k]
            u[i,j,k] = ( ur + Î*vr) / denom
            v[i,j,k] = (-Î*ur + vr) / denom
        end
    end

    # 3) Implicit vertical viscosity via tridiagonal Helmholtz in z
    if Î½ != 0 && hasfield(typeof(disc), :grid_z) && !isempty(disc.grid_z)
        dz = length(disc.grid_z) > 1 ? diff(disc.grid_z) : [1.0]
        Î = Î½ * dt
        u !== nothing && _helmholtz_z_serial!(u, Î, dz, theta)
        v !== nothing && _helmholtz_z_serial!(v, Î, dz, theta)
        w !== nothing && _helmholtz_z_serial!(w, Î, dz, theta)
        b !== nothing && _helmholtz_z_serial!(b, Î, dz, theta)
    end

    return solution
end

"""
    _helmholtz_z_serial!(A, alpha, dz, theta)

Solve (I - Î Î d2/dz2) A = rhs in-place along vertical columns using a
tridiagonal Thomas algorithm with simple Dirichlet boundary rows.
"""
function _helmholtz_z_serial!(A::AbstractArray, alpha::Real, dz::AbstractVector, theta::Real)
    Nz = size(A,3)
    Nz >= 3 || return A
    a = zeros(eltype(A), Nz)
    b = zeros(eltype(A), Nz)
    c = zeros(eltype(A), Nz)
    col = similar(view(A,1,1,:))
    for k in 2:Nz-1
        hm = dz[k-1]; hp = dz[k]
        L =  2/(hm*(hm+hp))
        D = -2/(hm*hp)
        U =  2/(hp*(hm+hp))
        a[k] = -alpha*theta*L
        b[k] = 1 - alpha*theta*D
        c[k] = -alpha*theta*U
    end
    b[1] = 1; c[1] = 0; a[1] = 0
    a[Nz] = 0; b[Nz] = 1; c[Nz] = 0
    @inbounds for j in axes(A,2), i in axes(A,1)
        # copy column
        for k in 1:Nz; col[k] = A[i,j,k]; end
        # forward sweep
        for k in 2:Nz
            m = a[k]/b[k-1]
            b[k] -= m*c[k-1]
            col[k] -= m*col[k-1]
        end
        # back substitution
        A[i,j,Nz] = col[Nz]/b[Nz]
        for k in Nz-1:-1:1
            A[i,j,k] = (col[k] - c[k]*A[i,j,k+1]) / b[k]
        end
    end
    return A
end
