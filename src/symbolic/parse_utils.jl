# Shared parsing utilities

function contains_time_variable(expr)
    if isa(expr, Symbol)
        return expr == :t
    elseif isa(expr, Expr)
        return any(contains_time_variable(arg) for arg in expr.args)
    else
        return false
    end
end

function substitute_time_variable(expr, t_val)
    if isa(expr, Symbol)
        return expr == :t ? t_val : expr
    elseif isa(expr, Expr)
        new_args = [substitute_time_variable(arg, t_val) for arg in expr.args]
        return Expr(expr.head, new_args...)
    else
        return expr
    end
end

# Term classification for IMEX timestepping
function parse_equation_terms(equation::Expr, parameters::Dict{Symbol, Float64})
    """
    Parse equation into linear terms (for implicit treatment) and nonlinear terms (for explicit treatment).
    
    Linear terms include:
    - Laplacian: lap(u), nu*lap(u)
    - Gradients: dx(p), dy(p), dz(p)
    - Coriolis: f*v, -f*u
    - Viscous/diffusion: nu*dz^2(u), kappa*lap(b)
    
    Nonlinear terms include:
    - Advection: u*dx(u), v*dy(u), w*dz(u)
    - Nonlinear buoyancy coupling
    - Products of fields
    """
    linear_terms = Dict{Symbol, Vector{Any}}()
    nonlinear_terms = Dict{Symbol, Vector{Any}}()
    
    # Extract the equation structure (assumes dt(field) - RHS = 0 form)
    if equation.head == :call && equation.args[1] == :(==)
        lhs = equation.args[2]
        rhs = equation.args[3]
        
        # Extract terms from LHS and RHS
        lhs_terms = extract_terms(lhs)
        rhs_terms = extract_terms(rhs)
        
        # Classify each term as linear or nonlinear
        for term in vcat(lhs_terms, rhs_terms)
            field_name = extract_primary_field(term)
            if field_name !== nothing
                if is_linear_term(term, parameters)
                    if !haskey(linear_terms, field_name)
                        linear_terms[field_name] = []
                    end
                    push!(linear_terms[field_name], term)
                else
                    if !haskey(nonlinear_terms, field_name)
                        nonlinear_terms[field_name] = []
                    end
                    push!(nonlinear_terms[field_name], term)
                end
            end
        end
    end
    
    return linear_terms, nonlinear_terms
end

function extract_field_from_equation(equation::Expr)
    """Extract the primary field being evolved from dt(field) term."""
    if equation.head == :call && equation.args[1] == :(==)
        lhs = equation.args[2]
        if isa(lhs, Expr) && lhs.head == :call
            if length(lhs.args) >= 2 && lhs.args[1] == :-
                # Handle dt(field) - terms = 0 structure
                for arg in lhs.args[2:end]
                    if isa(arg, Expr) && arg.head == :call && arg.args[1] == :dt
                        return arg.args[2]
                    end
                end
            elseif lhs.args[1] == :dt && length(lhs.args) >= 2
                # Handle direct dt(field) = RHS structure
                return lhs.args[2]
            end
        end
    end
    return nothing
end

function extract_terms(expr)
    """Extract individual terms from an expression."""
    terms = []
    if isa(expr, Expr) && expr.head == :call
        if expr.args[1] == :+ || expr.args[1] == :-
            # Sum/difference of terms
            for arg in expr.args[2:end]
                append!(terms, extract_terms(arg))
            end
        else
            # Single term
            push!(terms, expr)
        end
    else
        push!(terms, expr)
    end
    return terms
end

function extract_primary_field(term)
    """Extract the primary field name from a term."""
    if isa(term, Symbol)
        return term
    elseif isa(term, Expr) && term.head == :call
        if term.args[1] == :dt && length(term.args) >= 2
            return term.args[2]
        elseif term.args[1] in [:dx, :dy, :dz, :lap] && length(term.args) >= 2
            return extract_primary_field(term.args[2])
        elseif term.args[1] == :* && length(term.args) >= 3
            # For products, return the field being operated on (not parameters)
            for arg in term.args[2:end]
                field = extract_primary_field(arg)
                if field !== nothing && field ∉ [:nu, :kappa, :f, :Ra, :Pr]
                    return field
                end
            end
        end
        # Check all arguments for field names
        for arg in term.args[2:end]
            if isa(arg, Symbol) && arg ∉ [:nu, :kappa, :kappa_T, :kappa_s, :f, :Ra, :Pr, :beta_T, :beta_s, :x, :y, :z, :t]
                return arg
            end
        end
    end
    return nothing
end

function is_linear_term(term, parameters::Dict{Symbol, Float64})
    """
    Determine if a term should be treated as linear (implicit) or nonlinear (explicit).
    
    Linear terms (implicit):
    - Viscous diffusion: nu*lap(u), nu*d2z(u)
    - Thermal diffusion: kappa*lap(b)
    - Coriolis: f*v, -f*u
    - Pressure gradient: dx(p), dy(p), dz(p)
    - Buoyancy: b (in w-momentum equation)
    
    Nonlinear terms (explicit):
    - Advection: u*dx(u), v*dy(u), w*dz(u)
    - Any product of evolved fields
    """
    if isa(term, Expr) && term.head == :call
        op = term.args[1]
        
        # Diffusion terms (always linear)
        if op == :lap
            return true
        elseif op == :* && length(term.args) >= 3
            # Check for diffusion patterns: nu*lap(u), kappa*d2z(b), etc.
            has_diffusion_param = false
            has_diffusion_op = false
            has_field_product = false
            field_count = 0
            
            for arg in term.args[2:end]
                if isa(arg, Symbol)
                    if arg in [:nu, :kappa, :alpha, :kappa_T, :kappa_s] # diffusion parameters
                        has_diffusion_param = true
                    elseif arg ∉ [:x, :y, :z, :t, :f, :Ra, :Pr, :nu, :kappa, :alpha, :kappa_s, :kappa_T] # field variable
                        field_count += 1
                    end
                elseif isa(arg, Expr) && arg.head == :call
                    if arg.args[1] in [:lap, :d2x, :d2y, :d2z] # diffusion operators
                        has_diffusion_op = true
                    elseif arg.args[1] in [:dx, :dy, :dz] # gradient operators
                        # For gradients, don't automatically assume diffusion
                        # Only if there's a diffusion parameter present
                        if has_diffusion_param
                            has_diffusion_op = true
                        end
                    end
                end
            end
            
            # Check for advection patterns: u*dx(something), v*dy(something), etc.
            has_advection_pattern = false
            velocity_fields = [:u, :v, :w]
            
            # Look for velocity field multiplied by gradient
            has_velocity = false
            has_gradient = false
            
            for arg in term.args[2:end]
                if isa(arg, Symbol) && arg in velocity_fields
                    has_velocity = true
                elseif isa(arg, Expr) && arg.head == :call && arg.args[1] in [:dx, :dy, :dz]
                    has_gradient = true
                end
            end
            
            # If we have both velocity and gradient, it's advection (nonlinear)
            if has_velocity && has_gradient
                has_advection_pattern = true
            end
            
            # Multiple field variables or advection pattern indicates nonlinear
            if field_count >= 2 || has_advection_pattern
                has_field_product = true
            end
            
            # Coriolis terms: f*u, f*v (linear)
            if any(arg == :f for arg in term.args[2:end]) && field_count == 1
                return true
            end
            
            # Diffusion terms: nu*lap(u), kappa*d2z(b) (linear)
            if has_diffusion_param && has_diffusion_op && !has_field_product
                return true
            end
            
            # Advection terms: u*dx(v) (nonlinear)
            if has_field_product || has_advection_pattern
                return false
            end
            
            # Default for multiplication: check if it involves field products
            return !has_field_product
            
        # Direct gradient/laplacian operators (usually linear, but context matters)
        elseif op in [:dx, :dy, :dz]
            # Context-dependent: pressure gradients are linear, but gradients in advection are nonlinear
            if length(term.args) >= 2
                field = extract_primary_field(term.args[2])
                if field in [:p, :P, :pressure]
                    return true  # pressure gradients are linear
                else
                    return false # other gradients are typically nonlinear (advection context)
                end
            end
            return false
            
        # Pure field variables
        elseif op in [:u, :v, :w, :b, :p, :T, :theta, :s, :S]
            return true # buoyancy terms b, temperature T, salinity s are typically linear
        end
    elseif isa(term, Symbol)
        # Pure field variables (like buoyancy b in w-momentum)
        if term in [:b, :T, :theta, :p, :s, :S]
            return true
        end
        return false
    end
    
    # Default: treat unknown terms as nonlinear for safety
    return false
end

function setup_tensor_helper_mapping!(prob::SymbolicProblem)
    """Set up mappings to PencilFlow tensor helpers and nonlinear workspace functions."""
    disc = prob.discretization
    
    # Map linear operators to PencilFlow functions
    linear_map = Dict(
        :laplacian => "apply_laplacian_pencil",
        :diffusion => "apply_diffusion_pencil", 
        :coriolis => "apply_coriolis_pencil",
        :pressure_gradient => "apply_pressure_gradient_pencil"
    )
    
    # Map nonlinear operators to tensor helper functions
    nonlinear_map = Dict(
        :advection => "compute_advection_terms_pencil",
        :nonlinear_buoyancy => "compute_buoyancy_coupling_pencil"
    )
    
    # Store mappings for later use
    disc.linear_operators[:_function_map] = linear_map
    disc.nonlinear_functions[:_function_map] = nonlinear_map
    
    println("    Tensor helper mappings configured for PencilFlow integration")
end

