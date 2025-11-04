############################# aliases.jl (NameAliases) ############################

"""
Alias / canonical naming support for variables with Unicode physics symbols.

* Canonical internal names are **Unicode** (e.g. :ω̂, :ψ̂, :û).
* Users can request fields with ASCII aliases (e.g. :omega_hat, :psi_hat, :u_hat).
* All selection paths normalize through `canonical_symbol`.
"""

# ---------------- Core Alias Map (ASCII or alt -> canonical Unicode) ----------------
const alias_map = Dict{Symbol,Symbol}(
    # Vorticity / streamfunction
    :ω̂ => :ω̂, :omega_hat => :ω̂, :omegaHat => :ω̂,
    :ω  => :ω,  :omega => :ω,
    :ψ̂ => :ψ̂, :psi_hat => :ψ̂, :psiHat => :ψ̂,
    :ψ  => :ψ,  :psi => :ψ,

    # Shallow Water spectral
    :û => :û, :u_hat => :û,
    :v̂ => :v̂, :v_hat => :v̂,
    :ĥ => :ĥ, :h_hat => :ĥ,

    # Shallow Water physical
    :u => :u, :v => :v, :h => :h,

    # Generic fallbacks (optional; map to themselves)
    :state => :state
)

# ---------------- Optional ASCII Export Map (canonical -> preferred ASCII) ---------
const _ASCII_EXPORT = Dict(
    :ω̂ => :omega_hat, :ω => :omega,
    :ψ̂ => :psi_hat,   :ψ => :psi,
    :û => :u_hat,     :v̂ => :v_hat, :ĥ => :h_hat
)

"""
    ascii_export(sym::Symbol) -> Symbol

Return preferred ASCII form for canonical symbol `sym` (or `sym` unchanged if none).
"""
ascii_export(sym::Symbol) = get(_ASCII_EXPORT, sym, sym)

"""
    canonical_symbol(sym::Symbol) -> Symbol

Map `sym` (alias or canonical) to canonical Unicode symbol.
"""
canonical_symbol(sym::Symbol) = get(alias_map, sym, sym)

"""
    list_aliases(canonical::Symbol) -> Vector{Symbol}

Return all alias keys (including the canonical symbol) that map to `canonical`.
"""
function list_aliases(canonical::Symbol)
    [a for (a,c) in alias_map if c === canonical]
end

"""
    add_alias!(alias::Symbol, canonical::Symbol)

Register new alias -> canonical mapping. Errors if alias already mapped differently.
"""
function add_alias!(alias::Symbol, canonical::Symbol)
    if haskey(alias_map, alias) && alias_map[alias] !== canonical
        error("Alias $alias already maps to $(alias_map[alias]); cannot remap to $canonical.")
    end
    alias_map[alias] = canonical
    return canonical
end

# ---------------- Normalization Helpers ------------------------------------

"""
    normalize_field_spec(spec)

    Normalize user field spec (Symbol, String, Vector, Dict, NamedTuple)
    to use canonical symbols. Regex and FieldSpec are left unchanged (handled later).
"""
function normalize_field_spec(spec)
    if spec === :auto || spec === :all
        return spec
    elseif spec isa Symbol
        return canonical_symbol(spec)
    elseif spec isa String
        return canonical_symbol(Symbol(spec))
    elseif spec isa AbstractVector
        return [x isa Symbol ? canonical_symbol(x) : canonical_symbol(Symbol(x)) for x in spec]
    elseif spec isa Dict
        return Dict(canonical_symbol(Symbol(k)) => v for (k,v) in spec)
    elseif spec isa NamedTuple
        newkeys = Tuple(canonical_symbol(Symbol(k)) for k in keys(spec))
        return NamedTuple{newkeys}(values(spec))
    else
        return spec
    end
end


