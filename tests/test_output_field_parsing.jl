println("Output: computed field parsing and basic evaluation")

# Small mock problem with grids for finite-difference fallbacks
prob = Dict{Symbol,Any}(
    :vars => Dict{Symbol,Any}(
        :u => randn(8,8),
        :v => randn(8,8),
    ),
    :dx => 2*pi/8,
    :dy => 2*pi/8,
    :dz => 1.0,     # not used in 2D case but harmless
)

fields_spec = [:u, :v, :ζ => "dx(v) - dy(u)"]
regular, computed = PencilFlows.parse_computed_fields(fields_spec)

println("  regular: ", regular)
println("  computed: ", [c.name for c in computed])

(length(regular) == 3) || error("Expected 3 regular names (u,v,ζ)")
(length(computed) == 1 && computed[1].name == :ζ) || error("Expected exactly one computed field :ζ")

# Try an evaluation; non-error indicates path is wired. Size check is lenient.
val = PencilFlows.evaluate_computed_field(computed[1], prob)
(val === nothing) && error("Computed field evaluation returned nothing")
(size(val,1) == 8 && size(val,2) == 8) || error("Unexpected size from computed field evaluation")

println("OK: computed field parsing and evaluation succeeded")

