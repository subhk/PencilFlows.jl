using Test
using PencilFlows

# Run all test files
test_files = [
    "tests/test_smoke.jl",
    "tests/test_core_functionality.jl",
    "tests/test_boundary_conditions.jl",
    "tests/test_spatial_fields.jl",
    "tests/test_import_order.jl",
]

for test_file in test_files
    if isfile(test_file)
        println("Running $test_file...")
        include(test_file)
    else
        @warn "Test file not found: $test_file"
    end
end
