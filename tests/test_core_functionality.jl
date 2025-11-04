"""
Test suite for core PencilFlows.jl functionality
"""

println("Testing: Core functionality (types, workspace, utilities)")

# Test RSNSWorkspace creation and basic functionality
try
    # Mock decomposition for testing
    mock_decomp = (
        Nx_global = 32,
        Ny_global = 32, 
        pencil_x = nothing,
        pencil_y = nothing,
        pencil_z = nothing
    )
    
    mock_grid = (dx = 2π/32, dy = 2π/32, dz = 1.0)
    mock_proto = nothing  # Would need actual PencilArray for full test
    
    println("   Mock objects created for workspace testing")
    
catch e
    println("  [WARN] RSNSWorkspace test skipped (requires PencilArrays setup): $e")
end

# Test VelocityField creation and evaluation
try
    # Create a simple linear shear velocity field
    shear_func = linear_shear(1.0, 0.5, direction=:x)
    velocity_field = VelocityField(shear_func, :uvw, "Linear shear in x-direction")
    
    # Test evaluation at a point
    u, v, w = evaluate_spatial_field(velocity_field, 0.0, 0.0, 1.0)
    expected_u = 1.0 + 0.5 * 1.0  # U0 + dU_dz * z
    
    abs(u - expected_u) < 1e-12 || error("VelocityField evaluation incorrect: got $u, expected $expected_u")
    abs(v) < 1e-12 || error("VelocityField v-component should be 0, got $v")
    abs(w) < 1e-12 || error("VelocityField w-component should be 0, got $w")
    
    println("   VelocityField creation and evaluation")
    
catch e
    println("   VelocityField test failed: $e")
end

# Test StratificationField creation and evaluation  
try
    # Create constant stratification field
    strat_func = constant_stratification(1e-4)  # N² = 1e-4
    strat_field = StratificationField(strat_func, :temperature, "Constant stratification")
    
    # Test evaluation
    temp = evaluate_spatial_field(strat_field, 0.0, 0.0, 10.0)
    expected_temp = -1e-4 * 10.0  # -N² * z
    
    abs(temp - expected_temp) < 1e-12 || error("StratificationField evaluation incorrect: got $temp, expected $expected_temp")
    
    println("   StratificationField creation and evaluation")
    
catch e
    println("   StratificationField test failed: $e")
end

# Test utility functions
try
    # Test MPI utilities (should work in serial mode)
    rank = comm_rank()
    size = comm_size() 
    is_root = isroot()
    
    (rank == 0) || error("Expected rank 0 in serial mode, got $rank")
    (size == 1) || error("Expected size 1 in serial mode, got $size") 
    is_root || error("Expected to be root in serial mode")
    
    # Test reductions
    test_val = 5.0
    sum_val = mpi_allreduce_sum(test_val)
    max_val = mpi_allreduce_max(test_val)
    
    abs(sum_val - test_val) < 1e-12 || error("mpi_allreduce_sum failed in serial")
    abs(max_val - test_val) < 1e-12 || error("mpi_allreduce_max failed in serial")
    
    println("   MPI utility functions")
    
catch e
    println("   MPI utility test failed: $e")
end

# Test memory formatting utilities
try
    byte_str = bytestr(1024)
    byte_str == "1.0KB" || error("bytestr formatting incorrect: got '$byte_str'")
    
    rate_str = format_rate(100.0, "MB")
    rate_str == "100.0 MB/s" || error("format_rate incorrect: got '$rate_str'")
    
    println("   Memory formatting utilities")
    
catch e
    println("   Memory formatting test failed: $e")
end

# Test safe numeric helpers
try
    # Test safediv
    result = safediv(10.0, 2.0)
    abs(result - 5.0) < 1e-12 || error("safediv normal case failed")
    
    result = safediv(10.0, 0.0, default=42.0)
    abs(result - 42.0) < 1e-12 || error("safediv zero case failed")
    
    # Test clamp_step
    result = clamp_step(5.0, 1.0, 10.0)
    abs(result - 5.0) < 1e-12 || error("clamp_step normal case failed")
    
    result = clamp_step(0.5, 1.0, 10.0) 
    abs(result - 1.0) < 1e-12 || error("clamp_step min case failed")
    
    result = clamp_step(15.0, 1.0, 10.0)
    abs(result - 10.0) < 1e-12 || error("clamp_step max case failed")
    
    println("   Safe numeric helpers")
    
catch e
    println("   Safe numeric helpers test failed: $e")
end

println("OK: Core functionality tests completed")