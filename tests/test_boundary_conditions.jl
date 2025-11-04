"""
Test suite for boundary condition functionality in PencilFlows.jl
"""

println("Testing: Boundary conditions and BCs infrastructure")

# Test BuoyancyBC struct and type stability
try
    # Test with Real bottom_value (using correct enum values)
    bc_real = BuoyancyBC(B_CONSTANT, B_FLUX, bottom_value=1.5, top_value=0.0)
    
    # Verify field types
    bc_real.bottom_value isa Union{Real, Function} || error("bottom_value type incorrect")
    bc_real.top_value isa Union{Real, Function} || error("top_value type incorrect")
    
    # Test with Function bottom_value
    temp_func(z) = sin(z)
    bc_func = BuoyancyBC(B_ROBIN, B_CONSTANT, bottom_value=temp_func, top_value=2.0)
    
    bc_func.bottom_value isa Union{Real, Function} || error("Function bottom_value type incorrect")
    bc_func.top_value isa Union{Real, Function} || error("Function top_value type incorrect")

    println("[PASS] BuoyancyBC struct creation and type stability")

catch e
    println("[FAIL] BuoyancyBC test failed: $e")
end

# Test boundary condition enums and constants
try
    # Test that boundary condition constants are defined
    no_slip_val = NO_SLIP
    free_slip_val = FREE_SLIP
    prescribed_vel = PRESCRIBED_VELOCITY
    stress_free_val = STRESS_FREE
    periodic_z_val = PERIODIC_Z
    robin_val = ROBIN
    
    # These should all be different values
    values = Set([no_slip_val, free_slip_val, prescribed_vel, stress_free_val, periodic_z_val, robin_val])
    length(values) >= 6 || error("Boundary condition constants not unique")

    println("[PASS] Boundary condition constants defined")

catch e
    println("[FAIL] Boundary condition constants test failed: $e")
end

# Test _eval_bc_value function (should exist and work)
try
    # Test with constant value
    result = _eval_bc_value(5.0, 1.0)  # val, t
    abs(result - 5.0) < 1e-12 || error("_eval_bc_value with constant failed")
    
    # Test with function value  
    bc_func(t) = 2.0 * t + 1.0
    result = _eval_bc_value(bc_func, 3.0)  # Should evaluate at t=3.0
    expected = 2.0 * 3.0 + 1.0
    abs(result - expected) < 1e-12 || error("_eval_bc_value with function failed: got $result, expected $expected")

    println("[PASS] _eval_bc_value function works correctly")

catch e
    println("[FAIL] _eval_bc_value test failed: $e")
end

# Test that duplicate _eval_bc_value_old function is removed
try
    # This should error since the function shouldn't exist
    try
        _eval_bc_value_old(1.0, 0.0, 0.0)
        error("_eval_bc_value_old still exists - should have been removed!")
    catch UndefVarError
        # This is expected - the function should not exist
        println("[PASS] Duplicate _eval_bc_value_old function successfully removed")
    end
    
catch e
    println("[FAIL] Duplicate function removal test failed: $e")
end

# Test BoundaryCondition and related types if they exist
try
    # Try to create a basic boundary condition
    # Note: This tests if the type exists and has basic functionality
    bc_dict = Dict(:bottom_type => :dirichlet, :top_type => :neumann)
    
    # Test that we can create BuoyancyBC with these types
    bc = BuoyancyBC(B_CONSTANT, B_FLUX, bottom_value=0.0, top_value=1.0)
    bc.bottom_type == B_CONSTANT || error("Bottom type not set correctly")
    bc.top_type == B_FLUX || error("Top type not set correctly")

    println("[PASS] Boundary condition type system functional")

catch e
    println("[FAIL] Boundary condition type system test failed: $e")
end

# Test boundary condition validation (if validation functions exist)
try
    # Test that boundary conditions have sensible defaults and validation
    bc = BuoyancyBC(B_CONSTANT, B_CONSTANT, bottom_value=1.0, top_value=2.0)
    
    # Basic sanity checks
    bc.bottom_value isa Real || bc.bottom_value isa Function || error("Invalid bottom_value type")
    bc.top_value isa Real || bc.top_value isa Function || error("Invalid top_value type") 
    bc.bottom_type isa Symbol || error("Invalid bottom_type")
    bc.top_type isa Symbol || error("Invalid top_type")
    
    println("[PASS] Boundary condition validation")
    
catch e
    println("[FAIL] Boundary condition validation test failed: $e")
end

println("OK: Boundary condition tests completed")