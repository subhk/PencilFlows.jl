"""
Test suite for spatial field functionality in PencilFlows.jl
"""

println("Testing: Spatial fields and forcing functions")

# Test VelocityField creation with different parameters
try
    # Test different velocity field directions
    for direction in [:x, :y, :z]
        shear_func = linear_shear(2.0, 1.5, direction=direction)
        velocity_field = VelocityField(shear_func, :uvw, "Test shear $direction")
        
        # Test evaluation
        u, v, w = evaluate_spatial_field(velocity_field, 1.0, 2.0, 3.0)
        
        if direction == :x
            expected = (2.0 + 1.5 * 3.0, 0.0, 0.0)
        elseif direction == :y
            expected = (0.0, 2.0 + 1.5 * 3.0, 0.0)
        else  # :z
            expected = (0.0, 0.0, 2.0 + 1.5 * 3.0)
        end
        
        abs(u - expected[1]) < 1e-12 || error("VelocityField $direction u-component incorrect")
        abs(v - expected[2]) < 1e-12 || error("VelocityField $direction v-component incorrect") 
        abs(w - expected[3]) < 1e-12 || error("VelocityField $direction w-component incorrect")
    end
    
    println("   VelocityField creation and evaluation for all directions")
    
catch e
    println("   VelocityField direction test failed: $e")
end

# Test VelocityField component specification
try
    shear_func = linear_shear(1.0, 0.5, direction=:x)
    
    # Test different component specifications
    for comp in [:u, :v, :w, :uv, :uw, :vw, :uvw]
        vf = VelocityField(shear_func, comp, "Test field with $comp components")
        vf.components == comp || error("Component specification failed for $comp")
    end
    
    println("   VelocityField component specification")
    
catch e
    println("   VelocityField component test failed: $e")
end

# Test StratificationField creation with different field types
try
    strat_func = constant_stratification(2e-4)
    
    # Test different field types
    for field_type in [:temperature, :density, :buoyancy]
        sf = StratificationField(strat_func, field_type, "Test $field_type field")
        sf.field_type == field_type || error("Field type specification failed for $field_type")
        
        # Test evaluation
        result = evaluate_spatial_field(sf, 0.0, 0.0, 5.0)
        expected = -2e-4 * 5.0
        abs(result - expected) < 1e-12 || error("StratificationField evaluation failed for $field_type")
    end
    
    println("   StratificationField creation and field type specification")
    
catch e
    println("   StratificationField field type test failed: $e")
end

# Test spatial field evaluation with different coordinates
try
    # Create a more complex velocity field
    complex_func(x, y, z) = (sin(x) + cos(z), y*z, x*y + z^2)
    vf = VelocityField(complex_func, :uvw, "Complex test field")
    
    # Test at different points
    test_points = [(0.0, 0.0, 0.0), (π/2, 1.0, π/4), (π, 2.0, π/2)]
    
    for (x, y, z) in test_points
        u, v, w = evaluate_spatial_field(vf, x, y, z)
        expected_u = sin(x) + cos(z)
        expected_v = y * z
        expected_w = x * y + z^2
        
        abs(u - expected_u) < 1e-12 || error("Complex field u evaluation failed at ($x,$y,$z)")
        abs(v - expected_v) < 1e-12 || error("Complex field v evaluation failed at ($x,$y,$z)")
        abs(w - expected_w) < 1e-12 || error("Complex field w evaluation failed at ($x,$y,$z)")
    end
    
    println("   Complex spatial field evaluation")
    
catch e
    println("   Complex spatial field test failed: $e")
end

# Test constant_stratification function
try
    N2_values = [1e-6, 1e-4, 1e-2, 1.0]
    
    for N2 in N2_values
        strat_func = constant_stratification(N2)
        
        # Test at different z values
        for z in [0.0, 1.0, 10.0, -5.0]
            result = strat_func(0.0, 0.0, z)  # x,y don't matter for constant stratification
            expected = -N2 * z
            abs(result - expected) < 1e-12 || error("constant_stratification failed for N²=$N2, z=$z")
        end
    end
    
    println("   constant_stratification function")
    
catch e
    println("   constant_stratification test failed: $e")
end

# Test linear_shear function edge cases
try
    # Test zero shear
    zero_shear = linear_shear(5.0, 0.0, direction=:x)
    u, v, w = zero_shear(1.0, 2.0, 10.0)
    abs(u - 5.0) < 1e-12 || error("Zero shear u-component incorrect")
    abs(v) < 1e-12 || error("Zero shear v-component should be zero")
    abs(w) < 1e-12 || error("Zero shear w-component should be zero")
    
    # Test negative shear
    neg_shear = linear_shear(2.0, -1.5, direction=:y)
    u, v, w = neg_shear(0.0, 0.0, 4.0)
    abs(u) < 1e-12 || error("Negative shear u-component should be zero")
    abs(v - (2.0 - 1.5 * 4.0)) < 1e-12 || error("Negative shear v-component incorrect")
    abs(w) < 1e-12 || error("Negative shear w-component should be zero")
    
    println("   linear_shear edge cases")
    
catch e
    println("   linear_shear edge cases test failed: $e")
end

# Test invalid direction handling
try
    # This should error for invalid direction
    error_caught = false
    try
        invalid_shear = linear_shear(1.0, 0.5, direction=:invalid)
        # If we reach here, no error was thrown
        error_caught = false
    catch e
        if isa(e, ArgumentError) || isa(e, BoundsError)
            error_caught = true
            println("   Invalid direction properly rejected")
        else
            rethrow(e)
        end
    end
    
    if !error_caught
        error("Should have failed with invalid direction")
    end
    
catch e
    println("   Invalid direction test failed: $e")
end

# Test field descriptions and metadata
try
    desc = "Custom velocity profile for testing"
    vf = VelocityField(linear_shear(1.0, 0.5), :uvw, desc)
    vf.description == desc || error("VelocityField description not stored correctly")
    
    desc2 = "Temperature stratification profile"
    sf = StratificationField(constant_stratification(1e-4), :temperature, desc2)
    sf.description == desc2 || error("StratificationField description not stored correctly")
    
    println("   Field descriptions and metadata")
    
catch e
    println("   Field metadata test failed: $e")
end

println("OK: Spatial fields tests completed")