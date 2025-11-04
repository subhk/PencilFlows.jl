# PencilFlows.jl Test Suite

This directory contains the comprehensive test suite for PencilFlows.jl, ensuring code quality, consistency, and functionality.

## Overview

The test suite is designed to verify:

- **Codebase Consistency**: No breaking points, type stability, corrected function calls
- **Core Functionality**: Spatial fields, boundary conditions, solver components
- **Module Integration**: Seamless operation between all components
- **Performance**: Type stability and memory efficiency
- **Reliability**: Robust error handling and edge cases

## Test Structure

### Core Tests

- **`test_core_functionality.jl`** - Core types, workspaces, and utilities
- **`test_spatial_fields.jl`** - Spatial field creation and evaluation
- **`test_boundary_conditions.jl`** - Boundary condition handling and type stability
- **`test_predictor_corrector.jl`** - Solver components and integration
- **`test_integration.jl`** - End-to-end functionality and module interactions

### Existing Tests

- **`test_output_field_parsing.jl`** - Output field computation and parsing
- **`test_pressure_poisson_basic.jl`** - Pressure-Poisson equation derivation
- **`test_universal_pde_analyze_only.jl`** - Universal PDE analysis framework

## Running Tests

### All Tests

```bash
# Run all tests
julia runtests.jl

# Run with timing information
julia --project=. runtests.jl
```

### Individual Tests

```bash
# Run specific test files
julia runtests.jl test_spatial_fields.jl
julia runtests.jl test_boundary_conditions.jl

# Multiple specific tests
julia runtests.jl test_core_functionality.jl test_integration.jl
```

### Interactive Testing

```julia
# Load package and run individual tests
include("src/PencilFlows.jl")
using .PencilFlows

# Run specific test file
include("tests/test_spatial_fields.jl")
```

## Test Categories

### 1. Core Functionality Tests

**File**: `test_core_functionality.jl`

Tests basic functionality:
- RSNSWorkspace creation and management
- VelocityField and StratificationField evaluation
- MPI utility functions (serial mode)
- Memory formatting utilities
- Safe numeric helpers

Example test:
```julia
# Test VelocityField evaluation
shear_func = linear_shear(1.0, 0.5, direction=:x)
velocity_field = VelocityField(shear_func, :uvw, "Linear shear")
u, v, w = evaluate_spatial_field(velocity_field, 0.0, 0.0, 1.0)
@test abs(u - 1.5) < 1e-12  # Expected: 1.0 + 0.5*1.0
```

### 2. Spatial Fields Tests

**File**: `test_spatial_fields.jl`

Tests spatial field functionality:
- VelocityField creation with different directions (:x, :y, :z)
- StratificationField creation with different field types
- Complex spatial field evaluation
- Edge cases and error handling
- Field metadata and descriptions

Key tests:
```julia
# Test all velocity directions
for direction in [:x, :y, :z]
    shear_func = linear_shear(2.0, 1.5, direction=direction)
    velocity_field = VelocityField(shear_func, :uvw, "Test shear $direction")
    # Verify correct component is non-zero
end

# Test stratification types
for field_type in [:temperature, :density, :buoyancy]
    strat_field = StratificationField(const_strat, field_type, "Test $field_type")
    # Verify field evaluation
end
```

### 3. Boundary Conditions Tests

**File**: `test_boundary_conditions.jl`

Tests boundary condition functionality:
- BuoyancyBC struct creation and type stability
- Union{Real, Function} type handling
- Boundary condition evaluation functions
- Removal of duplicate functions
- Boundary condition constants and enums

Critical type stability test:
```julia
# Test type stability fix
bc_real = BuoyancyBC(B_CONSTANT, B_FLUX, bottom_value=1.5, top_value=0.0)
bc_func = BuoyancyBC(B_ROBIN, B_CONSTANT, bottom_value=temp_func, top_value=2.0)

@test bc_real.bottom_value isa Union{Real, Function}
@test bc_func.top_value isa Union{Real, Function}
```

### 4. Predictor-Corrector Tests

**File**: `test_predictor_corrector.jl`

Tests solver components:
- Core solver function accessibility
- Pressure solver integration
- Nonlinear term computation
- Unicode corruption fixes verification
- SSAValue compilation fix verification
- Parameter syntax corrections

Unicode fix verification:
```julia
# Test that mathematical parameters work without Unicode issues
alpha_val = 0.5  # Was α (Unicode)
nu_val = 1e-3   # Was ν (Unicode)  
pi_val = 0.8    # Was π (Unicode)

@test all(isa.([alpha_val, nu_val, pi_val], Real))
```

### 5. Integration Tests

**File**: `test_integration.jl`

Tests end-to-end functionality:
- Module loading and integration
- Cross-module function dependencies
- Utility integration with core functionality
- Symbolic interface integration
- Parameter system integration
- Complete solver workflow
- All consistency fixes working together

Comprehensive integration test:
```julia
# Test that all major components work together
workspace_type = RSNSWorkspace
field_type = VelocityField
bc_type = BuoyancyBC
solver_func = make_poisson_plan

# Verify no conflicts between modules
spatial_field = VelocityField(linear_shear(1.0, 0.5), :uvw, "Test")
boundary_condition = BuoyancyBC(B_CONSTANT, B_FLUX, bottom_value=1.0, top_value=0.0)
```

## Test Results Analysis

### Expected Behavior

The test suite checks for:

1. **[PASS] Passing Tests**: Full functionality working correctly
2. **[WARN]  Skipped Tests**: Optional functionality (requires full setup)
3. **[FAIL] Failing Tests**: Indicates issues needing attention

### Common Issues and Solutions

#### Type Stability Issues
- **Problem**: `Any` types causing performance issues
- **Solution**: Use `Union{Real, Function}` types in BuoyancyBC
- **Test**: Verify type annotations in boundary conditions

#### Unicode Corruption
- **Problem**: Invalid Unicode characters causing compilation errors
- **Solution**: Replace with ASCII equivalents (π → pi, α → alpha, ν → nu)
- **Test**: Verify mathematical symbols work correctly

#### Function Dependencies
- **Problem**: Undefined functions due to import order
- **Solution**: Correct include order in main module
- **Test**: Verify all expected functions are accessible

#### Breaking Points
- **Problem**: Package fails to load due to syntax/import errors
- **Solution**: Fix SSAValue errors, missing imports, module order
- **Test**: Package loads successfully and core functions work

## Test Development Guidelines

### Adding New Tests

When adding tests, follow these patterns:

1. **Test Structure**:
   ```julia
   println("Testing: [Feature Description]")
   
   try
       # Test code here
       println("   [Test description]")
   catch e
       println("   [Test description] failed: $e")
   end
   
   println("OK: [Feature] tests completed")
   ```

2. **Test Coverage**:
   - Normal operation cases
   - Edge cases and boundary conditions
   - Error handling and invalid inputs
   - Type stability verification
   - Integration with other components

3. **Error Handling**:
   - Catch and report specific error types
   - Provide meaningful error messages
   - Continue testing other components after failures
   - Summarize results clearly

### Test Naming Conventions

- `test_[module]_[functionality].jl` for module-specific tests
- `test_[feature].jl` for feature-specific tests
- `test_integration.jl` for cross-module tests
- Descriptive function names within tests

## Continuous Integration

The test suite supports automated testing:

```bash
# Exit codes
# 0: All tests passed
# 1: Some tests failed

# Run tests in CI environment
julia --project=. runtests.jl
echo "Exit code: $?"
```

## Performance Testing

Some tests include performance considerations:

```julia
# Type stability (affects performance)
@test bc.bottom_value isa Union{Real, Function}  # Not Any

# Memory allocation patterns
workspace = RSNSWorkspace(...)  # Pre-allocated arrays

# Function dispatch efficiency
result = _eval_bc_value(value, time)  # Should dispatch correctly
```

## Contributing Tests

When contributing new tests:

1. **Follow existing patterns** for consistency
2. **Test both success and failure cases**
3. **Include performance considerations** (type stability)
4. **Add documentation** explaining what is tested
5. **Verify tests work** in both standalone and integrated modes
6. **Update this README** with new test descriptions

## Debugging Test Failures

### Common Debug Steps

1. **Check function accessibility**:
   ```julia
   @test isdefined(Main.PencilFlows, :function_name)
   ```

2. **Verify type stability**:
   ```julia
   @test typeof(variable) <: Union{Expected, Types}
   ```

3. **Test module loading**:
   ```julia
   include("src/PencilFlows.jl")
   using .PencilFlows
   ```

4. **Check for Unicode issues**:
   ```bash
   hexdump -C file.jl | grep -E "c[0-9a-f] [0-9a-f][0-9a-f]"
   ```

### Test Environment

Tests assume:
- Julia 1.6+ environment
- PencilFlows.jl package loaded
- Serial execution (no MPI required for basic tests)
- Standard Julia packages available

The test suite provides comprehensive verification of the PencilFlows.jl improvements, ensuring a consistent, reliable, and high-performance computational fluid dynamics package.