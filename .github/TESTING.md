# Testing Guide for PencilFlows.jl

This guide explains how to run tests locally and in CI.

## Quick Start

### Run all tests locally
```bash
cd PencilFlows.jl
julia --project=. runtests.jl
```

### Run specific test
```bash
julia --project=. tests/test_smoke.jl
```

### Run with MPI (parallel)
```bash
mpiexec -n 2 julia --project=. runtests.jl
```

## GitHub Actions CI

Tests automatically run on:
- Push to `main`, `master`, or `develop` branches
- Pull requests to these branches

### CI Test Matrix

The CI runs tests with:
- **Julia versions**: 1.10, 1.11
- **Operating systems**: Ubuntu (Linux)
- **Configurations**: Serial and parallel (MPI)

### CI Workflow Structure

1. **Main Test Job** (`test`)
   - Installs system dependencies (OpenMPI)
   - Sets up Julia environment
   - Runs full test suite
   - Tests MPI functionality

2. **Serial Test Job** (`test-serial`)
   - Tests without MPI dependencies
   - Runs lightweight smoke tests

3. **Linting Job** (`lint`)
   - Checks package loading
   - Verifies import consistency
   - Validates code structure

## Test Categories

### 1. Smoke Tests (`test_smoke.jl`)
**Purpose**: Quick sanity checks
**Runtime**: < 10 seconds
**When to run**: Before committing

```bash
julia --project=. tests/test_smoke.jl
```

### 2. Consistency Tests (`test_pencil_consistency.jl`, `test_import_order.jl`)
**Purpose**: Verify API consistency and code quality
**Runtime**: < 30 seconds
**When to run**: After refactoring

```bash
julia --project=. tests/test_import_order.jl
julia --project=. tests/test_pencil_consistency.jl
```

### 3. Component Tests
**Purpose**: Test individual components
**Runtime**: 1-5 minutes each
**When to run**: After modifying specific components

```bash
julia --project=. tests/test_boundary_conditions.jl
julia --project=. tests/test_spatial_fields.jl
julia --project=. tests/test_predictor_corrector.jl
```

### 4. Integration Tests (`test_integration.jl`)
**Purpose**: Test components working together
**Runtime**: 5-10 minutes
**When to run**: Before merging PRs

```bash
julia --project=. tests/test_integration.jl
```

## Local Development Workflow

### Before Committing
1. Run smoke tests:
   ```bash
   julia --project=. tests/test_smoke.jl
   ```

2. Run consistency checks:
   ```bash
   julia --project=. tests/test_import_order.jl
   ```

3. If you modified specific components, run relevant tests:
   ```bash
   julia --project=. tests/test_<component>.jl
   ```

### Before Creating PR
1. Run full test suite:
   ```bash
   julia --project=. runtests.jl
   ```

2. Test with MPI (if possible):
   ```bash
   mpiexec -n 2 julia --project=. runtests.jl
   ```

3. Check that GitHub Actions passes on your branch

## Debugging Failed Tests

### Test fails locally but not in CI
- Check Julia version (CI uses 1.10 and 1.11)
- Check MPI configuration
- Verify all dependencies are up to date

### Test fails in CI but not locally
- Check the GitHub Actions logs for detailed error messages
- Look for environment differences (OS, MPI implementation)
- Verify the test works on Ubuntu Linux

### MPI-related failures
- Ensure MPI is properly initialized
- Check process counts match expectations
- Verify data decomposition is correct

## Adding New Tests

### 1. Create test file
```julia
# tests/test_mycomponent.jl
"""
Test suite for MyComponent
"""

println("Testing: MyComponent")

using Test

@testset "MyComponent Tests" begin
    # Your tests here
    @test true
end

println("✓ MyComponent tests passed!")
```

### 2. Verify it runs
```bash
julia --project=. tests/test_mycomponent.jl
```

### 3. Test will auto-run in CI
The `runtests.jl` automatically discovers and runs all `test_*.jl` files.

## Environment Setup

### Required packages
- MPI (with system MPI library)
- FFTW
- PencilFFTs
- PencilArrays
- LinearAlgebra
- Test

### Installing on Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y libopenmpi-dev openmpi-bin
```

### Installing on macOS
```bash
brew install open-mpi
```

### Julia environment
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Troubleshooting

### "MPI not initialized" error
```julia
using MPI
MPI.Init()
```

### "PencilArrays" or "PencilFFTs" not found
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Test timeout in CI
- Check if test is too slow
- Consider splitting into smaller tests
- Verify no infinite loops

### Import order errors
Run the consistency check:
```bash
julia --project=. tests/test_import_order.jl
```

Fix by ensuring `using PencilFFTs` comes before `using PencilArrays` in all files.

## Contact

For test-related issues:
1. Check existing GitHub issues
2. Create a new issue with:
   - Test file name
   - Error message
   - Julia version
   - MPI configuration (if relevant)
