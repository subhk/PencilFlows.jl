# PencilFlows.jl Demonstrations

This folder contains comprehensive demonstrations of PencilFlows.jl capabilities.

##  **Demo Files**

### `demo_integrated_system.jl`
**Complete Integrated Automatic Parameter Conversion Demo**

Shows how automatic parameter conversion is fully integrated across ALL PencilFlows.jl solver components:

- **Demo 1**: Reynolds number integration across all solvers
- **Demo 2**: Ekman number integration in rotating flows  
- **Demo 3**: Complex multi-parameter physics integration
- **Demo 4**: Performance impact comparison (before vs after)
- **Demo 5**: Real research applications
- **Demo 6**: Code comparison (traditional vs automatic)

**Key Features Demonstrated:**
- Automatic equation analysis and parameter detection
- Seamless integration with pressure solver, nonlinear terms, and predictor-corrector
- Real-world application examples (ocean, atmosphere, laboratory)
- Productivity improvements (50+ lines †’ 3 lines)

##  **Running Demos**

```bash
# Run the complete integrated system demo
julia demo_integrated_system.jl
```

**Expected Output:**
- Detailed walkthrough of automatic parameter conversion
- Examples across different physics regimes
- Performance comparisons and benefits
- Real research application scenarios

##  **What You'll Learn**

1. **How Automatic Conversion Works**: See step-by-step parameter detection and conversion
2. **Solver Integration**: Understand how converted parameters flow through all components  
3. **User Experience**: Compare traditional manual setup vs automatic system
4. **Research Applications**: See real examples from ocean/atmospheric/laboratory settings
5. **Performance Benefits**: Quantified improvements in setup time and error reduction

##  **Demo Highlights**

- **Reynolds Example**: Re=1000 †’ Î½=0.001 automatic conversion and solver setup
- **Ekman Example**: Rotating flows with geophysical parameters 
- **Thermal Example**: Multi-parameter convection (Re, Ra, Pr simultaneously)
- **Before/After**: 20+ line manual setup vs 3-line automatic solution

##  **Key Takeaway**

These demos show how PencilFlows.jl transforms computational fluid dynamics from:
- **Manual, error-prone parameter setup** †’ **Automatic, guaranteed-correct conversion**
- **Hours of configuration** †’ **Seconds of setup**  
- **50+ lines of boilerplate** †’ **3 lines of physics**

Perfect for researchers, students, and engineers who want to focus on physics rather than numerical setup!
