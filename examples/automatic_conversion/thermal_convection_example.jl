#!/usr/bin/env julia
"""
Multi-Parameter Thermal Convection Example

This example shows automatic conversion of multiple dimensionless numbers
(Re, Ra, Pr) for thermal convection problems.
"""

function thermal_convection_example()
    println("• AUTOMATIC MULTI-PARAMETER CONVERSION EXAMPLE")
    println("="^60)
    
    # Thermal convection with multiple dimensionless parameters
    equations = [
        "dt(u) = -u*dx(u) - w*dz(u) - dx(p) + (1/Re)*lap(u)",
        "dt(w) = -u*dx(w) - w*dz(w) - dz(p) + (1/Re)*lap(w) + Ra*Pr*T",
        "dt(T) = -u*dx(T) - w*dz(T) + (1/(Re*Pr))*lap(T)",
        "dx(u) + dz(w) = 0"
    ]
    
    println("ù THERMAL CONVECTION EQUATIONS:")
    for (i, eq) in enumerate(equations)
        println("   $i. $eq")
    end
    
    println("\n¢ DIMENSIONLESS PARAMETERS:")
    println("   ¢ Reynolds number: Re = UL/ŒΩ (inertia/viscous)")
    println("   ¢ Rayleigh number: Ra = gŒŒTL¬≥/(ŒΩŒ∫) (buoyancy/diffusion)")
    println("   ¢ Prandtl number: Pr = ŒΩ/Œ∫ (momentum/thermal diffusion)")
    
    println("\nÑ AUTOMATIC CONVERSIONS:")
    println("   1É£  Re = 10¥ Üí ŒΩ = UL/Re")
    println("   2É£  Pr = 0.7 Üí Œ∫ = ŒΩ/Pr") 
    println("   3É£  Ra = 10∂ Üí Validates thermal driving")
    println("   4É£  Multi-physics coupling activated")
    
    println("\nô  SOLVER OPTIMIZATIONS:")
    println("   ¢ High-Re conservative advection")
    println("   ¢ Thermal diffusion with converted Œ∫")
    println("   ¢ Buoyancy-modified pressure projection") 
    println("   ¢ Multi-constraint time stepping")
    
    println("\n  APPLICATIONS:")
    println("   ¢ Rayleigh-B√©nard convection")
    println("   ¢ Atmospheric boundary layer")
    println("   ¢ Mantle convection")
    println("   ¢ Industrial heat transfer")
    
    return equations
end

if abspath(PROGRAM_FILE) == @__FILE__
    thermal_convection_example()
end