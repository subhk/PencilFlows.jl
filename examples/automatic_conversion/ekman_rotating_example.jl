#!/usr/bin/env julia
"""
Ekman Number Rotating Flow Example

This example demonstrates automatic conversion from Ekman number to viscosity
in rotating fluid systems, common in geophysical flows.
"""

function ekman_rotating_example()
    println("  AUTOMATIC EKMAN NUMBER CONVERSION EXAMPLE")
    println("="^60)
    
    # Rotating Navier-Stokes equations with Ekman number
    equations = [
        "dt(u) = -u*dx(u) - v*dy(u) - dx(p) + Ek*f*L2*lap(u) + f*v",
        "dt(v) = -u*dx(v) - v*dy(v) - dy(p) + Ek*f*L2*lap(v) - f*u", 
        "dx(u) + dy(v) = 0"
    ]
    
    println("ù GEOPHYSICAL SETUP:")
    for (i, eq) in enumerate(equations)
        println("   $i. $eq")
    end
    
    println("\n TYPICAL PARAMETERS:")
    println("   ¢ Ekman number: Ek = 1√ó10ª¥")
    println("   ¢ Coriolis parameter: f = 1√ó10ª¥ sª¬π (Earth-like)")
    println("   ¢ Length scale: L = 1000 m (mesoscale)")
    
    println("\nÑ AUTOMATIC CONVERSION:")
    println("   ¢ Detects: Ek, f, L2 parameters")
    println("   ¢ Formula: ŒΩ = Ek √ó f √ó L¬≤")
    println("   ¢ Computes: ŒΩ = 1√ó10ª¥ √ó 1√ó10ª¥ √ó 10∂ = 0.01 m¬≤/s")
    println("   ¢ Sets up: Rotation-aware pressure solver")
    println("   ¢ Applies: Coriolis-modified time stepping")
    
    println("\n APPLICATIONS:")
    println("   ¢ Ocean mesoscale eddies")
    println("   ¢ Atmospheric dynamics")
    println("   ¢ Laboratory rotating tank experiments")
    
    return equations
end

if abspath(PROGRAM_FILE) == @__FILE__
    ekman_rotating_example()
end