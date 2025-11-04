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
    
