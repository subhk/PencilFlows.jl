#!/usr/bin/env julia
"""
Simple Reynolds Number Conversion Example

This example shows how PencilFlows.jl automatically converts Reynolds numbers
to viscosity values and sets up the complete solver system.
"""

# Load PencilFlows (in practice you would: using PencilFlows)
include("../../src/PencilFlows.jl")
using .PencilFlows

function reynolds_example()
    println(" AUTOMATIC REYNOLDS NUMBER CONVERSION EXAMPLE")
    println("="^60)
    
    # Define Navier-Stokes equations with Reynolds number
    equations = [
        "dt(u) = -u*dx(u) - v*dy(u) - dx(p) + (1/Re)*lap(u)",
        "dt(v) = -u*dx(v) - v*dy(v) - dy(p) + (1/Re)*lap(v)",
        "dx(u) + dy(v) = 0"
    ]
    
