# Aggregated Symbolic Interface (modular)
# This file re-exports the symbolic interface by including focused submodules.

using Symbolics, LinearAlgebra
using PencilFFTs
using PencilArrays
using Printf, Dates

# Public API exports (kept stable)
export SymbolicProblem, add_equation!, add_bc!, build_problem!, solve!
export add_parameter!, parse_boundary_condition_string, parse_symbolic_expression
export Coordinate, Basis, Domain, Field, Parameter
export Fourier, FiniteDifference
export set_grid!, set_domain!, get_basis, get_grid_points
export dx, dy, dz, dt, lap, div, grad, cross, curl
export left, right, bottom, top, front, back
export add_time_dependent_bc!, t
export pencilflow_header, pencilflow_banner, equation_summary, domain_summary, show_build_progress, reset_banner!
export enforce_imex_separation!, parse_equation_terms, is_linear_term

# New Improved IVP Parser exports
export ImprovedIVPParser, set_domain!, add_variable!, set_parameter!
export parse_equation!, add_boundary_condition!, validate_problem
export get_summary, build_system_matrices, suggest_equation_restructure

# Split implementation into submodules for maintainability
include("types.jl")
include("parse_utils.jl")
# parse_equation.jl removed - function available in parse.jl
include("parse_bc.jl")
include("validate.jl")
include("bc.jl")
include("printing.jl")
include("init.jl")
include("build.jl")
include("interface_functions.jl")
