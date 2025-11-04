module PencilFlows

    # load required packages
    using Printf
    using SpecialFunctions
    using Parameters
    using BenchmarkTools
    using PencilFFTs
    using PencilArrays
    using FFTW
    using LinearAlgebra
    using MPI
    using Dates
    using JLD2
    using NetCDF
    using Statistics
    using Random
    using Symbolics


# Export all public functions and types

    # MPI / Pencil plan utilities (defined in parallel/mpi.jl)
    export make_pencil_plan_rfft, forward_rfft!, backward_rfft!,
        alloc_arrays, alloc_phys, alloc_spec,
        localindices_dim, localrange_dim, localview,
        isparallel, isreal, get_comm,
        eltype_physical, eltype_spectral,
        physical_sizes, spectral_sizes,
        supported_schemes, create_stepper, step!, TimestepProblem, memory_usage
    
    # Selected convenience exports implemented elsewhere
    export apply_boundary_conditions!, compute_field_statistics

    # For boundary conditions
    export BoundaryType, BoundaryCondition,
        NO_SLIP, FREE_SLIP, PRESCRIBED_VELOCITY, STRESS_FREE, PERIODIC_Z, ROBIN,
        compute_bc_coefficients,
        apply_velocity_bcs_nonuniform!,
        dz_derivative_nonuniform_with_bcs!,
        d2z_derivative_nonuniform_with_bcs!,
        validate_bc_grid_compatibility,
        create_boundary_layer_grid,
        BuoyancyBCType, BuoyancyBC, apply_buoyancy_bcs_nonuniform!,
        dz_derivative_buoyancy_with_bcs!

    # For 2D pencil decomposition
    export PencilDecomposition, init_pencil_decomposition
    export create_distributed_fields, compute_horizontal_derivatives_2d!
    export compute_z_derivatives_2d!, demo_2d_decomposition
    export profile_decomposition_performance

    # Export analysis and diagnostics
    export AnalysisHandle, add_system!, add_task!, analysis_step!
    
    # Field analysis utilities from utils.jl
    export compute_kinetic_energy, compute_rms_field, compute_mean_field
    
    # Output and I/O functions from output.jl
    export write_state, add_output_task!, write_output!
    export should_write_output, write_handler_output!
    
    # Transform functions from unified transforms.jl
    export ddx!, ddy!, d2dx2!, d2dy2!
    export create_transform_plans_2d, find_optimal_process_grid, create_transform_fields
    export compute_horizontal_derivatives_fd!, get_horizontal_wavenumbers, get_vertical_grid
    
    # Optimized transform functions
    export OptimizedTransformPlans, create_optimized_transform_plans_2d
    export compute_horizontal_derivatives_optimized!
    # d2dx2_vectorized! and d2dy2_vectorized! exports removed
    export apply_multiple_derivatives!, laplacian_2d!
    export apply_dealias_optimized!, vertical_derivative_optimized!
    export batch_horizontal_fft!, batch_horizontal_ifft!
    
    # Compatibility functions for transform interoperability
    export CompatibleTransformPlans
    export get_wavenumbers, get_plan_local_size, get_working_arrays
    export get_local_range, get_local_size
    export validate_transform_compatibility, get_recommended_plan_type

    # For the Coriolis terms
    export FPlane, coriolis_terms!, coriolis_terms
    
    # Pressure/Poisson solvers
    export PoissonPlan, make_poisson_plan, solve_poisson!, eigenvalues_1d
    export MGPoissonPlan, MGDistPlan, make_mg_poisson, mg_solve!,
           make_mg_poisson_distributed, mg_solve_distributed!,
           auto_mg_levels, make_mg_poisson_distributed_auto


    # Additional exports for MPI, problem management, and I/O
    export PencilPlanRFFT, pencilize_array, ensure_comm, global_sizes,
        Problem, init_problem, advance!, compute_cfl, set_dt!,
        register_callback!, clear_callbacks!,
        canonical_symbol, list_aliases, normalize_field_spec, ascii_export,
        add_alias!, alias_map,
        write_state, gather_field,
        format_time_for_filename



    # Core functionality
    include("core/aliases.jl")
    include("core/utils.jl")
    include("timestep/timestep.jl")
    include("timestep/steppers_symbolic.jl")

    # Symbolic timestepper exports (serial fallback)
    export SymbolicRK4Stepper, EulerStepper
    export create_time_stepper, time_step!
    include("core/problem.jl")
    include("core/tensor_helpers.jl")
    
    # Physics and boundary conditions (needed before predictor_corrector.jl)
    include("physics/nonlinear.jl")
    include("physics/spatial_fields.jl")
    
    include("core/predictor_corrector.jl")
    include("physics/Coriolis.jl")
    include("physics/setBCs.jl")

    # Import and re-export boundary condition types from the submodule
    using .BoundaryConditionsNonUniform: BoundaryType, BoundaryCondition,
        NO_SLIP, FREE_SLIP, PRESCRIBED_VELOCITY, STRESS_FREE, PERIODIC_Z, ROBIN,
        compute_bc_coefficients,
        apply_velocity_bcs_nonuniform!,
        dz_derivative_nonuniform_with_bcs!,
        d2z_derivative_nonuniform_with_bcs!,
        validate_bc_grid_compatibility,
        create_boundary_layer_grid,
        BuoyancyBCType, BuoyancyBC,
        apply_buoyancy_bcs_nonuniform!,
        dz_derivative_buoyancy_with_bcs!

    # Transforms and decomposition (unified from previous transforms.jl and transforms_optimized.jl)
    include("transforms/transforms.jl")
    # include("pencil_compat.jl")  # currently unused in main exports
    include("decomposition/pencil_decomposition_2d.jl")

    # Solvers
    include("solvers/poisson_solver.jl")
    include("solvers/poisson_multigrid.jl")

    # Parallel and I/O
    include("parallel/mpi.jl")
    include("io/output.jl")

    # Interfaces and parameter systems
    # Modular symbolic interface entrypoint
    include("symbolic/main.jl")
    include("parameters/automatic_parameter_detection.jl")
    include("parameters/integrated_parameter_system.jl")
    include("interfaces/smart_interface.jl")
    include("interfaces/simulation_api.jl")
    include("interfaces/pressure_poisson_derivation.jl")

    export RSNSWorkspace, predictor_corrector_step!
    
    # Export spatial field functionality
    export SpatialField, VelocityField, StratificationField
    export apply_spatial_field!, evaluate_spatial_field
    export linear_shear, quadratic_profile, exponential_profile, sinusoidal_field
    export constant_stratification, linear_stratification, exponential_stratification
    export initialize_velocity_field!, initialize_stratification_field!
    export create_channel_flow, create_atmospheric_profile, create_ocean_stratification
    
    # Export symbolic interface functionality (core types and operators only)
    export SymbolicProblem, build_problem!, solve!
    export Coordinate, Basis, Domain, Field, Parameter
    export Fourier, FiniteDifference
    export set_domain!, get_basis, get_grid_points
    export dx, dy, dz, dt, lap, div, grad, cross, curl
    export left, right, bottom, top, front, back
    export add_time_dependent_bc!, t
    export pencilflow_header, pencilflow_banner, equation_summary, domain_summary, show_build_progress, reset_banner!
    
    # Export enhanced parameter handling
    export PhysicalParameters, extract_and_validate_parameters!, add_parameter!
    export validate_parameter_consistency!
    
    # Export automatic parameter detection
    export analyze_symbolic_equations, auto_build_problem!, EquationAnalysis
    
    # Export integrated parameter system
    export SolverParameters, create_solver_parameters
    export update_poisson_plan_with_parameters!
    
    # Export smart interface
    export build_problem_from_equations, analyze_and_suggest, quick_solve
    
    # Export high-level simulation API
    export run_simulation!, enhanced_predictor_corrector_step!
    
    # Export pressure Poisson equation derivation
    export derive_pressure_poisson_equation, PoissonEquationForm
    export analyze_pressure_terms, generate_poisson_solver_code
    
    # Include and export general PDE interface (ultimate generality)
    include("symbolic/enhanced_equation_analysis.jl")
    include("symbolic/equation_format_validation.jl")
    # Load general PDE solver types before any files that reference them
    include("interfaces/general_pde_solver.jl")
    # Load timestep utilities after to avoid type reference issues
    include("timestep/adaptive_stepper_selection.jl")
    include("timestep/imex_term_splitting.jl")
    include("interfaces/equation_boundary_interface.jl")
    
    # Include flexible parameter detection system (ultimate flexibility)
    include("parameters/flexible_parameter_detection.jl")
    include("symbolic/flexible_symbolic_interface.jl")
    
    # Export general PDE solver interface (handles ANY number of equations)
    export solve_arbitrary_pde_system, ArbitraryPDESystem, analyze_arbitrary_pde_system
    export GeneralPDESystem, analyze_general_pde_system, build_general_pde_problem
    export StepperRecommendation, select_optimal_stepper
    export IMEXSplitting, analyze_imex_splitting, create_imex_stepper
    export EquationFormatError, validate_equation_format
    export quick_pde_solve, demo_universal_interface
    
    # Export equation/boundary condition builder interface
    export ProblemBuilder, add_equation!, add_bc!, add_parameter!, set_domain!
    export demonstrate_equation_boundary_interface, run_interface_demo
    
    # Export pressure system analysis and integration
    export PressureEquationAnalysis, analyze_pressure_system, integrate_pressure_system!
    export create_pressure_poisson_solver, demonstrate_pressure_integration
    
    # Export enhanced boundary condition capabilities
    export EnhancedBoundaryCondition, add_enhanced_bc!, add_functional_bc!
    export analyze_boundary_condition_system, demonstrate_enhanced_boundary_conditions
    
    # Export flexible parameter detection system (ultimate flexibility)
    export FlexibleEquationAnalysis, analyze_flexible_equations, create_flexible_problem
    export FlexibleSymbolicProblem, add_flexible_equation!, set_flexible_parameter!
    export auto_detect_physics!, build_flexible_problem!, show_flexible_problem_summary
    
    # Convenience aliases for easier usage
    const solve_pde_system = solve_arbitrary_pde_system
    const quick_pde = quick_solve  # Points to smart_interface version (comprehensive)
    const quick_arbitrary_pde = quick_pde_solve  # Points to general_pde_solver version (minimal)
    const analyze_pde_system = analyze_arbitrary_pde_system
    const flexible_problem = FlexibleSymbolicProblem
    export solve_pde_system, quick_pde, quick_arbitrary_pde, analyze_pde_system, flexible_problem

end #module PencilFlows
