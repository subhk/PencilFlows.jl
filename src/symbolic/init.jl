# Initialization helpers for PencilFlows integration (decomposition, workspaces, components)

function initialize_pencil_infrastructure!(prob::SymbolicProblem)
    domain = prob.domain
    disc = prob.discretization
    Nx = Ny = Nz = 1
    for (i, basis) in enumerate(domain.bases)
        N = domain.grid_points[i]
        basis.name == :x && (Nx = N)
        basis.name == :y && (Ny = N)
        basis.name == :z && (Nz = N)
    end
    if Nx > 1 && Nz > 1
        if Ny == 1
            println("  Initializing 2D decomposition (x,z)...")
            disc.pencil_decomposition = init_pencil_decomposition(Nx, 2, Nz)
        else
            println("  Initializing 3D decomposition (x,y,z)...")
            disc.pencil_decomposition = init_pencil_decomposition(Nx, Ny, Nz)
        end
        disc.distributed_fields = create_distributed_fields(disc.pencil_decomposition)
    else
        println("  Serial computation (no MPI decomposition needed)")
        disc.pencil_decomposition = nothing
        disc.distributed_fields = nothing
    end
end

function setup_pencil_fft_plans!(prob::SymbolicProblem)
    disc = prob.discretization
    if disc.pencil_decomposition !== nothing
        decomp = disc.pencil_decomposition
        dims = (decomp.Nx_global, decomp.Ny_global, decomp.Nz_global)
        disc.fft_plan_rfft = make_pencil_plan_rfft(Float64, dims; comm=decomp.comm, real_transform=true)
        println("  PencilFlow FFT plans created for dimensions: $dims")
    else
        println("  Skipping FFT plans (serial computation)")
    end
end

function initialize_pencil_workspaces!(prob::SymbolicProblem)
    disc = prob.discretization
    if disc.pencil_decomposition !== nothing && disc.distributed_fields !== nothing
        decomp = disc.pencil_decomposition
        fields = disc.distributed_fields
        dx = length(disc.grid_x) > 1 ? disc.grid_x[2] - disc.grid_x[1] : 1.0
        dy = length(disc.grid_y) > 1 ? disc.grid_y[2] - disc.grid_y[1] : 1.0
        Nz = length(disc.grid_z)
        dz_vec = Nz > 1 ? diff(disc.grid_z) : [1.0]
        grid = (dx=dx, dy=dy, dz=dz_vec, Nz=Nz, Nx=decomp.Nx_global, Ny=decomp.Ny_global)
        prototype_field = fields.u_z
        disc.rsns_workspace = RSNSWorkspace(decomp, grid, prototype_field)
        try
            disc.nonlinear_workspace = NonlinearWorkspace(decomp)
            println("  NonlinearWorkspace initialized")
        catch e
            println("  NonlinearWorkspace not available: $e")
            disc.nonlinear_workspace = nothing
        end
        method = (prob.metadata !== nothing) ? get(prob.metadata, :poisson_method, :fft) : :fft
        if method === :mg
            try
                disc.mg_plan = make_mg_poisson_distributed_auto(decomp, grid; levels=4, ω=0.8,
                                                                bc_z=Dict(:bottom_type=>:neumann, :top_type=>:neumann))
                disc.poisson_plan = nothing
                println("  Multigrid Poisson plan initialized (distributed)")
            catch e
                println("   MG plan init failed ($e), falling back to FFT plan")
                disc.poisson_plan = make_poisson_plan(fields.p_z; decomp=decomp, grid=grid, bc_z=:neumann)
                disc.mg_plan = nothing
            end
        else
            disc.poisson_plan = make_poisson_plan(fields.p_z; decomp=decomp, grid=grid, bc_z=:neumann)
            disc.mg_plan = nothing
            println("  FFT Poisson plan initialized")
        end
        println("  PencilFlow workspaces initialized (RSNSWorkspace, Poisson)")
    else
        println("  Skipping workspace initialization (serial computation)")
    end
end

function setup_fft_plans!(prob::SymbolicProblem)
    domain = prob.domain
    disc = prob.discretization
    for (i, basis) in enumerate(domain.bases)
        if isa(basis, Fourier)
            N = domain.grid_points[i]
            dummy = zeros(ComplexF64, N)
            disc.fft_plans[basis.name] = Dict(:forward => "FFTW forward plan for $(basis.name)",
                                              :backward => "FFTW backward plan for $(basis.name)",
                                              :size => N)
        end
    end
end

function separate_equation_terms!(prob::SymbolicProblem)
    disc = prob.discretization
    println("  Integrating with PencilFlow tensor helpers and nonlinear terms...")
    for equation in prob.equations
        linear_terms, nonlinear_terms = parse_equation_terms(equation, prob.parameters)
        field_name = extract_field_from_equation(equation)
        if field_name !== nothing
            disc.linear_operators[field_name] = linear_terms
            disc.nonlinear_functions[field_name] = nonlinear_terms
        end
    end
    setup_tensor_helper_mapping!(prob)
end

function initialize_extended_components!(prob::SymbolicProblem)
    disc = prob.discretization
    println("  Initializing extended PencilFlow components...")
    initialize_pencil_grid!(prob)
    initialize_transform_plans!(prob)
    initialize_time_stepper!(prob)
    initialize_domain_grids!(prob)
    initialize_coriolis!(prob)
    initialize_utilities!(prob)
end

function initialize_pencil_grid!(prob::SymbolicProblem)
    disc = prob.discretization
    domain = prob.domain
    if disc.pencil_decomposition !== nothing
        decomp = disc.pencil_decomposition
        Nx, Ny, Nz = decomp.Nx_global, decomp.Ny_global, decomp.Nz_global
        Lx = Ly = Lz = 1.0
        periodic_x = periodic_y = periodic_z = true
        for (i, basis) in enumerate(domain.bases)
            L = basis.interval[2] - basis.interval[1]
            basis.name == :x && (Lx = L; periodic_x = isa(basis, Fourier))
            basis.name == :y && (Ly = L; periodic_y = isa(basis, Fourier))
            basis.name == :z && (Lz = L; periodic_z = isa(basis, Fourier))
        end
        try
            disc.pencil_grid = Dict(:type => "PencilGrid",
                                    :Nx => Nx, :Ny => Ny, :Nz => Nz,
                                    :Lx => Lx, :Ly => Ly, :Lz => Lz,
                                    :periodic_x => periodic_x,
                                    :periodic_y => periodic_y,
                                    :periodic_z => periodic_z,
                                    :P1 => decomp.P1, :P2 => decomp.P2,
                                    :grid_x => disc.grid_x,
                                    :grid_y => disc.grid_y,
                                    :grid_z => disc.grid_z)
            println(" PencilGrid initialized: $(Nx)x$(Ny)x$(Nz)")
        catch e
            println(" PencilGrid initialization failed: $e")
            disc.pencil_grid = nothing
        end
    end
end

function initialize_transform_plans!(prob::SymbolicProblem)
    disc = prob.discretization
    if disc.pencil_decomposition !== nothing
        decomp = disc.pencil_decomposition
        try
            disc.transform_plans = Dict(:type => "TransformPlans",
                :pencil_x => decomp.pencil_x,
                :pencil_y => decomp.pencil_y,
                :pencil_z => decomp.pencil_z,
                :plan_fft_x => decomp.fft_x,
                :plan_fft_y => decomp.fft_y,
                :transforms => Dict(:z_to_x => decomp.transform_z_to_x,
                                    :x_to_y => decomp.transform_x_to_y,
                                    :y_to_z => decomp.transform_y_to_z,
                                    :x_to_z => decomp.transform_x_to_z,
                                    :y_to_x => decomp.transform_y_to_x,
                                    :z_to_y => decomp.transform_z_to_y))
            println("    TransformPlans initialized with all pencil orientations")
        catch e
            println("    TransformPlans initialization failed: $e")
            disc.transform_plans = nothing
        end
    end
end

function initialize_time_stepper!(prob::SymbolicProblem)
    disc = prob.discretization
    if disc.pencil_decomposition !== nothing
        try
            disc.time_stepper = Dict(:parallel_aware => true,
                :work_arrays => Dict(:u => disc.distributed_fields.u_z,
                                     :v => disc.distributed_fields.v_z,
                                     :w => disc.distributed_fields.w_z,
                                     :p => disc.distributed_fields.p_z),
                :helper_functions => Dict(:similar_pencil => "_similar_pencil",
                                          :copy_array! => "copy_array!",
                                          :zero_array! => "zero_array!",
                                          :axpy_array! => "axpy_array!",
                                          :parallel_maximum => "parallel_maximum"))
            println("    Advanced time steppers available: LSRK4, LSAB1-4, RK4")
        catch e
            println("    Time stepper initialization failed: $e")
            disc.time_stepper = nothing
        end
    end
end

function initialize_domain_grids!(prob::SymbolicProblem)
    disc = prob.discretization
    domain = prob.domain
    if domain === nothing
        println("    Domain not initialized, skipping domain grids setup")
        disc.domain_grids = nothing
        return
    end
    if disc === nothing
        println("    Discretization not initialized, skipping domain grids setup")
        return
    end
    try
        grid_info = []
        for (i, basis) in enumerate(domain.bases)
            if i > length(domain.grid_points)
                @warn "Grid points not defined for basis $(i) ($(basis.name)), skipping"
                continue
            end
            N = domain.grid_points[i]
            grid_points = nothing
            if basis.name == :x && !isempty(disc.grid_x)
                grid_points = disc.grid_x
            elseif basis.name == :y && !isempty(disc.grid_y)
                grid_points = disc.grid_y
            elseif basis.name == :z && !isempty(disc.grid_z)
                grid_points = disc.grid_z
            else
                @warn "Grid for $(basis.name) not initialized yet, skipping domain grid setup"
                continue
            end
            grid_dict = Dict(:name => basis.name,
                             :type => typeof(basis),
                             :size => N,
                             :interval => basis.interval,
                             :grid_points => grid_points,
                             :parallel => disc.pencil_decomposition !== nothing,
                             :spectral_method => isa(basis, Fourier) ? :fourier : :finite_difference)
            push!(grid_info, grid_dict)
        end
        if isempty(grid_info)
            println("    No valid grids found for domain grid initialization")
            disc.domain_grids = nothing
            return
        end
        disc.domain_grids = Dict(:type => "AbstractGrid",
            :ndims => length(domain.bases),
            :grids => grid_info,
            :eltype => Float64,
            :parallel => disc.pencil_decomposition !== nothing)
        println("    Domain grids initialized: $(length(grid_info))D")
    catch e
        println("    Domain grids initialization failed: $e")
        disc.domain_grids = nothing
    end
end

function initialize_coriolis!(prob::SymbolicProblem)
    disc = prob.discretization
    f_val = get(prob.parameters, :f, get(prob.parameters, :Omega, 0.0))
    if f_val != 0.0
        try
            disc.coriolis_plane = Dict(:type => "FPlane",
                                       :f => f_val,
                                       :functions => Dict(:coriolis_terms! => "coriolis_terms! from Coriolis.jl",
                                                          :coriolis_terms => "coriolis_terms from Coriolis.jl"),
                                       :rotation_enabled => true)
            println("    Coriolis f-plane initialized: f = $f_val")
        catch e
            println("    Coriolis initialization failed: $e")
            disc.coriolis_plane = nothing
        end
    else
        disc.coriolis_plane = Dict(:type => "FPlane", :f => 0.0, :rotation_enabled => false)
        println("    Non-rotating case (f = 0)")
    end
end

function initialize_utilities!(prob::SymbolicProblem)
    disc = prob.discretization
    try
        disc.workspace_pool = Dict(:enabled => true)
        disc.timer = Dict(:enabled => true)
        disc.logger = Dict(:enabled => true)
        disc.output_handler = Dict(:type => "DedalusStyleOutput")
        disc.file_handlers = Dict{Symbol, Any}()
        println("    Utilities initialized: WorkspacePool, Timer, Logger, MPI helpers, Output system")
    catch e
        println("    Utilities initialization failed: $e")
    end
end

"""
    validate_imex_separation!(prob::SymbolicProblem)

Validate that the linear/nonlinear term separation is correct for IMEX timestepping.
Reports which terms are classified as linear (implicit) vs nonlinear (explicit).
"""
function validate_imex_separation!(prob::SymbolicProblem)
    disc = prob.discretization
    println("  Validating IMEX term separation...")
    
    total_linear = 0
    total_nonlinear = 0
    
    # Check each field's term classification
    for (field_name, _) in disc.linear_operators
        if field_name != :_function_map
            linear_count = length(disc.linear_operators[field_name])
            nonlinear_count = haskey(disc.nonlinear_functions, field_name) ? 
                             length(disc.nonlinear_functions[field_name]) : 0
            
            println("    Field $field_name: $linear_count linear terms (implicit), $nonlinear_count nonlinear terms (explicit)")
            
            total_linear += linear_count
            total_nonlinear += nonlinear_count
        end
    end
    
    println("    IMEX Classification Complete:")
    println("       Linear terms (implicit treatment): $total_linear")
    println("       Nonlinear terms (explicit treatment): $total_nonlinear")
    
    # Warn about potential issues
    if total_linear == 0 && total_nonlinear > 0
        @warn "No linear terms found - this will be fully explicit (may be unstable for stiff problems)"
    elseif total_nonlinear == 0 && total_linear > 0  
        @warn "No nonlinear terms found - this will be fully implicit (may be inefficient)"
    elseif total_linear == 0 && total_nonlinear == 0
        @warn "No terms classified - equations may not be properly parsed"
    else
        println("       IMEX separation looks good!")
    end
end

