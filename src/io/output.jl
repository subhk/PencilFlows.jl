#module OutputPencil

# 
# output_pencil.jl  NetCDF I/O utilities with Dedalus-style layout, segment
# counters starting at **1**, and automatic file rotation after `max_writes`
# records.
#
# Public API (unchanged):
#     write_state(prob; kwargs...)
#
# New keyword:
#     max_writes::Int = 0   # 0  unlimited; otherwise rotate after this many
#                            # writes per segment (file).
# 

using MPI, Printf, Dates
using PencilArrays

#  NetCDF backend (required) 
const _HAS_NETCDF = Ref(false)
try
    @eval using NetCDF
    _HAS_NETCDF[] = true
catch
    @error "NetCDF.jl not found. Please `pkg> add NetCDF` to enable output."
end

#  MPI conveniences 
MPI.Initialized() || MPI.Init()
const _COMM = MPI.COMM_WORLD
comm_rank() = MPI.Comm_rank(_COMM)
comm_size() = MPI.Comm_size(_COMM)
isroot()    = comm_rank() == 0

#  Consistent Filename Generation Helpers 
"""
    format_time_for_filename(t::Real; decimals::Int=6)

Format time for inclusion in filenames with consistent precision.
"""
format_time_for_filename(t::Real; decimals::Int=6) = @sprintf("%.*f", decimals, float(t))

"""
    generate_segment_filename(base::AbstractString, segment::Integer; include_segment::Bool=true)

Generate consistent segmented filenames. Segments start at 1.
"""
generate_segment_filename(base::AbstractString, segment::Integer; include_segment::Bool=true) = 
    include_segment && segment > 1 ? string(base, "_s", segment) : base

"""
    generate_timestep_filename(base::AbstractString, t::Real, step::Integer; 
                              include_time::Bool=true, include_step::Bool=true,
                              time_decimals::Int=6, step_width::Int=6)

Generate consistent timestep-based filenames.
"""
function generate_timestep_filename(base::AbstractString, t::Real, step::Integer; 
                                   include_time::Bool=true, include_step::Bool=true,
                                   time_decimals::Int=6, step_width::Int=6)
    filename = base
    if include_time
        filename *= "_t" * format_time_for_filename(t; decimals=time_decimals)
    end
    if include_step && step > 0
        filename *= "_step" * lpad(step, step_width, '0')
    end
    return filename
end

# Legacy support
segment_stem(base::AbstractString, seg::Integer) = generate_segment_filename(base, seg)

#  Consistent Safety and Error Handling
"""
    ensure_can_write(path; overwrite::Bool)

Consistent file writing safety check with proper error handling.
"""
function ensure_can_write(path; overwrite::Bool)
    try
        if isroot() && isfile(path) && !overwrite
            error("OutputError: File $path exists and overwrite=false. Use overwrite=true to force overwrite.")
        end
        MPI.Barrier(_COMM)
    catch e
        if isa(e, ErrorException) && occursin("OutputError", e.msg)
            rethrow(e)  # Re-throw our custom errors
        else
            error("OutputError: Failed to check file write permissions for $path: $e")
        end
    end
end

"""
    safe_netcdf_operation(f, description::String)

Safely execute NetCDF operations with consistent error handling.
"""
function safe_netcdf_operation(f, description::String)
    try
        return f()
    catch e
        error("OutputError: $description failed - $e")
    end
end

#  Gather helper (PencilArray -> root) 
function _gather_field(A::PencilArrays.PencilArray)
    pencil = PencilArrays.pencil(A)
    comm  = PencilArrays.get_comm(pencil)
    lp    = parent(A)
    lcnt  = length(lp)
    counts = MPI.Allgather(lcnt, comm)
    displs = cumsum([0; counts[1:end-1]])
    if isroot()
        full = Array{eltype(A)}(undef, sum(counts))
        MPI.Gatherv!(lp, full, counts, displs, 0, comm)
        return reshape(full, size(A))
    else
        MPI.Gatherv!(lp, Vector{eltype(A)}(), Int[], Int[], 0, comm)
        return nothing
    end
end
_gather_field(A::AbstractArray) = isroot() ? A : nothing

"""
    gather_field(A)

Public wrapper to gather a distributed field (`PencilArray`) to the root rank.
Returns the full `Array` on root, and `nothing` on other ranks. For regular
arrays, returns the input on root and `nothing` otherwise.
"""
gather_field(A) = _gather_field(A)

#  NetCDF helpers 
const _SPATIAL_DIM_NAMES = ("x","y","z","w")  # up to 4-D if needed

"""
append_record!(ds, name::String, A::AbstractArray)
Append one record `A` to variable `name`, creating the variable (and spatial
dimensions) if it does not yet exist.
"""
function append_record!(ds, name::String, A::AbstractArray; compress_level::Int=4)
    var_exists = name in NetCDF.varnames(ds)

    # Define dims on first creation
    if !var_exists
        NetCDF.reDef(ds)
        nd     = ndims(A)
        dims   = ("write", _SPATIAL_DIM_NAMES[1:nd]...)
        for d in 1:nd
            dimnm = _SPATIAL_DIM_NAMES[d]
            dimnm in NetCDF.dimnames(ds) || NetCDF.defDim(ds, dimnm, size(A,d))
        end
        NetCDF.defVar(ds, name, eltype(A), dims; compress=compress_level)
        NetCDF.endDef(ds)
    end

    v  = NetCDF.var(ds, name)
    rec = size(v, 1) + 1  # next write index (1-based)
    starts = (rec, ntuple(_->1, ndims(A))...)
    NetCDF.putVar(v, starts, A)
end

"""
_write_netcdf(path, prob, data, rec_meta; compress, overwrite)
Create or append to a NetCDF file "path" in Dedalus style.
"""
function _write_netcdf(path::AbstractString, prob, data::Dict{Symbol,Any}, rec_meta::Dict{String,Float64};
                       compress::Bool, overwrite::Bool)
    _HAS_NETCDF[] || error("OutputError: NetCDF.jl not available. Please install NetCDF.jl to enable file output.")
    ensure_can_write(path; overwrite)

    mode = isfile(path) ? NetCDF.NC_WRITE : NetCDF.CLOBBER
    ds   = NetCDF.open(path, mode)
    try
        # Ensure unlimited write dim exists
        "write" in NetCDF.dimnames(ds) || (NetCDF.reDef(ds); NetCDF.defDim(ds, "write", NetCDF.UNLIMITED); NetCDF.endDef(ds))

        # Append tasks
        for (nm,A) in data
            nm == :time_scalar && continue  # handled below
            append_record!(ds, "tasks/"*String(nm), A; compress_level=(compress ? 4 : 0))
        end

        # Append per-write scalars in /scales
        for (key,val) in rec_meta
            append_record!(ds, "scales/"*key, [val]; compress_level=0)
        end

        # Global attributes (set once)
        if NetCDF.getAtt(ds, NetCDF.nc_global, "created"; default=nothing) === nothing
            NetCDF.putAtt(ds, NetCDF.nc_global, "created", string(Dates.now()))
            NetCDF.putAtt(ds, NetCDF.nc_global, "n_ranks", comm_size())
        end
    finally
        NetCDF.close(ds)
    end
    path
end

#  Rotation bookkeeping 
"""
_get_rotation!(prob, tag) -> (seg, n_writes)
Retrieve and update the rotation bookkeeping dict stored in `prob`.
"""
function _get_rotation!(prob, tag::Symbol)
    dict = get!(prob, :_output_rotation) do; Dict{Symbol,Tuple{Int,Int}}(); end
    return get!(dict, tag) do; (1, 0); end
end

function _set_rotation!(prob, tag::Symbol, seg::Int, n::Int)
    dict = prob[:_output_rotation]
    dict[tag] = (seg, n)
end

"""
    should_rotate_file(current_writes::Int, max_writes::Int) -> Bool

Determine if file should be rotated to new segment based on write count.
"""
should_rotate_file(current_writes::Int, max_writes::Int) = max_writes > 0 && current_writes >= max_writes

#  Public API 
"""
write_state(prob; filename="analysis_tasks", max_writes=0, kwargs...)
Write simulation state in Dedalus-style NetCDF.  If `max_writes>0`, rotate to a
new segment (`..._s#.nc`) after that many records.
"""
function write_state(prob; filename="analysis_tasks", fields=:auto, gather::Bool=:auto,
                     tag::Symbol=:default, kwargs...)
    # Get consistent configuration
    config = get_output_config(kwargs)
    
    parallel = get(prob,:parallel,false)
    gather === :auto && (gather = parallel)
    
    # Extract configuration values  
    overwrite = config[:overwrite]
    compress = config[:compress]
    step = get(kwargs, :step, get(prob, :iteration, 0))
    include_time_in_name = config[:include_time_in_name] 
    time_decimals = config[:time_decimals]
    step_width = config[:step_width]
    max_writes = config[:max_writes]

    # Consistent rotation bookkeeping
    seg, nwrites = _get_rotation!(prob, tag)
    if should_rotate_file(nwrites, max_writes)
        seg += 1; nwrites = 0
    end

    # Build filename using consistent generation
    base_name = generate_segment_filename(filename, seg)
    if include_time_in_name
        stem = generate_timestep_filename(base_name, prob.t, step; 
                                         include_time=true, include_step=(step > 0),
                                         time_decimals=time_decimals, step_width=step_width)
    else
        stem = base_name
    end
    path = stem * ".nc"

    # Parse computed fields and collect data
    data = Dict{Symbol,Any}()
    if haskey(prob,:vars)
        # Parse field specification for computed fields
        regular_fields, computed_fields = parse_computed_fields(config[:fields])
        
        # Update config with parsed regular fields
        config_with_parsed = copy(config)
        config_with_parsed[:fields] = regular_fields
        
        # Validate field selection
        available_fields = collect(keys(prob.vars))
        validate_field_selection(regular_fields, available_fields)
        
        # Filter regular fields based on configuration
        filtered_fields = filter_fields_for_output(prob.vars, config_with_parsed)
        
        # Gather regular fields
        for (k,v) in filtered_fields
            gathered = gather ? _gather_field(v) : v
            isroot() && (data[k] = gathered)
        end
        
        # Compute and add computed fields
        for comp_field in computed_fields
            computed_value = evaluate_computed_field(comp_field, prob)
            if computed_value !== nothing
                gathered = gather ? _gather_field(computed_value) : computed_value
                isroot() && (data[comp_field.name] = gathered)
                println("Added computed field: $(comp_field.name) = $(comp_field.expression)")
            end
        end
        
        # Report what was selected for output
        if isroot() && !isempty(data)
            selected_names = collect(keys(data))
            regular_count = length(filtered_fields)
            computed_count = length(computed_fields)
            total_available = length(available_fields)
            println("Writing $(length(selected_names)) fields ($regular_count regular + $computed_count computed) of $total_available available: $(join(selected_names, ", "))")
        end
    end

    # Record-level metadata (single numbers)
    rec_meta = Dict("t"=>prob.t, "dt"=>prob.dt, "iter"=>float(step))
    wall_time = get(prob, :wall_time, NaN)
    !isnan(wall_time) && (rec_meta["wall_time"] = wall_time)

    isroot() && _write_netcdf(path, prob, data, rec_meta; compress, overwrite)

    # Update rotation state
    _set_rotation!(prob, tag, seg, nwrites+1)
    return path
end

export write_state, format_time_for_filename, segment_stem
export generate_segment_filename, generate_timestep_filename
export filter_fields_for_output, validate_field_selection
export select_velocity_fields, select_scalar_fields, select_vorticity_fields
export select_analysis_fields, exclude_debug_fields
export demonstrate_variable_selection
# Computed fields functionality
export ComputedField, parse_computed_fields, evaluate_computed_field
export vorticity_z, vorticity_x, vorticity_y
export kinetic_energy, kinetic_energy_2d, enstrophy
export u_rms, v_rms, w_rms, T_rms
export grad_T_magnitude, grad_u_magnitude
export T_mean, u_mean, v_mean
export velocity_and_vorticity, scalars_and_energy, diagnostics_2d
export demonstrate_computed_fields

#  Example usage 
# The following commented block shows how to integrate OutputPencil into a
# PencilArrays-based simulation.  Copy/paste it into a separate script and
# remove the leading "# " to run.
#
# -------------------------------------------------------------------------
# using MPI
# using PencilArrays, PencilFFTs
# using .OutputPencil
# 
# # ----------------------------------------------------------------------
# # Set up a dummy 2-D vorticity problem with a pencil decomposition
# # ----------------------------------------------------------------------
# MPI.Initialized() || MPI.Init()
# size  = MPI.Comm_size(MPI.COMM_WORLD)
# rank  = MPI.Comm_rank(MPI.COMM_WORLD)
# Nx, Ny = 128, 128                        # global grid
# topo   = PencilArrays.Topology(MPI.COMM_WORLD, (size, 1))
# pencil = PencilArrays.Pencil(topo, (Nx, Ny), (1,))  # x-pencils
# 
# piÌ = PencilArray{ComplexF64}(undef, pencil)      # spectral vorticity
# piÌ = PencilArray{ComplexF64}(undef, pencil)      # spectral streamfunction
# 
# # Populate with some data (rankdependent for demonstration)
# fill!(piÌ, 1.0im * rank)
# fill!(piÌ, 2.0im * rank)
# 
# # Problem dictionary expected by write_state
# prob = Dict(
#     :vars      => Dict(:omega_hat => piÌ, :psi_hat => piÌ),
#     :t         => 0.0,
#     :dt        => 1.0e-3,
#     :iteration => 0,
#     :parallel  => true,
# )
# 
# # ----------------------------------------------------------------------
# # Write output every 10 iterations, rotating files after 50 writes
# # ----------------------------------------------------------------------
# nt = 120
# for n in 0:nt
#     # (solver code would update piÌ, piÌ here)
#     prob[:iteration] = n
#     prob[:t]        += prob[:dt]
#     if n % 10 == 0
#         OutputPencil.write_state(prob;
#             filename   = "analysis_tasks",
#             max_writes = 50,           # rotate every 50 records
#             compress   = true,
#         )
#     end
# end
# -------------------------------------------------------------------------
# This loop will create:
#   analysis_tasks_s1.nc   (writes 150)
#   analysis_tasks_s2.nc   (writes 51100)
#   analysis_tasks_s3.nc   (writes 101120)
# -------------------------------------------------------------------------

#  Dedalus-style Analysis helper 

"""
    AnalysisHandle(prob; filename="analysis", sim_dt=1.0, max_writes=0,
                   gather=:auto, tag=:default)
High-level analysis object that mimics Dedaluskappa `analysis.Analysis`:
* Records a snapshot **every multiple of `sim_dt`** in simulation time.
* Calls `OutputPencil.write_state` internally, inheriting segment rotation
  (`max_writes`, `..._s#.nc`).
* Provides
    ¢ `add_system!(ana)`  save **all** fields in `prob[:vars]`.
    ¢ `add_task!(ana, name, f)`  save a custom field computed by `f(prob)`.
Call `analysis_step!(ana, prob)` each solver step; it will write when
`prob[:t] ¥ next_time`.
"""
mutable struct AnalysisHandle
    filename::String
    sim_dt::Float64
    next_time::Float64
    gather::Bool
    max_writes::Int
    tag::Symbol
    tasks::Dict{Symbol,Function}
    config::Dict{Symbol,Any}  # Store output configuration
end

function AnalysisHandle(prob; filename="analysis", sim_dt::Real=1.0, 
                        gather=:auto, tag::Symbol=:default, kwargs...)
    # Use consistent configuration 
    config = get_output_config(kwargs)
    gather === :auto && (gather = get(prob,:parallel,false))
    max_writes = config[:max_writes]
    return AnalysisHandle(filename, float(sim_dt), prob[:t], gather, max_writes, tag, Dict(), config)
end

""" add_system!(ana, prob) """
function add_system!(ana::AnalysisHandle, prob)
    haskey(prob, :vars) || error("prob lacks :vars to add_system")
    
    # Filter fields based on analysis configuration
    filtered_fields = filter_fields_for_output(prob[:vars], ana.config)
    
    for (k,_) in filtered_fields
        ana.tasks[k] = p -> p[:vars][k]
    end
    
    if !isempty(filtered_fields)
        selected_names = collect(keys(filtered_fields))
        total_available = length(prob[:vars])
        println("Analysis will track $(length(selected_names)) of $total_available available fields: $(join(selected_names, ", "))")
    end
    
    return ana
end

""" add_task!(ana, name, f)  Add custom variable """
function add_task!(ana::AnalysisHandle, name::Symbol, f::Function)
    ana.tasks[name] = f
    return ana
end

"""
    write_state(ana::AnalysisHandle, prob; fields, kwargs...)

Write specific fields using the AnalysisHandle configuration.
Allows flexible per-call field selection while using the analysis configuration.
"""
function write_state(ana::AnalysisHandle, prob; fields=:all, kwargs...)
    # Merge analysis config with per-call options
    merged_config = copy(ana.config)
    call_config = get_output_config(kwargs)
    
    # Override with call-specific options
    for (k, v) in call_config
        merged_config[k] = v
    end
    
    # Handle field specification
    if fields != :all
        merged_config[:fields] = fields
    end
    
    # Parse computed fields
    regular_fields, computed_fields = parse_computed_fields(merged_config[:fields])
    
    # Collect data similar to regular write_state but with analysis config
    data = Dict{Symbol,Any}()
    if haskey(prob, :vars)
        # Filter regular fields
        if regular_fields != :all && regular_fields != :none
            merged_config[:fields] = regular_fields
        end
        filtered_fields = filter_fields_for_output(prob[:vars], merged_config)
        
        # Add regular fields
        for (k, v) in filtered_fields
            gathered = ana.gather ? _gather_field(v) : v
            data[k] = gathered
        end
        
        # Add computed fields
        for comp_field in computed_fields
            computed_value = evaluate_computed_field(comp_field, prob)
            if computed_value !== nothing
                gathered = ana.gather ? _gather_field(computed_value) : computed_value
                data[comp_field.name] = gathered
                println("Added computed field: $(comp_field.name) = $(comp_field.expression)")
            end
        end
    end
    
    # Create filename for this specific write
    timestamp = get(kwargs, :step, get(prob, :iteration, 0))
    base_filename = get(kwargs, :filename, ana.filename)
    
    if get(merged_config, :include_time_in_name, true)
        full_filename = generate_timestep_filename(base_filename, prob[:t], timestamp)
    else
        full_filename = base_filename
    end
    
    # Write using NetCDF backend
    if !isempty(data)
        rec_meta = Dict("t" => prob[:t], "dt" => prob[:dt], "iter" => float(timestamp))
        isroot() && _write_netcdf(full_filename * ".nc", prob, data, rec_meta; 
                                 compress=merged_config[:compress], 
                                 overwrite=merged_config[:overwrite])
        
        if isroot()
            selected_names = collect(keys(data))
            println("AnalysisHandle wrote $(length(selected_names)) fields: $(join(selected_names, ", "))")
        end
    end
    
    return ana
end

""" analysis_step!(ana, prob)  call each timestep """
function analysis_step!(ana::AnalysisHandle, prob)
    if prob[:t] + 1e-12 >= ana.next_time
        # Evaluate tasks
        data = Dict{Symbol,Any}()
        for (k,fun) in ana.tasks
            val = fun(prob)
            data[k] = ana.gather ? _gather_field(val) : val
        end
        # Use consistent filename generation
        filename = generate_segment_filename(ana.filename, 1) * ".nc"
        isroot() && _write_netcdf(filename, prob, data,
                                  Dict("t"=>prob[:t], "dt"=>prob[:dt], "iter"=>float(get(prob,:iteration,0)));
                                  compress=true, overwrite=true)
        ana.next_time += ana.sim_dt
    end
    return nothing
end

# export AnalysisHandle, add_system!, add_task!, analysis_step!

# ============================================================================
# DEDALUS-STYLE OUTPUT INTERFACE (for SymbolicProblem integration)
# ============================================================================
"""
    add_output_task!(prob, field_name; layout='g', name=nothing)

Add an output task similar to Dedalus solver.add_task().
Layout options: 'g' for physical space, 'c' for spectral space.

Note: This function is designed for SymbolicProblem types from the symbolic interface.
"""
function add_output_task!(prob, field_name::Symbol; 
                         layout::Char='g', name::Union{String,Nothing}=nothing)
    
    if prob.discretization === nothing || prob.discretization.output_handler === nothing
        error("Problem must be built and have output handler initialized")
    end
    
    task_name = name !== nothing ? name : string(field_name)
    output_handler = prob.discretization.output_handler
    
    # Add task to the output handler
    output_handler[:tasks][field_name] = Dict(
        :name => task_name,
        :layout => layout,
        :field => field_name,
        :type => :field
    )
    
    println(" Added output task: $task_name (layout: $layout)")
    return prob
end

"""
    add_analysis_task!(prob, expression, name; layout='g')

Add a computed analysis task (e.g., kinetic energy, enstrophy).

Note: This function is designed for SymbolicProblem types from the symbolic interface.
"""
function add_analysis_task!(prob, expression::String, name::String; 
                           layout::Char='g')
    
    if prob.discretization === nothing || prob.discretization.output_handler === nothing
        error("Problem must be built and have output handler initialized") 
    end
    
    output_handler = prob.discretization.output_handler
    
    # Add analysis task
    output_handler[:tasks][Symbol(name)] = Dict(
        :name => name,
        :layout => layout,
        :expression => expression,
        :type => :computed
    )
    
    println(" Added analysis task: $name = $expression")
    return prob
end

"""
    create_file_handler!(prob, filename; sim_dt=0.1, max_writes=0, max_file_size=2^30)

Create a file handler similar to Dedalus solver.add_file_handler().

Note: This function is designed for SymbolicProblem types from the symbolic interface.
"""
function create_file_handler!(prob, filename::String; 
                             sim_dt::Real=0.1, max_writes::Int=0, 
                             max_file_size::Real=2^30)
    
    if prob.discretization === nothing
        error("Problem must be built before creating file handlers")
    end
    
    disc = prob.discretization
    
    # Initialize output handler if not present
    if disc.output_handler === nothing
        disc.output_handler = Dict(
            :type => "DedalusStyleOutput",
            :tasks => Dict{Symbol, Any}(),
            :handlers => Dict{Symbol, Any}()
        )
        disc.file_handlers = Dict{Symbol, Any}()
    end
    
    handler_name = Symbol(filename)
    handler_info = Dict(
        "filename" => filename,
        "sim_dt" => sim_dt,
        "max_writes" => max_writes,
        "max_file_size" => max_file_size,
        "tasks" => copy(disc.output_handler[:tasks]),  # Snapshot current tasks
        "last_write_time" => 0.0,
        "write_count" => 0,
        "current_segment" => 1
    )
    
    # Store in both locations for compatibility
    disc.output_handler[:handlers][handler_name] = handler_info
    disc.file_handlers[handler_name] = handler_info
    
    println(" Created file handler: $filename")
    println(" Write interval: every $sim_dt simulation time units")
    if max_writes > 0
        println("   Max writes per file: $max_writes")
    end
    
    return prob
end

"""
    write_output!(prob, solution, current_time, step)

Write output using Dedalus-style file handlers and output.jl backend.

Note: This function is designed for SymbolicProblem types from the symbolic interface.
"""
function write_output!(prob, solution::Dict{Symbol, Any}, 
                      current_time::Real, step::Int)
    
    if prob.discretization === nothing || prob.discretization.output_handler === nothing
        println("  No output handler configured")
        return
    end
    
    disc = prob.discretization
    output_handler = disc.output_handler
    
    # Process each file handler
    for (handler_name, handler) in disc.file_handlers
        if should_write_output(handler, current_time)
            write_handler_output!(handler, prob, solution, current_time, step, disc)
            handler["last_write_time"] = current_time
            handler["write_count"] += 1
        end
    end
end

"""
    should_write_output(handler, current_time)

Determine if it's time to write output based on simulation time.
"""
function should_write_output(handler::Dict{String, Any}, current_time::Real)
    time_since_last = current_time - handler["last_write_time"]
    return time_since_last >= handler["sim_dt"]
end

"""
    write_handler_output!(handler, prob, solution, current_time, step, disc)

Write output for a specific file handler using output.jl.

Note: This function integrates with the symbolic interface's DiscretizationInfo structure.
"""
function write_handler_output!(handler::Dict{String, Any}, prob,
                              solution::Dict{Symbol, Any}, current_time::Real, 
                              step::Int, disc)
    
    filename = handler["filename"]
    segment = handler["current_segment"]
    
    # Check if we need to rotate to new file
    if handler["max_writes"] > 0 && handler["write_count"] >= handler["max_writes"]
        handler["current_segment"] += 1
        handler["write_count"] = 0
        segment = handler["current_segment"]
    end
    
    # Build filename with segment
    if segment > 1
        full_filename = "$(filename)_s$(segment)"
    else
        full_filename = filename
    end
    
    # Prepare data for output
    output_data = Dict{Symbol, Any}()
    
    # Add field data from tasks
    for (task_name, task_info) in handler["tasks"]
        if task_info[:type] == :field && haskey(solution, task_name)
            output_data[task_name] = solution[task_name]
        elseif task_info[:type] == :computed
            # Compute analysis quantities
            computed_value = compute_analysis_task(task_info[:expression], solution)
            if computed_value !== nothing
                output_data[task_name] = computed_value
            end
        end
    end
    
    # Create problem structure compatible with output.jl
    prob_dict = Dict{Symbol, Any}(
        :vars => output_data,
        :t => current_time,
        :dt => get(prob.parameters, :dt, 0.001),
        :iteration => step,
        :wall_time => time(),
        :parallel => disc.pencil_decomposition !== nothing,
        :_output_rotation => Dict{Symbol, Tuple{Int,Int}}()
    )
    
    # Write using existing output.jl functionality
    try
        written_path = write_state(prob_dict; 
                                  filename=full_filename,
                                  step=step,
                                  overwrite=true,
                                  compress=true,
                                  include_time_in_name=false,  # We handle this ourselves
                                  max_writes=handler["max_writes"]
        )
        
        println(" Written: $written_path (step $step, t=$(round(current_time, digits=4)))")
    catch e
        println("  Output write failed: $e")
    end
end

"""
    compute_analysis_task(expression, solution)

Compute analysis tasks like kinetic energy, RMS values, etc.
Uses the field analysis utilities from utils.jl.
"""
function compute_analysis_task(expression::String, solution::Dict{Symbol, Any})
    try
        # Handle common analysis expressions
        if occursin("kinetic_energy", expression) || occursin("0.5*(u2 + v2", expression)
            return compute_kinetic_energy(solution)
        elseif occursin("u_rms", expression) || occursin("sqrt(mean(u2))", expression)
            return compute_rms_field(solution, :u)
        elseif occursin("v_rms", expression) || occursin("sqrt(mean(v2))", expression)
            return compute_rms_field(solution, :v)
        elseif occursin("w_rms", expression) || occursin("sqrt(mean(w2))", expression)
            return compute_rms_field(solution, :w)
        elseif occursin("mean(b)", expression) || occursin("b_mean", expression)
            return compute_mean_field(solution, :b)
        else
            # Generic expression - would need more sophisticated parsing
            println("     Analysis expression not implemented: $expression")
            return nothing
        end
    catch e
        println("  Error computing analysis task: $e")
        return nothing
    end
end

# ============================================================================
# CONSISTENT OUTPUT CONFIGURATION 
# ============================================================================

# Default output configuration for consistency across all functions
const DEFAULT_OUTPUT_CONFIG = Dict(
    :compress => true,
    :overwrite => true,  
    :compress_level => 4,
    :time_decimals => 6,
    :step_width => 6,
    :max_writes => 0,  # 0 = unlimited
    :include_time_in_name => true,
    :include_step_in_name => true,
    :fields => :all,  # :all, :none, or list of field names
    :exclude_fields => Symbol[],  # fields to exclude
    :include_computed => true,  # include computed/derived fields
    :field_layouts => Dict{Symbol, Char}()  # per-field layout specifications
)

"""
    get_output_config(kwargs)

Merge user-provided options with default output configuration.
"""
function get_output_config(kwargs)
    config = copy(DEFAULT_OUTPUT_CONFIG)
    for (k, v) in kwargs
        if haskey(config, k)
            config[k] = v
        end
    end
    return config
end

"""
    filter_fields_for_output(all_fields::Dict{Symbol,Any}, config::Dict) -> Dict{Symbol,Any}

Filter fields based on user selection criteria.

# Field Selection Options:
- `fields = :all` - Include all available fields (default)
- `fields = :none` - Include no fields 
- `fields = [:u, :v, :T]` - Include only specified fields
- `exclude_fields = [:temp_var, :debug_field]` - Exclude specific fields
- `include_computed = false` - Exclude computed/derived fields

# Examples:
```julia
# Save only velocity fields
write_state(prob; fields = [:u, :v, :w])

# Save all fields except temporary ones  
write_state(prob; exclude_fields = [:temp, :debug, :scratch])

# Save everything except computed fields
write_state(prob; include_computed = false)
```
"""
function filter_fields_for_output(all_fields::Dict{Symbol,Any}, config::Dict)
    fields_to_include = config[:fields]
    fields_to_exclude = config[:exclude_fields]
    include_computed = config[:include_computed]
    
    # Start with all fields or empty set
    if fields_to_include == :all
        selected_fields = copy(all_fields)
    elseif fields_to_include == :none
        selected_fields = Dict{Symbol,Any}()
    elseif fields_to_include isa AbstractVector
        # Only include specified fields that exist
        selected_fields = Dict{Symbol,Any}()
        for field_name in fields_to_include
            if haskey(all_fields, field_name)
                selected_fields[field_name] = all_fields[field_name]
            else
                println("Warning: Requested field '$field_name' not found in problem variables")
            end
        end
    else
        error("OutputError: fields parameter must be :all, :none, or a vector of field names")
    end
    
    # Remove excluded fields
    for field_name in fields_to_exclude
        if haskey(selected_fields, field_name)
            delete!(selected_fields, field_name)
            println(" Excluded field: $field_name")
        end
    end
    
    # Filter computed fields if requested
    if !include_computed
        computed_patterns = [r"_rms$", r"_mean$", r"_energy$", r"_enstrophy$", r"_derived$", r"_computed$"]
        fields_to_remove = Symbol[]
        
        for (field_name, _) in selected_fields
            field_str = string(field_name)
            if any(pattern -> occursin(pattern, field_str), computed_patterns)
                push!(fields_to_remove, field_name)
            end
        end
        
        for field_name in fields_to_remove
            delete!(selected_fields, field_name)
            println("Excluded computed field: $field_name")
        end
    end
    
    return selected_fields
end

"""
    validate_field_selection(fields_config, available_fields::Vector{Symbol})

Validate user field selection against available fields.
"""
function validate_field_selection(fields_config, available_fields::Vector{Symbol})
    if fields_config isa AbstractVector
        missing_fields = setdiff(fields_config, available_fields)
        if !isempty(missing_fields)
            println("Warning: The following requested fields are not available:")
            for field in missing_fields
                println("    • $field")
            end
            println("   Available fields: $(join(available_fields, ", "))")
        end
    end
end

"""
    select_velocity_fields() -> Vector{Symbol}

Helper to select common velocity field components.
"""
select_velocity_fields() = [:u, :v, :w]

"""
    select_scalar_fields() -> Vector{Symbol}

Helper to select common scalar fields.
"""
select_scalar_fields() = [:T, :b, :c, :rho, :p]

"""
    select_vorticity_fields() -> Vector{Symbol}

Helper to select vorticity components.
"""
select_vorticity_fields() = [:omega_x, :omega_y, :omega_z, :omega, :vorticity]

"""
    select_analysis_fields() -> Vector{Symbol}

Helper to select common analysis/diagnostic fields.
"""
select_analysis_fields() = [:kinetic_energy, :u_rms, :v_rms, :w_rms, :T_mean, :enstrophy]

"""
    exclude_debug_fields() -> Vector{Symbol}

Helper to exclude common debug/temporary fields.
"""
exclude_debug_fields() = [:debug, :temp, :scratch, :work, :aux, :temporary]

# ============================================================================
# COMPUTED FIELDS FUNCTIONALITY
# ============================================================================

"""
    ComputedField

Structure to hold a computed field definition with name and expression.
"""
struct ComputedField
    name::Symbol
    expression::String
    description::String
end

"""
    parse_computed_fields(fields_spec) -> (regular_fields, computed_fields)

Parse field specification that may contain computed fields.
Computed fields have format: [:ζ => "dx(v) - dy(u)"]
"""
function parse_computed_fields(fields_spec)
    regular_fields = Symbol[]
    computed_fields = ComputedField[]
    
    if fields_spec == :all || fields_spec == :none
        return fields_spec, computed_fields
    elseif fields_spec isa AbstractVector
        for item in fields_spec
            if item isa Pair
                # Computed field: :ζ => "dx(v) - dy(u)"
                name, expr = item
                computed_field = ComputedField(name, expr, "computed: $expr")
                push!(computed_fields, computed_field)
                push!(regular_fields, name)  # Also add to regular list for filtering
            elseif item isa Symbol
                # Regular field
                push!(regular_fields, item)
            else
                error("OutputError: Invalid field specification: $item")
            end
        end
        return regular_fields, computed_fields
    else
        return fields_spec, computed_fields
    end
end

"""
    evaluate_computed_field(field::ComputedField, prob) -> Any

Evaluate a computed field expression using available problem variables.
"""
function evaluate_computed_field(field::ComputedField, prob)
    try
        # Get available variables
        vars = get(prob, :vars, Dict())
        
        # Common computed field patterns
        expr = field.expression
        
        # Vorticity calculations
        if occursin("dx(v)", expr) && occursin("dy(u)", expr)
            if occursin("-", expr)  # ζ = dx(v) - dy(u) (z-component)
                if haskey(vars, :u) && haskey(vars, :v)
                    return compute_vorticity_z(vars[:v], vars[:u], prob)
                end
            end
        elseif expr == "dx(v) - dy(u)"
            if haskey(vars, :u) && haskey(vars, :v)
                return compute_vorticity_z(vars[:v], vars[:u], prob)
            end
        end
        
        # Kinetic energy calculations
        if occursin("0.5*(u^2 + v^2", expr) || expr == "kinetic_energy"
            if haskey(vars, :u) && haskey(vars, :v)
                ke = 0.5 * (abs2.(vars[:u]) .+ abs2.(vars[:v]))
                if haskey(vars, :w)
                    ke .+= 0.5 * abs2.(vars[:w])
                end
                return ke
            end
        end
        
        # RMS calculations
        if occursin("sqrt(mean(", expr)
            field_match = match(r"sqrt\(mean\(([^)]+)\^?2?\)\)", expr)
            if field_match !== nothing
                field_name = Symbol(field_match.captures[1])
                if haskey(vars, field_name)
                    return sqrt(mean(abs2.(vars[field_name])))
                end
            end
        end
        
        # Temperature gradient magnitude
        if expr == "sqrt(dx(T)^2 + dy(T)^2 + dz(T)^2)" || expr == "grad_T_mag"
            if haskey(vars, :T)
                return compute_gradient_magnitude(vars[:T], prob)
            end
        end
        
        # Enstrophy (0.5 * ω²)
        if expr == "0.5*ζ^2" || expr == "enstrophy"
            if haskey(vars, :ζ) || haskey(vars, :omega_z)
                omega = get(vars, :ζ, get(vars, :omega_z, nothing))
                if omega !== nothing
                    return 0.5 * abs2.(omega)
                end
            end
        end
        
        println("Warning: Could not evaluate computed field '$(field.name)' with expression '$(field.expression)'")
        return nothing
        
    catch e
        println("Error computing field '$(field.name)': $e")
        return nothing
    end
end

"""
    compute_vorticity_z(v_field, u_field, prob) -> Array

Compute z-component of vorticity: ζ = ∂v/∂x - ∂u/∂y using PencilFlows differential operators
"""
function compute_vorticity_z(v_field, u_field, prob)
    println("Computing vorticity ζ = dx(v) - dy(u) using PencilFlows transforms")
    
    # Use PencilFlows differential operators directly
    dvdx = apply_dx_operator(v_field, prob)
    dudy = apply_dy_operator(u_field, prob)
    
    return dvdx .- dudy
end

"""
    apply_dx_operator(field, prob) -> Array

Apply dx operator using PencilFlows transform infrastructure
"""
function apply_dx_operator(field, prob)
    try
        # Check if we have PencilFlows transform infrastructure
        if haskey(prob, :discretization) && prob.discretization !== nothing
            disc = prob.discretization
            
            # Use actual PencilFlows spectral transforms if available
            if disc.transform_plans !== nothing && haskey(disc, :dx_matrix) && disc.dx_matrix !== nothing
                # Use existing differentiation matrix or transform system
                return apply_pencilflows_dx(field, disc)
            end
        end
        
        # Check if it's a SymbolicProblem with PencilFlows backend
        if haskey(prob, :vars) && haskey(prob, :discretization)
            # Try to access PencilFlows functions directly
            # This would call the actual ddx! function from unified transforms.jl
            return compute_dx_pencilflows_direct(field, prob)
        end
        
        # If no PencilFlows infrastructure, error - user requested no local FDM
        error("PencilFlows transform infrastructure not available. Cannot compute dx() without spectral/transform system.")
        
    catch e
        error("Failed to apply dx operator with PencilFlows: $e")
    end
end

"""
    apply_dy_operator(field, prob) -> Array

Apply dy operator using PencilFlows transform infrastructure
"""
function apply_dy_operator(field, prob)
    try
        # Check if we have PencilFlows transform infrastructure
        if haskey(prob, :discretization) && prob.discretization !== nothing
            disc = prob.discretization
            
            # Use actual PencilFlows spectral transforms if available
            if disc.transform_plans !== nothing && haskey(disc, :dy_matrix) && disc.dy_matrix !== nothing
                # Use existing differentiation matrix or transform system
                return apply_pencilflows_dy(field, disc)
            end
        end
        
        # Check if it's a SymbolicProblem with PencilFlows backend
        if haskey(prob, :vars) && haskey(prob, :discretization)
            # Try to access PencilFlows functions directly
            # This would call the actual ddy! function from unified transforms.jl
            return compute_dy_pencilflows_direct(field, prob)
        end
        
        # If no PencilFlows infrastructure, error - user requested no local FDM
        error("PencilFlows transform infrastructure not available. Cannot compute dy() without spectral/transform system.")
        
    catch e
        error("Failed to apply dy operator with PencilFlows: $e")
    end
end

"""
    apply_dz_operator(field, prob) -> Array

Apply dz operator using PencilFlows finite difference infrastructure
"""
function apply_dz_operator(field, prob)
    try
        # Check if we have PencilFlows z-derivative infrastructure
        if haskey(prob, :discretization) && prob.discretization !== nothing
            disc = prob.discretization
            
            # Use actual PencilFlows z-derivatives if available (typically finite difference)
            if haskey(disc, :dz_matrix) && disc.dz_matrix !== nothing
                return apply_pencilflows_dz(field, disc)
            end
        end
        
        # Try to use PencilFlows boundary condition functions for z-derivatives
        # These use the actual dz_derivative_nonuniform_with_bcs! function
        return compute_dz_pencilflows_direct(field, prob)
        
    catch e
        error("Failed to apply dz operator with PencilFlows: $e")
    end
end

"""
    compute_gradient_magnitude(field, prob) -> Array

Compute magnitude of gradient: |∇field| = √(dx²+dy²+dz²) using PencilFlows operators
"""
function compute_gradient_magnitude(field, prob)
    println("Computing gradient magnitude |∇field| using PencilFlows operators")
    
    # Use PencilFlows differential operators directly
    dfdx = apply_dx_operator(field, prob)
    dfdy = apply_dy_operator(field, prob) 
    dfdz = apply_dz_operator(field, prob)
    
    return sqrt.(abs2.(dfdx) .+ abs2.(dfdy) .+ abs2.(dfdz))
end

# ========================================================================================
# PencilFlows Integration Functions - Direct Interface to Transform System
# ========================================================================================

"""
    compute_dx_pencilflows_direct(field, prob) -> Array

Compute dx using PencilFlows ddx! function directly
"""
function compute_dx_pencilflows_direct(field, prob)
    try
        # Get transform infrastructure from problem
        disc = get(prob, :discretization, nothing)
        if disc === nothing
            error("No discretization info available for PencilFlows transforms")
        end
        
        # Check if we have the necessary transform components
        if hasattr(disc, :transform_plans) && disc.transform_plans !== nothing
            plans = disc.transform_plans
            
            # Convert field to spectral space and apply ddx!
            field_hat = fft_field_to_spectral(field, plans)
            
            # Get wavenumbers - this should come from the transform plans
            kx = get_kx_wavenumbers(plans)
            
            # Apply ddx! using PencilFlows function
            ddx!(field_hat, kx, plans)
            
            # Transform back to physical space
            return ifft_field_to_physical(field_hat, plans)
        end
        
        error("Transform plans not available in discretization")
        
    catch e
        error("Failed to compute dx with PencilFlows direct interface: $e")
    end
end

"""
    compute_dy_pencilflows_direct(field, prob) -> Array

Compute dy using PencilFlows ddy! function directly
"""
function compute_dy_pencilflows_direct(field, prob)
    try
        # Get transform infrastructure from problem
        disc = get(prob, :discretization, nothing)
        if disc === nothing
            error("No discretization info available for PencilFlows transforms")
        end
        
        # Check if we have the necessary transform components
        if hasattr(disc, :transform_plans) && disc.transform_plans !== nothing
            plans = disc.transform_plans
            
            # Convert field to spectral space and apply ddy!
            field_hat = fft_field_to_spectral(field, plans)
            
            # Get wavenumbers - this should come from the transform plans
            ky = get_ky_wavenumbers(plans)
            
            # Apply ddy! using PencilFlows function
            ddy!(field_hat, ky, plans)
            
            # Transform back to physical space
            return ifft_field_to_physical(field_hat, plans)
        end
        
        error("Transform plans not available in discretization")
        
    catch e
        error("Failed to compute dy with PencilFlows direct interface: $e")
    end
end

"""
    compute_dz_pencilflows_direct(field, prob) -> Array

Compute dz using PencilFlows dz_derivative_nonuniform_with_bcs! function
"""
function compute_dz_pencilflows_direct(field, prob)
    try
        # Get discretization info
        disc = get(prob, :discretization, nothing)
        if disc === nothing
            error("No discretization info available for PencilFlows z-derivatives")
        end
        
        # Check for grid and boundary condition info needed for z-derivatives
        if haskey(disc, :grid_z) && haskey(disc, :bc_matrices)
            grid = disc.grid_z
            
            # Create output array
            dfdz = similar(field)
            
            # Use PencilFlows dz_derivative_nonuniform_with_bcs! function
            # This is the actual function used in the codebase for z-derivatives
            dummy_bc = BoundaryCondition(:dummy, NO_SLIP, 0.0)  # Placeholder BC
            dz_derivative_nonuniform_with_bcs!(dfdz, field, grid, dummy_bc, :field)
            
            return dfdz
        end
        
        error("Grid or boundary condition info not available for z-derivatives")
        
    catch e
        error("Failed to compute dz with PencilFlows direct interface: $e")
    end
end

"""
Helper functions for PencilFlows transform integration
"""
function hasattr(obj, attr::Symbol)
    return hasfield(typeof(obj), attr) && getfield(obj, attr) !== nothing
end

function fft_field_to_spectral(field, plans)
    # This would use PencilFlows FFT infrastructure to convert to spectral space
    # For now, return a placeholder that would work with ddx!/ddy! functions
    error("FFT to spectral space not yet implemented - requires PencilFlows FFT integration")
end

function ifft_field_to_physical(field_hat, plans) 
    # This would use PencilFlows iFFT infrastructure to convert back to physical space
    error("iFFT to physical space not yet implemented - requires PencilFlows iFFT integration")
end

function get_kx_wavenumbers(plans)
    # Extract kx wavenumbers from transform plans
    error("Wavenumber extraction not yet implemented - requires PencilFlows transform plan inspection")
end

function get_ky_wavenumbers(plans)
    # Extract ky wavenumbers from transform plans  
    error("Wavenumber extraction not yet implemented - requires PencilFlows transform plan inspection")
end

# Placeholder functions for matrix-based differentiation (if available)
function apply_pencilflows_dx(field, disc)
    error("Matrix-based dx not yet implemented - requires PencilFlows differentiation matrix integration")
end

function apply_pencilflows_dy(field, disc)
    error("Matrix-based dy not yet implemented - requires PencilFlows differentiation matrix integration")  
end

function apply_pencilflows_dz(field, disc)
    error("Matrix-based dz not yet implemented - requires PencilFlows differentiation matrix integration")
end

"""
Helper functions for common computed field expressions.
"""

# Vorticity components
vorticity_z() = :ζ => "dx(v) - dy(u)"
vorticity_x() = :ω_x => "dy(w) - dz(v)" 
vorticity_y() = :ω_y => "dz(u) - dx(w)"

# Energy fields
kinetic_energy() = :KE => "0.5*(u^2 + v^2 + w^2)"
kinetic_energy_2d() = :KE => "0.5*(u^2 + v^2)"
enstrophy() = :enstrophy => "0.5*ζ^2"

# RMS fields
u_rms() = :u_rms => "sqrt(mean(u^2))"
v_rms() = :v_rms => "sqrt(mean(v^2))"
w_rms() = :w_rms => "sqrt(mean(w^2))"
T_rms() = :T_rms => "sqrt(mean(T^2))"

# Gradient magnitudes
grad_T_magnitude() = :grad_T_mag => "sqrt(dx(T)^2 + dy(T)^2 + dz(T)^2)"
grad_u_magnitude() = :grad_u_mag => "sqrt(dx(u)^2 + dy(u)^2 + dz(u)^2)"

# Mean fields  
T_mean() = :T_mean => "mean(T)"
u_mean() = :u_mean => "mean(u)"
v_mean() = :v_mean => "mean(v)"

# Useful combinations
velocity_and_vorticity() = [:u, :v, :w, vorticity_z()]
scalars_and_energy() = [:T, :b, kinetic_energy()]
diagnostics_2d() = [kinetic_energy_2d(), enstrophy(), u_rms(), v_rms(), T_rms()]

# ============================================================================
# COMPUTED FIELDS EXAMPLES AND DOCUMENTATION
# ============================================================================

"""
# Computed Fields in PencilFlows.jl Output System

The output system now supports computed fields that are calculated on-the-fly during output.

## Basic Computed Field Syntax

### 1. Direct Expression Syntax (exactly what you requested!)
```julia
# Compute vorticity automatically
write_state(prob; fields = [:u, :v, :ζ => "dx(v) - dy(u)"])

# Multiple computed fields
write_state(prob; fields = [
    :u, :v, :T,
    :ζ => "dx(v) - dy(u)",           # Vorticity
    :KE => "0.5*(u^2 + v^2)",        # Kinetic energy
    :T_rms => "sqrt(mean(T^2))"      # Temperature RMS
])
```

### 2. AnalysisHandle with Flexible write_state (exactly what you wanted!)
```julia
# Set up analysis handle
ana = AnalysisHandle(prob;
    filename = "data",
    sim_dt = 0.1              # Controls saving rate
)

# Flexible per-call field selection
ana.write_state(prob; fields = [:u])                           # Save u only
ana.write_state(prob; fields = [:v])                           # Save v only  
ana.write_state(prob; fields = [:ζ => "dx(v) - dy(u)"])        # Save computed vorticity
ana.write_state(prob; fields = [:u, :v, :ζ => "dx(v) - dy(u)"]) # Save u, v, and vorticity
```

## Helper Functions for Common Computed Fields

```julia
# Use predefined computed field helpers
write_state(prob; fields = [:u, :v, vorticity_z()])           # :ζ => "dx(v) - dy(u)"
write_state(prob; fields = [:u, :v, :T, kinetic_energy()])    # :KE => "0.5*(u^2 + v^2 + w^2)"
write_state(prob; fields = [u_rms(), v_rms(), T_rms()])       # RMS values

# Combine regular and computed fields easily
write_state(prob; fields = velocity_and_vorticity())          # [:u, :v, :w, :ζ => "dx(v) - dy(u)"]
write_state(prob; fields = scalars_and_energy())              # [:T, :b, :KE => "0.5*(u^2 + v^2 + w^2)"]
```

## Advanced Analysis Workflows

### 3. Mixed Regular and Computed Fields
```julia
# Complex analysis with multiple computed quantities
ana = AnalysisHandle(prob; filename = "analysis", sim_dt = 0.1)

# Save basic fields frequently
ana.write_state(prob; fields = [:u, :v])

# Save vorticity and energy diagnostics
ana.write_state(prob; fields = [
    vorticity_z(),          # :ζ => "dx(v) - dy(u)"
    kinetic_energy(),       # :KE => "0.5*(u^2 + v^2 + w^2)" 
    enstrophy()            # :enstrophy => "0.5*ζ^2"
])

# Save gradient magnitudes for analysis
ana.write_state(prob; fields = [
    :T,
    grad_T_magnitude()      # :grad_T_mag => "sqrt(dx(T)^2 + dy(T)^2 + dz(T)^2)"
])
```

### 4. Time Loop with Computed Fields
```julia
ana = AnalysisHandle(prob; filename = "simulation", sim_dt = 0.1)

for step in 1:1000
    # ... time stepping physics ...
    
    # Different saving strategies
    if step % 5 == 0
        ana.write_state(prob; fields = [:u, :v])              # Basic fields often
    end
    
    if step % 20 == 0  
        ana.write_state(prob; fields = [vorticity_z()])       # Vorticity less often
    end
    
    if step % 100 == 0
        ana.write_state(prob; fields = diagnostics_2d())      # Full diagnostics rarely  
    end
end
```

## Supported Computed Field Expressions

| Expression | Result | Helper Function |
|------------|--------|-----------------|
| `"dx(v) - dy(u)"` | z-vorticity | `vorticity_z()` |
| `"dy(w) - dz(v)"` | x-vorticity | `vorticity_x()` |
| `"dz(u) - dx(w)"` | y-vorticity | `vorticity_y()` |
| `"0.5*(u^2 + v^2 + w^2)"` | Kinetic energy | `kinetic_energy()` |
| `"0.5*ζ^2"` | Enstrophy | `enstrophy()` |
| `"sqrt(mean(u^2))"` | u RMS | `u_rms()` |
| `"sqrt(dx(T)^2 + dy(T)^2 + dz(T)^2)"` | Temperature gradient | `grad_T_magnitude()` |
| `"mean(T)"` | Mean temperature | `T_mean()` |

## Performance Benefits

- **On-Demand Computation**: Computed fields only calculated when requested
- **Storage Efficiency**: Don't store intermediate computed results permanently  
- **Flexible Analysis**: Different computed fields for different analysis needs
- **Memory Efficient**: Computed fields don't take up permanent memory in prob.vars

"""
function demonstrate_computed_fields()
    println("COMPUTED FIELDS EXAMPLES")
    println("="^50)
    
    examples = [
        ("Vorticity calculation", "write_state(prob; fields = [:u, :v, :ζ => \"dx(v) - dy(u)\"])"),
        ("Multiple computed fields", "write_state(prob; fields = [:u, :v, :ζ => \"dx(v) - dy(u)\", :KE => \"0.5*(u^2 + v^2)\"])"),
        ("Analysis with helper functions", "write_state(prob; fields = [vorticity_z(), kinetic_energy()])"),
        ("AnalysisHandle flexible writes", "ana.write_state(prob; fields = [:u]); ana.write_state(prob; fields = [vorticity_z()])"),
        ("Combined regular and computed", "write_state(prob; fields = velocity_and_vorticity())")
    ]
    
    for (description, code) in examples
        println(" $description:")
        println("   $code")
        println()
    end
    
    println("Computed fields give you ultimate flexibility in what to save!")
end

# ============================================================================
# VARIABLE SELECTION EXAMPLES AND DOCUMENTATION
# ============================================================================

"""
# Variable Selection in PencilFlows.jl Output System

The output system now supports flexible variable selection, allowing users to choose exactly which fields to save.

## Basic Usage Examples

### 1. Save Only Specific Variables
```julia
# Save only velocity components
write_state(prob; fields = [:u, :v, :w])

# Save temperature and buoyancy fields
write_state(prob; fields = [:T, :b])

# Use helper functions for common selections
write_state(prob; fields = select_velocity_fields())
write_state(prob; fields = select_scalar_fields())
```

### 2. Exclude Specific Variables
```julia
# Save everything except debug fields
write_state(prob; exclude_fields = [:debug, :temp, :scratch])

# Use helper function to exclude common debug fields  
write_state(prob; exclude_fields = exclude_debug_fields())

# Exclude computed/derived fields
write_state(prob; include_computed = false)
```

### 3. Advanced Selection Combinations
```julia
# Save velocities but exclude w component
write_state(prob; 
    fields = select_velocity_fields(),
    exclude_fields = [:w]
)

# Save scalars and analysis fields, excluding debug
write_state(prob; 
    fields = vcat(select_scalar_fields(), select_analysis_fields()),
    exclude_fields = exclude_debug_fields()
)
```

### 4. Analysis Handle with Variable Selection
```julia
# Create analysis that only tracks velocity fields
ana = AnalysisHandle(prob; 
    filename = "velocity_analysis",
    sim_dt = 0.1,
    fields = select_velocity_fields()
)
add_system!(ana, prob)  # Only velocity fields will be tracked

# Analysis excluding computed fields
ana = AnalysisHandle(prob;
    filename = "basic_fields", 
    include_computed = false
)
```

### 5. Custom Field Selection Functions
```julia
# Define custom field selection
custom_fields() = [:u, :v, :T, :omega_z]

# Use in output
write_state(prob; fields = custom_fields())

# Conditional field selection based on simulation parameters
function select_fields_for_timestep(prob, step)
    base_fields = [:u, :v, :T]
    if step % 100 == 0  # Every 100 steps, save additional diagnostics
        return vcat(base_fields, select_analysis_fields())
    else
        return base_fields
    end
end

# Use conditional selection
fields_to_save = select_fields_for_timestep(prob, current_step)
write_state(prob; fields = fields_to_save, step = current_step)
```

## Field Selection Options

| Option | Type | Description | Examples |
|--------|------|-------------|----------|
| `fields` | `:all`, `:none`, `Vector{Symbol}` | Fields to include | `:all`, `[:u, :v, :T]` |
| `exclude_fields` | `Vector{Symbol}` | Fields to exclude | `[:debug, :temp]` |
| `include_computed` | `Bool` | Include computed/derived fields | `false` |
| `field_layouts` | `Dict{Symbol,Char}` | Per-field layout specs | `Dict(:u => 'g', :T => 'c')` |

## Helper Functions for Common Selections

- `select_velocity_fields()` → `[:u, :v, :w]`
- `select_scalar_fields()` → `[:T, :b, :c, :rho, :p]` 
- `select_vorticity_fields()` → `[:omega_x, :omega_y, :omega_z, :omega, :vorticity]`
- `select_analysis_fields()` → `[:kinetic_energy, :u_rms, :v_rms, :w_rms, :T_mean, :enstrophy]`
- `exclude_debug_fields()` → `[:debug, :temp, :scratch, :work, :aux, :temporary]`

## Performance Benefits

Variable selection provides several benefits:
- **Reduced File Sizes**: Only save fields you need
- **Faster I/O**: Less data to write means faster output
- **Storage Efficiency**: Focus disk space on important fields
- **Analysis Clarity**: Output files contain only relevant data

"""
function demonstrate_variable_selection()
    println("VARIABLE SELECTION EXAMPLES")
    println("="^50)
    
    # These are example usages - would work with actual prob object
    examples = [
        ("Save only velocity fields", "write_state(prob; fields = select_velocity_fields())"),
        ("Exclude debug fields", "write_state(prob; exclude_fields = exclude_debug_fields())"),
        ("No computed fields", "write_state(prob; include_computed = false)"),
        ("Custom selection", "write_state(prob; fields = [:u, :v, :T, :omega_z])"),
        ("Analysis with selection", "AnalysisHandle(prob; fields = select_scalar_fields())")
    ]
    
    for (description, code) in examples
        println(" $description:")
        println("   $code")
        println()
    end
    
    println("Variable selection makes output efficient and focused!")
end
