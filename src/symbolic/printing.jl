# Pretty printing and progress display helpers for the symbolic interface

function pencilflow_header()
    header = """
    ================================================================================================================
      ____                _ _ _____ _                 _ _        
     |  _ \\ ___ _ __   ___(_) |  ___| | _____      __| | |       
     | |_) / _ \\ '_ \\ / __| | | |_  | |/ _ \\ \\/ / / /| | |       
     |  __/  __/ | | | (__| | |  _| | | (_) \\ V  V / | | |       
     |_|   \\___|_| |_|\\___|_|_|_|   |_|\\___/ \\_/\\_(_)|_|_|       
                                                                 
               Pseudospectral PDE Solver for Julia                
            Built on PencilArrays & PencilFFTs for HPC            
                                                                 
    ================================================================================================================
    """
    printstyled(header, color=:cyan, bold=true)
    println()
    pencilflow_banner()
end

function pencilflow_banner()
    banner = """
    ------------------------------------------------------------
      Symbolic PDE Interface  |  Automatic Parameter System
      Domain + Equations + BCs  ->  Discretization + Solve
    ------------------------------------------------------------
    """
    printstyled(banner, color=:blue, bold=true)
end

const BANNER_SHOWN = Ref(false)

function ensure_banner_shown!(prob::SymbolicProblem)
    if !BANNER_SHOWN[]
        pencilflow_banner()
        BANNER_SHOWN[] = true
    end
end

function reset_banner!()
    BANNER_SHOWN[] = false
end

function equation_summary(prob::SymbolicProblem)
    printstyled("", color=:blue, bold=true)
    printstyled("                    EQUATION SYSTEM                         ", color=:yellow, bold=true)
    printstyled("\n", color=:blue, bold=true)
    # Fields
    if !isempty(prob.fields)
        field_names = [string(f.name) for f in prob.fields]
        printstyled(" Fields: ", color=:blue, bold=true)
        printstyled("$(join(field_names, ", "))", color=:green)
        println()
    end
    # Equations
    if !isempty(prob.equations)
        printstyled(" Governing Equations:", color=:blue, bold=true)
        println()
        for (i, eq) in enumerate(prob.equations)
            eq_str = string(eq)
            printstyled("   $i. ", color=:blue)
            printstyled(eq_str, color=:white)
            println()
        end
    end
    # BCs
    if !isempty(prob.boundary_conditions)
        printstyled(" Boundary Conditions:", color=:blue, bold=true)
        println()
        for bc in prob.boundary_conditions
            bc_str = "$(bc.location)($(bc.field)) = $(bc.value)"
            printstyled("    ", color=:blue)
            printstyled(bc_str, color=:magenta)
            println()
        end
    end
    # Params
    if !isempty(prob.parameters)
        printstyled(" Parameters:", color=:blue, bold=true)
        println()
        for (k,v) in prob.parameters
            param_str = "$(k) = $(v)"
            printstyled("    ", color=:blue)
            printstyled(param_str, color=:yellow)
            println()
        end
    end
    domain_summary(prob)
end

function domain_summary(prob::SymbolicProblem)
    printstyled("", color=:blue, bold=true)
    printstyled("                 COMPUTATIONAL DOMAIN                       ", color=:cyan, bold=true)
    printstyled("\n", color=:blue, bold=true)
    if prob.domain !== nothing
        for (i, basis) in enumerate(prob.domain.bases)
            coord_name = basis.name
            N = prob.domain.grid_points[i]
            interval = basis.interval
            interval_str = "$(interval[1]) .. $(interval[2])"
            basis_type = typeof(basis)
            printstyled(" ", color=:blue)
            printstyled("$(coord_name): ", color=:green, bold=true)
            printstyled("$basis_type", color=:yellow)
            printstyled(" on [$interval_str] with $N points", color=:white)
            println()
        end
    else
        for (coord, N) in prob.grid_points
            printstyled(" ", color=:blue)
            printstyled("$(coord): ", color=:green, bold=true)
            printstyled("$N points", color=:white)
            println()
        end
    end
end

function show_build_progress(stage::String, details::String="")
    timestamp = Dates.format(Dates.now(), "HH:MM:SS")
    printstyled("[$timestamp] ", color=:light_black)
    printstyled(">> ", color=:blue, bold=true)
    printstyled("$stage", color=:green, bold=true)
    if !isempty(details)
        printstyled(" - ", color=:light_black)
        printstyled(details, color=:white)
    end
    println()
end

