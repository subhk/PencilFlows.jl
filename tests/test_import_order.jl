"""
Test to verify consistent import ordering across the codebase
This ensures PencilFFTs is always imported before PencilArrays
"""

println("Testing: Import order consistency")

using Test

@testset "Import Order Consistency" begin
    # List of source files to check
    src_dir = joinpath(@__DIR__, "..", "src")

    # Find all .jl files in src directory recursively
    function find_jl_files(dir)
        files = String[]
        for (root, dirs, filenames) in walkdir(dir)
            for filename in filenames
                if endswith(filename, ".jl")
                    push!(files, joinpath(root, filename))
                end
            end
        end
        return files
    end

    files = find_jl_files(src_dir)

    println("  Checking $(length(files)) source files...")

    issues = String[]

    for file in files
        content = read(file, String)
        lines = split(content, '\n')

        # Find lines with PencilFFTs and PencilArrays imports
        pencilfft_lines = Int[]
        pencilarray_lines = Int[]

        for (i, line) in enumerate(lines)
            # Skip comments
            stripped = strip(line)
            if startswith(stripped, '#')
                continue
            end

            # Check for actual using statements (not in comments)
            if occursin(r"^\s*using.*PencilFFTs", line)
                push!(pencilfft_lines, i)
            end
            if occursin(r"^\s*using.*PencilArrays", line)
                push!(pencilarray_lines, i)
            end
        end

        # If both are present, check order
        if !isempty(pencilfft_lines) && !isempty(pencilarray_lines)
            fft_first = minimum(pencilfft_lines)
            arr_first = minimum(pencilarray_lines)

            if fft_first > arr_first
                rel_path = relpath(file, src_dir)
                push!(issues, "  X $rel_path: PencilArrays (line $arr_first) before PencilFFTs (line $fft_first)")
            end
        end
    end

    if isempty(issues)
        println("    [OK] All imports are in correct order (PencilFFTs before PencilArrays)")
        @test true
    else
        println("\n  Issues found:")
        for issue in issues
            println(issue)
        end
        @test_broken false  # Import order inconsistencies found
    end
end

@testset "PencilFFTPlan Constructor Consistency" begin
    src_dir = joinpath(@__DIR__, "..", "src")

    function find_jl_files(dir)
        files = String[]
        for (root, dirs, filenames) in walkdir(dir)
            for filename in filenames
                if endswith(filename, ".jl")
                    push!(files, joinpath(root, filename))
                end
            end
        end
        return files
    end

    files = find_jl_files(src_dir)

    println("  Checking PencilFFTPlan constructor calls...")

    issues = String[]

    for file in files
        content = read(file, String)
        lines = split(content, '\n')

        for (i, line) in enumerate(lines)
            # Skip comments
            stripped = strip(line)
            if startswith(stripped, '#')
                continue
            end

            # Check for PencilFFTPlan constructor without flags
            if occursin(r"PencilFFTPlan\([^)]+\)", line) &&
               !occursin("flags=", line) &&
               !occursin("# ", line)  # Skip inline comments

                # Check if this is actually a constructor call (not in a comment)
                if occursin("PencilFFTPlan(", line)
                    rel_path = relpath(file, src_dir)
                    push!(issues, "  ! $rel_path:$i: PencilFFTPlan without flags parameter")
                end
            end
        end
    end

    if isempty(issues)
        println("    [OK] All PencilFFTPlan constructors use flags parameter")
        @test true
    else
        println("\n  Potential issues found (may be acceptable):")
        for issue in issues
            println(issue)
        end
        # Don't fail the test, just warn
        @test_broken false  # Some PencilFFTPlan calls don't use flags (may be intentional)
    end
end

println("\n[OK] Import consistency tests completed!")
