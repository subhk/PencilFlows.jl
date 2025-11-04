using Documenter
using PencilFlows

makedocs(
    sitename = "PencilFlows.jl",
    authors = "Subhajit Kar",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://subhk.github.io/PencilFlows.jl",
        assets = String[],
        sidebar_sitename = true,
    ),
    modules = [PencilFlows],
    pages = [
        "Home" => "index.md",
        "Getting Started" => [
            "Installation" => "installation.md",
            "Quick Start" => "quickstart.md",
        ],
        "User Guide" => [
            "Core Concepts" => "guide/concepts.md",
            "Boundary Conditions" => "guide/boundary_conditions.md",
            "MPI Parallelization" => "guide/mpi.md",
            "Poisson Solvers" => "guide/poisson.md",
            "Time Stepping" => "guide/timestepping.md",
        ],
        "Examples" => [
            "Basic Flow Simulation" => "examples/basic_flow.md",
            "Parallel Computing" => "examples/parallel.md",
            "Custom Boundary Conditions" => "examples/custom_bc.md",
        ],
        "API Reference" => [
            "Core Functions" => "api/core.md",
            "Solvers" => "api/solvers.md",
            "I/O" => "api/io.md",
            "Physics" => "api/physics.md",
            "Utilities" => "api/utilities.md",
        ],
    ],
    doctest = false,
    checkdocs = :none,
)

deploydocs(
    repo = "github.com/subhk/PencilFlows.jl.git",
    devbranch = "main",
    push_preview = true,
    versions = ["stable" => "v^", "v#.#", "dev" => "main"],
)
