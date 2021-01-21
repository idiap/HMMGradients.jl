using Documenter #, DocumenterLaTeX
using HMMGradients

makedocs(
    sitename = "HMMGradients",
    format = [Documenter.HTML()],#, DocumenterLaTeX.LaTeX()],
    authors = "Niccolò Antonello",
    modules = [HMMGradients],
    pages = [
        "Home" => "index.md",
        "Theory and Notation" => "1_intro.md",
        "Forward-Backward functions" => "2_fb.md",
        "Computing gradients" => "3_grads.md",
        "Demo" => "4_demo.md",
    ],
    doctest=false,
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/idiap/HMMGradients.jl.git",
    devbranch = "main",
)
