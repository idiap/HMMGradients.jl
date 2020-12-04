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
    ],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "https://github.com/idiap/HMMGradients.jl"
)
