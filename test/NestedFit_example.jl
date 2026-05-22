#Example
using CairoMakie
using DataFrames
using Distributions, Statistics, Random

include("..//src//TreeStructure.jl")
include("..//src//StochPaths.jl")
include("..//src//tree_approx_nested.jl")
include("..//src//trees_plot.jl")


using Clustering

trr = Tree(Int32[1,2,2,2,2])
tree_nested_approx!(trr::Tree,BMotion_sampler2)
tree_plot(trr)