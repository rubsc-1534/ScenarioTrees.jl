module ScenarioTrees

using CairoMakie
using DataFrames
using Distributions, Statistics, Random
rng = MersenneTwister(01012019);

include("TreeStructure.jl")

include("StochPaths.jl")

include("TreeApprox.jl")
include("tree_approx_alloc_free.jl")
include("tree_approx_alloc_free_buffer.jl")

export tree_approximation!,lattice_approximation,
        Tree, tree_approximation_alloc!,tree_approximation_alloc_buf!


#        stage,height,leaves,nodes,root,
#        part_tree,build_probabilities!,
#        gaussian_path1D,gaussian_path2D,
#        running_maximum1D,running_maximum2D,path,kernel_scenarios, checkTree, 
#        tree_path,
#        tree_plot, plot_hd, plot_lattice



end
