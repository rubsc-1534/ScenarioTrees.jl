# Required packages: 
using Random
using Statistics
using Distributions
using CairoMakie
using Clustering
using DataFrames

include("..//src//TreeStructure.jl")
include("..//src//StochPaths.jl")
include("..//src//tree_approx_nested.jl")
include("..//src//trees_plot.jl")
########################################################
function simulate_pure_stochastic_longevity(MI_param)
    Random.seed!(MI_param.seed)
    N_steps = round(Int, MI_param.T / MI_param.dt) + 1
    time_grid = range(0, MI_param.T, length=N_steps)

    # 1. Simulate improvement rate phi_t (OU Process)
    phi = zeros(MI_param.N_paths, N_steps)
    phi[:, 1] .= MI_param.phi_0
    
    # Pre-generate standard normal noise
    Z = randn(MI_param.N_paths, N_steps - 1)
    
    for i in 2:N_steps
        dW = sqrt(MI_param.dt) .* Z[:, i-1]
        # Euler-Maruyama discretization for OU process
        phi[:, i] = phi[:, i-1] .+ MI_param.kappa .* (MI_param.phi_bar .- phi[:, i-1]) .* MI_param.dt .+ MI_param.sigma_phi .* dW
    end
    
    # 2. Cumulative Improvement Y_t = ∫ φ_s ds
    # Because there's no short-term noise, Y_t is purely the smooth integral of phi_t
    Y = cumsum(phi .* MI_param.dt, dims=2)
    
    return time_grid, phi, Y
end

# ============================================================
# TREE NESTED APPROXIMATION adjusted for this example
# ============================================================
function tree_nested_approx2!(trr::Tree,g,MI_test)
    stages = 1:height(trr)

    # Step size (Robbins–Monro)
    ak(k) = 0.2 / (k+30)^0.75

    # Initialize the root node
    trr.state[1] = 0.015
    nestedDistance = 0
    #go through the tree from beginning
    for s in stages
        nodes = get_nodes(trr.structure)[trr.structure.stage.==s-1]
        for k in nodes
            println("Evaluation node $k in stage $s")
            history = trr.state[root_path(trr.structure,Int32(k))] #provides states of path until node k as a vector
            f() = g(history,MI_test) #just defines the cond. sampling function

            children = trr.structure.children[k] #get indices for nodes being updated in this round
            b = length(children)    #number of nodes to be simulated in this round
            init = vec(kmeans(reshape(0.005*randn(100_000),1,100_000) .+last(history), b).centers)    #get good starting value based on clustering
            #update state and probability in one pass using SA algorithm with cluster starting values
            trr.state[children], trr.p_edge[children] = stochastic_nodes(f;b = b,nsteps = 1_000,ak = ak,init=init)
        end
    end
end


#define a conditional 1-step sampling function
function projection_step_wrapper(history::Vector{Float64},MI_test)
    if(length(history)==1)
        phi = 0.015;
    else
        phi = round(last(history))
    end
    phi_new = simulate_scenario_projection_step(phi,MI_test)
    return(phi_new)
end

function simulate_scenario_projection_step(phi_last,MI_test)
    Z = randn(1)
    dW = sqrt(MI_test.dt) * Z
    # Euler-Maruyama discretization for OU process
    phi = phi_last + MI_test.kappa * (MI_test.phi_bar - phi_last)*MI_test.dt + MI_test.sigma_phi*dW[1]
    return(phi)
end

#####################################
# Go from period qx to survivors
#####################################
"""
    compute_survivors(tree::Tree, qx_col::Int=1, N0::Float64=1000.0) -> Vector{Float64}

Computes the number of survivors (lx) at each node of the tree.
- `qx_col`: column index in `tree.state` containing the mortality rate qx
- `N0`: initial cohort size at the root (node 1)
"""
function compute_survivors(tree::Tree, qx_col::Int = 1, N0::Float64 = 1000.0)
    n_nodes = length(tree.structure.parent)
    survivors = zeros(Float64, n_nodes)
    # Root node initial population
    survivors[1] = N0
    # Traverse nodes in topological order
    for i in 2:n_nodes
        p = tree.structure.parent[i]
        # Mortality rate during the transition from parent node p
        qx = tree.state[p, qx_col]
        # Survivor propagation: lx_child = lx_parent * (1 - qx)
        survivors[i] = survivors[p] * (1.0 - qx)
    end
    return survivors
end


function tree_to_survivor_tree(tree::Tree, qx_col::Int = 1, N0::Float64 = 1000.0)
    survivors = compute_survivors(tree, qx_col, N0)
    # Create new state matrix with lx values
    new_state = reshape(survivors, :, 1)
    return Tree(
        tree.name * " [Survivors]",
        tree.structure,
        tree.PathBundle,
        new_state,
        copy(tree.p_edge),
        copy(tree.p_cum),
        tree.dist
    )
end
 
# ============================================================
# Risk measures
# ============================================================
function MeanSD(state,prob,beta,p)
    tmp = sum(state.*prob)
    mSD = tmp + beta * (sum(max.(state.-tmp,0).^p .* prob))^(1/p)
    return(mSD)
end
