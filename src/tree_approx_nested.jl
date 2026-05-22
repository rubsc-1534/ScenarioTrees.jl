#Implements Algorithm 6 of Pflug&Pichler Dynamic generation of scenario trees

#The general idea is as follows
#
#1.) provide tree with fixed branching structure
#2.) from root to last stage generate new samples and based on branching structure simulate new states
#    and transition probabilities (edge_prob). 


using Clustering
using LinearAlgebra
using Statistics
using CairoMakie

function tree_nested_approx!(trr::Tree,g)

    stages = 1:height(trr)

    # Step size (Robbins–Monro)
    ak(k) = 0.2 / (k+30)^0.75

    # Initialize the root node
    trr.state[1] = 0
    nestedDistance = 0
    #go through the tree from beginning
    for s in stages
        nodes = get_nodes(trr.structure)[trr.structure.stage.==s-1]
        for k in nodes
            println("Evaluation node $k in stage $s")
            history = trr.state[root_path(trr.structure,Int32(k))] #provides states of path until node k as a vector
            f() = g(history) #just defines the cond. sampling function

            children = trr.structure.children[k] #get indices for nodes being updated in this round
            b = length(children)    #number of nodes to be simulated in this round
            init = vec(kmeans(reshape(randn(100_000),1,100_000), b).centers)    #get good starting value based on clustering
            #update state and probability in one pass using SA algorithm with cluster starting values
            trr.state[children], trr.p_edge[children] = stochastic_nodes(f;b = b,nsteps = 1_000_000,ak = ak,init=init)
        end

    end
end



"""
    stochastic_nodes(f; b, nsteps, ak, init=nothing)

Input:
- f        : sampler () -> Vector{Float64}
- b        : number of nodes
- nsteps   : number of SA iterations
- ak       : step-size function ak(k)
- init     : optional initial nodes (d × b matrix)

Output:
- nodes    : d × b matrix of support points
- p        : probability vector (length b)
"""
function stochastic_nodes(f;b::Int,nsteps::Int,ak,init = nothing)
    # --- 1) Initialization ---
    if init === nothing
        x0 = f()
        d = length(x0)
        nodes = vcat([f() for _ in 1:b]...)  # sensible random start
    else
        nodes = copy(init)
        b = length(nodes)
    end

    counts = zeros(Int, b)

    # --- 2) Stochastic approximation loop ---
    for k in 1:nsteps
        ξ = f()
        a = ak(k)

        # find closest node
        dists = [abs.(ξ .- nodes[i]) for i in 1:b]
        i_star = argmin(dists)

        counts[i_star] += 1

        # gradient update
        bstar = nodes[i_star]
        delta = bstar .- ξ
        dist = (delta.^2)[1]

        if dist > 0
            #grad = r * delta #.* abs.(delta).^(s - 1) .* sign.(delta)
            nodes[i_star] -= a .* 2 .* dist * sign.(delta)[1] #grad
        end
    end

    # --- 3) Probabilities ---
    p = counts ./ sum(counts)

    return nodes, p
end



function BMotion_sampler2(history::Vector)
    if length(history) == 0
        return(randn())
    end


    return(last(history)+randn())
end

