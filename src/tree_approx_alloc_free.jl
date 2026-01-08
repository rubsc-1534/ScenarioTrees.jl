

#############################################################
"""
    tree_approximation!(
        tree,
        path,
        nIterations;
        batchsize=32,
        p=2,
        r=2,
    )
"""
function tree_approximation_alloc!(
    tree::Tree,
    f,
    nIterations::Int;
    batchsize::Int = 32,
    p::Int = 2,
    r::Int = 2,
)
    T = height(tree)
    nleaf = size(tree.state, 1)

    # ---- Preallocation ----
    probaLeaf = zeros(Float64, nleaf)
    d = zeros(Float64, nleaf)
    samplepaths = Array{Float64}(undef, batchsize, T + 1)
    
    # ---- Main loop ----
    nBatches = cld(nIterations, batchsize)
    Threads.@threads for x = 1:nBatches
        if (rem(x,10,)==0)
		        print("Progress: $(round(x/nBatches*100,digits=2))%   \r")
		    flush(stdout)
	    end
        # Generate batch
        @inbounds for b = 1:batchsize
                @views f(samplepaths[b, :])
        end

        # Apply SA steps
        @inbounds for b = 1:batchsize
            sa_step!(
                tree,
                view(samplepaths, b, :),
                probaLeaf,
                d,
                p,
                r,
            )
        end
    end

    # ---- Finalization ----
    probaLeaf = probaLeaf[get_leaves(tree.structure)[1]]
    probabilities = probaLeaf ./ sum(probaLeaf)
    d = d[get_leaves(tree.structure)[1]]
    t_dist = (d' * probabilities / nIterations).^(1 / r)

    tree.name = "$(tree.name) with d=$(t_dist) at $(nIterations) iterations"
    tree.p_edge .= build_probabilities!(tree, probabilities)
    tree.dist = t_dist[1]

    return tree
end

##################################################################################################
# sa_step update
##################################################################################################

"""
    sa_step!(
        tree,
        samplepath,
        leaf,
        path_to_leaves,
        path_to_all_nodes,
        probaLeaf,
        d,
        p,
        r,
    )

Single stochastic approximation update for one sample path.
"""
function sa_step!(trr::Tree, samplepath::SubArray{Float64}, 
    probaLeaf::Vector{Float64},
    d::Vector{Float64}, p::Int, r::Int) 
       
    T = size(samplepath, 1) - 1
    offs = trr.PathBundle.path_nodes_offset
    P_nodes = trr.PathBundle.path_nodes
    A = samplepath
    B = trr.state
        
    # ---- Projection step (tree traversal) ---- 
    endleaf = 1  
    @inbounds for t = 2:T+1
        children = trr.structure.children[endleaf]
        bestdist = Inf
        bestchild = endleaf
        
        

        @inbounds for i in children
            @inbounds begin
                s = 0.0
                τ = 1
                for k = offs[i]:(offs[i+1]-1)
                    node = P_nodes[k]
                    @simd for j in axes(A, 2)
                        dumb = A[τ, j] - B[node, j]
                        s += dumb*dumb
                    end
                    τ += 1
                end
            end
            if s < bestdist
                bestdist = s
                bestchild = i
            end
        end
        endleaf = bestchild
    end

    #---- Probability update ---- 
    probaLeaf[endleaf] += 1.0 
    
    # ---- Stochastic gradient step ---- 
    @inbounds begin
        # compute distance^p
        dist_p = 0.0
        τ = 1
        for k = offs[endleaf]:(offs[endleaf+1]-1)
            node = P_nodes[k]
            @simd for j in axes(A, 2)
                x = B[node, j] - A[τ, j]
                dist_p += abs(x)^p
            end
            τ += 1
        end

        dist = dist_p^(1/p)
        d[endleaf] += dist^r

        ak = 1.0 / (30.0 + probaLeaf[endleaf])
        coeff = r * dist^(r - p) * ak

        # gradient update
        τ = 1
        for k = offs[endleaf]:(offs[endleaf+1]-1)
            node = P_nodes[k]
            @simd for j in axes(A, 2)
                x = B[node, j] - A[τ, j]
                B[node, j] -= coeff * abs(x)^(p - 1) * sign(x)
            end
            τ += 1
        end
    end
    trr.state=B
    return nothing 
end
