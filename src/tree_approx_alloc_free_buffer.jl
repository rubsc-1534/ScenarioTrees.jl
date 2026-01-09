mutable struct SABuffer
    leaf_count :: Vector{Float64}   # size = nleaf
    d_accum    :: Vector{Float64}   # size = nleaf #accumulated distance in buffer
    newGrad    :: Matrix{Float64}   # state_dim × dim
    used
end



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
function tree_approximation_alloc_buf!(tree::Tree,f,nIterations::Int;batchsize::Int = 32,p::Int = 2,r::Int = 2)
    T = height(tree)
    nleaf = size(tree.state, 1)

    # ---- Preallocation ----
    probaLeaf = zeros(Float64, nleaf)
    d = zeros(Float64, nleaf)
    samplepaths = Array{Float64}(undef, batchsize, T + 1)
    nBatches = cld(nIterations, batchsize)

    # ---- Create thread-local buffers ----
    nthreads = Threads.nthreads()
    nbuf = Threads.maxthreadid()


    buffers = [
        SABuffer(
            zeros(Float64, nleaf),              # leaf_count
            zeros(Float64, nleaf),              # d_accum
            zeros(Float64,size(tree.state)),    # gradient
            false                               # 
        )
        for _ in 1:nbuf #nthreads
    ]

    # ---- Main loop ----
    

    Threads.@threads for x = 1:nBatches
        tid = Threads.threadid()
        buf = buffers[tid]
        buf.used = true

        # optional progress output (safe enough)
        if tid == 1 && x % 10 == 0
            print("Progress: $(round(x/nBatches*100, digits=2))%   \r")
            flush(stdout)
        end

        # ---- Generate batch ----
        @inbounds for b = 1:batchsize
            @views f(samplepaths[b, :])
        end

        # ---- SA steps (buffered) ----
        for b = 1:batchsize #add inbounds later after fixing segfaults
            sa_step_buffered!(tree,view(samplepaths, b, :),buf,p,r)
        end
    end

    # ---- Reduction step (single-threaded, safe) ----
    apply_buffers!(tree,buffers,probaLeaf,d)

    # ---- Finalization ----
    leaf_idx = get_leaves(tree.structure)[1]

    probaLeaf = probaLeaf[leaf_idx]
    probabilities = probaLeaf ./ sum(probaLeaf)

    d = d[leaf_idx]
    t_dist = (d' * probabilities / nIterations).^(1 / r)

    tree.name = "$(tree.name) with d=$(t_dist) at $(nIterations) iterations"
    tree.p_edge .= build_probabilities!(tree, probabilities)
    tree.dist = t_dist[1]

    return tree
end

#################################################
# sa step update (Buffered version)
#################################################
function sa_step_buffered!(
    trr::Tree,
    samplepath::SubArray{Float64},   # (T+1) × dim
    buf::SABuffer,
    p::Int,
    r::Int,
)
    T = size(samplepath, 1) - 1
    dim = size(samplepath, 2)

    offs    = trr.PathBundle.path_nodes_offset
    Pnodes  = trr.PathBundle.path_nodes
    state   = trr.state
    children = trr.structure.children

    A = samplepath
    B = state

    # -------------------------------
    # Projection step (tree traversal)
    # -------------------------------
    endleaf = 1

    for t = 2:T+1 #add inbounds later after fixing segfaults    
        bestdist  = Inf
        bestchild = endleaf

        for child in children[endleaf]
            s = 0.0
            τ = 1

            for k = offs[child]:(offs[child+1]-1)
                node = Pnodes[k]
                @simd for j = 1:dim
                    x = A[τ, j] - B[node, j]
                    s += x * x
                end
                τ += 1
            end

            if s < bestdist
                bestdist  = s
                bestchild = child
            end
        end

        endleaf = bestchild
    end

    # -------------------------------
    # Probability update (buffered)
    # -------------------------------
    buf.leaf_count[endleaf] += 1.0

    # -------------------------------
    # Distance and gradient
    # -------------------------------
    dist_p = 0.0
    τ = 1

    pathlen = offs[endleaf+1] - offs[endleaf]

    idx = 1
    for k = offs[endleaf]:(offs[endleaf+1]-1)
        node = Pnodes[k]
        buf.path_nodes[idx] = node

        @simd for j = 1:dim
            x = B[node, j] - A[τ]
            #buf.grad[idx, j] = x        #grad update is saved per path with idx of length of path, this is why buffer for path is needed but still wrong
            buf.newGrad[node,j] = x
            dist_p += abs(x)^p
        end

        idx += 1
        τ += 1
    end

    dist = dist_p^(1/p)
    buf.d_accum[endleaf] += dist^r

    # -------------------------------
    # Gradient scaling (buffered)
    # -------------------------------
    ak    = 1.0 / (30.0 + buf.leaf_count[endleaf])
    coeff = r * dist^(r - p) * ak

    for k = offs[endleaf]:(offs[endleaf+1]-1)  #add inbounds later after fixing segfaults  
        @simd for j = 1:dim
            node = Pnodes[k]
            x = buf.newGrad[node, j]
            buf.newGrad[node, j] = -coeff * abs(x)^(p - 1) * sign(x)
        end
    end

    #Gradient needs to be size of state. This is basically a resize
    #nodes = Pnodes[offs[endleaf]:(offs[endleaf+1]-1)]
    #buf.newGrad[nodes,:] = buf.grad[1:pathlen,:]

    #buf.newGrad = stateLike[path,dim] = buf.grad[:,dim] #something like this is needed to not save the path in the struct

    return nothing
end


function apply_buffers!(
    tree::Tree,
    buffers::Vector{SABuffer},
    probaLeaf::Vector{Float64},
    d::Vector{Float64},
)
    B = tree.state
    buffers = filter(buf -> buf.used, buffers)
    for buf in buffers
        probaLeaf .+= buf.leaf_count
        d         .+= buf.d_accum
        
        #currently buf.path_nodes saves only the first path ever; current setup requieres saving all paths (BAD!)
        for i = 1:length(buf.path_nodes) #add inbounds later after fixing segfaults  
            node = buf.path_nodes[i]
            @simd for j in axes(B,2)
                B[node, j] += buf.grad[i, j]
            end
        end

        fill!(buf.leaf_count, 0.0)
        fill!(buf.d_accum, 0.0)
    end

    return nothing
end
