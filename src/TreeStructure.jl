##############################################################################
# Tree Structure (immutable, cached)
##############################################################################

struct TreeStructure
    parent   :: Vector{Int32}                  # parent[i] = parent of node i (0 for root)
    children :: Vector{Vector{Int32}}          # children[i] = children of node i
    stage    :: Vector{Int32}                  # stage (depth) of each node
    leaves   :: Vector{Int32}                  # indices of leaf nodes
end


struct PathBundle
    path_nodes          :: Vector{Int32}          # path to all nodes contiguous
    path_nodes_offset   :: Vector{Int32}          # offset for path to all nodes contiguous
end



##############################################################################
# Tree Data (mutable, optimized for solvers)
##############################################################################

mutable struct Tree
    name        :: String
    structure   :: TreeStructure
    PathBundle  :: PathBundle
    state       :: Matrix{Float64}              # (n_nodes × dimension)
    p_edge      :: Vector{Float64}              # P(node | parent)
    p_cum       :: Vector{Float64}              # cumulative probability from root
    dist        :: Float64
end


##############################################################################
# Tree creation
##############################################################################
function Tree(bstructure::Vector{Int32}, dimension::Int = 1; rng = Random.GLOBAL_RNG)

    @assert bstructure[1] == 1 "Branching must start with 1 (root)"

    # ---- total number of nodes ----
    n_nodes = sum(cumprod(bstructure))

    parent   = zeros(Int32, n_nodes)
    stage    = zeros(Int32, n_nodes)
    children = [Int32[] for _ in 1:n_nodes]

    # ---- build structure ----
    idx = 1
    level_nodes = [1]

    for t = 2:length(bstructure)
        new_level = Int[]
        for p in level_nodes
            for _ in 1:bstructure[t]
                idx += 1
                parent[idx] = p
                stage[idx]  = stage[p] + 1
                push!(children[p], idx)
                push!(new_level, idx)
            end
        end
        level_nodes = new_level
    end

    leaves = [i for i in 1:n_nodes if isempty(children[i])]

    structure = TreeStructure(parent, children, stage, leaves)

    # ---- initialize data ----
    state  = randn(rng, n_nodes, dimension)
    p_edge = ones(n_nodes)
    p_cum  = ones(n_nodes)

    # normalize probabilities per parent
    for i in 1:n_nodes
        if !isempty(children[i])
            w = rand(rng, length(children[i]))
            w ./= sum(w)
            for (k, c) in enumerate(children[i])
                p_edge[c] = w[k]
            end
        end
    end

    # cumulative probabilities
    update_cumulative_probabilities!(structure, p_edge, p_cum)

    name = "Tree(" * join(bstructure, "×") * ")"

    paths = build_paths(structure)
    return Tree(name, structure, paths, state, p_edge, p_cum, 0.0)
end

function update_cumulative_probabilities!(ts::TreeStructure,p_edge :: Vector{Float64},p_cum  :: Vector{Float64})
    p_cum .= 0.0
    p_cum[1] = 1.0

    for i in 2:length(p_edge)
        p_cum[i] = p_cum[ts.parent[i]] * p_edge[i]
    end
    return p_cum
end



"""
	get_stage(trr::Tree, node=Int64[])

Returns the stage of each node in the tree.

Args:
- trr - an instance of a Tree.
- node - the number of node in the scenario tree you want to know its stage.
"""
function get_stage(trr::Tree, node = nothing)
    if isnothing(node)
        return trr.structure.stage
    else
        return(trr.structure.stage[node])
    end
end

"""
	height(trr::Tree)

Returns the height of the tree which is just the maximum number of the stages of each node.

Args:
- trr - an instance of a Tree.
"""
function height(trr::Tree)
    return maximum(get_stage(trr))
end


"""
	nodes(trr::Tree,t=Int64[])

Returns the nodes in the tree, at stages t. Generally the range of the nodes in the tree.

Args:
- trr - an instance of a Tree.
- t  - stage in the tree.

Example : nodes(trr,2) - gives all nodes at stage 2.
"""
function nodes(ts::TreeStructure, t=nothing)
    if isnothing(t)
        return(1:length(ts.stage))
    else
        return(findall(==(t), ts.stage))
    end
end



"""
	root_path(trr::Tree,nodes=Int64[])
Returns the root of the tree if the node is not specified.

Args:
- trr - an instance of Tree.
- nodes - node in the tree you want to know the sequence from the root.

If `nodes` is not specified, it returns the root of the tree.

If `nodes` is specified, it returns a sequence of nodes from the root to the specified node.
"""
function root_path(ts::TreeStructure, node::Int32)
    path = Int32[]
    while node > 0
        push!(path, node)
        node = ts.parent[node]
    end
    return reverse(path)
end


"""
	split_dim_tree(trr::Tree)

Returns a vector of trees in d-dimension.

Args:
- trr - an instance of Tree.
"""
function split_dim__tree(tr::Tree)
    trees = Tree[]
    for d in 1:size(tr.state, 2)
        push!(trees,
            Tree(
                "Tree state $d",
                tr.struct,
                tr.PathBundle,
                tr.state[:, d:d],
                copy(tr.p_edge),
                copy(tr.p_cum),
                tr.dist
            )
        )
    end
    return trees
end



"""
    build_probabilities!(tr::Tree, p::AbstractVector{<:Real})

Builds valid edge probabilities for the tree.

If `length(p) == number of nodes`, `p[i]` is interpreted as P(node i | parent).

If `length(p) == number of leaves`, `p` is interpreted as probabilities
assigned to leaf nodes, and internal probabilities are reconstructed.

All probabilities are projected to ≥ 0 and normalized per parent.
"""
function build_probabilities!(tr::Tree, p::AbstractVector{<:Real})

    ts = tr.structure
    n  = length(ts.parent)

    # --------------------------------------------------
    # Case 1: edge probabilities given for all nodes
    # --------------------------------------------------
    if length(p) == n

        @inbounds for i in 1:n
            tr.p_edge[i] = max(0.0, p[i])
        end

    # --------------------------------------------------
    # Case 2: probabilities given only for leaves
    # --------------------------------------------------
    elseif length(p) == length(ts.leaves)

        tr.p_edge .= 0.0

        # assign leaf probabilities
        @inbounds for (k, leaf) in enumerate(ts.leaves)
            tr.p_cum[leaf] = max(0.0, p[k])
        end

        # bottom-up accumulation
        for i in reverse(2:n)
            par = ts.parent[i]
            if par > 0
                tr.p_cum[par] += tr.p_cum[i]
            end
        end

        # derive edge probabilities
        for i in 2:n
            par = ts.parent[i]
            if tr.p_cum[par] > 0
                tr.p_edge[i] = tr.p_cum[i] / tr.p_cum[par]
            else
                tr.p_edge[i] = 0.0
            end
        end

        tr.p_edge[1] = 1.0   # root

    else
        error("Probability vector has incompatible length.")
    end

    # --------------------------------------------------
    # Normalize probabilities per parent (simplex)
    # --------------------------------------------------
    for i in 1:n
        ch = ts.children[i]
        if !isempty(ch)
            s = 0.0
            @inbounds for c in ch
                s += tr.p_edge[c]
            end
            if s > 0
                @inbounds for c in ch
                    tr.p_edge[c] /= s
                end
            end
        end
    end

    # --------------------------------------------------
    # Update cumulative probabilities
    # --------------------------------------------------
    update_cumulative_probabilities!(ts, tr.p_edge, tr.p_cum)

    return tr.p_edge
end



"""
    partition_tree(tr::Tree, t::Int)

Partitions the tree into independent subtrees rooted at stage `t`.

Each subtree:
- has compact node indexing
- has renormalized probabilities
- is fully independent
- is GPU-ready for parallel optimization
"""
function partition_tree(tr::Tree, t::Int)

    ts = tr.struct
    roots = findall(==(t), ts.stage)

    subtrees = Vector{Tree}(undef, length(roots))

    for (k, root) in enumerate(roots)

        # --------------------------------------------------
        # 1. Collect subtree nodes (BFS)
        # --------------------------------------------------
        stack = [root]
        nodes = Int[]

        while !isempty(stack)
            i = pop!(stack)
            push!(nodes, i)
            append!(stack, ts.children[i])
        end

        n = length(nodes)

        # --------------------------------------------------
        # 2. Build index mapping (global → local)
        # --------------------------------------------------
        idxmap = Dict{Int,Int}()
        for (i, v) in enumerate(nodes)
            idxmap[v] = i
        end

        # --------------------------------------------------
        # 3. Build compact structure arrays
        # --------------------------------------------------
        parent   = zeros(Int, n)
        stage    = zeros(Int, n)
        children = [Int[] for _ in 1:n]

        for (i, v) in enumerate(nodes)
            p = ts.parent[v]
            if p != 0 && haskey(idxmap, p)
                parent[i] = idxmap[p]
                push!(children[parent[i]], i)
                stage[i] = stage[parent[i]] + 1
            end
        end

        leaves = [i for i in 1:n if isempty(children[i])]

        struct_sub = TreeStructure(parent, children, stage, leaves)

        # --------------------------------------------------
        # 4. Extract data (contiguous)
        # --------------------------------------------------
        state  = tr.state[nodes, :]
        p_edge = tr.p_edge[nodes]

        # renormalize root
        p_edge[1] = 1.0

        p_cum = similar(p_edge)
        update_cumulative_probabilities!(struct_sub, p_edge, p_cum)

        subtrees[k] = Tree(
            "$(tr.name)_stage$t_$k",
            struct_sub,
            placeholderPathbundle,      #figure out the correct PathBundle logic for the subtrees
            state,
            p_edge,
            p_cum,
            0
        )
    end

    return subtrees
end


"""
    build_paths(ts::TreeStructure) -> PathBundle

Builds a PathBundle containing all root-to-leaf paths in the tree.

- `nodes`   : concatenated node indices along all paths
- `offsets` : starting index of each path in `nodes`
- `lengths` : length of each path
"""
function build_paths(ts::TreeStructure)::PathBundle
    # Now calculate the path to all nodes as Vector(Vector(Int)) such that path_nodes[i] shows the way to i and stops there. 
    tmp = [root_path(ts, Int32(j)) for j in nodes(ts)]

    
    path_nodes, path_nodes_offset = Contiguous_paths(tmp)


    return PathBundle(path_nodes, path_nodes_offset)
end




"""
    get_leaves(ts::TreeStructure; node::Union{Nothing,Int,Vector{Int}}=nothing, p_edge::Union{Nothing, Vector{Float64}}=nothing)

Return leaf nodes of the tree.

Arguments:
- `node` (optional) : restrict to leaves that are descendants of this node or nodes
- `p_edge` (optional): vector of edge probabilities to compute conditional probability for each leaf

Returns:
- `leaf_nodes` : indices of leaves
- `prob`       : conditional probability of each leaf (if p_edge given, otherwise all ones)
"""
function get_leaves(ts::TreeStructure; node=nothing, p_edge=nothing)
    # Start with all leaves
    leaf_nodes = ts.leaves

    # If a subset node is specified, keep only descendants
    if node !== nothing
        # allow single int or vector
        nodeset = isa(node, Int) ? [node] : node
        # BFS to find leaves that are descendants
        descendants = Vector{Int}()
        for n in nodeset
            stack = [n]
            while !isempty(stack)
                cur = pop!(stack)
                # add children to stack
                append!(stack, ts.children[cur])
                # if current node is a leaf, keep it
                if cur in ts.leaves
                    push!(descendants, cur)
                end
            end
        end
        leaf_nodes = unique(descendants)
    end

    # Compute conditional probability if p_edge given
    prob = ones(length(leaf_nodes))
    if p_edge !== nothing
        @inbounds for (i, leaf) in enumerate(leaf_nodes)
            p = 1.0
            node = leaf
            while node != 0
                p *= p_edge[node]
                node = ts.parent[node]
            end
            prob[i] = p
        end
    end

    return leaf_nodes, prob
end

function get_leaves(tr::Tree; node=nothing, p_edge=nothing)
    return get_leaves(tr.structure; node, p_edge)
end


"""
    Contiguous_paths(paths::Vector{Vector{Int}})

Creates a contiguous version of the path to all nodes in PathBundle

Arguments:
- `path`  : A Vector(Vector(int) object containing paths to all nodes of the tree

Returns:
- Contiguous memory version of the input path object
"""

function Contiguous_paths(paths::Vector{Vector{Int32}})
    n = length(paths)
    offsets = Vector{Int32}(undef, n + 1)
    offsets[1] = 1

    total = 0
    for i in 1:n
        total += length(paths[i])
        offsets[i+1] = total + 1
    end

    data = Vector{Int32}(undef, total)
    k = 1
    for p in paths
        copyto!(data, k, p, 1, length(p))
        k += length(p)
    end
    return (data, offsets)
end




#################################
# Examples of small trees for testing

function Tree(identifier::Int64)
        if identifier == 302
            trr = Tree(Int32[1,2,2],1)
            trr.name = "Tree 1x2x2"
            trr.state = [2.0 2.1 1.9 4.0 1.0 3.0]
            trr.p_edge = [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            trr.p_cum = update_cumulative_probabilities!(trr.structure,trr.p_edge,trr.p_cum)
            trr.PathBundle = build_paths(trr.structure)
        elseif identifier == 303
            trr = Tree(Int32[1,1,4],1)
            trr.name = "Tree 1x1x4"
            trr.state = [3.0 3.0 6.0 4.0 2.0 0.0]
            trr.p_edge= [1.0, 1.0, 0.25, 0.25, 0.25, 0.25]
            trr.p_cum = update_cumulative_probabilities!(trr.structure,trr.p_edge,trr.p_cum)
            trr.PathBundle = build_paths(trr.structure)
        elseif identifier == 304
            trr = Tree(Int32[1,4,1,1],1)
            trr.name = "Tree 1x4x1x1"
            trr.state = [0.1 2.1 3.0 0.1 1.9 1.0 0.0 -2.9 -1.0 -0.1 -3.1 -4.0]
            trr.p_edge = [0.14, 1.0, 1.0, 0.06, 1.0, 1.0, 0.48, 1.0, 1.0, 0.32, 1.0, 1.0]
            trr.p_cum = update_cumulative_probabilities!(trr.structure,trr.p_edge,trr.p_cum)
            trr.PathBundle = build_paths(trr.structure)
        elseif identifier == 305
            trr = Tree(Int32[1,1,4],1)
            trr.name = "Tree 1x1x4"
            trr.state = [0.0 10.0 28.0 22.0 21.0 20.0]
            trr.p_edge = [1.0 1.0 0.25 0.25 0.25 0.25]
            trr.p_cum = update_cumulative_probabilities!(trr.structure,trr.p_edge,trr.p_cum)
            trr.PathBundle = build_paths(trr.structure)
        elseif identifier == 306
            trr = Tree(Int32[1,2,2],1)
            trr.name = "Tree 1x2x2"
            trr.state = [0.0 10.0 10.0 28.0 22.0 21.0 20.0]
            trr.p_edge = [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            trr.p_cum = update_cumulative_probabilities!(trr.structure,trr.p_edge,trr.p_cum)
            trr.PathBundle = build_paths(trr.structure)
        elseif identifier == 307
            trr = Tree(Int32[1,4,1],1)
            trr.name = "Tree 1x4x1"
            trr.state = [0.0 10.0 10.0 10.0 10.0 28.0 22.0 21.0 20.0]
            trr.p_edge = [1.0, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0]
            trr.p_cum = update_cumulative_probabilities!(trr.structure,trr.p_edge,trr.p_cum)
            trr.PathBundle = build_paths(trr.structure)
        elseif identifier == 401
            trr = Tree(Int32[1,1,2,2],1)
            trr.name = "Tree 1x1x2x2"
            trr.state = [10.0 10.0 8.0 12.0 9.0 6.0 10.0 13.0]
            trr.p_edge = [1.0, 1.0, 0.66, 0.34, 0.24, 0.76, 0.46, 0.54]
            trr.p_cum = update_cumulative_probabilities!(trr.structure,trr.p_edge,trr.p_cum)
            trr.PathBundle = build_paths(trr.structure)
        elseif identifier == 402
            trr = Tree(Int32[1,2,2,2],1)
            trr.name = "Tree 1x2x2x2"
            trr.state = [10.0 12.0 8.0 15.0 11.0 9.0 5.0 18.0 16.0 13.0 11.0 10.0 7.0 6.0 3.0]
            trr.p_edge = [1.0, 0.8, 0.2, 0.3, 0.7, 0.8, 0.2, 0.6, 0.4, 0.5, 0.5, 0.4, 0.6, 0.7, 0.3]
            trr.p_cum = update_cumulative_probabilities!(trr.structure,trr.p_edge,trr.p_cum)
            trr.PathBundle = build_paths(trr.structure)
        elseif identifier == 4022
            trr = Tree(Int32[1,2,2,2],2)
            trr.name = "2dim-Tree 1x2x2x2"
            trr.state = [10.0 0.0; 12.0 1.0; 8.0 -1.0; 15.0 2.0; 11.0 1.0; 9.0 -0.5; 5.0 -2.0; 18.0 3.0; 16.0 1.8; 13.0 0.9; 11.0 0.2; 10.0 0.0; 7.0 -1.2; 6.0 -2.0; 3.0 -3.2]
            trr.p_edge = [1.0, 0.8, 0.7, 0.3, 0.2, 0.8, 0.4, 0.6, 0.2, 0.5, 0.5, 0.4, 0.6, 0.7, 0.3]
            trr.p_cum = update_cumulative_probabilities!(trr.structure,trr.p_edge,trr.p_cum)
            trr.PathBundle = build_paths(trr.structure)
        elseif identifier ==404
            trr = Tree(Int32[1,2,2,2],1)
            trr.name = "Tree 1x2x2x2"
            trr.state = (0.2+0.6744).*[0.0 1.0 -1.0 2.0 0.1 0.0 -2.0 3.0 1.1 0.9 -1.1 1.2 -1.2 -0.8 -3.2]
            trr.p_edge = [1.0, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9, 0.5, 0.5, 0.6, 0.4, 0.4, 0.6, 0.3, 0.7]
            trr.p_cum = update_cumulative_probabilities!(trr.structure,trr.p_edge,trr.p_cum)
            trr.PathBundle = build_paths(trr.structure)
        end
        return trr
    end




