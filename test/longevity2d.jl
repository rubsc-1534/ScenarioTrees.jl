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
#########################################################
#Setting up parameters
phi_0 = 0.015       # Initial improvement rate (1.5%)
phi_bar = 0.015     # Long-term mean improvement rate (1.5%)
kappa = 0.1         # Speed of mean reversion
sigma_phi = 0.15   # Volatility of the structural trend
T = 5.0            # Simulation horizon (years)
dt = 12/12           # Time step (monthly)
N_paths = 100    # Number of Monte Carlo trajectories
seed = 42



########################################################
function simulate_pure_stochastic_longevity(phi_0,phi_bar,kappa,sigma_phi,T,dt,N_paths,seed)
    Random.seed!(seed)
    N_steps = round(Int, T / dt) + 1
    time_grid = range(0, T, length=N_steps)

    # 1. Simulate improvement rate phi_t (OU Process)
    phi = zeros(N_paths, N_steps)
    phi[:, 1] .= phi_0
    
    # Pre-generate standard normal noise
    Z = randn(N_paths, N_steps - 1)
    
    for i in 2:N_steps
        dW = sqrt(dt) .* Z[:, i-1]
        # Euler-Maruyama discretization for OU process
        phi[:, i] = phi[:, i-1] .+ kappa .* (phi_bar .- phi[:, i-1]) .* dt .+ sigma_phi .* dW
    end
    
    # 2. Cumulative Improvement Y_t = ∫ φ_s ds
    # Because there's no short-term noise, Y_t is purely the smooth integral of phi_t
    Y = cumsum(phi .* dt, dims=2)
    
    return time_grid, phi, Y
end

# --- RUNNING THE SIMULATION ---
println("Running Monte Carlo longevity simulation (10,000 paths)...")
time_grid, phi_paths, Y_paths = simulate_pure_stochastic_longevity(phi_0,phi_bar,kappa,sigma_phi,T,dt,N_paths,seed)

# --- VISUALIZATION ---
# Plot 1: mortality Improvement (phi_t) Paths (Sample 100 paths)
fig = Figure(size = (800, 600))
ax = Axis(fig[1, 1],title = "Mortality Improvement paths (φ_t)",xlabel = "Years from Valuation",
    ylabel = "Stagewise Improvement")

for i in 1:100
    lines!(ax,time_grid,phi_paths[i, :],color = (:blue, 0.3))
end
fig
save("MI_paths.png",fig)

# ============================================================
# TREE NESTED APPROXIMATION adjusted for this example
# ============================================================
function tree_nested_approx2!(trr::Tree,g)
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
            f() = g(history,s) #just defines the cond. sampling function

            children = trr.structure.children[k] #get indices for nodes being updated in this round
            b = length(children)    #number of nodes to be simulated in this round
            init = vec(kmeans(reshape(0.005*randn(100_000),1,100_000) .+last(history), b).centers)    #get good starting value based on clustering
            #update state and probability in one pass using SA algorithm with cluster starting values
            trr.state[children], trr.p_edge[children] = stochastic_nodes(f;b = b,nsteps = 1_000,ak = ak,init=init)
        end
    end
end


#define a conditional 1-step sampling function
function projection_step_wrapper(history::Vector{Float64},year)
    if(length(history)==1)
        phi = 0.015;
    else
        phi = round(last(history))
    end
    phi_new = simulate_scenario_projection_step(phi)
    return(phi_new)
end

function simulate_scenario_projection_step(phi_last)
    phi_bar = 0.015     # Long-term mean improvement rate (1.5%)
    kappa = 0.1      # Speed of mean reversion
    sigma_phi = 0.15   # Volatility of the structural trend
    dt = 1    
    Z = randn(1)
    dW = sqrt(dt) * Z
    # Euler-Maruyama discretization for OU process
    phi = phi_last + kappa * (phi_bar - phi_last)*dt + sigma_phi*dW[1]
    return(phi)
end

#Add every stage >1 turn every node into 3 with 95%,100%,105% of the appropriate qx for this stage
# and multiply by cumulative improvement factor given by tree version of Y_paths



#Now create new tree for just the improvement factors
    #for this one step or conditional simulation functions have to be defined given state Y_t and phi_t what happens
# phi_t+1 and Y_t+1 and all future periods??
Random.seed!(42)
trr = Tree(Int32[1;fill(2,5)])
tree_nested_approx2!(trr::Tree,projection_step_wrapper)
figMI = tree_plot2(trr,title="Mortality improvements per stage",density=false)
save("figMI.png",figMI, update=false) 

trr1 = deepcopy(trr)
#Get the cumulative improvement factors
#trr1.state[1] = 0.0
for i=2:length(trr1.state)
    trr1.state[i] += trr1.state[trr1.structure.parent[i]]
end
figcumMI = tree_plot2(trr1,title="Cumulative mortality improvement process",density=false)
save("figcumMI.png",figcumMI, update=false) 

##############################
# BASE_Qx Volatility
trr2 = Tree(Int32[1;fill(3,5)])
#Set the 3 different states for each node
#For every parent update all children nodes 
trr2.state[1] = 0.0242
for i in unique(trr2.structure.parent)
    if i==0
        trr2.state[trr2.structure.parent.==i] .= trr2.state[1]
    else
        trr2.state[trr2.structure.parent.==i] = trr2.state[i] .* [0.7,1.0,1.15]
    end

end
figQx=tree_plot2(trr2, title="Base qx process",density=false)
save("figQx.png",figQx, update=false) 

#################
# Multiplication of trees
Mtrr = merge_trees(trr1,trr2,name="Test_multiplication")
figMerge = tree_plot2(Mtrr, title="qx process under mortality improvements",density=false)
save("MergedTree.png",figMerge, update=false) 


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

S_tree=tree_to_survivor_tree(Mtrr, 1, 100.0)
fig_SurvivorTree = tree_plot2(S_tree,title="Survivor process",density=false)
save("SurvivorTree.png",fig_SurvivorTree, update=false)



payout_tree = deepcopy(S_tree);
# Summing payments over time. Process now shows cumulative payments. 
for i = 2:length(payout_tree.state)
    payout_tree.state[i] += payout_tree.state[payout_tree.structure.parent[i]]
end
fig_payout = tree_plot(payout_tree,title="Payout process",density=false)
save("payoutTree.png",fig_payout, update=false) 
# ============================================================
# Risk measures
# ============================================================

function MeanSD(state,prob,beta,p)
    tmp = sum(state.*prob)
    mSD = tmp + beta * (sum(max.(state.-tmp,0).^p .* prob))^(1/p)
    return(mSD)
end

# ============================================================
# Calculating dynamic risk measures BE
# ============================================================
Reserve_tree = deepcopy(payout_tree)
start_stage = maximum(Reserve_tree.structure.stage)-1
beta = 0.0

for s=start_stage:-1:0 #going backward through all relevant stages
    nodes_idx = findall(x -> x == s, Reserve_tree.structure.stage)
    for idx in nodes_idx
        children_nodes = Reserve_tree.structure.children[idx]
        state = Reserve_tree.state[children_nodes] 
        prob = Reserve_tree.p_edge[children_nodes]
        Reserve_tree.state[idx] = MeanSD(state,prob,beta,2)
    end
end

fig_BE_reserveTree = tree_plot(Reserve_tree, title="Best estimate reserving process",density=false)
save("BE_ReserveTree.png",fig_BE_reserveTree, update=false)

Reserve_tree2 = deepcopy(payout_tree)
start_stage = maximum(Reserve_tree2.structure.stage)-1
beta = 0.8

for s=start_stage:-1:0 #going backward through all relevant stages
    nodes_idx = findall(x -> x == s, Reserve_tree2.structure.stage)
    for idx in nodes_idx
        children_nodes = Reserve_tree2.structure.children[idx]
        state = Reserve_tree2.state[children_nodes]
        prob = Reserve_tree2.p_edge[children_nodes]
        Reserve_tree2.state[idx] = MeanSD(state,prob,beta,2)
    end
end

fig_risk_reserveTree = tree_plot(Reserve_tree2, title="Risk-averse reserving process",density=false)
save("risk_ReserveTree.png",fig_risk_reserveTree, update=false) 