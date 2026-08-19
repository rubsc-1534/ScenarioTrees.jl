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
include("ARC_Longevity_setup.jl")
#########################################################
#Setting up parameters
struct MI_param
    phi_0::Float64      # Initial improvement rate (1.5%)
    phi_bar::Float64     # Long-term mean improvement rate (1.5%)
    kappa::Float64        # Speed of mean reversion
    sigma_phi::Float64   # Volatility of the structural trend
    T::Int            # Simulation horizon (years)
    dt::Float64           # Time step (monthly)
    N_paths::Int   # Number of Monte Carlo trajectories
    seed::Int
end

MI_test = MI_param(0.015,0.015,0.1,0.15,5,12/12,100,42)


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
save("plots/MI_paths.png",fig)


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



S_tree=tree_to_survivor_tree(Mtrr, 1, 100.0)
fig_SurvivorTree = tree_plot2(S_tree,title="Survivor process",density=false)
save("SurvivorTree.png",fig_SurvivorTree, update=false)


payout_tree = deepcopy(S_tree);
# Summing payments over time. Process now shows cumulative payments. 
for i = 2:length(payout_tree.state)
    payout_tree.state[i] += payout_tree.state[payout_tree.structure.parent[i]]
end
fig_payout = tree_plot2(payout_tree,title="Payout process",density=false)
save("payoutTree.png",fig_payout, update=false) 


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

fig_BE_reserveTree = tree_plot2(Reserve_tree, title="Best estimate reserving process",density=false)
save("BE_ReserveTree.png",fig_BE_reserveTree, update=false)

Reserve_tree2 = deepcopy(payout_tree)
start_stage = maximum(Reserve_tree2.structure.stage)-1
beta = 0.995

for s=start_stage:-1:0 #going backward through all relevant stages
    nodes_idx = findall(x -> x == s, Reserve_tree2.structure.stage)
    for idx in nodes_idx
        children_nodes = Reserve_tree2.structure.children[idx]
        state = Reserve_tree2.state[children_nodes]
        prob = Reserve_tree2.p_edge[children_nodes]
        Reserve_tree2.state[idx] = MeanSD(state,prob,beta,2)
    end
end

fig_risk_reserveTree = tree_plot2(Reserve_tree2, title="Risk-averse reserving process",density=false)
save("risk_ReserveTree.png",fig_risk_reserveTree, update=false) 