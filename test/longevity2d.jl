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



function simulate_pure_stochastic_longevity(;
    B = 0.0001,          # Baseline mortality scale
    C = 0.085,           # Baseline aging rate
    x = 65.0,            # Initial age
    r = 0.03,            # Risk-free discount rate (3%)
    phi_0 = 0.015,       # Initial improvement rate (1.5%)
    phi_bar = 0.015,     # Long-term mean improvement rate (1.5%)
    kappa = 0.1,         # Speed of mean reversion
    sigma_phi = 0.004,   # Volatility of the structural trend
    T = 5.0,            # Simulation horizon (years)
    dt = 12/12,           # Time step (monthly)
    N_paths = 100,    # Number of Monte Carlo trajectories
    seed = 42
)
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
time_grid, phi_paths, Y_paths = simulate_pure_stochastic_longevity()

# --- VISUALIZATION ---
# Plot 1: Cumulative Improvement (Y_t) Paths (Sample 100 paths)
p1 = plot(time_grid, Y_paths[1:100, :]', 
          title="Cumulative Mortality Improvement (Y_t)",
          xlabel="Years from Valuation", 
          ylabel="Cumulative Improvement",
          legend=false, 
          alpha=0.3, 
          linecolor=:blue)


fig = Figure(size = (800, 600))
ax = Axis(
    fig[1, 1],
    title = "Cumulative Mortality Improvement (Y_t)",
    xlabel = "Years from Valuation",
    ylabel = "Cumulative Improvement"
)

for i in 1:100
    lines!(
        ax,
        time_grid,
        phi_paths[i, :],
        color = (:blue, 0.3)   # blue with 30% opacity
    )
end
fig


# ============================================================
# TREE NESTED APPROXIMATION adjusted for example
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
    sigma_phi = 0.004   # Volatility of the structural trend
    dt = 0.1    
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
trr = Tree(Int32[1;fill(2,2)])
tree_nested_approx2!(trr::Tree,projection_step_wrapper)
tree_plot(trr)

#Get the cumulative improvement factors
for i=2:length(trr.state)
    trr.state[i] += trr.state[trr.structure.parent[i]]
end
tree_plot(trr)


##############################
# BASE_Qx Volatility
trr2 = Tree(Int32[1;fill(3,2)])
#Set the 3 different states for each node
#For every parent update all children nodes 
trr2.state[1] = 0.0100
for i in unique(trr2.structure.parent)
    if i==0
        trr2.state[trr2.structure.parent.==i] .= trr2.state[1]
    else
        trr2.state[trr2.structure.parent.==i] = trr2.state[i] .* [0.95,1.0,1.05]
    end

end
tree_plot(trr2)

#################
# Multiplication of trees
Mtrr = merge_trees(trr,trr2,name="Test_multiplaction")
tree_plot(Mtrr)

