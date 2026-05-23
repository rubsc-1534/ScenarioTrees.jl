# ============================================================
# Load Model and helper functions
# ============================================================
include("Longevity_model_setup.jl")
seed = 12345; rng = MersenneTwister(seed)

HORIZON = 10
MI_exa = MI_model(AGE0,HORIZON,BENEFIT,DISCOUNT,N_LIVES,N_SCENARIOS,MI_MEAN,MI_SD,MI_LOWER,MI_UPPER,zeros(N_LIVES),BASE_Q,
                            1,  #year
                            trues(N_LIVES), #alive
                            ones(HORIZON),  #improvement_factors
                            zeros(Int,HORIZON),  #deaths_by_year
                            ones(Int,HORIZON)*N_LIVES #survivors_at_payment
                            )

trr = Tree(Int32[1;fill(2,9)])

# ============================================================
# RUN RESULTS for illustration
# ============================================================

results = run_simulation(1000,MI_exa)



# ============================================================
# Plot RESULTS
# ============================================================

reserves = results.survivors_matrix

x = 1:size(reserves, 2)
n_scenarios = size(reserves, 1)
fig = Figure(size = (900, 600))
ax = Axis(
    fig[1, 1],
    xlabel = "Projection Year",
    ylabel = "Survivors",
    title = "Simulated Portfolio sizes",
)

for i in 1:n_scenarios
    lines!(ax, x, reserves[i, :], color = (:steelblue, 0.35), linewidth = 1)
end

fig

# ============================================================
# TREE NESTED APPROXIMATION adjusted for example
# ============================================================

tree_nested_approx2!(trr::Tree,projection_step_wrapper)
tree_plot(trr)

# Summing payments over time. Process now shows cumulative payments. 
trr2 = deepcopy(trr)
for i = 2:length(trr2.state)
    trr2.state[i] += trr2.state[trr2.structure.parent[i]]
end
tree_plot(trr2)

# ============================================================
# Risk measures
# ============================================================

function MeanSD(state,prob,beta)
    tmp = sum(state.*prob)
    mSD = tmp + beta * sum(max.(state.-tmp,0) .* prob)
    return(mSD)
end

# ============================================================
# Calculating dynamic risk measures
# ============================================================
start_stage = maximum(trr.structure.stage)-1
beta = 1.0

for s=start_stage:-1:0 #going backward through all relevant stages
    nodes_idx = findall(x -> x == s, trr.structure.stage)
    for idx in nodes_idx
        children_nodes = trr.structure.children[idx]
        state = trr.state[children_nodes]
        prob = trr.p_edge[children_nodes]
        trr.state[idx] = MeanSD(state,prob,beta)
    end
end

