using Random
using Statistics
using Distributions
using CairoMakie
#using ScenarioTrees

# ============================================================
# SETTINGS
# ============================================================
include("Longevity_model_setup.jl")
seed = 12345; rng = MersenneTwister(seed)

# ============================================================
# ALGORITHM
# ============================================================
# For each scenario 
#   Simulate Portfolio (meaning different base death distributions)
#   Simulate projections years step by step depending on previous year (picking MI improvement, base fraility and all survivors)



# ============================================================
# HELPER FUNCTIONS
# ============================================================
function projection_step_wrapper(history::Vector{Float64},year)
    if(length(history)==1)
        MI_exa.n_lives = N_LIVES
    else
        MI_exa.survivors_at_payment[year-1] = round(last(history))
    end
    simulate_scenario_projection_step(MI_exa)
    return(MI_exa.survivors_at_payment[year])
end



"""
Simulate one projection year within a scenario.
"""
function simulate_scenario_projection_step(MI_exa)
    #Set frailty and get number of people alive
    if MI_exa.year == 1
        alive = trues(MI_exa.n_lives)
        mu = -0.5 * FRAILTY_SIGMA^2
        frailty_dist = LogNormal(mu, FRAILTY_SIGMA)
        MI_exa.frailty = rand(frailty_dist, MI_exa.n_lives)
    else
        alive = trues(MI_exa.survivors_at_payment[MI_exa.year-1])
        mu = -0.5 * FRAILTY_SIGMA^2
        frailty_dist = LogNormal(mu, FRAILTY_SIGMA)
        MI_exa.frailty = rand(frailty_dist, MI_exa.survivors_at_payment[MI_exa.year-1])
    end


    mi_rate = rand(truncated(Normal(MI_exa.MI_mean, MI_exa.MI_sd), MI_exa.MI_low, MI_exa.MI_high))
    if MI_exa.year > 1
        MI_exa.MI[MI_exa.year] = 1.0 - mi_rate
    end
    cum_MI = cumprod(MI_exa.MI)

    q_t = clamp.(MI_exa.base_q[MI_exa.year] .* MI_exa.frailty .* cum_MI[end], 0.0, 1.0)
    u = rand(length(MI_exa.frailty))

    #die_this_year = MI_exa.alive .& (u .< q_t)
    #survive_to_payment = MI_exa.alive .& .!die_this_year

    MI_exa.deaths_by_year[MI_exa.year] = count(alive .& (u .< q_t))
    MI_exa.survivors_at_payment[MI_exa.year] = count(alive .& .!(alive .& (u .< q_t)))

    return (0)
end

"""
Sample a random portfolio of frailty multipliers.
"""
function sample_portfolio(
    n_lives::Integer;
    frailty_sigma::Float64 = FRAILTY_SIGMA,
)
    mu = -0.5 * frailty_sigma^2
    frailty_dist = LogNormal(mu, frailty_sigma)
    return rand(frailty_dist, n_lives)
end



"""
Simulate one scenario for a fixed frailty portfolio.
"""
function simulate_scenario(MI_exa)
    #maybe create a different model here so call MI_exa(asdkjasdksahdksahd)
    for t in 1:horizon
        MI_exa.year = t
        simulate_scenario_projection_step(MI_exa)
    end

    return(0)
end




# ============================================================
# MAIN MONTE CARLO
# ============================================================

function run_simulation(n_scenarios,MI_exa)
    
    deaths_matrix = zeros(Int, n_scenarios, MI_exa.Horizon)
    survivors_matrix = zeros(Int, n_scenarios, MI_exa.Horizon)

    for s in 1:n_scenarios
        frailty = sample_portfolio(MI_exa.n_lives; frailty_sigma = FRAILTY_SIGMA)
        MI_exa = MI_model(AGE0,HORIZON,BENEFIT,DISCOUNT,N_LIVES,N_SCENARIOS,MI_MEAN,MI_SD,MI_LOWER,MI_UPPER,frailty,BASE_Q,
                            1,  #year
                            trues(N_LIVES), #alive
                            ones(HORIZON),  #improvement_factors
                            zeros(Int,HORIZON),  #deaths_by_year
                            ones(Int,HORIZON)*N_LIVES #survivors_at_payment
                            )

        simulate_scenario(MI_exa)

        deaths_matrix[s, :] .= MI_exa.deaths_by_year
        survivors_matrix[s, :] .= MI_exa.survivors_at_payment
    end

    return (
        deaths_matrix = deaths_matrix,
        survivors_matrix = survivors_matrix,
    )
end




# ============================================================
# RUN RESULTS
# ============================================================

results = run_simulation(1000,MI_exa)



# ============================================================
# Plot RESULTS
# ============================================================

function reserve_calculation(results; benefit::Float64 = BENEFIT)
    survivors = results.survivors_matrix
    return(survivors)
    #return cumsum(survivors, dims = 2) .* benefit
end

reserves = reserve_calculation(results)

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
# TREE NESTED APPROXIMATION EXAMPLE
# ============================================================

function tree_nested_approx2!(trr::Tree,g)

    stages = 1:height(trr)

    # Step size (Robbins–Monro)
    ak(k) = 0.2 / (k+30)^0.75

    # Initialize the root node
    trr.state[1] = 1000
    nestedDistance = 0
    #go through the tree from beginning
    for s in stages
        nodes = get_nodes(trr.structure)[trr.structure.stage.==s-1]
        MI_exa.year = s
        for k in nodes
            println("Evaluation node $k in stage $s")
            history = trr.state[root_path(trr.structure,Int32(k))] #provides states of path until node k as a vector
            f() = g(history,s) #just defines the cond. sampling function

            children = trr.structure.children[k] #get indices for nodes being updated in this round
            b = length(children)    #number of nodes to be simulated in this round
            init = vec(kmeans(reshape(randn(100_000),1,100_000) .+last(history) .-10, b).centers)    #get good starting value based on clustering
            #update state and probability in one pass using SA algorithm with cluster starting values
            trr.state[children], trr.p_edge[children] = stochastic_nodes(f;b = b,nsteps = 1_000,ak = ak,init=init)
        end
        

    end
end

HORIZON = 21
MI_exa = MI_model(AGE0,HORIZON,BENEFIT,DISCOUNT,N_LIVES,N_SCENARIOS,MI_MEAN,MI_SD,MI_LOWER,MI_UPPER,zeros(N_LIVES),BASE_Q,
                            1,  #year
                            trues(N_LIVES), #alive
                            ones(HORIZON),  #improvement_factors
                            zeros(Int,HORIZON),  #deaths_by_year
                            ones(Int,HORIZON)*N_LIVES #survivors_at_payment
                            )

trr = Tree(Int32[1;fill(2,20)])
tree_nested_approx2!(trr::Tree,projection_step_wrapper)
tree_plot(trr)