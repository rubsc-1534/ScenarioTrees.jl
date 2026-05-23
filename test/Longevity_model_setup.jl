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


mutable struct MI_model
    age             :: Int64
    Horizon         :: Int64
    Benefit         :: Float64
    discount        :: Float64
    n_lives         :: Int64
    n_scen          :: Int64
    MI_mean         :: Float64
    MI_sd           :: Float64
    MI_low          :: Float64
    MI_high         :: Float64
    frailty        :: Vector{Float64}
    base_q          :: Vector{Float64}
    year            :: Int64
    alive           :: BitVector
    MI              :: Vector{Float64}
    deaths_by_year  :: Vector{Int64}
    survivors_at_payment :: Vector{Int64}
end

# ============================================================
# Setting parameters
# ============================================================

AGE0        = 65
HORIZON     = 30
BENEFIT     = 1_000.0
DISCOUNT    = 0.03

N_LIVES     = 1_000      # number of lives in each selected portfolio
N_SCENARIOS = 5_000      # number of Monte Carlo scenarios


# Average annual mortality improvement assumption from earlier example
MI_MEAN     = 0.015      # 1.5%

# Volatility around mortality improvement assumption
# You can tune this if you want more / less randomness
MI_SD       = 0.005      # 0.5% standard deviation

# Truncation bounds to keep annual improvement in a sensible range
MI_LOWER    = -0.005     # -0.5%
MI_UPPER    = 0.035      # +3.5%

# Portfolio selection randomness:
# Frailty multiplier with mean 1.0
# If sigma is larger, the selected portfolios vary more from average mortality
FRAILTY_SIGMA = 0.20

# Exact base mortality table from the deterministic example
# Ages 65 to 110
BASE_Q = [
    0.0100, # age 65
    0.0108, # age 66
    0.0117, # age 67
    0.0127, # age 68
    0.0139, # age 69
    0.0152, # age 70
    0.0167, # age 71
    0.0184, # age 72
    0.0203, # age 73
    0.0225, # age 74
    0.0242, # age 75
    0.0265, # age 76
    0.0290, # age 77
    0.0317, # age 78
    0.0347, # age 79
    0.0380, # age 80
    0.0415, # age 81
    0.0455, # age 82
    0.0498, # age 83
    0.0545, # age 84
    0.0596, # age 85
    0.0652, # age 86
    0.0714, # age 87
    0.0781, # age 88
    0.0855, # age 89
    0.0936, # age 90
    0.1024, # age 91
    0.1121, # age 92
    0.1227, # age 93
    0.1343, # age 94
    0.1470, # age 95
    0.1608, # age 96
    0.1760, # age 97
    0.1926, # age 98
    0.2108, # age 99
    0.2307, # age 100
    0.2525, # age 101
    0.2764, # age 102
    0.3025, # age 103
    0.3311, # age 104
    0.3623, # age 105
    0.3966, # age 106
    0.4340, # age 107
    0.4750, # age 108
    0.5199, # age 109
    0.5689  # age 110
]


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
    for t in 1:MI_exa.Horizon
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
# TREE NESTED APPROXIMATION adjusted for example
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
