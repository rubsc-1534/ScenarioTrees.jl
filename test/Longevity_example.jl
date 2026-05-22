using Random
using Statistics
using Printf
using Distributions
using CairoMakie
using ScenarioTrees

# ============================================================
# SETTINGS
# ============================================================

const AGE0        = 65
const HORIZON     = 30
const BENEFIT     = 1_000.0
const DISCOUNT    = 0.03
const V           = 1 / (1 + DISCOUNT)

const N_LIVES     = 1_000      # number of lives in each selected portfolio
const N_SCENARIOS = 5_000      # number of Monte Carlo scenarios
const SEED        = 12345

# Average annual mortality improvement assumption from earlier example
const MI_MEAN     = 0.015      # 1.5%

# Volatility around mortality improvement assumption
# You can tune this if you want more / less randomness
const MI_SD       = 0.005      # 0.5% standard deviation

# Truncation bounds to keep annual improvement in a sensible range
const MI_LOWER    = -0.005     # -0.5%
const MI_UPPER    = 0.035      # +3.5%

# Portfolio selection randomness:
# Frailty multiplier with mean 1.0
# If sigma is larger, the selected portfolios vary more from average mortality
const FRAILTY_SIGMA = 0.20

# Exact base mortality table from the deterministic example
# Ages 65 to 110
const BASE_Q = [
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
# HELPER FUNCTIONS
# ============================================================
"""
Sample a random portfolio of frailty multipliers.
"""
function sample_portfolio(
    rng::AbstractRNG,
    n_lives::Integer;
    frailty_sigma::Float64 = FRAILTY_SIGMA,
)
    mu = -0.5 * frailty_sigma^2
    frailty_dist = LogNormal(mu, frailty_sigma)
    return rand(rng, frailty_dist, n_lives)
end

"""
Sample annual mortality improvement rates for a scenario.
"""
function sample_mortality_improvement_path(
    rng::AbstractRNG,
    horizon::Integer;
    mi_mean::Float64 = MI_MEAN,
    mi_sd::Float64 = MI_SD,
    lower::Float64 = MI_LOWER,
    upper::Float64 = MI_UPPER,
)
    mi_dist = truncated(Normal(mi_mean, mi_sd), lower, upper)
    return rand(rng, mi_dist, horizon)
end

"""
Simulate one scenario for a fixed frailty portfolio.
"""
function simulate_scenario(
    rng::AbstractRNG,
    frailty::AbstractVector{<:Real};
    horizon::Integer = HORIZON,
    base_q::AbstractVector{<:Real} = BASE_Q,
    mi_mean::Float64 = MI_MEAN,
    mi_sd::Float64 = MI_SD,
    mi_lower::Float64 = MI_LOWER,
    mi_upper::Float64 = MI_UPPER,
)
    n_lives = length(frailty)
    alive = trues(n_lives)
    deaths_by_year = zeros(Int, horizon)
    survivors_at_payment = zeros(Int, horizon)
    cumulative_improvement_factor = 1.0

    for t in 1:horizon
        step = simulate_scenario_projection_step(
            rng,
            t,
            frailty,
            alive,
            cumulative_improvement_factor,
            deaths_by_year,
            survivors_at_payment,
            base_q;
            mi_mean = mi_mean,
            mi_sd = mi_sd,
            mi_lower = mi_lower,
            mi_upper = mi_upper,
        )

        alive = step.alive
        cumulative_improvement_factor = step.cumulative_improvement_factor
        deaths_by_year = step.deaths_by_year
        survivors_at_payment = step.survivors_at_payment
    end

    return (
        deaths_by_year = deaths_by_year,
        survivors_at_payment = survivors_at_payment,
    )
end

"""
Simulate one projection year within a scenario.
"""
function simulate_scenario_projection_step(
    rng::AbstractRNG,
    year::Integer,
    frailty::AbstractVector{<:Real},
    alive::BitVector,
    cumulative_improvement_factor::Float64,
    deaths_by_year::Vector{Int},
    survivors_at_payment::Vector{Int},
    base_q::AbstractVector{<:Real};
    mi_mean::Float64 = MI_MEAN,
    mi_sd::Float64 = MI_SD,
    mi_lower::Float64 = MI_LOWER,
    mi_upper::Float64 = MI_UPPER,
)
    mi_rate = rand(rng, truncated(Normal(mi_mean, mi_sd), mi_lower, mi_upper))
    if year > 1
        cumulative_improvement_factor *= 1.0 - mi_rate
    end

    q_t = clamp.(base_q[year] .* frailty .* cumulative_improvement_factor, 0.0, 1.0)
    u = rand(rng, length(frailty))

    die_this_year = alive .& (u .< q_t)
    survive_to_payment = alive .& .!die_this_year

    deaths_by_year[year] = count(die_this_year)
    survivors_at_payment[year] = count(survive_to_payment)

    return (
        alive = survive_to_payment,
        cumulative_improvement_factor = cumulative_improvement_factor,
        deaths_by_year = deaths_by_year,
        survivors_at_payment = survivors_at_payment,
    )
end



# ============================================================
# MAIN MONTE CARLO
# ============================================================

function run_simulation(
    ;
    n_scenarios::Int = N_SCENARIOS,
    n_lives::Int = N_LIVES,
    horizon::Int = HORIZON,
    seed::Int = SEED,
    frailty_sigma::Float64 = FRAILTY_SIGMA,
    base_q::AbstractVector{<:Real} = BASE_Q,
)
    rng = MersenneTwister(seed)
    deaths_matrix = zeros(Int, n_scenarios, horizon)
    survivors_matrix = zeros(Int, n_scenarios, horizon)

    for s in 1:n_scenarios
        frailty = sample_portfolio(rng, n_lives; frailty_sigma = frailty_sigma)
        result = simulate_scenario(
            rng,
            frailty;
            horizon = horizon,
            base_q = base_q,
        )

        deaths_matrix[s, :] .= result.deaths_by_year
        survivors_matrix[s, :] .= result.survivors_at_payment
    end

    return (
        deaths_matrix = deaths_matrix,
        survivors_matrix = survivors_matrix,
    )
end




# ============================================================
# RUN RESULTS
# ============================================================

results = run_simulation()



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
    ylabel = "Cumulative payments",
    title = "Simulated Payment Paths",
)

for i in 1:n_scenarios
    lines!(ax, x, reserves[i, :], color = (:steelblue, 0.35), linewidth = 1)
end

fig

# ============================================================
# TREE NESTED APPROXIMATION EXAMPLE
# ============================================================

function run_tree_nested_example(
    ;
    structure::Vector{Int32} = Int32[1, 2, 2, 2, 2],
    seed::Int = SEED,
)
    rng = MersenneTwister(seed)
    trr = Tree(structure)
    trr.state[1] = 1.0

    sampler = history -> simulate_scenario_projection_sampler(
        rng,
        history;
        base_q = BASE_Q,
        mi_mean = MI_MEAN,
        mi_sd = MI_SD,
        mi_lower = MI_LOWER,
        mi_upper = MI_UPPER,
    )

    tree_nested_approx!(trr, sampler)
    return tree_plot(trr)
end

nested_fig = run_tree_nested_example()
nested_fig
