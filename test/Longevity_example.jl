using Random
using Statistics
using Printf
using Distributions

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
Sample a random portfolio:
- all lives start at age 65 (to stay exactly aligned with the earlier example)
- each life gets a frailty multiplier
  frailty > 1.0 => heavier mortality
  frailty < 1.0 => lighter mortality
Mean frailty is 1.0.
"""
function sample_portfolio(rng::AbstractRNG, n_lives::Int, frailty_sigma::Float64)
    # Lognormal parameterization chosen so E[frailty] = 1
    mu = -0.5 * frailty_sigma^2
    frailty_dist = LogNormal(mu, frailty_sigma)
    frailty = rand(rng, frailty_dist, n_lives)

    return frailty
end

"""
Sample a random annual mortality-improvement path for one scenario.
Each year's improvement rate is random but centered around 1.5%.
"""
function sample_mortality_improvement_path(rng::AbstractRNG, horizon::Int)
    mi_dist = truncated(Normal(MI_MEAN, MI_SD), MI_LOWER, MI_UPPER)
    return rand(rng, mi_dist, horizon)
end

"""
Simulate one scenario:
1) select portfolio randomly (frailty factors)
2) draw random annual mortality improvement path
3) simulate deaths year by year

Payment rule:
- annuity is paid at end of year only if alive at end of that year
"""
function simulate_scenario(rng::AbstractRNG, n_lives::Int, frailty::Vector{Float64})
    
    alive = trues(n_lives)

    deaths_by_year = zeros(Int, HORIZON)
    survivors_at_payment = zeros(Int, HORIZON)

    # cumulative improvement factor applied to base mortality
    # year 1 => no prior improvement => factor = 1.0
    cumulative_improvement_factor = 1.0


    for t in 1:HORIZON
        one_step = simulate_scenario_projection_step(t, frailty,n_lives, alive, cumulative_improvement_factor, deaths_by_year, survivors_at_payment)
        alive = one_step.alive
        cumulative_improvement_factor = one_step.cumulative_improvement_factor
        deaths_by_year = one_step.deaths_by_year
        survivors_at_payment = one_step.survivors_at_payment
    end

    return (
        deaths_by_year = deaths_by_year,
        survivors_at_payment = survivors_at_payment
    )
end

function simulate_scenario_projection_step(t,frailty,n_lives,alive,cumulative_improvement_factor,deaths_by_year,survivors_at_payment)
        mi_path = sample_mortality_improvement_path(rng, 1)
        if t > 1
            cumulative_improvement_factor *= (1.0 - mi_path[1])
        end
        # Actual mortality for each life in this year:
        # base table * portfolio frailty * realized cumulative improvement
        q_t = BASE_Q[t] .* frailty .* cumulative_improvement_factor

        # Just for safety
        q_t = clamp.(q_t, 0.0, 1.0)

        # Simulate deaths during year t
        u = rand(rng, n_lives)
        die_this_year = alive .& (u .< q_t)

        # Must survive the year to receive end-of-year annuity payment
        survive_to_payment = alive .& .!die_this_year

        deaths_by_year[t] = count(die_this_year)
        survivors_at_payment[t] = count(survive_to_payment)

        # Roll forward
        alive = survive_to_payment
        
        return(alive=alive,
        cumulative_improvement_factor=cumulative_improvement_factor,
        deaths_by_year=deaths_by_year,
        survivors_at_payment=survivors_at_payment)

    end

# ============================================================
# MAIN MONTE CARLO
# ============================================================

function run_simulation(; n_scenarios::Int=N_SCENARIOS, n_lives::Int=N_LIVES, seed::Int=SEED)
    rng = MersenneTwister(seed)

    deaths_matrix = zeros(Int, n_scenarios, HORIZON)
    survivors_matrix = zeros(Int, n_scenarios, HORIZON)


    frailty = sample_portfolio(rng, n_lives, FRAILTY_SIGMA)
    for s in 1:n_scenarios
        result = simulate_scenario(rng, n_lives,frailty)

        deaths_matrix[s, :] .= result.deaths_by_year
        survivors_matrix[s, :] .= result.survivors_at_payment
    end

    return (
        deaths_matrix = deaths_matrix,
        survivors_matrix = survivors_matrix
    )
end




# ============================================================
# RUN RESULTS
# ============================================================

results = run_simulation()



# ============================================================
# Plot RESULTS
# ============================================================


function reserve_calculation(results)
    survivors = results.survivors_matrix
    reserves = cumsum(survivors,dims=2)*BENEFIT

end


# x-axis = projection year
x = 1:size(survivors, 2)

n_scenarios = size(survivors, 1)
n_years = size(survivors, 2)


fig = Figure(size = (900, 600))
ax = Axis(
    fig[1, 1],
    xlabel = "Projection Year",
    ylabel = "Cumulative payments",
    title = "Simulated Payment Paths"
)

for i in 1:n_scenarios
    lines!(ax, x, reserves[i, :], color = (:steelblue, 0.35), linewidth = 1)
end

fig


  deaths_by_year = zeros(Int, HORIZON)
    survivors_at_payment = zeros(Int, HORIZON)

f() = simulate_scenario_projection_step(1,frailty,n_lives,trues(n_lives),1.0,deaths_by_year ,survivors_at_payment)[:survivors_at_payment][1]
simulate_scenario_projection_step(t,frailty,n_lives,alive,cumulative_improvement_factor,deaths_by_year,survivors_at_payment)
f()