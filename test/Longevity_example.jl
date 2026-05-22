using Random
using Statistics
using Printf
using Distributions

# ============================================================
# SETTINGS
# ============================================================

const AGE0        = 65
const HORIZON     = 10
const BENEFIT     = 10_000.0
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
# Ages 65 to 74
const BASE_Q = [
    0.0100,  # age 65
    0.0108,  # age 66
    0.0117,  # age 67
    0.0127,  # age 68
    0.0139,  # age 69
    0.0152,  # age 70
    0.0167,  # age 71
    0.0184,  # age 72
    0.0203,  # age 73
    0.0225   # age 74
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

"""
Deterministic benchmark PV per life using the original 1.5% mortality improvement.
This reproduces the earlier 10-year annuity example.
"""
function deterministic_pv_per_life(base_q::Vector{Float64}, mi_mean::Float64)
    survival = 1.0
    pv = 0.0

    for t in 1:length(base_q)
        q_t = base_q[t] * (1 - mi_mean)^(t - 1)
        survival *= (1 - q_t)  # alive at payment date
        pv += BENEFIT * survival * (V^t)
    end

    return pv
end

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
4) calculate discounted annuity PV

Payment rule:
- annuity is paid at end of year only if alive at end of that year
"""
function simulate_scenario(rng::AbstractRNG, n_lives::Int)
    frailty = sample_portfolio(rng, n_lives, FRAILTY_SIGMA)
    mi_path = sample_mortality_improvement_path(rng, HORIZON)

    alive = trues(n_lives)

    pv = 0.0
    deaths_by_year = zeros(Int, HORIZON)
    survivors_at_payment = zeros(Int, HORIZON)
    avg_q_by_year = zeros(Float64, HORIZON)

    # cumulative improvement factor applied to base mortality
    # year 1 => no prior improvement => factor = 1.0
    cumulative_improvement_factor = 1.0

    for t in 1:HORIZON
        if t > 1
            cumulative_improvement_factor *= (1.0 - mi_path[t - 1])
        end

        # Actual mortality for each life in this year:
        # base table * portfolio frailty * realized cumulative improvement
        q_t = BASE_Q[t] .* frailty .* cumulative_improvement_factor

        # Just for safety
        q_t = clamp.(q_t, 0.0, 1.0)

        # Track average actual q among currently alive lives
        if any(alive)
            avg_q_by_year[t] = mean(q_t[alive])
        else
            avg_q_by_year[t] = 0.0
        end

        # Simulate deaths during year t
        u = rand(rng, n_lives)
        die_this_year = alive .& (u .< q_t)

        # Must survive the year to receive end-of-year annuity payment
        survive_to_payment = alive .& .!die_this_year

        deaths_by_year[t] = count(die_this_year)
        survivors_at_payment[t] = count(survive_to_payment)

        # Discounted payment
        pv += BENEFIT * (V^t) * survivors_at_payment[t]

        # Roll forward
        alive = survive_to_payment
    end

    return (
        pv = pv,
        deaths_by_year = deaths_by_year,
        survivors_at_payment = survivors_at_payment,
        mi_path = mi_path,
        avg_q_by_year = avg_q_by_year,
        avg_frailty = mean(frailty),
        portfolio_frailty = frailty
    )
end

# ============================================================
# MAIN MONTE CARLO
# ============================================================

function run_simulation(; n_scenarios::Int=N_SCENARIOS, n_lives::Int=N_LIVES, seed::Int=SEED)
    rng = MersenneTwister(seed)

    scenario_pv = zeros(Float64, n_scenarios)
    scenario_avg_frailty = zeros(Float64, n_scenarios)
    scenario_avg_mi = zeros(Float64, n_scenarios)

    deaths_matrix = zeros(Int, n_scenarios, HORIZON)
    survivors_matrix = zeros(Int, n_scenarios, HORIZON)
    avg_q_matrix = zeros(Float64, n_scenarios, HORIZON)
    mi_matrix = zeros(Float64, n_scenarios, HORIZON)

    first_scenario = nothing

    for s in 1:n_scenarios
        result = simulate_scenario(rng, n_lives)

        scenario_pv[s] = result.pv
        scenario_avg_frailty[s] = result.avg_frailty
        scenario_avg_mi[s] = mean(result.mi_path)

        deaths_matrix[s, :] .= result.deaths_by_year
        survivors_matrix[s, :] .= result.survivors_at_payment
        avg_q_matrix[s, :] .= result.avg_q_by_year
        mi_matrix[s, :] .= result.mi_path

        if s == 1
            first_scenario = result
        end
    end

    deterministic_per_life = deterministic_pv_per_life(BASE_Q, MI_MEAN)
    deterministic_portfolio = n_lives * deterministic_per_life

    return (
        scenario_pv = scenario_pv,
        scenario_avg_frailty = scenario_avg_frailty,
        scenario_avg_mi = scenario_avg_mi,
        deaths_matrix = deaths_matrix,
        survivors_matrix = survivors_matrix,
        avg_q_matrix = avg_q_matrix,
        mi_matrix = mi_matrix,
        first_scenario = first_scenario,
        deterministic_per_life = deterministic_per_life,
        deterministic_portfolio = deterministic_portfolio
    )
end

# ============================================================
# RUN AND PRINT RESULTS
# ============================================================

results = run_simulation()

pv = results.scenario_pv

mean_pv = mean(pv)
sd_pv   = std(pv)
p05     = quantile(pv, 0.05)
p50     = quantile(pv, 0.50)
p95     = quantile(pv, 0.95)

avg_deaths_by_year = vec(mean(results.deaths_matrix, dims=1))
avg_survivors_by_year = vec(mean(results.survivors_matrix, dims=1))
avg_q_by_year = vec(mean(results.avg_q_matrix, dims=1))
avg_mi_by_year = vec(mean(results.mi_matrix, dims=1))

println("============================================================")
println("STOCHASTIC PORTFOLIO ANNUITY SIMULATION")
println("============================================================")
@printf("Lives per portfolio                : %d\n", N_LIVES)
@printf("Number of scenarios                : %d\n", N_SCENARIOS)
@printf("Benefit per life per year          : %.2f\n", BENEFIT)
@printf("Discount rate                      : %.2f%%\n", 100 * DISCOUNT)
@printf("Target mean mortality improvement  : %.2f%%\n", 100 * MI_MEAN)
println()

println("Deterministic benchmark (same assumptions as earlier example)")
@printf("PV per life                        : %.2f\n", results.deterministic_per_life)
@printf("PV for %d lives                    : %.2f\n", N_LIVES, results.deterministic_portfolio)
println()

println("Monte Carlo distribution of total portfolio PV")
@printf("Mean PV                            : %.2f\n", mean_pv)
@printf("Std dev PV                         : %.2f\n", sd_pv)
@printf("5th percentile PV                  : %.2f\n", p05)
@printf("Median PV                          : %.2f\n", p50)
@printf("95th percentile PV                 : %.2f\n", p95)
println()

println("Average selected portfolio characteristics")
@printf("Average frailty across scenarios   : %.6f\n", mean(results.scenario_avg_frailty))
@printf("Average annual MI across scenarios : %.4f%%\n", 100 * mean(results.scenario_avg_mi))
println()

println("Average year-by-year experience across scenarios")
println("Year   Avg MI    Avg actual q    Avg deaths    Avg survivors at payment")
println("-----------------------------------------------------------------------")
for t in 1:HORIZON
    @printf(
        "%4d   %6.3f%%    %10.6f    %10.2f    %24.2f\n",
        t,
        100 * avg_mi_by_year[t],
        avg_q_by_year[t],
        avg_deaths_by_year[t],
        avg_survivors_by_year[t]
    )
end

println()
println("First scenario mortality improvement path")
for t in 1:HORIZON
    @printf("Year %2d MI = %6.3f%%\n", t, 100 * results.first_scenario.mi_path[t])
end

println()
println("First scenario deaths by year")
for t in 1:HORIZON
    @printf(
        "Year %2d: deaths = %4d, survivors at payment = %4d\n",
        t,
        results.first_scenario.deaths_by_year[t],
        results.first_scenario.survivors_at_payment[t]
    )
end
