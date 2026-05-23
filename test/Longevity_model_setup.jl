using Random
using Statistics
using Printf
using Distributions
using CairoMakie



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
