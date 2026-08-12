# Required packages: 
# Pkg.add(["Distributions", "Plots"])
using Random, Statistics, Distributions
using Plots

function simulate_pure_stochastic_longevity(;
    B = 0.0001,          # Baseline mortality scale
    C = 0.085,           # Baseline aging rate
    x = 65.0,            # Initial age
    r = 0.03,            # Risk-free discount rate (3%)
    phi_0 = 0.015,       # Initial improvement rate (1.5%)
    phi_bar = 0.015,     # Long-term mean improvement rate (1.5%)
    kappa = 0.1,         # Speed of mean reversion
    sigma_phi = 0.004,   # Volatility of the structural trend
    T = 50.0,            # Simulation horizon (years)
    dt = 1/12,           # Time step (monthly)
    N_paths = 10_000,    # Number of Monte Carlo trajectories
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
    
    # 3. Force of Mortality & Survival Probability
    # Baseline mortality at each time step: μ(x+t, 0)
    mu_base = B .* exp.(C .* (x .+ time_grid')) # Broadcasting over row vector
    
    # Stochastic mortality paths: μ(x+t, t) = μ(x+t, 0) * exp(-Y_t)
    mu_path = mu_base .* exp.(-Y)
    
    # Cumulative hazard H_t = ∫ μ(x+s, s) ds
    H = cumsum(mu_path .* dt, dims=2)
    
    # Survival probabilities S(x, t) = exp(-H_t)
    S = exp.(-H)
    
    # 4. Annuity Reserve Calculation (Continuous Integration)
    # Discount curve: e^{-rt}
    discount = exp.(-r .* time_grid') 
    
    # a_x = ∫ e^{-rt} S(t) dt
    reserves = sum(discount .* S .* dt, dims=2) |> vec
    
    return time_grid, phi, Y, reserves
end

# --- RUNNING THE SIMULATION ---
println("Running Monte Carlo longevity simulation (10,000 paths)...")
time_grid, phi_paths, Y_paths, reserves = simulate_pure_stochastic_longevity()

# --- CALCULATING RISK METRICS ---
mean_res = mean(reserves)
p95_res = quantile(reserves, 0.95)
p995_res = quantile(reserves, 0.995) # Solvency II VaR equivalent

println("\n=== ANNUITY RESERVE METRICS (Age 65) ===")
println("Expected Reserve (Best Estimate): \$$(round(mean_res, digits=3))")
println("95th Percentile Reserve:        \$$(round(p95_res, digits=3))")
println("99.5th Percentile (VaR SCR):    \$$(round(p995_res, digits=3))")
println("Implied Longevity Capital (SCR): \$$(round(p995_res - mean_res, digits=3)) per \$1 of annuity")

# --- VISUALIZATION ---
# Plot 1: Cumulative Improvement (Y_t) Paths (Sample 100 paths)
p1 = plot(time_grid, Y_paths[1:100, :]', 
          title="Cumulative Mortality Improvement (Y_t)",
          xlabel="Years from Valuation", 
          ylabel="Cumulative Improvement",
          legend=false, 
          alpha=0.3, 
          linecolor=:blue)

# Plot 2: Annuity Reserve Distribution
p2 = histogram(reserves, 
               bins=50, 
               normalize=:probability, 
               title="Distribution of Annuity Reserves",
               xlabel="Reserve Value (\$)", 
               ylabel="Probability", 
               legend=false,
               fillcolor=:steelblue, 
               linecolor=:white)

# Add vertical lines for Mean and 99.5% VaR
vline!(p2, [mean_res], line=(:red, 2, :dash), label="Mean")
vline!(p2, [p995_res], line=(:darkred, 2, :solid), label="99.5% VaR")

# Display plots side-by-side
plot(p1, p2, layout=(1, 2), size=(1000, 400), margin=5Plots.mm)