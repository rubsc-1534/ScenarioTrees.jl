using LinearAlgebra
using Statistics
using CairoMakie

"""
    stochastic_nodes(f; b, nsteps, ak, r=2.0, s=2.0, init=nothing)

Input:
- f        : sampler () -> Vector{Float64}
- b        : number of nodes
- nsteps   : number of SA iterations
- ak       : step-size function ak(k)
- r, s     : parameters in update
- init     : optional initial nodes (d × b matrix)

Output:
- nodes    : d × b matrix of support points
- p        : probability vector (length b)
"""
function stochastic_nodes(
    f;
    b::Int,
    nsteps::Int,
    ak,
    r::Float64 = 2.0,
    init = nothing
)

    # --- 1) Initialization ---
    if init === nothing
        x0 = f()
        d = length(x0)
        nodes = hcat([f() for _ in 1:b]...)  # sensible random start
    else
        nodes = copy(init)
        b = length(nodes)
    end

    counts = zeros(Int, b)

    # --- 2) Stochastic approximation loop ---
    for k in 1:nsteps
        ξ = f()
        a = ak(k)

        # find closest node
        dists = [norm(ξ .- nodes[i]) for i in 1:b]
        i_star = argmin(dists)

        counts[i_star] += 1

        # gradient update
        bstar = nodes[i_star]
        delta = bstar .- ξ
        dist = norm(delta)^2

        if dist > 0
            #grad = r * delta #.* abs.(delta).^(s - 1) .* sign.(delta)
            nodes[i_star] -= a .* r .* dist * sign.(delta)[1] #grad
        end
    end

    # --- 3) Probabilities ---
    p = counts ./ sum(counts)

    return nodes, p
end










########################################
# example
########################################
# Sampler: 2D Gaussian
f() = randn(1)

# Step size (Robbins–Monro)
ak(k) = 0.2 / (k+30)^0.75

b=20
init = collect(range(-3, stop=3, length=b))

nodes, p = stochastic_nodes(
    f;
    b = b,
    nsteps = 1_000_000,
    ak = ak,
    r = 2.0,
    init
)

println("Nodes:")
println(nodes)

println("Probabilities:")
println(p)
println("Sum(p) = ", sum(p))

fig = Figure(size= (600, 400))
ax = Axis(fig[1, 1],
    xlabel = "Node index",
    ylabel = "Probability",
    title = "Probabilities of quantization nodes"
)

barplot!(ax, vec(nodes), p)



test = randn(1_000_000)
hist!(ax,test)

fig

