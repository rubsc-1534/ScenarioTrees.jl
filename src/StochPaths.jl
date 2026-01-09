#All these examples of path are in 4 stages

using Distributions, Random
rng = MersenneTwister(01012019);

"""
	gaussian_path1D(out::AbstractVector{Float64})

Returns a 'nx1' dimensional array of Gaussian random walk where n is determined by the length of out. 
"""
function gaussian_path1D!(out::AbstractArray{Float64})
    s = 0.0
    @inbounds for i in eachindex(out)
        s += randn(rng)
        out[i] = s
    end
    out[1] = 0.0
    return nothing
end

"""
	gaussian_path2D()

Returns a '4x2' dimensional array of Gaussian random walk
"""
function gaussian_path2D(n=4)
    gsmatrix = randn(rng, n, 2) * [1.0 0.0 ; 0.9 0.3] #will create an (dimension x nstages) matrix
    gsmatrix[1,:] .= 0.0
    return cumsum(gsmatrix .+ [1.0 0.0], dims = 1)
end

"""
	running_maximum1D()

Returns a '4x1' dimensional array of Running Maximum process.
"""
function running_maximum1D(n=4)
    rmatrix = vcat(0.0, cumsum(randn(rng, n-1, 1), dims = 1))
    for i = 2 : 4
        rmatrix[i] = max.(rmatrix[i-1], rmatrix[i])
    end
    return rmatrix
end

"""
	running_maximum2D()

Returns a '4x2' dimensional array of Running Maximum process.
"""
function running_maximum2D(n=4)
    rmatrix = vcat(0.0, cumsum(randn(rng, n-1, 1), dims = 1))
    rmatrix2D = zeros(n, 2)
    rmatrix2D[:,1] .= vec(rmatrix)
    for j = 2 : 2
        for i = 2 : n
            rmatrix2D[i,j] = max.(rmatrix[i-1], rmatrix[i])
        end
    end
    return rmatrix2D * [1.0 0.0; 0.9 0.3]
end

"""
	path()

Returns a sample of stock prices following the a simple random random process.
"""
function path(n=4)
    return  100 .+ 50 * vcat(0.0, cumsum(randn(rng, n-1, 1), dims = 1))
end
