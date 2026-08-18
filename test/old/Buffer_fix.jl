
trr = Tree(Int32[1,2,2,2])
trr2 = Tree(Int32[1,2,2,2])
samplesize = 10_000_000
p = 2; r = 2
batchsize = 1 #512
g = gaussian_path1D!

# Warm-up runs to compile
rng = MersenneTwister(01012019);
trr = tree_approximation_alloc!(trr, g, samplesize;batchsize=batchsize, p=p, r=r)
rng = MersenneTwister(01012019);
trr2 = tree_approximation_alloc_buf!(trr2, g, samplesize;batchsize=batchsize, p=p, r=r)

tree_plot(trr)
tree_plot(trr2)
###########
#trr and trr2 are not identical -> buffered version creates errors
#first check "correct version tree_approximation_alloc!
nIterations = 1

##########


T = height(trr)
nleaf = size(trr.state, 1)

# ---- Preallocation ----
probaLeaf = zeros(Float64, nleaf)
d = zeros(Float64, nleaf)
samplepaths = Array{Float64}(undef, min(nIterations,batchsize), T + 1)

# ---- Main loop ----
nBatches = cld(nIterations, batchsize)
batchsize= min(nIterations,batchsize)
   
@views g(samplepaths[1, :])
samplepaths

# Apply SA steps
@inbounds for b = 1:batchsize
    sa_step!(
        trr,
        view(samplepaths, 1, :),
        probaLeaf,
        d,
        p,
        r,
    )
end






















rng = MersenneTwister(01012019);


#######################################################
# Now buffered version step-by-step
T = height(trr2)
nleaf = size(trr2.state, 1)

# ---- Preallocation ----
probaLeaf = zeros(Float64, nleaf)
d = zeros(Float64, nleaf)
samplepaths = Array{Float64}(undef, min(nIterations,batchsize), T + 1)
nBatches = cld(nIterations, batchsize)
batchsize = min(nIterations,batchsize)

# ---- Create thread-local buffers ----
nthreads = Threads.nthreads()
nbuf = Threads.maxthreadid()


buffers = [
    SABuffer(
        zeros(Float64, nleaf),              # leaf_count
        zeros(Float64, nleaf),              # d_accum
        zeros(Float64,size(trr2.state)),    # gradient
        false                               # 
    )
    for _ in 1:nbuf #nthreads
]

# ---- Main loop ----


tid = Threads.threadid()
buf = buffers[tid]
buf.used = true

# ---- Generate batch ----
@views g(samplepaths[1, :])
samplepaths

# ---- SA steps (buffered) ----
sa_step_buffered!(trr2,view(samplepaths, 1, :),buf,p,r)
buf