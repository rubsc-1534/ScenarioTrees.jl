using BenchmarkTools

trr = Tree(Int32[1,2,2,2])
trr2 = Tree(Int32[1,2,2,2])
#trr2 = Tree(Int32[1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])
#trr3 = Tree(Int32[1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,2,2])
samplesize = 1_000_000
p = 2
r = 2
batchsize = 512 #512
g = gaussian_path1D!

# Warm-up runs to compile
rng = MersenneTwister(01012019);
trr = tree_approximation_alloc!(trr, g, 1;batchsize=batchsize, p=p, r=r)
rng = MersenneTwister(01012019);
trr2 = tree_approximation_alloc_buf!(trr2, g, 1;batchsize=batchsize, p=p, r=r) #creates segfaults and other crashes

rng = MersenneTwister(01012019);
tmp = [1.0, 1, 1,1]
g(tmp)
println(tmp)
rng = MersenneTwister(01012019);
g(tmp)
println(tmp)

@benchmark tree_approximation_alloc!(
    tree,
    path,
    $samplesize;
    batchsize=$batchsize,
    p=$p,
    r=$r
) setup = (tree = deepcopy(trr);path = g)



########################
using CUDA

if has_cuda()
    device = CUDA.device()
    sm_count = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    println("Number of Streaming Multiprocessors (SMs): ", sm_count)
else
    println("No CUDA-enabled GPU found.")
end
