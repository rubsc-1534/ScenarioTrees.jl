using BenchmarkTools

trr = Tree(Int32[1,2,2,2])
trr2 = Tree(Int32[1,2,2,2])
#trr2 = Tree(Int32[1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])
#trr3 = Tree(Int32[1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,2,2])
samplesize = 1_000_000
p = 2
r = 2
batchsize = 1024 #512
g = gaussian_path1D!

# Warm-up runs to compile
rng = MersenneTwister(01012019);
trr = tree_approximation_alloc!(trr, g, samplesize;batchsize=batchsize, p=p, r=r)

tree_plot(trr)


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


for i in 1:length(trr.structure.children)
    println(trr.structure.children[i])
end