#using ScenarioTrees
using Test

@testset "ScenarioTrees.jl" begin
    @testset "Predefined tree - Tree 402" begin
        a = Tree(402)
        @test typeof(a) == Tree
        @test length(a.structure.parent) == 15
        @test length(a.state) == length(a.p_edge) == length(a.structure.parent) == 15
        @test sum(a.p_edge) == 8.0
        @test length(a.structure.children) == 15
    end
    @testset "Initial Trees" begin
        init = Tree([1,2,2,2],1)
        @test typeof(init) == Tree
        @test length(init.structure.parent) == 15
        @test length(init.state) == length(init.p_edge) == length(init.structure.parent) == 15
        @test length(init.structure.children) == 15
        @test length(get_stage(init)) == 15
        @test height(init) == 3
        @test length(get_leaves(init)) == 2
        @test nodes(init) == 1:15
        @test length(nodes(init)) == 15

    end
    @testset "A sample of a Scenario Tree 1D" begin
        x = Tree([1,3,3,3,3],1)
        @test typeof(x) == Tree
        @test length(x.structure.parent) == 121
        @test length(x.state) == length(x.p_edge) == length(x.structure.parent) == 121
        @test sum(x.p_edge) ≈ 41.0
        @test length(x.structure.children) == 121
        @test length(get_leaves(x)) == 2
    end
    @testset "A sample of a Scenario Tree 2D" begin
        y = Tree([1,3,3,3,3],2)
        @test typeof(y) == Tree
        @test length(y.structure.parent) == 121
        @test length(y.p_edge) == length(y.structure.parent) == 121
        @test length(y.state) == length(y.structure.parent)*2 == 242
        @test sum(y.p_edge) ≈ 41.0
        @test length(y.structure.children) == 121
    end
    @testset "Sample stochastic functions" begin
        a = gaussian_path1D(4)
        b = running_maximum1D()
        c = path()
        d = gaussian_path2D()
        e = running_maximum2D()
        @test length(a) == 4
        @test length(b) == 4
        @test length(c) == 4
        @test size(d) == (4,2)
        @test size(e) == (4,2)
    end
    @testset "ScenTrees.jl - Tree Approximation 1D" begin
        paths = [gaussian_path1D,running_maximum1D]
        trees = [Tree([1,2,2,2]),Tree([1,3,3,3])]
        samplesize = 100000
        p = 2
        r = 2

        for path in paths
            for newtree in trees
                f = ()->path(4)
                tree_approximation!(newtree,()->path(4),samplesize,p,r)
                @test length(newtree.parent) == length(newtree.state)
                @test length(newtree.parent) == length(newtree.probability)
                @test length(stage(newtree)) == length(newtree.parent)
                @test height(newtree) == maximum(stage(newtree))
                @test round(sum(leaves(newtree)[3]),digits=1) == 1.0   #sum of unconditional probabilities of the leaves
                @test length(root(newtree)) == 1
            end
        end
    end
     @testset "ScenTrees.jl - Tree Approximation batched 1D" begin
        paths = [gaussian_path1D,running_maximum1D]
        trees = [Tree([1,2,2,2]),Tree([1,3,3,3])]
        samplesize = 100000
        p = 2
        r = 2
        for path in paths
            for newtree in trees
                tree_approximation_new!(newtree,path,samplesize,batchsize=32,p=p,r=r)
                @test length(newtree.parent) == length(newtree.state)
                @test length(newtree.parent) == length(newtree.probability)
                @test length(stage(newtree)) == length(newtree.parent)
                @test height(newtree) == maximum(stage(newtree))
                @test round(sum(leaves(newtree)[3]),digits=1) == 1.0   #sum of unconditional probabilities of the leaves
                @test length(root(newtree)) == 1
            end
        end
    end
    @testset "ScenTrees.jl - Tree Approximation 2D" begin
        twoD = tree_approximation!(Tree([1,3,3,3],2),gaussian_path2D,100000,2,2)
        @test size(twoD.state,2) == 2
        @test size(twoD.state,1) == length(twoD.parent) == length(twoD.probability)
    end

    @testset "ScenTrees.jl - Lattice Approximation" begin
        tstLat = lattice_approximation([1,2,3,4],gaussian_path1D,500000,2,1)
        @test length(tstLat.state) == length(tstLat.probability)
        @test round.(sum.(tstLat.probability), digits = 1)  == [1.0, 1.0, 1.0, 1.0] #sum of probs at every stage
    end

    @testset "ScenTrees.jl - Lattice Approximation 2D" begin
        lat2 = lattice_approximation([1,2,3,4],gaussian_path2D,500000,2,2)
        @test length(lat2) == 2 # resultant lattices are 2
        @test length(lat2[1].state) == length(lat2[1].probability)
        @test round.(sum.(lat2[1].probability), digits = 1)  == [1.0, 1.0, 1.0, 1.0]
        @test round.(sum.(lat2[2].probability), digits = 1)  == [1.0, 1.0, 1.0, 1.0]
    end
   
end
