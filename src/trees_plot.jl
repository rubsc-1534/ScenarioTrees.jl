"""
	tree_plot(trr::Tree,fig=1)

Returns the plot of the input tree annotated with density of
probabilities of reaching the leaf nodes in the tree.
Args:
- trr - A scenario tree.
- fig - Specifies the size of the image you want to be returned, default = 1.

Using the Makie version no Python has to be installed.
"""
function tree_plot(trr::Tree;fig = nothing, title = "Tree Model", simple= false, density=true)

    stg = get_stage(trr)
    
    if fig === nothing
        f = Figure(backgroundcolor = :gray80, size = (1000, 700))
    else
        f = fig;
    end

    ga = f[1, 1] = GridLayout();

    axmain = Axis(ga[1, 1], xlabel = "Stages / Time", ylabel = "Values");
    if(density==true)
        axright = Axis(ga[1, 2]);
    end

    if (simple == true)
        for i = 1 : length(trr.structure.parent)
            if stg[i] > 0
                if (trr.state[trr.structure.parent[i]] != 0 && trr.p_edge[trr.structure.parent[i]] > 0)
                    if trr.state[i] != 0 && trr.p_edge[i] >0
                        tmp = DataFrame(x=[stg[i],stg[i]+1], y= [trr.state[trr.parent[i]],trr.state[i]])
                        lines!(axmain,tmp.x,tmp.y)
                    end
                end
            end
        end

    else # simple= false, DEFAULT
        for i = 1 : length(trr.structure.parent)
            if stg[i] > 0
                tmp = DataFrame(x=[stg[i],stg[i]+1], y= [trr.state[trr.structure.parent[i]],trr.state[i]])
                lines!(axmain,tmp.x,tmp.y)
            end
        end

    end
    
    if density==true
        (Yi,_) = get_leaves(trr)
        Yi = [trr.state[i] for i in Yi]
        

        density!(axright, Yi, direction = :y)

        xlims!(axright, low = 0)
        #hidedecorations!(axright, grid = false)
        colgap!(ga, 10)
        rowgap!(ga, 10)
        Label(ga[1, 1:2, Top()], title, valign = :bottom, padding = (0, 0, 5, 0))

        return(f)
    else
        colgap!(ga, 10)
        rowgap!(ga, 10)
        Label(ga[1, 1, Top()], title, valign = :bottom, padding = (0, 0, 5, 0))

        return(f)
    end
end



"""
	plot_hd(newtree::Tree)

Returns a plots of trees in higher dimensions.
"""
function plot_hd(newtree::Tree,fig = nothing, tit = nothing, simple= false)
    
    NumPlot = size(newtree.state,2);
	

    if fig === nothing
        f = Figure(backgroundcolor = :gray80, size = (1000, 700))
    else
        f = fig;
    end
	for i=1:NumPlot
		f[i] = GridLayout()		
	end

    stg = get_stage(newtree)
    
    for rw = 1:size(newtree.state,2)
      axmain = Axis(f[rw][1, 1], xlabel = "Stages / Time", ylabel = "Values");
      axright = Axis(ga[1, 2]);

      for i in range(1,stop = length(newtree.structure.parent))
          if stg[i] > 0
		tmp = DataFrame(x=[stg[i],stg[i]+1], y= [newtree.state[:,rw][newtree.strucutre.parent[i]], newtree.state[:,rw][i]])
                lines!(axmain,tmp.x,tmp.y)
          end
      end
    end
    colgap!(ga, 10)
    rowgap!(ga, 10)
    Label(f[1, Top()], "Tree Model", valign = :bottom, padding = (0, 0, 5, 0))

    return(f)
end


#########################
function tree_plot2(trr::Tree; fig = nothing, title = "Tree Model", simple = false, density = true)

    stg = get_stage(trr)
    parents = trr.structure.parent
    n_edges = length(parents)

    f = fig === nothing ? Figure(backgroundcolor = :gray80, size = (1000, 700)) : fig
    ga = f[1, 1] = GridLayout()

    axmain = Axis(ga[1, 1], xlabel = "Stages / Time", ylabel = "Values")
    axright = density ? Axis(ga[1, 2]) : nothing

    # Pre-allocate coordinates for NaN-separated line segments
    # Each valid edge adds 3 entries: (start, end, NaN)
    x_coords = Float64[]
    y_coords = Float64[]
    sizehint!(x_coords, 3 * n_edges)
    sizehint!(y_coords, 3 * n_edges)

    if simple
        for i in 1:n_edges
            p_idx = parents[i]
            if stg[i] > 0
                if trr.state[p_idx] != 0 && trr.p_edge[p_idx] > 0 && trr.state[i] != 0 && trr.p_edge[i] > 0
                    push!(x_coords, stg[i], stg[i] + 1, NaN)
                    push!(y_coords, trr.state[p_idx], trr.state[i], NaN)
                end
            end
        end
    else
        for i in 1:n_edges
            p_idx = parents[i]
            if stg[i] > 0
                push!(x_coords, stg[i], stg[i] + 1, NaN)
                push!(y_coords, trr.state[p_idx], trr.state[i], NaN)
            end
        end
    end

    # Single plot call for all line segments
    lines!(axmain, x_coords, y_coords)

    if density
        Yi, _ = get_leaves(trr)
        @views leaf_states = trr.state[Yi]

        density!(axright, leaf_states, direction = :y)
        xlims!(axright, low = 0)

        colgap!(ga, 10)
        rowgap!(ga, 10)
        Label(ga[1, 1:2, Top()], title, valign = :bottom, padding = (0, 0, 5, 0))
    else
        colgap!(ga, 10)
        rowgap!(ga, 10)
        Label(ga[1, 1, Top()], title, valign = :bottom, padding = (0, 0, 5, 0))
    end

    return f
end