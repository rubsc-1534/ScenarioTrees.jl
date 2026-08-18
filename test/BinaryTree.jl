using Graphs
using GraphMakie
using CairoMakie

# 1. Construct the graph with 15 nodes
g = SimpleDiGraph(15)

# Standard tree edges (black lines)
edges = [
    (1, 2), (1, 3),
    (2, 4), (2, 5),
    (3, 6), (3, 7),
    (6, 12), (6, 13)
]

for (u, v) in edges
    add_edge!(g, u, v)
end

# Arrow edges pointing back into nodes 4, 5, 7 (red arrows)
red_edges = [
    (8, 4), (9, 4),
    (10, 5), (11, 5),
    (14, 7), (15, 7)
]

for (u, v) in red_edges
    add_edge!(g, u, v)
end

# 2. Define exact node coordinates (horizontal tree layout)
positions = [
    Point2f(0, 0),      # Node 1
    
    Point2f(1, 1.5),    # Node 2
    Point2f(1, -1.5),   # Node 3
    
    Point2f(2, 2.25),   # Node 4
    Point2f(2, 0.75),   # Node 5
    Point2f(2, -0.75),  # Node 6
    Point2f(2, -2.25),  # Node 7
    
    Point2f(3, 2.6),    # Node 8
    Point2f(3, 1.9),    # Node 9
    Point2f(3, 1.1),    # Node 10
    Point2f(3, 0.4),    # Node 11
    Point2f(3, -0.4),   # Node 12
    Point2f(3, -1.1),   # Node 13
    Point2f(3, -1.9),   # Node 14
    Point2f(3, -2.6)    # Node 15
]

# 3. Styling definitions
red_nodes = [8, 9, 10, 11, 14, 15]

node_colors = [i in red_nodes ? :crimson : :black for i in 1:15]
node_text_colors = node_colors

# Edge colors and arrow settings
edge_colors = [dst(e) in [4, 5, 7] && src(e) in red_nodes ? :crimson : :black for e in Graphs.edges(g)]
arrow_sizes = [color == :crimson ? 15 : 0 for color in edge_colors]

# 4. Plot the graph
fig, ax, p = graphplot(
    g,
    layout = _ -> positions,
    node_color = :white,
    node_strokecolor = node_colors,
    node_strokewidth = 2,
    node_size = 35,
    nlabels = string.(1:15),
    nlabels_color = node_text_colors,
    nlabels_align = (:center, :center),
    nlabels_distance = 0,
    edge_color = edge_colors,
    edge_width = 1.5,
    arrow_size = arrow_sizes,
    arrow_shift = :end
)

hidedecorations!(ax)
hidespines!(ax)

# Save figure
save("binary_tree.png", fig)