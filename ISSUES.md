# ISSUES

Use this file to list and control issues/features/to-dos

(At this moment, this repository is mirrored on GitHub, which has its own project management tools.
At this moment, we prefer to use a ISSUES file.)

Possible values for status: OPEN, COMPLETE, IGNORED, ONGOING.

## ISSUE #0001 COMPLETE

Define good way to versioning (and integrate paradigm to Git branching)

## ISSUE #0002 COMPLETE

Create test suite, aiming to integrate it into code development and production.
Initially using built-in package unittest.

## ISSUE #0003 COMPLETE

Split classes (currently in single file) into 4 modules:
vertices_arrows_and_edges (Edge/Arrow/Vertex)
path_and_cycles (VertexPath and subclasses)
graphs_and_digraphs (Digraph and subclasses)
algorithm_oriented_classes (the auxiliary classes created to help with specific, complex digraph algorithms)

Should be done after #0004

## ISSUE #0004 COMPLETE

Move classes Edge, Arrow, Vertex outside Digraph, refactor to accomodate changes

## ISSUE #0005 OPEN

Refactor methods to integrate VertexPath/VertexCycle
Methods using paths and cycles currently use their own implementation of paths and cycles

## ISSUE #0006 OPEN

Finish writing method get_all_paths_shortest_paths_via_Johnsons

## ISSUE #0007 OPEN

Eliminate bugs on method get_all_paths_shortest_paths_via_Floyd_Warshals

## ISSUE #0008 OPEN

Eliminate bugs on method get_single_source_shortest_paths_via_Bellman_Fords

## ISSUE #0009 COMPLETE

Implement methods get_hamiltonian_path()/solve_traveling_salesman_problem()
Also means sorting out the methods in class StateDigraphSolveTSP

## ISSUE #0010 OPEN

Implement methods to merge vertices into a single vertex (collapse vertex)
This is useful for certain algorithms

Should be done after #0011

## ISSUE #0011 OPEN

Implement methods to delete arrows/edges/vertices of graph/digraph

## ISSUE #0012 COMPLETE

Build a Python package out of the code
Currently uses a setup.py to use with pip.
The package is named homemadegraph - pip doesn't work well with underscores
The project is currently called homemade_graphs

## ISSUE #0013 OPEN

Potentially release this repository as a public package.
PyPI is an option.

Should be done after #0012

## ISSUE #0014 OPEN

Add option require namedtuple to methods of VertexPath (to add flexibility).

## ISSUE #0015 COMPLETE

Simplify/strealine the flow of method StateDigraphSolveTSP.solve_full_problem,
which has currently about 350 lines and many if/else's.
Idea: use submethods to split top-down/bottom-up (memoization/tabulation) variants.
The logic for each case discriminated by the if/else's goes into a submethod.
Can also split into the path and the cycle variants.
Maybe store variables in class attributes, or carry them as method arguments,
as the operations and procedures demand.

## ISSUE #0016 COMPLETE

Consider renaming top-down/bottom-up to memoization/tabulation in
solve_full_problem and solve_subproblem from class StateDigraphSolveTSP.

## ISSUE #0017 COMPLETE

In a certain way, redo Issue 0003 to make it better and make the imports work.
The intention is: to have each class to be referred to its name in its native module.
For example, when calling/referring/using/creating a Vertex, use Vertex
instead of src.vertices_arrows_and_edges.Vertex or anything like that.
(Or being happily able to use homemade_graphs.Vertex, if using 
"import homemade_graphs" instead of "from homemade_graphs import \*".)
(P.S.: In MarkDown, \* triggers italic, so we precede it by a backslash.)

## ISSUE #0018 COMPLETE

Work on methods in Graph/Digraph connected to TSP problem.
(Right now most of the work is in the class StateDigraphSolveTSP
but that work needs to be "redirected" the right way.)

## ISSUE #0019 OPEN

Work on method to get vertex covers: Digraph.get_vertex_cover_with_limited_size

## ISSUE #0020 OPEN

Work on hypohamiltonian (di)graphs: where the (di)graph is not hamiltonian but
the remotion of any vertex makes the resulting (di)graph hamiltonian.

Should be done after #0011

## ISSUE #0021 COMPLETE

Define equality of (di)graphs. To be equal, digraphs must have vertices of same
names and the same arrows/edges (including their weight) between them.
Note: always checks arrows

## ISSUE #0022 OPEN

Design way to verify two (di)graphs are isomorphic.

## ISSUE #0023 ONGOING

Design good ways to print a (di)graph. The solution can be a textual or a
visual representations.

## ISSUE #0024 COMPLETE

Verify that initiation in Digraph works correctly when having data_type
being "all_vertices_and_all_arrows" and variations

## ISSUE #0025 COMPLETE

Abandon the idea of a "flat" package (simulating a single-module package)
and move to have relative/siblings/parents modules. Set up abbreviations
in siblings to allow for successful call without blotting up the code.

## ISSUE #0026 OPEN

Consider using setup.cfg which is more flexible/modern than setup.py
A .toml file also generalizes setup.py. Consult PEP 518

## ISSUE #0027 OPEN

Since package name is currently homemadegraphs, consider renaming things.
One thing that should be renamed is the header of every file, which mentions
"homemade_graphs" instead of "homemadegraphs".

## ISSUE #0028 ONGOING

Create tests for every method/function/operation/functionality in package.
(This is bound to be an almost permanent issue, as any new, future functionality
should also have corresponding testing implemented.)

## ISSUE #0029 COMPLETE

Work on sanitize_arrow_or_edge to have an optional argument
also_sanitize_vertices which, if True, orders sanitization of the vertices
(that is, the first and second items of the Arrow/Edge namedtuple)

## ISSUE #0030 COMPLETE

Make a better Digraph.__repr__ to write the name of the class without
<class ...>.

## ISSUE #0031 COMPLETE

Create function that takes cities with coordinates on a map and makes a
complete weighted graph whose vertices are the cities and the distances
becomes the weights of the edges.
Motivation: traveling salesman problem.

## ISSUE #0032 OPEN

Find consistent way to raise AssertionError and other Errors in whole code.
Maybe also do good Warnings in good places.

## ISSUE #0033 OPEN

Consider extending Issue #0031 to enable building Graphs/Digraphs from files.
The file might have info on arrows, on edges, or even pre a pre-input for
WeightedGraph.from_names_and_coordinates.

## ISSUE #0034 OPEN

Consider renaming functions in Digraph to increase clarity
For example, the current method "get_arrows_out" does not indicate in its name
whether it returns self._arrows_out or the value of self._arrows_out at
a specific key/vertex.

## ISSUE #0035 ONGOING

Since many methods have many arguments, consider always "towering" them
(that is, putting one in each line, as done for a few function/method calls)
in their definitions

## ISSUE #0036 COMPLETE

Generalize testing by the use of generic testing classes. For example,
a class for initialization (that tests all ways something can be initialized
and whether they return the correct results) which is subclassed and
applied in different contexts (p. ex. Digraphs, Graphs, empty digraph)

## ISSUE #0037 ONGOING

Create template for testing classes (template meaning an "abstract" class)
with multiple objects and methods. Test each object against a subset of the methods,
trying to match with specification.
Probably need a class more complicated than GenericObjectTestCase,
and maybe more complex PropertySpecifications.

## ISSUE #0038 COMPLETE

Created a common method which serves both paths and cycles
on collecting solutions for all the pairs of initial and final vertices.

Original, discarded idea (might be implemented but it is unlikely):
Change StateDigraphSolveTSP to only essentially handle cycles.
In case of paths, modify graph to have a new vertex. This vertex would have
arrows of weight 0 going into all original initial vertices and receiving
arrows of weight 0 from all final vertices. The solution of the path problem
would be the solution of the cycle problem for the extended graph (removing
the new vertex).

## ISSUE #0039 COMPLETE

The determination of omit_minimizing_path was split off from
StateDigraphSolveSTP._produce_auxiliary_constructs into separate method
should_omit_minimizing_path

## ISSUE #0040 ONGOING

Consider option the vertices of a VertexPath or VertexCycle using their names only.
That is, generate 'A' instead of Vertex(name='A').
Can also do this on other classes such as Digraph and subclasses. For example,
a method called get_vertex_names.
This is done through method OperationsVAE.simplified_representation.

## ISSUE #0041 COMPLETE

Allow get_neighbors/get_neighbors_in/get_neighbors/out methods in Digraph
to allow or disallow for repeated vertices.

## ISSUE #0042 COMPLETE

In VertexPath/VertexCycle, move 'arrows' and 'vertices' to be lists '_arrows' and '_vertices'.
Define get_arrows() and get_vertices() as interface.

## ISSUE #0043 OPEN

Idea: implement classes such as EnhancedVertex, EnhancedArrow
and EnhancedEdge, to be classes corresponding to the namedtuples Vertex, Arrow, Edge.
This way, namedtuples can be used for expensive computations, and the enhanced
versions add flexibility to the concepts and specific methods.
Counter-argument: lot of this flexibility is already achieved by OperationsVAE.

## ISSUE #0044 ONGOING

Work on improving performance on StateDigraphSolveTSP.
Currently: doing memoization is about 10 times faster than tabulation,
but execution time about doubles for every additional vertex in a complete graph
(as expected from the algorithm). If it reaches RAM limit, even worse,
so it's important to downsize also the memory consumption.
