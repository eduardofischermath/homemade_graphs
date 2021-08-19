# ISSUES

Use this file to list and control issues/features/to-dos

(At this moment, this repository is mirrored on GitHub, which has its own project management tools.
At this moment, we prefer to use a ISSUES file.)

Possible values for status: OPEN, COMPLETE, IGNORED, ONGOING.

## ISSUE #0001 COMPLETE

Define good way to versioning (and integrate paradigm to Git branching)

## ISSUE #0002 OPEN

Create test suite, integrate to code development and production

## ISSUE #0003 COMPLETE

Split classes (currently in single file) into 4 modules:
namedtuples (Edge/Arrow/Vertex)
path/cycles (VertexPath and subclasses)
digraphs/graphs (Digraph and subclasses)
others (the auxiliary classes created to help with specific, complex digraph algorithms)

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

# ISSUE #0009 COMPLETE

Implement methods get_hamiltonian_path()/solve_traveling_salesman_problem()
Also means sorting out the methods in class StateDigraphSolveTSP

# ISSUE #0010 OPEN

Implement methods to merge vertices into a single vertex (collapse vertex)
This is useful for certain algorithms

Should be done after #0011

## ISSUE #0011 OPEN

Implement methods to delete arrows/edges/vertices of graph/digraph

## ISSUE #0012 OPEN

Build a Python package out of the code
(using __init__.py files, ideally with good import statements, optionally also folders)

## ISSUE #0013 OPEN

Potentially release this repository as a public package.
PyPI is an option.

Should be done after #0012

## ISSUE #0014 OPEN

Add option require namedtuple to methods of VertexPath (to add flexibility).

## ISSUE #0015 OPEN

Simplify/strealine the flow of method StateDigraphSolveTSP.solve_full_problem,
which has currently about 350 lines and many if/else's.
Idea: se submethods to split top-down/bottom-up (memoization/tabulation) variants.
The logic for each case discriminated by the if/else's goes into a submethod.
Can also split into the path and the cycle variants.
Maybe store variables in class attributes, or carry them as method arguments,
as the operations and procedures demand.

## ISSUE #0016 OPEN

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

## ISSUE #0018 OPEN

Work on methods in Graph/Digraph connected to TSP problem.
(Right now most of the work is in the class StateDigraphSolveTSP
but that work needs to be "redirected" the right way.)

## ISSUE #0019 OPEN

Work on method to get vertex covers: get_vertex_cover_with_limited_size

## ISSUE #0020 OPEN

Work on hypohamiltonian (di)graphs: where the (di)graph is not hamiltonian but
the remotion of any vertex makes the resulting (di)graph hamiltonian.

Should be done after #0011

## ISSUE #0021 OPEN

Define equality of (di)graphs. To be equal, digraphs must have vertices of same
names and the same arrows/edges (including their weight) between them.

## ISSUE #0022 OPEN

Design way to verify two (di)graphs are isomorphic.

## ISSUE #0023 OPEN

Design good ways to print a (di)graph. The solution can be a textual or a
visual representations.

## ISSUE #0024 OPEN

Verify that initiation in Digraph works correctly when having data_type
being "all_vertices_and_all_arrows" and variations
