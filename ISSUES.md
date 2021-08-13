Use this file to list and control issues/features/to-dos

(At this moment, this repository is mirrored on GitHub, which has its own project management tools.
At this moment, we prefer to use a ISSUES file.)

Possible values for status: OPEN, COMPLETED, IGNORED

# ISSUE #0001 COMPLETED

Define good way to versioning (and integrate paradigm to Git branching)

# ISSUE #0002 OPEN

Create test suite, integrate to code development and production

# ISSUE #0003 OPEN

Split classes (currently in single file) into 4 modules:
namedtuples (Edge/Arrow/Vertex)
path/cycles (VertexPath and subclasses)
digraphs/graphs (Digraph and subclasses)
others (the auxiliary classes created to help with specific, complex digraph algorithms)

Should be done after #0004

# ISSUE #0004 OPEN

Move classes Edge, Arrow, Vertex outside Digraph, refactor to accomodate changes

# ISSUE #0005 OPEN

Refactor methods to integrate VertexPath/VertexCycle
Methods using paths and cycles currently use their own implementation of paths and cycles

# ISSUE #0006 OPEN

Finish writing method get_all_paths_shortest_paths_via_Johnsons

# ISSUE #0007 OPEN

Eliminate bugs on method get_all_paths_shortest_paths_via_Floyd_Warshals

# ISSUE #0008 OPEN

Eliminate bugs on method get_single_source_shortest_paths_via_Bellman_Fords

# ISSUE #009 OPEN

Implement methods get_hamiltonian_path()/solve_traveling_salesman_problem()
Also means sorting out the methods in class StateDigraphSolveTSP

# ISSUE #0010 OPEN

Implement methods to merge vertices into a single vertex (collapse vertex)
This is useful for certain algorithms

Should be done after #0011

# ISSUE #0011 OPEN

Implement methods to delete arrows/edges/vertices of graph/digraph

# ISSUE #0012 OPEN

Build a Python package out of the code
(using __init__.py files, ideally with good import statements, optionally also folders)

# ISSUE #0013 OPEN

Potentially release this repository as a public package.
PyPI is an option.

Should be done after #0012
