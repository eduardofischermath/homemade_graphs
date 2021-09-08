########################################################################
# DOCUMENTATION / README
########################################################################

# File belonging to package "homemade_graphs"
# Implements classes and algorithms related to graphs and digraphs.

# For more information on functionality, see README.md
# For more information on bugs and planned features, see ISSUES.md
# For more information on the versioning, see RELEASES.md

# Copyright (C) 2021 Eduardo Fischer

# This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU Affero General Public License version 3
#as published by the Free Software Foundation. See LICENSE.
# Alternatively, see https://www.gnu.org/licenses/.

# This program is distributed in the hope that it will be useful,
#but without any warranty; without even the implied warranty of
#merchantability or fitness for a particular purpose.

########################################################################
# External imports
########################################################################

from itertools import product as itertools_product
from math import inf as math_inf
# Since cache from functools was introduced in Python version >= 3.9,
#we check for it. If not new enough, we go with lru_cache(maxsize = None)
#(available from Python 3.2 and up, the minimum for the package)
#and bind the decorator to the name functools_cache
# Alternative is try/except, but comparing versions is also ok
from sys import version_info as sys_version_info
if sys_version_info >= (3, 9):
  from functools import cache as functools_cache
else:
  from functools import lru_cache as functools_lru_cache
  functools_cache = functools_lru_cache(maxsize=None)
  # In code always call it through functools_cache

########################################################################
# Internal imports
########################################################################

from homemadegraphs.paths_and_cycles import VertexPath, VertexCycle
from homemadegraphs.vertices_arrows_and_edges import OperationsVAE

########################################################################
# Class StateDigraphGetCC
########################################################################

class StateDigraphGetSCC(object):
  '''
  Instances used to record the state of method Digraph.get_sccs()
  
  Attributes:
  _graph
  _vertices_ranked
  _n
  
  Temporary attributes (using only for SCC algorithm, then deleted):
  t
  s
  leaders
  explored
  new_rank
  '''
  
  def __init__(self, graph):
    '''
    Initializes the instace.
    '''
    self._graph = graph
    self._vertices_ranked = self._graph.get_vertices() # Lists the keys, the vertices
    self._n = len(self._vertices_ranked) # Same as self.get_number_of_vertices()
    
  def manually_change_graph(self, graph):
    '''
    Changes the graph to another graph.
    
    This new graph should have the same vertices (p. ex. inverted graph.)
    '''
    # This should not alter self._n nor self._vertices_ranked
    self._graph = graph

  def manually_change_vertices_ranked(self, vertices_ranked):
    '''
    Changes the order of the vertices in the instance.
    
    This new rank should have the same vertices.
    '''
    self._vertices_ranked = vertices_ranked
    
  def dfs_outer_loop(self):
    # This is the right place. These are all new definitions
    # (Since we assume delete them attributes at the end.)
    self.t = 0 # A number from 0 to n-1
    self.s = None # Should be None or a vertex
    self.leaders = {vertex: None for vertex in self._vertices_ranked}
    self.explored = {vertex: False for vertex in self._vertices_ranked}
    self.new_rank = [None]*self._n
    # Now we execute the algorithm
    for i in range(self._n - 1, -1, -1):
      current_vertex = self._vertices_ranked[i]
      if not self.explored[current_vertex]:
        #print(f'{i} vertices yet to explore')
        self.s = current_vertex
        self.dfs_inner_loop(current_vertex)
    # We "deliver" the results which are new_rank (useful for first pass)
    #and leaders (useful for second pass)
    # We could make new_rank replace self._vertices_ranked but we won't
    # We will even delete the reference to the attributes
    new_rank = list(self.new_rank) # Makes a fresh copy
    leaders = dict(self.leaders)
    delattr(self, 't')
    delattr(self, 's')
    delattr(self, 'leaders')
    delattr(self, 'explored')
    delattr(self, 'new_rank')
    return (new_rank, leaders)

  def dfs_inner_loop(self, fixed_vertex):
    '''
    Performs an internal operation on self, based on fixed_vertex of self._graph.
    
    It assumes the existence of the attibute new_rank. Called only by dfs_loop.
    '''
    self.explored[fixed_vertex] = True
    self.leaders[fixed_vertex] = self.s
    for vertex in self._graph.get_neighbors_out(fixed_vertex):
      if not self.explored[vertex]:
        self.dfs_inner_loop(vertex)
    self.new_rank[self.t] = fixed_vertex # Puts fixed_vertex at rank self.t
    # A little debug routine
    #print(f'Marked vertex {fixed_vertex} as rank {self.t}')
    #print(f'{self.explored=}')
    #print(f'{self.leaders=}\n')
    # Could use append instead of t with the same effect, since it goes in order
    # Nonetheless, we prefer to follow the original algorithm closely
    self.t += 1
    # Note that this is a procedure and not a method and thus we don't return anything

########################################################################
# Class StateGraphGetCC
########################################################################

class StateGraphGetCC(object):
  '''
  Instances used to record the state of method
  HomemadeGraph.get_ccs()
  
  Attributes:
  _graph
  
  Temporary attributes (using only for CC algorithm, then deleted):
  self.current_label
  
  Goal attributes:
  vertices_by_label
  components
  number_components
  '''
  
  def __init__(self, graph):
    '''
    Initializes the instance
    '''
    self._graph = graph
  
  def dfs_outer_loop(self):
    '''
    Returns the connected components of self._graph using depth-first search.
    '''
    # In this process we scan all vertices to find the connected components
    # We can do with either depth-first or breadth-first search either way
    # We can do with either depth-first or breadth-first search either way
    # Each time we explore a vertex, we mark it with a label (a number)
    # (This is how we mark it explored for DFS)
    # To make it easy, let's do component 0, 1, 2, and so on
    # Two vertices will have the same label iff they are in the same component
    # Let's do it with a dictionary. We start with the None label
    self.components = {vertex:None for vertex in self._graph.get_vertices()}
    # We assign labels to unlabeled vertices, starting from 0
    self.current_label = 0
    # We also control the vertices with each existing label using a dict
    self.vertices_by_label = {}
    # Regarding the vertices, we can scan them in order
    # (And update the label for the whole connected component)
    for vertex in self._graph.get_vertices():
      if self.components[vertex] is None:
        # We mark as explored right away using self.components[]
        self.components[vertex] = self.current_label
        # We could use a default dict but we'll do it manually
        if self.current_label not in self.vertices_by_label:
          self.vertices_by_label[self.current_label] = []
        self.vertices_by_label[self.current_label].append(vertex)
        # The following being right here characterizes depth-first search
        # (Technically it's a depth-based search and not a depth-first search, but ok)
        for other_vertex in self._graph.get_neighbors(vertex, skip_checks = True):
          if self.components[other_vertex] is None:
            self.dfs_inner_loop(other_vertex)
        # Now that everyone reachable using DFS was reached, we close the loop
        self.current_label += 1
    # The number of components is exactly the current value of self.current_label
    self.number_components = self.current_label
    # We output what is relevant, and delete the rest
    # That is, we output the components as given by self.components,
    #self.vertices_by_label, self.components, and self.number_components
    vertices_by_label = dict(self.vertices_by_label)
    components = dict(self.components)
    number_components = self.number_components
    delattr(self, 'vertices_by_label')
    delattr(self, 'components')
    delattr(self, 'number_components')
    delattr(self, 'current_label')
    return (vertices_by_label, components, number_components)
          
  def dfs_inner_loop(self, fixed_vertex):
    '''
    Inner loop of deapth-first search.
    
    Does not return anything, only changes self.
    '''
    # We mark fixed_vertex as explored, update the status of self
    self.components[fixed_vertex] = self.current_label
    if self.current_label not in self.vertices_by_label:
      self.vertices_by_label[self.current_label] = []
    self.vertices_by_label[self.current_label].append(fixed_vertex)
    # We know try all edges and do the process
    for other_vertex in self._graph.get_neighbors(fixed_vertex, skip_checks = True):
      if self.components[other_vertex] is None:
        self.dfs_inner_loop(other_vertex)
    # We don't return anything, only change self

########################################################################
# Class StateDigraphSolveTSP
########################################################################

class StateDigraphSolveTSP(object):
  '''
  Used to help with method solve_traveling_salesman_problem.
  '''
  
  def __init__(self, digraph):
    '''
    Magic method. Initializes the instance.
    '''
    self.digraph = digraph
    self.n = self.digraph.get_number_of_vertices()
    # Let's create a relationship between vertices and their indices
    self.number_by_vertex = {}
    self.vertex_by_number = {}
    # Note Vertex is a namedtuple and thus it is hasheable
    for idx, vertex in enumerate(self.digraph.get_vertices()):
      self.number_by_vertex[vertex] = idx
      self.vertex_by_number[idx] = vertex

  @staticmethod
  def produce_boolean_tuples_with_fixed_sum(given_length, given_sum,
      output_as_generator = False):
    '''
    Returns the tuples of True/False with a specific number of each.
    
    Used to generate presence sets with a specific number of total vertices
    [given_length] and a specific number of present vertices [given_sum]
    '''
    # We generate all tuples of given length with True/False values
    generator_of_all_tuples = itertools_product((True, False), repeat = given_length)
    # We restrict the tuples to the ones with given number of True and Falses
    # (Given by sum, meaning this sum is the number of True values)
    # We use a Boolean lambda function and a filter
    filter_criterion = lambda tuplee: (sum(tuplee) == given_sum)
    generator_for_needed_tuples = filter(filter_criterion, generator_of_all_tuples)
    if output_as_generator:
      return generator_for_needed_tuples
    else:
      return list(generator_for_needed_tuples)

  @functools_cache
  def solve_subproblem(self, initial_vertex, final_vertex, presence_set,
      use_memoization_instead_of_tabulation = False, omit_minimizing_path = False, skip_checks = False):
    '''
    Computes the minimal path length given specific parameters: given
    initial and final vertices and a set of vertices of underlying graph
    self.digraph [given by a tuple of Booleans], finds minimal among paths
    traveling once though each vertex satisfyng the boundary conditions.
    
    Returns the minimal weight of such path, and also, if requested,
    also one of these minimizing paths as VertexPath instance. (If request is
    for its ommission, produces None as such path to fill the space.)
    
    [Note this method is only about paths without any self-crossings, not about cycles.]
    '''
    # Our subproblems are: Consider we have a fixed initial vertex
    #(which might be passed as argument as source_vertex), a fixed
    #final vertex, and a set of the vertices including those two. We want
    #the minimal length of the paths going through each vertex once
    #(if there are such paths), starting at initial and ending on the final
    # We will parametrize these subproblems by initial, final, presence_set,
    #where presence_set is a tuple of n Booleans, True meaning the corresponding
    #vertex in its position is an element of the set (and thus part of path)
    initial_number = self.number_by_vertex[initial_vertex]
    final_number = self.number_by_vertex[final_vertex]
    if not skip_checks:
      print(f'{self.n=}\n{initial_number=} ({initial_vertex=})\n{final_number} ({final_vertex=})\n{presence_set=}\n')
      # Expect arg to be a tuple of Booleans with length self.n
      assert len(presence_set) == self.n, 'Internal logic error, presence_set should be as long as the number of vertices'
      # Check that initial_vertex and final_vertex are present [i. e. True]
      assert presence_set[initial_number], 'Internal logic error, initial vertex must be in presence set'
      assert presence_set[final_number], 'Internal logic error, final vertex must be in presence set'
    # We get rid of the boundary cases
    # We impose that if the final vertex coincides with the initial vertex,
    #the only possible path is the no-arrow path (of length 0 arrows)
    if initial_number == final_number:
      # Want that vertex as the only True, otherwise no path (distance math_inf)
      sought_presence_set = tuple((idx == initial_number) for idx in range(self.n))
      assert presence_set == sought_presence_set, 'Internal logic error, cannot have cycles [with more than one vertex] in subproblem'

        # No previous vertex, so previous path should be the "quasi empty path" to work well later
        # By "quasi empty path" we mean the path with initial_vertex and no arrows
        quasi_empty_path = VertexPath(
            data = [initial_vertex],
            data_type = 'vertices',
            verify_validity_on_initialization = not skip_checks)
        # (This is a nondegenerate path, and works fine with arrow addition)
        # If only lengths are asked, we produce None instead of [], for consistency
        if omit_minimizing_path:
          return (0, None)
        else:
          return (0, quasi_empty_path)

    else:
      # We essentially recur on "previous subproblems"
      # That is, for all arrows landing on final_vertex, we ask which
      #could be the last one, and pick the one producing the smallest
      #weight (assuming we solve the subproblems without this last vertex)
      min_among_all_last_arrows = math_inf
      whole_path_as_arrows = None
      for last_arrow in self.digraph.get_arrows_in(final_vertex):
        # last_arrow has information last_arrow.source, last_arrow.target
        #which is final_vertex, and last_arrow.weight.
        # We verify the source does belong to the presence_set
        last_arrow_source_as_number = self.number_by_vertex[last_arrow.source]
        if presence_set[last_arrow_source_as_number]:
          # We "remove" last_arrow.source by flipping True to False
          # We need to create a temporary mutable object first
          presence_set_as_list = list(presence_set)
          presence_set_as_list[last_arrow_source_as_number] = False
          last_off_presence_set = tuple(presence_set_as_list)
          # Total weight is then the solution of that problem,
          #plus the weight of this last arrow
          # Note that we also keep a list of arrows going back to start
          # It's probably easier than start a VertexPath instance every time
          if use_memoization_instead_of_tabulation:
            # In this case we simply call the suproblem method again
            solution_of_smaller_subproblem = self.solve_subproblem(
                initial_vertex = initial_vertex,
                final_vertex = last_arrow.source,
                presence_set = last_off_presence_set,
                omit_minimizing_path = omit_minimizing_path,
                skip_checks = skip_checks)
          else:
            # In this case the result should be stored in self._table_of_results
            solution_of_smaller_subproblem = self._table_of_results[
                (initial_vertex, final_vertex, last_off_presence_set)]
          previous_length, previous_path = solution_of_smaller_subproblem
          this_distance = arrow.weight + previous_length
          if this_distance < min_among_all_last_arrows:
            # Update the minimal distance, if this is minimal
            min_among_all_last_arrows = this_distance
            if omit_minimizing_path:
              # To save memory during execution, if we only want the minimal length
              #we will not conserve information on how to reconstruct the path
              # We use the very default object None for this objective
              whole_path_as_arrows = None
            else:
              # Need to update the last arrow (last arrow in path)
              # We will use the VertexPath method, returning a new instance
              whole_path_as_arrows = previous_path.append_to_path(data = arrow,
                  data_type = 'arrow', modify_self = False, skip_checks = skip_checks)
      # With the loop ended, the best should be recorded
      # (None whole_path_as_arrows contains None if output_as is 'length')
      return (best_distance, whole_path_as_arrows)

  def solve_full_problem(self, compute_path_instead_of_cycle,
      initial_vertex = None, final_vertex = None,
      use_memoization_instead_of_tabulation = False, output_as = None, skip_checks = False):
    '''
    Solves the Traveling Salesman Problem for the graph.
    
    output_as goes through VertexPath.reformat_paths()
    '''
    # Default output_as = None to a better option: 'length'
    if output_as is None:
      output_as = 'length'
    # Prepare initial and final vertices for path/cycle-searching
    initial_vertex, final_vertex, initial_and_final_vertices = self._prepare_initial_and_final_vertices(
        compute_path_instead_of_cycle, initial_vertex, final_vertex)
    # We check that there is at least one vertex
    if not bool(self.digraph):
      # Returns path/cycle with no vertices
      if compute_path_instead_of_cycle:
        path = VertexPath(self.digraph, [], 'vertices')
        return path.reformat_path_from_path(output_as = output_as, skip_checks = skip_checks)
      else:
        cycle = VertexCycle(self.digraph, [], 'vertices')
        return path.reformat_path_from_cycle(output_as = output_as, skip_checks = skip_checks)
    else:
      # That is, self.digraph is non-empty
      # Subdivide into the four possible cases according to the variables
      #compute_path_instead_of_cycle and use_memoization_instead_of_tabulation
      if compute_path_instead_of_cycle:
        # To solve the problem for a path (for a cycle it involves an extra step;
        #we will do it at the end, and only if required), we need to use some
        #form or recursion, or dynamic programming, which is carried out
        #in a separate method, solve_subproblem
        # For all vertices but the initial_vertex, we compute the possible
        #paths starting on initial_vertex, passing through all others exactly once
        #and ending on them
        if use_memoization_instead_of_tabulation:
          pre_output = self._solve_full_problem_for_path_and_memoization(
              initial_and_final_vertices = initial_and_final_vertices,
              output_as = output_as, # for omit_minimizing_path only
              skip_checks = skip_checks)
        else:
          pre_output = self._solve_full_problem_for_path_and_tabulation(
              initial_vertex = initial_vertex,
              final_vertex = final_vertex,
              initial_and_final_vertices = initial_and_final_vertices,
              output_as = output_as, # for omit_minimizing_path only
              skip_checks = skip_checks)
      else:
        # Here: compute_path_instead_of_cycle == False
        # We briefly explain the logic
        # We consider all cycles starting at given cycle
        # We consider all possibilities for the penultimate vertex of the cycle
        #(the final vertex, by definition, coincides with the initial)
        # We pick the best one after closing the cycle with the last arrow
        #(from the penultimate to the initial/final vertex)
        # Note also this penultimate cannot be the initial vertex
        # (To read arrows ending at initial=final, we use get_arrows_in)
        if use_memoization_instead_of_tabulation:
          pre_output = self._solve_full_problem_for_cycle_and_memoization(
              initial_vertex = initial_vertex,
              output_as = output_as, # for omit_minimizing_path only
              skip_checks = skip_checks)
        else:
          pre_output = self._solve_full_problem_for_cycle_and_tabulation(
              initial_vertex = initial_vertex,
              output_as = output_as, # for omit_minimizing_path only
              skip_checks = skip_checks)
      # Prepares output
      final_output = self._prepare_output(pre_output, compute_path_instead_of_cycle, output_as)
      return output

  def _prepare_initial_and_final_vertices(self, compute_path_instead_of_cycle,
      initial_vertex, final_vertex, skip_checks = False):
    '''
    Prepares possible values for initial_vertex and final_vertex for use in solve_full_problem method.
    '''
    # We first sanitize the input vertices (if not None)
    if initial_vertex is not None:
      initial_vertex = OperationsVAE.sanitize_vertex(initial_vertex, require_vertex_namedtuple = False)
    if final_vertex is not None:
      final_vertex = OperationsVAE.sanitize_vertex(final_vertex, require_vertex_namedtuple = False)
    # Produce initial_and_final_vertices tailored for paths and for cycles
    if compute_path_instead_of_cycle:
      # In this case, if there is no initial given vertex (i. e. None),
      #we assume we must scan paths through all possible initial vertices
      # (There might be a non-None explicitly given final_vertex, no problem)
      # We need to ensure final_vertex is different than initial_vertex,
      #if those are given
      if (not initial_vertex is None) and (not final_vertex is None):
        assert final_vertex != initial_vertex, 'Solution of shortest path cannot be a cycle'
      # We will generate all possible paths for given initial and final vertices
      # Note initial and final vertices cannot coincide
      if initial_vertex is None:
        # All are allowed as initial. We use list on dict self.number_by_vertex
        initial_vertices = all_vertices
      else:
        initial_vertices = [initial_vertex]
      initial_and_final_vertices = []
      for vertex in initial_vertices:
        # If final_vertex is specified, only such vertex can be final
        # Otherwise, all (except initial_vertex; would form cycle) are allowed
        if final_vertex is None:
          for another_vertex in all_vertices:
            if vertex != another_vertex:
              initial_and_final_vertices.append((vertex, another_vertex))
        else:
          if vertex != final_vertex:
            initial_and_final_vertices.append((vertex, final_vertex))
    else:
      # If we want a cycle, a specified initial_vertex is only for convenience
      #of the output (will be displayed with that initial vertex)
      # There should be no specified final_vertex, unless it is equal to initial_vertex,
      #which would be a redundant way to reinforce the request for a cycle
      if initial_vertex is None:
        # As mentioned, final_vertex should be None, otherwise it introduces potential to confusion
        #[would final_vertex mean the final or the last before the final?]
        assert final_vertex is None, 'Cannot specify end of cycle if start is not specified'
        # In this case we pick a "random one" to be initial, for the reasons above
        #[it matters only for exhibition, not for calculation]
        initial_vertex = self.vertex_by_number[0]
        final_vertex = initial_vertex
      else:
        assert final_vertex == initial_vertex, 'Cycles start and end at same place'
      # Thus, a cycle will always have the same initial and final vertex
      initial_and_final_vertices = [(initial_vertex, final_vertex)] # Both are the same
    # We return initial_and_final vertices as well as the sanitized inputs (which might be None)
    return (initial_vertex, final_vertex, initial_and_final_vertices)

  def _produce_auxiliary_constructs(self, output_as):
    '''
    Produces useful objects for solve_full_problem.
    
    tuple_of_trues
    min_distance_overall
    min_path_overall
    omit_minimizing_path
    '''
    # (Having this separate as a method is good for the DRY principle)
    # Tuple of trues useful for calling methods (operates as a "presence set")
    list_with_tuple_of_trues = self.produce_boolean_tuples_with_fixed_sum(
        given_length = self.n,
        given_sum = self.n,
        output_as_generator = False)
    tuple_of_trues = list_with_tuple_of_trues[0]
    # We also create the variables for initialize the variables in
    #the minimization problem
    # If no path is valid, it should be math_inf, so that is how we start
    min_distance_overall = math_inf
    min_path_overall = None
    # Depending on the output option the intermediate paths will be computed or not
    # (If omit_minimizing_path is True, min_path_overall is never updated)
    if output_as.lower() in ['length']:
      omit_minimizing_path = True
    else:
      omit_minimizing_path = False
    # Return all
    return (tuple_of_trues, min_distance_overall, min_path_overall, omit_minimizing_path)
        
  def _solve_full_problem_for_path_and_memoization(self, initial_and_final_vertices,
      output_as, skip_checks = False):
    '''
    Subroutine of method solve_full_problem invoked when
    compute_path_instead_of_cycle is True and use_memoization_instead_of_tabulation is True
    '''
    # Create useful objects
    tuple_of_trues, min_distance_overall, min_path_overall, omit_minimizing_path = self._produce_auxiliary_constructs(
          output_as = output_as)
    # We compute all possibilities, and record the best
    # Dynamic programming [memoization] is automatically done within solve_subproblem
    for pair in initial_and_final_vertices:
      local_distance, local_path = self.solve_subproblem(
          initial_vertex = pair[0],
          final_vertex = pair[1],
          presence_set = tuple_of_trues,
          use_memoization_instead_of_tabulation = True,
          omit_minimizing_path = omit_minimizing_path,
          skip_checks = skip_checks)
      # Since initial_and_final_vertices might not be a singleton:
      if local_distance < min_distance_overall:
        min_distance_overall = local_distance
        min_path_overall = local_path
    # Return is pre_output which is the best distance and the best path
    pre_output = (min_distance_overall, min_path_overall)
    return pre_output

  def _solve_full_problem_for_path_and_tabulation(self, initial_vertex, final_vertex,
      initial_and_final_vertices, output_as, skip_checks):
    '''
    Subroutine of method solve_full_problem invoked when
    compute_path_instead_of_cycle is True and use_memoization_instead_of_tabulation is False
    '''
    # Create useful objects
    tuple_of_trues, min_distance_overall, min_path_overall, omit_minimizing_path = self._produce_auxiliary_constructs(
          output_as = output_as)
    # The table for the tabulation process:
    self._table_of_results = {}
    # Note that tabulation is done in order of incresing vertices present
    # That is, the "size" (number of Trues) of presence_set
    for length_of_path in range(1, self.n + 1):
      # We find all presence sets of size length_of_path
      right_size_presence_sets = self.produce_boolean_tuples_with_fixed_sum(
          self.n, length_of_path, output_as_generator = True)
      for presence_set in right_size_presence_sets:
        # Verify initial and last vertices are present in presence_set
        # We have the possible initial and final on initial_and_final_vertices
        # But that is only for the full-sized paths
        # But for intermediate paths, the final vertex may be anything
        #while the initial is still the initial
        # We will do the following: read the initial and final from presence_set
        for initial_index in range(self.n):
          for final_index in range(self.n):
            if length_of_path == 1 or initial_index != final_index:
              # Only tabulate cases that matter
              if presence_set[initial_index] and presence_set[final_index]:
                # We now check that the initial_index correspond to a valid
                #choice of initial vertex in initial_and_final_vertices
                local_initial_vertex = self.vertex_by_number[initial_index]
                if initial_vertex is None or local_initial_vertex == initial_vertex:
                  local_final_vertex = self.vertex_by_number[final_index]
                  if (length_of_path < self.n) or (
                      final_vertex is None or local_final_vertex == final_vertex):
                    # We compute the value and store it on the table
                    local_min_distance, local_min_path = self.solve_subproblem(
                        initial_vertex = local_initial_vertex,
                        final_vertex = local_final_vertex,
                        presence_set = presence_set,
                        use_memoization_instead_of_tabulation = False,
                        omit_minimizing_path = omit_minimizing_path,
                        skip_checks = skip_checks)
                    self._table_of_results[(initial_vertex, final_vertex, presence_set)] = (
                        local_min_distance, local_min_path)
    # We now use the opportunity to update the best overall
    # For that, we measure the paths with length self.n
    # We use initial_and_final_vertices and tuple_of_trues, already available
    for pair in initial_and_final_vertices:
      # Simply consult table
      local_min_distance, local_min_path = self._table_of_results[(
          pair[0], pair[1], tuple_of_trues)]
      if local_min_distance < min_distance_overall:
        # Update the variables
        min_distance_overall = local_min_distance
        min_path_overall = local_min_path
    # To reinforce that we achieved the minimum we sought, we delete the table
    del self._table_of_results
    # Return is pre_output which is the best distance and the best path
    pre_output = (min_distance_overall, min_path_overall)
    return pre_output

  def _solve_full_problem_for_cycle_and_memoization(self, initial_vertex,
      output_as, skip_checks = False):
    '''
    Subroutine of method solve_full_problem invoked when
    compute_path_instead_of_cycle is False and use_memoization_instead_of_tabulation is True
    '''
    # Create useful objects
    tuple_of_trues, min_distance_overall, min_path_overall, omit_minimizing_path = self._produce_auxiliary_constructs(
          output_as = output_as)
    # In this case a direct call does the job (but we need to search all last arrows)
    # Note that this is easier as an algorithm, but might consume more memory
    # (Also has the risk of breaking the default Python shell recursion limit)
    for arrow in self.get_arrows_in(initial_vertex):
      if arrow.source != initial_vertex:
        # Compute the paths from initial to penultimate [using the last arrow]
        length_up_to_penultimate, path_up_to_penultimate = self.solve_subproblem(
            initial_vertex = initial_vertex,
            final_vertex = arrow.source,
            presence_set = tuple_of_trues,
            use_memoization_instead_of_tabulation = True,
            omit_minimizing_path = omit_minimizing_path,
            skip_checks = skip_checks)
        # Comparisons involves always the last edge, whose weight must be factored in
        #to close the cycle
        this_distance = length_up_to_penultimate + arrow.weight
        if this_distance < min_distance_overall:
          min_distance_overall = this_distance
          # To save processing time we only record the path if required
          if output_as.lower() == 'length':
            min_path_overall = None
          else:
            # We create a new path instance using append_to_path
            min_path_overall = path_up_to_penultimate.append_to_path(
                data = arrow, data_type = 'arrow', modify_self = False,
                verify_validity_on_initiation = not skip_checks)
    # Return is pre_output which is the best distance and the best path
    pre_output = (min_distance_overall, min_path_overall)
    return pre_output
    
  def _solve_full_problem_for_cycle_and_tabulation(self, initial_vertex,
      output_as, skip_checks = False):
    '''
    Subroutine of method solve_full_problem invoked when
    compute_path_instead_of_cycle is False and use_memoization_instead_of_tabulation is False
    '''
    # Create useful objects
    tuple_of_trues, min_distance_overall, min_path_overall, omit_minimizing_path = self._produce_auxiliary_constructs(
          output_as = output_as)
    # We have a initial vertex. We associate to it its number
    initial_number = self.number_by_vertex[initial_vertex]
    # In this case we must organize the variables for tabulation
    # Note that every recurrence of the subproblem is for a path which
    #is one vertex shorter
    # Thus the key is working with the presence set, which should be
    #of ever increasing length, and always include the initial_vertex
    # We start the "table" (a dict) for the tabulation
    self._table_of_results = {}
    # We iterate through the arguments
    # Recall the first is the number of vertices in path (controlled by presence_set)
    for length_of_path in range(1, self.n + 1):
      # All presence sets using a special method
      right_size_presence_sets = self.produce_boolean_tuples_with_fixed_sum(
          self.n, length_of_path, output_as_generator = True)
      for presence_set in right_size_presence_sets:
        for final_number in range(self.n):
          # We now verify the arguments make sense
          # Need initial and final vertex present [controlled by numbers]
          if presence_set[initial_number] and presence_set[final_number]:
            # Need arrow from last to initial
            final_vertex = self.vertex_by_number[final_number]
            if final_vertex in neighbors_in_as_dict:
              # In this case we go ahead
              length_up_to_penultimate, path_up_to_penultimate = self.solve_subproblem(
                  initial_vertex = initial_vertex,
                  final_vertex = final_vertex,
                  presence_set = presence_set,
                  use_memoization_instead_of_tabulation = False,
                  omit_minimizing_path = omit_minimizing_path,
                  skip_checks = skip_checks)
              # We do the tabulation
              # (functools_cache would also do it, but storing in a separate variable
              #is more in line with the proposal of tabulation)
              # Note the only changing arguments here are presence_set and the last/penultimate vertex
              #but initial_vertex might change if the problem is about paths instead of cycles, so we include it
              self._table_of_results[(initial_vertex, final_vertex, presence_set)] = (length_up_to_penultimate, path_up_to_penultimate)
    # We now look at how to close the cycle. We iterate through the possible arrows
    # Note that unless there is a last arrow (from last to initial)
    #it is not possible to complete the cycle (once all vertices are in, that is)
    for arrow in self.digraph.get_arrows_in(initial_vertex):
      length_up_to_penultimate, path_up_to_penultimate = self._table_of_results[
          (initial_vertex, arrow.source, tuple_of_trues)]
      this_distance = length_up_to_penultimate + arrow.weight
      if this_distance < min_distance_overall:
        min_distance_overall = this_distance
        # We only record the path if needed
        # That comes from omit_minimining_path which is derived from output_as
        if omit_minimining_path:
          min_path_overall = None
        else:
          # Create new instance using append_to_path
          min_path_overall = path_up_to_penultimate.append_to_path(
              data = arrow, data_type = 'arrow', modify_self = False,
              skip_checks = skip_checks)
    # To reinforce that we achieved the minimum we sought, we delete the table
    del self._table_of_results
    # Return is pre_output which is the best distance and the best path
    pre_output = (min_distance_overall, min_path_overall)
    return pre_output

  def _prepare_output(self, pre_output, compute_path_instead_of_cycle, output_as,
      skip_checks = False):
    '''
    Prepares requested output from information provided.
    
    Last step of solve_full_problem.
    '''
    # pre_output is collected from methods split into paths/cycles/memoization/tabulation
    # Formatting is carried out according to output_as, which offloads to reformat_paths
    min_distance_overall, min_path_overall = pre_output
    if min_path_overall is None:
      # This can be either because there is no path meeting the conditions
      #or because omit_minimizing_path was triggered to True
      min_path_overall = 'Minimizing path not calculated'
    else:
      min_path_overall = min_path_overall.reformat_paths(
          underlying_graph = self.digraph,
          data = min_path_overall,
          data_type = ('path' if compute_path_instead_of_cycle else 'cycle'),
          output_as = output_as,
          skip_checks = skip_checks)
    return (min_distance_overall, min_path_overall)

########################################################################
