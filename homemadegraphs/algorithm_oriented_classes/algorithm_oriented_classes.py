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

  def produce_minimization_constructs(self):
    '''
    Produces useful objects for path minimization.
    
    More specifically, it creates:
    
    min_distance_overall, which is math.inf
    min_path_overall, which is None
    '''
    # We also create the variables for initialize the variables in
    #the path/cycle minimization problem
    # If no path is valid, it should be math_inf, so that is how we start
    min_distance_overall = math_inf
    min_path_overall = None
    return (min_distance_overall, min_path_overall)

  def produce_tuple_of_trues(self):
    # Tuple of trues useful for calling methods (operates as a "presence set")
    # produce_boolean_tuples_with_fixed_sum would be too costly, so we do it directly
    list_with_tuple_of_trues = list(itertools_product((True,), repeat = self.n))
    tuple_of_trues = list_with_tuple_of_trues[0]
    return tuple_of_trues

  def should_omit_minimizing_paths(self, output_as):
    '''
    Returns whether the full problem in class requires a specific path given.
    '''
    # Depending on the output option the intermediate paths will be computed or not
    # (If omit_minimizing_path is True, min_path_overall is never updated)
    if output_as.lower() in ['length']:
      omit_minimizing_path = True
    else:
      omit_minimizing_path = False
    return omit_minimizing_path

  @functools_cache
  def solve_subproblem(self, initial_vertex, final_vertex, presence_set,
      use_memoization_instead_of_tabulation = False, omit_minimizing_path = False, skip_checks = False):
    '''
    Computes the minimal path length given specific parameters: given
    initial and final vertices and a set of vertices of underlying graph
    self.digraph [given by a tuple of Booleans], finds minimal among all paths
    traveling once though each vertex satisfying the boundary conditions.
    
    Returns the minimal weight of such path, and also, if requested,
    also one of these minimizing paths as VertexPath instance. [If request is
    for its ommission, produces None as such path to fill the spot.]
    
    [Note this method is only about paths without any self-crossings;
    the path must be injective what also rules out any cycles. In particular,
    if the initial and final vertices are the same, the problem has no solution
    (returning infinite distance and None as path) unless this
    initial-and-final-vertex is the only vertex in the subproblem.]
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
    #print(f'\nSOLVING SUBPROBLEM\n{initial_number=}, {initial_vertex=}\n{final_number=}, {final_vertex=}\n{presence_set=} (Total: {sum(presence_set)})')
    if not skip_checks:
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
      if presence_set == sought_presence_set:
        # No previous vertex, so previous path should be the "quasi empty path" to work well later
        # By "quasi empty path" we mean the path with initial_vertex and no arrows
        quasi_empty_path = VertexPath(
            underlying_digraph = self.digraph,
            data = [initial_vertex],
            data_type = 'vertices',
            verify_validity_on_initialization = not skip_checks)
        # [This is a nondegenerate path, and works fine with arrow addition]
        # If only lengths are asked, we produce None instead of [], for consistency
        #print(f'Solution (one-vertex path): {0}')
        if omit_minimizing_path:
          return (0, None)
        else:
          return (0, quasi_empty_path)
      else:
        # In this case there are more than one vertex, thus making it impossible
        #to provide a solution to the subproblem
        #print(f'Solution (impossible set): {math_inf}')
        if omit_minimizing_path:
          return (math_inf, None)
        else:
          return (math_inf, None)
    else:
      # In this case final_number != initial_number
      # We essentially recur on "previous subproblems"
      # That is, for all arrows landing on final_vertex, we ask which
      #could be the last one, and pick the one producing the smallest
      #weight (assuming we solve the subproblems without this last vertex)
      min_among_all_last_arrows = math_inf
      whole_path_as_arrows = None
      for last_arrow in self.digraph.get_arrows_in(final_vertex):
        # We need to exclude self-arrows as they might throw the algorithm into a loop
        if not OperationsVAE.is_self_arrow_or_self_edge(last_arrow,
            use_edges_instead_of_arrows = False,
            require_namedtuple = False,
            request_vertex_sanitization = False,
            require_vertex_namedtuple = False):
          # last_arrow has information last_arrow.source, last_arrow.target
          #which is final_vertex, and last_arrow.weight.
          # We verify the source does belong to the presence_set
          last_arrow_source_as_number = self.number_by_vertex[last_arrow.source]
          if presence_set[last_arrow_source_as_number]:
            # We "remove" final_number by flipping True to False
            # We need to create a temporary mutable object first
            presence_set_as_list = list(presence_set)
            presence_set_as_list[final_number] = False
            last_off_presence_set = tuple(presence_set_as_list)
            # Total weight is then the solution of that problem,
            #plus the weight of this last arrow
            # Note that adding this last arrow is done using a VertexPath method
            # [It's probably easier than build a VertexPath instance every time]
            if use_memoization_instead_of_tabulation:
              # In this case we simply call the suproblem method again
              solution_of_smaller_subproblem = self.solve_subproblem(
                  initial_vertex = initial_vertex,
                  final_vertex = last_arrow.source,
                  presence_set = last_off_presence_set,
                  use_memoization_instead_of_tabulation = True,
                  omit_minimizing_path = omit_minimizing_path,
                  skip_checks = skip_checks)
            else:
              # In this case the result should be stored in self._subproblem_solutions
              solution_of_smaller_subproblem = self._subproblem_solutions[
                  (initial_vertex, last_arrow.source, last_off_presence_set)]
            previous_length, previous_path = solution_of_smaller_subproblem
            this_distance = last_arrow.weight + previous_length
            # Update the minimal distance, if it is minimal
            if this_distance < min_among_all_last_arrows:
              min_among_all_last_arrows = this_distance
              if omit_minimizing_path:
                # To save memory during execution, if we only want the minimal length
                #we will not conserve information on how to reconstruct the path
                # We use the very default object None for this objective
                whole_path_as_arrows = None
              else:
                # Need to update the last arrow (last arrow in path)
                # We will use the VertexPath method, returning a new instance
                whole_path_as_arrows = previous_path.append_to_path(
                    data = arrow,
                    data_type = 'arrow',
                    modify_self = False,
                    skip_checks = skip_checks)
      # With the loop ended, the best should be recorded [unless omit_minimizing_path
      #is True, in which case whole_path_as_arrows is simply None]
      #print(f'Solution (after recurrence): {min_among_all_last_arrows}')
      return (min_among_all_last_arrows, whole_path_as_arrows)

  def solve_full_length_subproblems_for_initial_and_final_vertices(self,
      initial_vertex, final_vertex, initial_and_final_vertices,
      use_memoization_instead_of_tabulation, omit_minimizing_path, skip_checks = False):
    '''
    Solves the subproblems for paths of maximum length (goes through all vertices),
    given the specified pairs of initial and final vertices [the boundary conditions].
    
    Returns a dictionary in which the keys are the pairs of possible initial
    and final vertices, and whose values are the optimizing distances and paths.
    
    SEE ALSO: solve_subproblem
    '''
    # Useful construct
    tuple_of_trues = self.produce_tuple_of_trues()
    # Create empty dict
    solutions = {}
    if use_memoization_instead_of_tabulation:
      # In this case initial_vertex and final_vertex are not needed
      del initial_vertex, final_vertex
      for pair in initial_and_final_vertices:
        # In this case just do the calls and store in the right dict key
        # [Dynamic programming [memoization] is automatically done within solve_subproblem
        #due to functools.cache]
        local_initial_vertex, local_final_vertex = pair
        local_distance, local_path = self.solve_subproblem(
            initial_vertex = local_initial_vertex,
            final_vertex = local_final_vertex,
            presence_set = tuple_of_trues,
            use_memoization_instead_of_tabulation = True,
            omit_minimizing_path = omit_minimizing_path,
            skip_checks = skip_checks)
        solutions[pair] = (local_distance, local_path)
    else:
      # In this case we create the solution of all subproblems, even the
      #ones smaller than full length, making use of tabulation
      # The table (a dictionary) for the tabulation process:
      self._subproblem_solutions = {}
      # Note that tabulation is done in increasing order of number of vertices present
      # That is, the "size" (number of Trues) of presence_set
      for length_of_path in range(1, self.n + 1):
        # We find all presence sets of size length_of_path
        right_size_presence_sets = self.produce_boolean_tuples_with_fixed_sum(
            self.n, length_of_path, output_as_generator = True)
        for presence_set in right_size_presence_sets:
          # To determine initial_vertex and final_vertex passed to subproblem
          #[note final_vertex is not the same as the one or None given to
          #solve_full_problem, since there we do a recursion on the last vertex]
          #we will scan through all possible values but only compute the subproblem
          #when the values make sense and are useful
          for local_initial_index in range(self.n):
            for local_final_index in range(self.n):
              local_initial_vertex = self.vertex_by_number[local_initial_index]
              local_final_vertex = self.vertex_by_number[local_final_index]
              # We would like to tabulate only cases that matter.
              # However, sometimes a recursion asks the solution of a subproblem
              #which obviously has no solution (for example in violation of
              #any condition below). In this case it becomes necessary,
              #for the recursion process, to have such entries stored in the
              #table as problems without solution, that is, (math.inf, None)
              # We control it using the variable: is_subproblem_certainly_impossible
              is_subproblem_certainly_impossible = False
              # Eliminate nontrivial [more than one vertex] cycles:
              if length_of_path == 1 or local_initial_index != local_final_index:
                # Ensure the vertices corresponding to the indices are present in presence_set
                if presence_set[local_initial_index] and presence_set[local_final_index]:
                  # We now check that the local_initial_index correspond 
                  #to a valid choice under the initial_vertex input
                  if initial_vertex is None or local_initial_vertex == initial_vertex:
                    # If the path is full-sized, the local_final_vertex variable
                    #has to match the final_vertex input (unless this is None)
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
                      self._subproblem_solutions[(local_initial_vertex, local_final_vertex, presence_set)] = (
                          local_min_distance, local_min_path)
                    else:
                      # For a full-length path, local_final_vertex has to match the final_vertex input
                      is_subproblem_certainly_impossible = True
                  else:
                    # Initial vertex has to always match the specified (unless initial_vertex is None)
                    is_subproblem_certainly_impossible = True
                else:
                  # Initial and final vertices don't belong to presence set
                  is_subproblem_certainly_impossible = True
              else:
                # More than one vertex, initial and final different
                is_subproblem_certainly_impossible = True
              # We store (math.inf, None) in the table to indicate unsolvable subproblem
              #(for subproblems marked unsolvable)
              if is_subproblem_certainly_impossible:
                self._subproblem_solutions[(local_initial_vertex, local_final_vertex, presence_set)] = (
                    math_inf, None)
      # Read the values from self._subproblem_solutions to prepare to return
      for pair in initial_and_final_vertices:
        local_initial_vertex, local_final_vertex = pair
        solutions[pair] = self._subproblem_solutions[(local_initial_vertex, local_final_vertex, presence_set)]
      # To show everything is complete, delete self._subproblem_solutions
      del self._subproblem_solutions
    # In both cases, return the dict solutions
    return solutions

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
        compute_path_instead_of_cycle = compute_path_instead_of_cycle,
        initial_vertex = initial_vertex,
        final_vertex = final_vertex)
    # Determine omit_minimizing_path which is used below
    omit_minimizing_path = self.should_omit_minimizing_paths(output_as)
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
        # To solve the problem for a path (for a cycle it would involves
        #an extra step, done obviously only for cycles), we need to use some
        #form or recursion, or dynamic programming, which is carried out
        #in a separate method, solve_subproblem
        # For all vertices but the initial_vertex, we compute the possible
        #paths starting on initial_vertex, passing through all others exactly once
        #(no non-trivial cycles allowed) and ending on final_vertex
        pre_output = self._solve_full_problem_for_paths(
              initial_vertex = initial_vertex,
              final_vertex = final_vertex,
              initial_and_final_vertices = initial_and_final_vertices,
              use_memoization_instead_of_tabulation = use_memoization_instead_of_tabulation,
              omit_minimizing_path = omit_minimizing_path,
              skip_checks = skip_checks)
      else:
        # For cycle the process is similar: it is similar to a path with a last arrow
        #which closes the cycle. We need to take special care of the last arrow
        # This is done and explained in methods _prepare_initial_and_final_vertices
        #and _solve_full_problem_for_cycles
        pre_output = self._solve_full_problem_for_cycles(
            initial_vertex = initial_vertex,
            final_vertex = final_vertex,
            initial_and_final_vertices = initial_and_final_vertices,
            use_memoization_instead_of_tabulation = use_memoization_instead_of_tabulation,
            omit_minimizing_path = omit_minimizing_path,
            skip_checks = skip_checks)
      # Prepares output
      final_output = self._prepare_output(pre_output, compute_path_instead_of_cycle, output_as)
      return final_output

  def _prepare_initial_and_final_vertices(self, compute_path_instead_of_cycle,
      initial_vertex, final_vertex, skip_checks = False):
    '''
    Prepares possible values for initial_and_final_vertices to be used in
    method solve_full_length_subproblems_for_initial_and_final_vertices,
    and also sanitizes initial_vertex and final_vertex.
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
        initial_vertices = self.digraph.get_vertices()
      else:
        initial_vertices = [initial_vertex]
      initial_and_final_vertices = []
      for vertex in initial_vertices:
        # If final_vertex is specified, only such vertex can be final
        # Otherwise, all (except initial_vertex; would form cycle) are allowed
        if final_vertex is None:
          for another_vertex in self.digraph.get_vertices():
            if vertex != another_vertex:
              initial_and_final_vertices.append((vertex, another_vertex))
        else:
          if vertex != final_vertex:
            initial_and_final_vertices.append((vertex, final_vertex))
    else:
      # If we want a cycle, a specified initial_vertex is only for convenience
      #of the output as the cycle will be essentially the same, only rotated
      #(but will be displayed with that initial vertex nonetheless)
      # There should be no specified final_vertex, unless it is equal to initial_vertex,
      #which would be a redundant way to reinforce the request for a cycle
      if initial_vertex is None:
        # In this case we pick a "random one" to be initial, for the reasons above
        #[it matters only for exhibition, not for calculation]
        initial_vertex = self.vertex_by_number[0]
        # As mentioned, final_vertex should be None, otherwise it introduces potential to confusion
        #[would final_vertex mean the final or the last before the final?]
        assert final_vertex is None, 'Cannot specify end of cycle if start is not specified'
      else:
        assert final_vertex == initial_vertex, 'Cycles start and end at same place'
      # To make it compatible with solve_full_length_subproblems_for_initial_and_final_vertices,
      #(done via tabulation) we set final_vertex = None
      final_vertex = None
      # The variable initial_and_final_vertices will be used to discriminate the boundary
      #conditions in solve_full_length_subproblems_for_initial_and_final_vertices
      # In particular, note initial_and_final_vertices != [(initial_vertex, final_vertex)]
      # This is done with the arrows into the initial_vertex. (Though we use the
      #neighbors to avoid repetitions). They are supposed to close the cycle,
      #so the source of the arrow is exactly a possible final vertex for
      #solve_full_length_subproblems_for_initial_and_final_vertices
      initial_and_final_vertices = []
      for neighbor in self.digraph.get_neighbors_in(initial_vertex):
        initial_and_final_vertices.append((initial_vertex, neighbor))
    # We return initial_and_final vertices as well as the sanitized inputs (which might be None)
    return (initial_vertex, final_vertex, initial_and_final_vertices)

  def _solve_full_problem_for_paths(self, initial_vertex, final_vertex,
      initial_and_final_vertices, use_memoization_instead_of_tabulation,
      omit_minimizing_path, skip_checks):
    '''
    Subroutine of method solve_full_problem for paths.
    '''
    # Create useful objects
    min_distance_overall, min_path_overall = self.produce_minimization_constructs()
    # Get data using other method
    minimizing_data = self.solve_full_length_subproblems_for_initial_and_final_vertices(
        initial_vertex = initial_vertex,
        final_vertex = final_vertex,
        initial_and_final_vertices = initial_and_final_vertices,
        use_memoization_instead_of_tabulation = use_memoization_instead_of_tabulation,
        omit_minimizing_path = omit_minimizing_path,
        skip_checks = skip_checks)
    # Simply find the shortest
    for pair in initial_and_final_vertices:
      local_distance, local_path = minimizing_data[pair]
      # Since initial_and_final_vertices might not be a singleton:
      if local_distance < min_distance_overall:
        min_distance_overall = local_distance
        min_path_overall = local_path
    # Return is pre_output which is the best distance and the best path
    pre_output = (min_distance_overall, min_path_overall)
    return pre_output

  def _solve_full_problem_for_cycles(self, initial_vertex, final_vertex, initial_and_final_vertices,
      use_memoization_instead_of_tabulation, omit_minimizing_path, skip_checks = False):
    '''
    Subroutine of method solve_full_problem for cycles.
    '''
    # Create useful objects for minimization
    min_distance_overall, min_cycle_overall = self.produce_minimization_constructs()
    # To solve the problem, first we consider a path which is one vertex short
    #of closing the cycle
    # We compute all possibilities of full length subproblems by obtaining the initial
    #and final vertices from the arrows arriving at initial vertex; note these
    #arrows are captured correctly by initial_and_final_vertices
    minimizing_data = self.solve_full_length_subproblems_for_initial_and_final_vertices(
        initial_vertex = initial_vertex,
        final_vertex = final_vertex,
        initial_and_final_vertices = initial_and_final_vertices,
        use_memoization_instead_of_tabulation = use_memoization_instead_of_tabulation,
        omit_minimizing_path = omit_minimizing_path,
        skip_checks = skip_checks)
    # We find the minimum for this data, ensuring the weight of the last arrow
    #is also added and factored in
    for arrow in self.digraph.get_arrows_in(initial_vertex):
      # To ensure initial_and_final_vertices was correct
      pair = (initial_vertex, arrow.source)
      assert pair in minimizing_data, 'Internal logic error, wrong subproblems solved.'
      local_distance, local_path = minimizing_data[pair]
      # To minimize the cycle we need to add the weight of the last arrow
      if local_distance + arrow.weight < min_distance_overall:
        min_distance_overall = local_distance + arrow.weight
        if omit_minimizing_path:
          min_cycle_overall = None
        else:
          # We also need to extend the path (now cycle) to include the arrow
          # Since it changes from VertexPath to VertexCycle we create a new instance
          all_previous_arrows = local_path.get_arrows()
          min_cycle_overall = VertexCycle(
              underlying_digraph = self.digraph,
              data = all_previous_arrows + [arrow],
              data_type = 'arrows',
              verify_validity_on_initialization = True)
    # Return is pre_output which is the best distance and the best cycle
    pre_output = (min_distance_overall, min_cycle_overall)
    return pre_output

  def _prepare_output(self, pre_output, compute_path_instead_of_cycle, output_as,
      skip_checks = False):
    '''
    Prepares requested output from information provided.
    
    Last step of solve_full_problem.
    '''
    # We need omit_minimizing_path, derivable from output_as
    omit_minimizing_path = self.should_omit_minimizing_paths(output_as)
    # pre_output is collected from methods split into paths/cycles/memoization/tabulation
    # Formatting is carried out according to output_as, which offloads to reformat_paths
    # (With the exception of when omit_minimizing_path is True)
    min_distance_overall, min_path_overall = pre_output
    if omit_minimizing_path:
      return min_distance_overall
    else:
      # In this case min_distance_overall is obsolete, since it should be
      #the length of the resulting path (unless there was no solution to
      #the problem, which resulted in min_path_overall being None)
      if min_path_overall is None:
        # No solution to the full problem
        assert min_distance_overall == math_inf, 'Without an existing path or cycle solution the distance shuld be infinite.'
        return (math_inf, 'There is no path/cycle solving the Traveling Salesman Problem.')
      else:
        # min_distance_overall is discarded; should be the same as length of VertexPath
        #as calculated by method VertexPath.get_total_weight(self, request_none_if_unweighted = False
        length_of_min_path = min_path_overall.get_total_weight(request_none_if_unweighted = False)
        assert length_of_min_path == min_distance_overall, 'Total weight/length of Path/cycle solution should match the value in the solution.'
        # To return use formatting from VertexPath.reformat_path
        return min_path_overall.reformat_paths(
            underlying_graph = self.digraph,
            data = min_path_overall,
            data_type = ('path' if compute_path_instead_of_cycle else 'cycle'),
            output_as = output_as,
            skip_checks = skip_checks)

########################################################################
