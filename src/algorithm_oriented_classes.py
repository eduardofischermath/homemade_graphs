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
    self.components = {vertex:None for vertex in self.get_vertices()}
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
    for idx, vertex in enumerate(self.get_vertices()):
      number_by_vertex[vertex] = idx
      vertex_by_number[idx] = vertex

  @functools_cache
  def solve_subproblem(self, initial_vertex, final_vertex, presence_set,
      use_top_down_instead_of_bottom_up = False, omit_minimizing_path = False, skip_checks = False):
    '''
    Computes the minimal path length given specific parameters: given
    initial and final vertices and a set of vertices [given by a tuple of
    Booleans], finds minimal among paths traveling once though each vertex.
    
    Returns the minimal weight of such path, and also, if requested,
    also one of these minimizing paths as VertexPath instance. (If request is
    for its ommission, produces None as such path to fill the space.)
    
    [Note this method is only about paths without self-crossings, never cycles.]
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
      # Expect arg to be a tuple of Booleans with length n
      assert len(presence_set) == self.n, 'Internal logic error'
      # Check that initial_vertex and final_vertex are present [i. e. True]
      assert presence_set[initial_number], 'Internal logic error'
      assert presence_set[final_number], 'Internal logic error'
    # We get rid of the boundary cases
    # We impose that if the final vertex coincides with the initial vertex,
    #the only possible path is the no-arrow path (of length 0)
    if initial_number == final_number:
      # Want only that vertex as True, otherwise no path (distance math_inf)
      sought_presence_set = tuple((idx == initial_number) for idx in range(self.n))
      if presence_set == sought_presence_set:
        # No previous vertex, so previous path should be the "quasi empty path" to work well later
        # By "quasi empty path" we mean the path with initial_vertex and no arrows
        quasi_empty_path = VertexPath(data = [initial_vertex], data_type = 'vertices',
            verify_validity_on_initialization = not skip_checks)
        # (This is a nondegenerate path, and works fine with arrow addition)
        # If only lengths are asked, we produce None instead of [], for consistency
        if omit_minimizing_path:
          return (0, None)
        else:
          return (0, quasi_empty_path)
      else:
        if omit_minimizing_path:
          return (math_inf, None)
        else:
          return (math_inf, quasi_empty_path)
    else:
      # We essentially recur on "previous subproblems"
      # That is, for all arrows landing on final_vertex, we ask which
      #could be the last one, and pick the one producing the smallest
      #weight (assuming we solve the subproblems without this last vertex)
      min_among_all_last_arrows = math_inf
      whole_path_as_arrows = None
      for arrow in self.digraph.get_arrows_in(final_vertex):
        # arrow has information arrow.source, arrow.target which is final
        #vertex, and arrow.weight.
        # We verify the source does belong to the presence_set
        arrow_source_as_number = number_by_vertex[arrow.source]
        if presence_set[arrow_source_as_number]:
          # We "remove" arrow.source by flipping True to False
          # We need to create a temporary mutable object first
          presence_set_as_list = list(presence_set)
          presence_set_as_list[arrow_source_as_number] = False
          last_off_presence_set = tuple(presence_set_as_list)
          # Total weight is then the solution of that problem,
          #plus the weight of this last arrow
          # Note that we also keep a list of arrows going back to start
          # It's probably easier than start a VertexPath instance every time
          if use_top_down_instead_of_bottom_up:
            # In this case we simply call the suproblem method again
            solution_of_smaller_subproblem = self.solve_subproblem(
                initial_vertex = initial_vertex,
                final_vertex = arrow.source,
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
      use_top_down_instead_of_bottom_up = False, output_as = None, skip_checks = False):
    '''
    Solves the Traveling Salesman Problem for the graph.
    
    output_as: 'path', 'vertices', 'vertices_and_arrows', 'arrows', 'length'
    '''
    # We determine whether full determination of path is required.
    # This is derived from output_as
    if output_as in ['length']:
      omit_minimizing_path = True
    else:
      omit_minimizing_path = False
    # We check that there is at least one vertex
    if not bool(self.digraph):
      # Returns path/cycle with no vertices
      if compute_path_instead_of_cycle:
        return VertexPath(self.digraph, [], 'vertices')
      else:
        return VertexCycle(self.digraph, [], 'vertices')
    else:
      # To solve the problem for a path (for a cycle it involves an extra step;
      #we will do it at the end, and only if required), we need to use some
      #form or recursion, or dynamic programming, which is carried out
      #in a separate method
      # For all vertices but the initial_vertex, we compute the possible
      #paths starting on initial_vertex, passing through all others exactly once
      #and ending on them
      # To simplify:
      all_vertices = list(self.number_by_vertex)
      # We prepare a tuple of n Trues to be the presence set, we will need it
      tuple_of_trues = tuple([True]*(self.n))
      # We also create the variables for storing the minima while we search for it
      # If no path is valid, it should be math_inf, so that is how we start
      min_distance_overall = math_inf
      min_path_overall = None
      if compute_path_instead_of_cycle:
        # Here: compute_path_instead_of_cycle == True
        # In this case, if there is no initial given vertex (i. e. None),
        #we assume we must scan through all possible initial vertices
        # We need to ensure final_vertex is different than initial_vertex,
        #if those are given
        if (not initial_vertex is None) and (not final_vertex is None):
          assert final_vertex != initial_vertex, 'Hamiltonian path cannot be a cycle'
        # We will generate all possible paths for given initial and final vertices
        # Note initial and final vertices cannot coincide
        if initial_vertex is None:
          # All are allowed. We use list on dict self.number_by_vertex
          initial_vertices = all_vertices
        else:
          initial_vertices = [initial_vertex]
        initial_and_final = []
        for vertex in initial_vertices:
          # If final_vertex is specified, only such vertex can be final
          # Otherwise, all (except initial_vertex; would form cycle) are allowed
          if final_vertex is None:
            for another_vertex in all_vertices:
              if vertex != another_vertex:
                initial_and_final.append((vertex, another_vertex))
          else:
            if vertex != final_vertex:
              initial_and_final.append((vertex, final_vertex))
        if use_top_down_instead_of_bottom_up:
          # Here: use_top_down_instead_of_bottom_up == True, use_top_down_instead_of_bottom_up == True
          # We compute all possibilities, and record the best
          for pair in initial_and_final:
            local_distance, local_path = self.solve_subproblem(
                initial_vertex = pair[0],
                final_vertex = pair[1],
                presence_set = tuple_of_trues,
                use_top_down_instead_of_bottom_up = True,
                omit_minimizing_path = omit_minimizing_path,
                skip_checks = skip_checks)
            # Note local_last_arrow is ignored... we still don't know how to
            #build the data using the arrows
            if local_distance < min_distance_overall:
              min_distance_overall = local_distance
        else:
          # Here: compute_path_instead_of_cycle == True, use_top_down_instead_of_bottom_up == False
          # The table for the tabulation process:
          self._table_of_results = {}
          # Note that tabulation is done in order of incresing vertices present
          # That is, the "size" (number of Trues) of presence_set
          for length_of_path in range(1, self.n + 1):
            # Use itertools_combinations on (True, ..., True, False, ..., False)
            true_false_list = [number < length_of_path for number in range(self.n)]
            right_size_presence_sets = itertools_combinations(true_false_list)
            for presence_set in right_size_presence_sets:
              # Verify initial and last vertices are present in presence_set
              # We have the possible initial and final on initial_and_final
              # But that is only for the full-sized paths
              # But for intermediate paths, the final vertex may be anything
              #while the initial is still the initial
              # We will do the following: read the initial and final from presence_set
              for initial_index in range(self.n):
                for final_index in range(self.n):
                  if length_of_path == 1 or initial_index != final_index:
                    if presence_set[initial_index] and presence_set[final_index]:
                      # We now check for initial and final vertex
                      # Note that if they are None, then they can be any
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
                              use_top_down_instead_of_bottom_up = False,
                              omit_minimizing_path = omit_minimizing_path,
                              skip_checks = skip_checks)
                          self._table_of_results[(initial_vertex, final_vertex, presence_set)] = (
                              local_min_distance, local_min_path)
          # We now use the opportunity to update the best overall
          # For that, we measure the paths with length self.n
          # We use initial_and_final and tuple_of_trues, already available
          for pair in initial_and_final:
            # Simply consult table
            local_min_distance, local_min_path = self._table_of_results[(
                pair[0], pair[1], tuple_of_trues)]
            if local_min_distance < min_distance_overall:
              # Update the variables
              min_distance_overall = local_min_distance
              min_path_overall = local_min_path
          # To reinforce that we achieved the minimum we sought, we delete the table
          del self._table_of_results
      else:
        # Here: compute_path_instead_of_cycle == False
        # If we want a cycle, there should be no specified final_vertex
        # [Unless the same vertex is entered as initial_vertex, which would be allowed
        #under certain interpretation to reinforce the request for a cycle]
        if initial_vertex is None:
          assert final_vertex is None, 'Cannot specify end of cycle if start is not specified'
        else:
          assert final_vertex == initial_vertex, 'Cycles start and end at same place'
        # In any case, the variable final_vertex will be ignored
        # We only want to raise the error to make it clearer
        # We will even delete the variable from this scope to reinforce the idea
        del final_vertex
        # In this case, the same cycle will be generated independently
        #of the first vertex
        # We pick the first listed if not given as argument
        # If passed as argument, we keep it [it doesn't really matter]
        if initial_vertex is None:
          initial_vertex = self.vertex_by_number[0]
        # We consider all cycles starting at given cycle
        # We consider all possibilities for the penultimate vertex of the cycle
        #(the final vertex, by definition, coincides with the initial)
        # We pick the best one after closing the cycle with the last arrow
        #(from the penultimate to the initial/final vertex)
        # Note also this penultimate cannot be the initial vertex
        # (To read arrows ending at initial=final, we use get_arrows_in)
        if use_top_down_instead_of_bottom_up:
          # Here: compute_path_instead_of_cycle == False, use_top_down_instead_of_bottom_up == False
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
                  use_top_down_instead_of_bottom_up = True,
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
        else:
          # Here: compute_path_instead_of_cycle == True, use_top_down_instead_of_bottom_up == False
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
            # Use itertools_combinations on (True, ..., True, False, ..., False)
            true_false_list = [number < length_of_path for number in range(self.n)]
            right_size_presence_sets = itertools_combinations(true_false_list)
            for presence_set in right_size_presence_sets:
              for final_number in range(self.n):
                # We now verify the arguments make sense
                # Need initial and final vertex present [controlled by numbers]
                if presence_set[initial_number] and presence_set[final_number]:
                  # Need arrow from last to initial
                  final_vertex = self.number_to_vertex[final_number]
                  if final_vertex in neighbors_in_as_dict:
                    # In this case we go ahead
                    length_up_to_penultimate, path_up_to_penultimate = self.solve_subproblem(
                        initial_vertex = initial_vertex,
                        final_vertex = final_vertex,
                        presence_set = presence_set,
                        use_top_down_instead_of_bottom_up = False,
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
      # This is the end of method, for paths/cycles/memoization/tabulation
      # By now, we have min_distance_overall and minimizing_path
      # Formatting is carried out by output_as, which offloads to reformat_paths
      if min_path_overall is None:
        min_path_overall = 'Minimizing path not calculated'
      else:
        min_path_overall = min_path_overall.reformat_paths(
            underlying_graph = self.digraph,
            data = minimizing_path,
            data_type = ('path' if compute_path_instead_of_cycle else 'cycle'),
            output_as = output_as)
      return (min_distance_overall, min_path_overall)

########################################################################
