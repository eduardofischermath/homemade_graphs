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

from collections import namedtuple as collections_namedtuple
from itertools import zip_longest as itertools_zip_longest
from itertools import chain as itertools_chain

########################################################################
# Vertex, Arrow and Edge namedtuples
########################################################################

# Name of vertex should be preferably hashable because they are often keys of dicts
# On the other hand, we implement no checks for hashableness
Vertex = collections_namedtuple('Vertex', 'name')
# Make weight to be None if unweighted (default value)
Arrow = collections_namedtuple('Arrow', 'source,target,weight', defaults = (None,))
Edge = collections_namedtuple('Edge', 'first,second,weight', defaults = (None,))

def sanitize_vertex(obj, require_namedtuple = False):
  '''
  Given an object, returns a Vertex namedtuple containing it as a name,
  or containing its content as a name.
  
  If starting with a Vertex namedtuple, return the Vertex.
  
  Can also require that the object was a Vertex namedtuple to start with.
  '''
  if isinstance(obj, Digraph.Vertex):
    return obj
  else:
    if require_namedtuple:
      raise ValueError('Require a Vertex namedtuple.')
    else:
      # We do many tests to determine what to do with the object
      # It should have length one in this case (and in particular a __len__ method)
      if not hasattr(obj, '__len__'):
        # A namedtuple always has length, so object is not one
        return Digraph.Vertex(obj)
      # We finally check the length (if we arrive here, there is length)
      elif len(obj) != 1:
        # In this case we know it is not a Vertex instance, and we can envelop it
        return Digraph.Vertex(obj)
      else:
        # In this case it is has length 1
        # We create a vertex from the first and only item
        # Note that one consequence of this is that [item] and item
        #will produce the same Vertex (one with name=item).
        # Note that for a single-char string, its first item is itself
        vertex_from_object = Digraph.Vertex(obj[0])

def sanitize_vertices(vertices, require_namedtuple = False, output_as_generator = False):
  '''
  Iterable version of sanitize_vertex.
  '''
  # We have options for output, as list or generator. Default is list
  #(that is, output_as_generator is defaulted to False)
  # We use lambda and map, making it a list if requested
  # Same could be accomplished via partial from functools, but we prefer lambda
  partial_function = lambda vertex: sanitize_vertex(vertex, require_namedtuple = require_namedtuple)
  as_generator = map(partial_function, vertices)
  if output_as_generator:
    return as_generator
  else:
    return list(as_generator)

def sanitize_arrow_or_edge(tuplee, use_edges_instead_of_arrows,
    require_namedtuple = False):
  '''
  Returns an Arrow or Edge namedtuple with the given information.
  
  If require_namedtuple, ensures argument is an Arrow/Edge namedtuple.
  
  [Tuples are imutable; this always produces a new namedtuple.]
  '''
  if use_edges_instead_of_arrows:
    selected_class = Digraph.Edge
  else:  
    selected_class = Digraph.Arrow
  # We want this to be very quick if tuple is alerady Arrow or Edge
  # Thus, we start by test for being an instance of selected_class
  if isinstance(tuplee, selected_class):
    return tuplee
  else:
    if require_namedtuple:
      raise TypeError('Require tuple to be an Arrow or Edge namedtuple')
    else:
      # In this case we attempt the conversion, and return the required Arrow/Edge
      # This will work if and only if len(tuplee) is 2 or 3 (unweighted or not)
      try:
        new_tuplee = selected_class(*tuplee)
        return new_tuplee
      except SyntaxError:
        # Likely in case the unpacking with * doesn't work
        raise TypeError('Expect tuple to be an iterable/container.')
      except TypeError:
        # This is likely due to not having 2 or 3 arguments given
        raise ValueError('Expect tuple to have length 2 or 3')

def sanitize_arrows_or_edges(tuplees, use_edges_instead_of_arrows,
    require_namedtuple = False, output_as_generator = False):
  '''
  Iterable version of sanitize_arrow_or_edge.
  '''
  partial_function = lambda tuplee: sanitize_arrow_or_edge(tuplee,
      use_edges_instead_of_arrows = use_edges_instead_of_arrows,
      require_namedtuple = require_namedtuple)
  as_generator = map(partial_function, tuplees)
  if output_as_generator:
    return as_generator
  else:
    return list(as_generator)

def get_edges_from_sanitized_arrows(data, are_arrows_from_digraph = False,
    is_multiarrow_free = False):
  '''
  From some sanitized arrows returns the edges which would form them
  (if each edge was decomposed into two mutually-reverse arrows).
  
  Using are_arrows_from_digraph, we obtain all edges derived from the arrows of
  the digraph given as data (using data.get_arrows(), which doesn't read its edges).
  All arrows should automatically be Arrow nameduples.
  
  Otherwise, data is assumed to be an iterable of Arrow namedtuples.
  
  If the digraph has no multiarrows we use a faster algorithm.
  
  If pairing arrows into edges is not possible, returns None.
  '''
  # We first obtain the arrows
  if are_arrows_from_digraph:
    # In this case data is a Digraph instance
    arrows = data.get_arrows()
  else:
    # We ensure list for uniformity, but any iterable is fine
    arrows = list(data)
  # One possible shortcut: if number of arrows is odd, they cannot form a Graph
  if len(arrows) % 2 == 1: 
    return None
  else:
    if is_multiarrow_free:
      # Algorithm with frozenset and hashing: we do it with lists in multiple steps
      #to facilitate understanding, even if it might cost more memory
      arrows_and_its_reverses = [[arrow, Digraph.get_reversed_arrow(arrow, skip_checks = True)] for arrow in arrows]
      list_of_frozensets = [frozenset(pair) for pair in arrows_and_reversed_arrows]
      # We use frozenset to do all the comparing (sets are not hasheable)
      # It is not a problem to fit them all in a set
      set_of_frozensets = frozenset(list_of_frozensets)
      # If we have exactly half elements as we had for arrows, it means they form pairs
      if len(frozenset_of_frozensets) == len(arrows) // 2:
        # To extract one edge from each frozenset
        edges = []
        for pair_of_arrows in set_of_frozensets:
          # We don't have pop() for frozensets, so we make them into lists
          # These lists have two elements; we extract the first
          arrow = list(pair_of_arrows)[0]
          new_edge = Digraph.Edge(arrow.source, arrow.target, arrow.weight)
          edges.append(new_edge)
        return edges
      else:
        # Pairs not perfectly formed
        return None
    else:
      # We are in the case where there are multi-arrows
      # Since two arrows with same source and target don't form an edge,
      #we need to use a slower, O(n^2) algorithm
      # We eliminate the arrows in pairs [each arrow and its reverse] from the list
      # If at any moment we fail to see the reverse of an arrow in the list,
      #we break because that means it forms no edge
      # If we arrive at [] as final list, we are done
      # We use a list to store the new edges (for the possibility we return them)
      new_edges = []
      found_arrow_without_edge = False
      while new_edges: # i. e. new_edges nonempty
        last_arrow = arrows.pop() # pop() removes last and return it
        last_arrow_reversed = Digraph.get_reversed_arrow(last_arrow, skip_checks = True)
        # Try to remove the reversed arrow. If it fails, they don't form edges
        try:
          arrows.remove(last_arrow_reversed)
        except ValueError:
          found_arrow_without_edge = True
          break
      if found_arrow_without_edge:
        return None
      else:
        return new_edges

def sanitize_arrows_and_return_formed_edges(arrows, require_namedtuple = False,
    raise_error_if_edges_not_formed = False):
  '''
  Sanitizes arrows, and returns the corresponding formed edges.
  
  Returns a tuple (arrows, edges). If edges cannot be formed, raise an error
  or return (arrows, None) depending on raise_error_if_edges_not_formed.
  '''
  new_arrows = sanitize_arrows(arrows, require_namedtuple)
  new_edges = get_edges_from_sanitized_arrows(new_arrows)
  # We deal with the situation in which we couldn't properly form edges
  if new_edges is None:
    # In this case we could not form the edges correctly
    if raise_error_if_edges_not_formed:
      raise ValueError('Cannot form edges with given arrows')
    else:
      # Simply return None as the new edges
      return (new_arrows, None)
  else:
    return (new_arrows, new_edges)

def get_reversed_arrow_or_equivalent_edge(tuplee, use_edges_instead_of_arrows,
    require_namedtuple = False):
  '''
  Given an arrow, returns the opposite arrow, going in the opposite direction.
  
  Given an edge, returns the equivalent edge in which its two inciding
  vertices are listed in reverse order.
  
  Has the option to accept or not a tuple which is not Arrow/Edge namedtuple.
  '''
  # To make the work simpler we factor through sanitization
  tuplee = Digraph.sanitize_arrow_or_edge(tuplee,
      use_edges_instead_of_arrows = use_edges_instead_of_arrows,
      require_namedtuple = require_namedtuple)
  # We can read the namedtuple from tuplee now; Arrow or Edge
  selected_class = type(tuplee)
  # We reverse the first and second items (arrow/source, first/second)
  # We number the items to be a common approach to Arrow and Edge
  return selected_class(tuplee[1], tuplee[0], tuplee[2])

def get_reversed_arrows_or_edges(tuplees, use_edges_instead_of_arrows,
    require_namedtuple = False, output_as_generator = False):
  '''
  Iterable version of get_reversed_arrow_or_edge.
  '''
  # We use lambda and map
  partial_function = lambda tuplee: get_reversed_arrow(tuplee,
      use_edges_instead_of_arrows = use_edges_instead_of_arrows,
      require_namedtuple = require_namedtuple)
  as_generator = map(partial_function, tuplees)
  if output_as_generator:
    return as_generator
  else:
    return list(as_generator)

def get_arrows_from_edge(edge, require_namedtuple = False):
  '''
  From an edge creates a list with the two corresponding arrows.
  
  Has the option to accept or not a tuple which is not Edge.
  '''
  # To save time we factor though sanitization
  edge = Digraph.sanitize_arrow_or_edge(edge,
      use_edges_instead_of_arrows = use_edges_instead_of_arrows,
      require_namedtuple = require_namedtuple)
  # We are guaranteed to have an Edge now
  # We could use get_reversed_arrow_or_equivalent_edge, but we won't
  first_arrow = Digraph.Arrow(edge.first, edge.second, edge.weight)
  second_arrow = Digraph.Arrow(edge.second, edge.first, edge.weight)
  return [first_arrow, second_arrow]

def get_arrows_from_edges(edges, require_namedtuple = False, output_as_generator = False):
  '''
  Iterable version of get_arrows_from_edge.
  '''
  # We produce a suitable generator using chain from itertools
  # First, we produce a generator to produce smaller (length 2) generators
  # We use a lambda to create those small generators
  # We modify the output to be a generator for each edge
  small_generator = lambda edge: (arrow for arrow in Digraph.get_arrows_from_edge(
      edge, require_namedtuple = require_namedtuple))
  # We group them together in a generator of generators
  generator_of_generators = map(small_generator, edges)
  # We produce a single generator
  big_generator = itertools_chain(*generator_of_generators)
  if output_as_generator:
    return big_generator
  else:
    return list(big_generator)

def remove_weight_from_arrow_or_edge(tuplee,
    use_edges_instead_of_arrows, require_namedtuple = False):
  '''
  Returns a new Arrow or Edge with no weight (that is, weight None).
  
  Has the option to accept or not tuples which are not Arrows or Edges.
  '''
  # We factor through sanitization to save time
  tuplee = Digraph.sanitize_arrow_or_edge(tuplee,
      use_edges_instead_of_arrows = use_edges_instead_of_arrows,
      require_namedtuple = require_namedtuple)
  # We have an Edge or Arrow namedtuple. We can capture class via type()
  selected_class = type(tuplee)
  return selected_class(tuplee[0], tuplee[1], None)
  # Note we don't really check if tuplee was weighted to begin with

def remove_weight_from_arrows_or_edges(weighted_tuples,
    use_edges_instead_of_arrows, require_namedtuple = False,
    output_as_generator = False):
  '''
  Iterable version of remove_weight_from_arrow_or_edge.
  '''
  # The following can also be accomplised via partial() from functools library
  partial_function = lambda x: remove_weight_from_arrow_or_edge(
      unweighted_tuple = x,
      use_edges_instead_of_arrows = use_edges_instead_of_arrows,
      require_namedtuple = require_namedtuple)
  as_generator = map(partial_function, weighted_tuples)
  if output_as_generator:
    return as_generator
  else:
    return list(as_generator)

def write_weight_into_arrow_or_edge(tuplee, use_edges_instead_of_arrows,
    new_weight = None, require_namedtuple = False):
  '''
  Writes a weight (default 1) to an Arrow or Edge, returning a new one.
  
  If arrow/edge originally unweighted, adds the given value as weight.
  
  If originally weighted, this modifies the weight to be the given value.
  
  Has the option to accept or not tuples which are not Arrows or Edges.
  '''
  # Tipically, None is used to denote weight in unweighted arrows.
  # In this function/method we do differently.
  # If None is new_weight (argument of this function), we change it to 1
  # (This use of None as a default argument only casually coincides with
  #the use of None as weight attribute of unweighted arrows/edges.)
  if new_weight == None:
    new_weight = 1
  # We can factor through sanitization
  tuplee = Digraph.sanitize_arrow_or_edge(tuplee,
      use_edges_instead_of_arrows = use_edges_instead_of_arrows,
      require_namedtuple = require_namedtuple)
  # Now tuplee is guaranteed to be Arrow or Edge
  selected_class = type(tuplee)
  return selected_class(tuplee[0], tuplee[1], )
  # Note that in no moment we verify we started with an unweighted Arrow/Edge

def write_weights_into_arrows_or_edges(tuplees, use_edges_instead_of_arrows,
    new_weights = None, require_namedtuple = False, output_as_generator = False):
  '''
  Iterable version of write_weight_into_arrow_or_edge
  '''
  # We use lambda. Using partial from functools could potentially work too
  #if the issues with argument ordering were solved
  partial_function = lambda tuplee, weight: write_weight_into_arrow_or_edge(
      tuplee = tuplee,
      use_edges_instead_of_arrows = use_edges_instead_of_arrows,
      new_weight = weight,
      require_namedtuple = require_namedtuple)
  # We prepare the arguments together. Note zip_longest from itertools
  #will automatically fill the pairings with None if the iterables have
  #different sizes (we hope that those are the weights and not the tuples)
  # To do so we must guarantee new_weights is iterable. This could be
  #a concern if we want the default value by setting new_weights = None
  if new_weights is None:
    new_weights = []
  tuples_and_weights = itertools_zip_longest(tuplees, new_weights)
  # We build the generator using map(), and, if asked, as list
  as_generator = map(partial_function, tuples_and_weights)
  if output_as_generator:
    return as_generator
  else:
    return list(as_generator)

def get_namedtuples_from_neighbors(dict_of_neighbors, output_as, namedtuple_choice = None):
  '''
  From a dictionary of neighbors, creates all corresponding tuples.
  
  Can give the answer as a list or as a dict with same keys.
  
  From dic[a] = [a1, a2, ..., an] produce [(a, a1), (a, a2), ..., (a, an)]
  Same if we have [[a1], [a2], ..., [an]] or [(a1), (a2), ..., (an)]
  From dic[a] = [[a11,... ,a1m], [a21,... ,a2m], ..., [an1,... ,anm]]
  produce [(a, a11, ..., a1m), (a, a21, ..., a2m), ..., (a, an1, ..., anm)]
  
  If namedtuple is given [it must be given as a class, not a string],
  produce that namedtuple. Otherwise, plain tuple.
  '''
  # It all depends if the values in the dict are iterable or not
  # To save time and testing strings fewer times, we do:
  if output_as.lower() == 'list':
    output_as_dict_instead_of_list = False
  elif output_as.lower() == 'dict':
    output_as_dict_instead_of_list = True
  else:
    raise ValueError('Can only output as dict or as list')
  # To save time while testing subclassing, we do:
  if namedtuple_choice is None or namedtuple_choice == tuple:
    # Note that tuple takes an iterable, while namedtuple specific values
    # We want to uniformize the input
    # Easier way is to use a lambda with unpacking
    namedtuple_choice = lambda *x: tuple(x) # Default to regular tuple
  else:
    # We assert we do have a tuple subclass (just to avoid errors later)
    # (Probably the best way to meet the proposal of the method.)
    assert issubclass(namedtuple_choice, tuple), 'Needs subclass of tuple'
    pass
  edited_dict = {}
  edited_list = []
  for key in dict_of_neighbors:
    edited_dict[key] = []
    for item in dict_of_neighbors[key]:
      # We check if __len__ is a valid method to determine how to proceed
      if hasattr(item, '__len__'):
        # Use * for easier unpacking. This works for length 0, 1 or bigger
        pre_appending_tuple = (key, *item)
      else:
        # No length method, even easier
        pre_appending_tuple = (key, item)
      # We make the right type of tuple or namedtuple
      # The packing and unpacking appear weird but is likely the best way
      pre_appending_tuple = namedtuple_choice(*pre_appending_tuple)
      # We append to the big list or to the individual key as requested
      if output_as_dict_instead_of_list:
        edited_dict[key].append(appending_tuple)
      else:
        edited_list.append(appending_tuple)

########################################################################