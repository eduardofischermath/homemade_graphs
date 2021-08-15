######################################################################
# DOCUMENTATION / README
######################################################################

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
# Imports
########################################################################
from collections import namedtuple as collections_namedtuple
from itertools import zip_longest as itertools_zip_longest
from itertools import chain as itertools_chain
from itertools import product as itertools_product
from copy import copy as copy_copy
from random import choices as random_choices
from math import log2 as math_log2
from math import inf as math_inf
from heapq import heapify as heapq_heapify
from heapq import heappush as heapq_heappush
from heapq import heappop as heapq_heappop
# Since cache from functools was introduced in Python version >= 3.9,
#we check for it. If not new enough, we go with lru_cache(maxsize = None)
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
# Class VertexPath
########################################################################

class VertexPath(object):
  '''
  A VertexPath in a Digraph is a sequence of arrows in the digraph such
  that the source of any arrow [after the first] is the target of the previous.
  
  Accepts a single-vertex-no-arrow path and even two "degenerate" cases:
  a single-vertex-single-arrow path, and a no-vertex-no-arrow path.
  
  Attributes:
  underlying_digraph: a Digraph or subclass where the path comes from
  vertices: list of Vertices
  arrows: list of Arrows
  '''
  
  def __init__(self, underlying_digraph, data, data_type,
      verify_validity_on_initialization = False):
    '''
    Magic method. Initializes the instance.
    '''
    self.underlying_digraph = underlying_digraph
    if data_type.lower() == 'vertices_and_arrows':
      # In this case we expect data to be (vertices, arrows)
      # For uniformization, make them lists
      pre_vertices, pre_arrows = data
      self.vertices = list(pre_vertices)
      self.vertices = list(pre_arrows)
    elif data_type.lower() == 'arrows':
      # If data_type is 'arrows', they are pretty much what we need
      # We expect a list, but we can do with any iterable
      self.arrows = list(data)
      # The vertices can be easily derived from the arrows
      if not self.arrows:
        self.vertices = []
      else:
        self.vertices = []
        # We add the source of the first arrow, and then the targets of all arrows
        #(including the first)
        self.vertices.append(self.arrows[0].source)
        for arrow in self.arrows:
          self.vertices.append(self.arrow.target)
    elif data_type.lower() == 'vertices':
      # In this case the vertices are ready, and we need to get the arrows
      # We use get_shortest_arrow_between_vertices
      # If there are multiple (meaning the digraph is not simple), it will
      #produce the shortest. If there are none, an exception will be raised,
      #because in this case it is not a real path
      self.vertices = list(data)
      if not self.vertices:
        self.arrows = []
      else:
        self.arrows = []
        for idx in range(len(self.vertices) - 1):
          # Form the arrows the only possible way
          # There will be one fewer arrows than vertices
          self.arrows.append(self.underlying_digraph.get_shortest_arrow_between_vertices(
              source = self.vertices[idx], target = self.vertices[idx+1],
              skip_checks = False))
    else:
      raise ValueError('Option not recognized')
    if verify_validity_on_initialization:
      self.verify_validity()
    else:
      pass

  def __bool__(self):
    '''
    Magic method. Returns the boolean value of self.
    '''
    # Our convention: true if there is at least one vertex
    # False only if it's the path with no vertices
    return bool(self.vertices)
    
  def __len__(self):
    '''
    Magic method. Returns the size of self.
    '''
    # Convention: size/length is the number of arrows
    # Note empty path and one-vertex path both have length 0, but the
    #former has bool False and the second bool True
    return self.get_number_of_arrows()
    
  def get_number_of_arrows(self):
    '''
    Returns number of arrows.
    '''
    return len(self.arrows)
    
  def get_number_of_vertices_as_path(self):
    '''
    Returns number of vertices of self as a path.
    
    [Note that, with exception of the degenerate cases, a path/cycle will
    have one more vertex as a path than it has arrows.]
    '''
    return len(self.vertices)
    
  def get_number_of_vertices_as_cycle(self):
    '''
    Returns number of vertices of self as a cycle.
    
    Requires instance to be a cycle. In this case, we return the number of arrows.
    '''
    assert self.is_cycle(), 'Need to be a cycle'
    return self.get_number_of_arrows()
    
  def is_degenerate(self):
    '''
    Returns whether path is degenerate.
    
    A path is degenerate if and only if either condition happens:
    Type-I: it has zero vertices [and thus no arrows]
    Type-II: it has a self-arrow and self.vertices has a single vertex
    
    [In both cases, the degenerate paths are cycles.]
    
    [A path with a single vertex and no arrows is not degenerate, nor is
    a cycle with one arrow if self.vertices has two equal elements.]
    '''
    if self.is_degenerate_type_i():
      return True
    elif self.is_degenerate_type_ii():
      return True
    else:
      # In this case we should even have that self.vertices is one element
      #longer than self.arrows, independently of being a cycle or not
      return False
      
  def is_degenerate_type_i(self):
    '''
    Returns whether path is degenerate Type-I: it has no vertices.
    '''
    if self.get_number_of_vertices_as_path() == 0:
      return True
    else:
      return False
  
  def is_degenerate_type_ii(self):
    '''
    Returns whether path is degenerate Type-II: has one vertex and one self-arrow.
    '''
    if self.get_number_of_vertices_as_path() == 1 and self.get_number_of_arrows() == 1:
      return True
    else:
      return False

  def __repr__(self):
    '''
    Magic method. Returns faithful representation of instance.
    '''
    return f'{type(self)}(underlying_digraph = {self.underlying_digraph},\
         data = {self.arrows}, data_type = \'arrows\', verify_validity = True)'
    
  def __str__(self):
    '''
    Magic method. Returns user-friendly representation for instance.
    '''
    # Note we don't mention the graph the instance comes from
    if isinstance(self, VertexCycle):
      single_name = 'Cycle'
    else:
      single_name = 'Path'
    # We want to have it slightly different if there are no vertices.
    if bool(self):
      return f'{single_name} with vertices {self.vertices}\nand arrows {self.arrows}'
    else:
      return f'Empty {single_name} with no vertices nor arrows.'

  def __eq__(self, other):
    '''
    Magic method. Determines equality between two instances.
    '''
    # First we need self and other to be VertexPath (self already is)
    if not isinstance(other, VertexPath):
      return False
    # To be equal, two instances must have the same underlying graph
    elif self.underlying_digraph != other.underlying_digraph:
      return False
    # We then compare equality of the arrows. That is a necessary and
    #sufficient condition
    elif self.arrows != other.arrows:
      return False
    else:
      return True
    # (Note we don't read the class. So a VertexPath instance can be
    #evaluated as equal to a VertexCycle instance)

  def __hash__(self):
    '''
    Magic method. Produces a hash of the instance.
    '''
    # Simplest is to take self.arrows, which pretty much determines the instance
    #(assuming the underlying vertex is fixed for out purposes), and compute hash
    # Arrows are namedtuples. self.arrows is list but is hashable if tuplefied
    return __hash__(tuple(self.arrows))

  def verify_validity(self):
    '''
    Verifies that instance represents a path in a digraph.
    '''
    # First we ensure vertices and arrows do belong to the digraph
    for vertex in self.vertices:
      assert vertex in self.underlying_digraph
    for arrow in self.arrows:
      # To facilitate searching for the arrow, we use the self._neighbors_out
      assert arrow in self.underlying_digraph.get_arrows_out(arrow.source)
    # We verify it is indeed a path, and that the vertices match with the arrows
    # Part of this is automatically set during __init__, but not all.
    # Also, if data_type == 'vertices_and_arrows' on __init__, nothing is
    # We need to excise the no-vertex path
    if not self.vertices: # Measuring length
      assert len(self.arrows) == 0, 'Without vertices there should be no arrows'
    else:
      assert len(self.arrows) == len(self.vertices) - 1, 'There should be one more vertex than arrow'
      for idx, arrow in enumerate(self.arrows):
        assert arrow.source == self.vertices[idx], 'Incoherent vertices and arrows'
        assert arrow.target == self.vertices[idx+1], 'Incoherent vertices and arrows'
      # To ensure we have a cycle if VertexCycle
      if isinstance(self, VertexCycle):
        assert self.is_cycle(), 'Need path to be a cycle'
      
  def get_total_weight(self, request_none_if_unweighted = False):
    '''
    Returns the total weight/length of the path, which is the result of
    adding the weights of the arrows.
    
    If instance has no arrows, returns 0.
    
    If arrows are unweighted, arrows count as having length/weight 1, but
    with the option to return None instead.
    '''
    # Note that if one arrow is weighted, all are.
    try:
      return sum(arrow.weight for arrow in self.arrows)
      # This will produce TypeError if trying to sum even a single None
    except TypeError:
      # In this case, the only possible information is the number of arrows
      # We can return either None or the number of arrows, depending on need
      if request_none_if_unweighted:
        return None
      else:
        return self.get_number_of_arrows()

  @staticmethod
  def reformat_paths(underlying_digraph, data, data_type, output_as,
      skip_checks = False):
    '''
    Given data configuring an instance of the class, path or cycle,
    returns same path or cycle given by equivalent information as requested.
    
    [With the exception that if the underlying digraph is a weighted multidigraph,
    giving information by the vertices picks the arrows of least weight.
    Nonetheless, we consider the information determines the path or cycle uniquely.]
    
    Options for data_type:
    'path'
    'cycle' [only if indeed cycle]
    'vertices'
    'arrows'
    'vertices_and_arrows'
    
    Options for output_as:
    [all options for data_type are acceptable for output_as]
    'length'
    'length_and_vertices'
    'length_and_arrows'
    'length_and_vertices_and_arrows'
    'str'
    'repr'
    'nothing'
    '''
    # We prepare the strings to have only lowercase characters
    # This is useful as they will be evaluated multiple times
    data_type = data_type.lower()
    output_as = output_as.lower()
    # We return None if asked to return nothing [it is useful to do this first]
    if output_as == 'nothing':
      return None
    # We build an instance from the data (if not already starting with one)
    if data_type == 'cycle':
      as_instance = data
      if not skip_checks:
        assert isinstance(as_instance, VertexCycle), 'Need to be a VertexCycle'
        assert underlying_digraph == as_instance.underlying_digraph, 'Underlying digraph must be correct'
        as_instance.verify_validity()
    elif data_type == 'path':
      as_instance = data
      if not skip_checks:
        assert isinstance(as_instance, VertexPath), 'Need to be a VertexPath'
        assert underlying_digraph == as_instance.underlying_digraph, 'Underlying digraph must be correct'
        as_instance.verify_validity()
    else:
      # If output_as is cycle, we aim for VertexCyle. Otherwise, VertexPath is good enough
      if output_as == 'cycle':
        selected_class = VertexCycle
      else:
        selected_class = VertexPath
      # Building the instance with __init__
      verify_validity_on_initiation = not skip_checks
      as_instance = selected_class(underlying_digraph = underlying_digraph,
          data = data, data_type = data_type,
          verify_validity_on_initiation = verify_validity_on_initiation)
    # Now that we have as_instance, we work into producing the requested information
    if output_as == 'str':
      return str(as_instance) # as_instance.__str__()
    elif output_as == 'repr':
      return repr(as_instance) # as_instance.__repr__()
    elif output_as in ['path', 'cycle']:
      return as_instance
    else:
      # We prepare the variables according what is required
      # Three pieces of information: length, vertices, arrows
      pre_data = []
      if 'length' in output_as:
        # Note that 'lengths' will produce the same effect as 'length'
        # (This makes the method slightly more flexible, which is good)
        length = self.get_total_weight(request_none_if_unweighted = False)
        pre_data.append(length)
      if 'vertices' in output_as:
        vertices = self.get_vertices()
        pre_data.append(vertices)
      if 'arrows' in output_as:
        arrows = self.get_arrows()
        pre_data.append(arrows)
      # We now return the output
      # If pre_data has 2 or more items, it is returned as a tuple
      # If it has 1 item, we return that single item
      # If it has 0 items, this means the request was bed
      if len(pre_data) >= 2:
        return tuple(pre_data)
      elif len(pre_data) == 1:
        return pre_data[0]
      else:
        raise ValueError('Option not recognized')

  def is_hamiltonian_path(self):
    '''
    Returns whether path is a Hamiltonian path.
    
    [Note that every cycle is a path but, unless on a one-vertex graph,
    a Hamiltonian path and a Hamiltonian cycle are strictly different.]
    '''
    # First we check lengths which is easy
    length_underlying_digraph = len(self.underlying_digraph)
    if len(self.vertices) != length_underlying_digraph:
      return False
    # We now check the vertices in self.vertices are distinct using set()
    elif len(self.vertices) != len(set(self.vertices)):
      return False
    else:
      # If passed the two tests, it is a Hamiltonian path
      return True

  def is_cycle(self):
    '''
    Returns whether path is a cycle.
    '''
    return (self.vertices[0] == self.vertices[-1])

  def is_hamiltonian_cycle(self):
    '''
    Returns whether path or cycle is a Hamiltonian cycle.
    '''
    # Adapted from is_hamiltonian_path, with a few differences
    # A Hamiltonian cycle becomes a Hamiltonian path without its first vertex
    # Also, it needs to be a cycle
    if not self.is_cycle():
      return False
    length_underlying_digraph = len(self.underlying_digraph)
    vertices_except_first = self.vertices[1:]
    if len(vertices_except_first) != length_underlying_digraph:
      return False
    elif len(vertices_except_first) != len(set(vertices_except_first)):
      return False
    else:
      # Survived all tests, thus is Hamiltonian cycle
      return True

  def shorten_path(self, number_to_shorted, modify_self = False, skip_checks = False):
    '''
    Removes a number of arrows and vertices from the end of path.
    
    Can either modify self (returning None) or create a new instance.
    '''
    raise NotImplementedError('WORK HERE')

  def append_to_path(self, data, data_type, modify_self = False, skip_checks = False):
    '''
    Extend path by adding a vertex and an arrow to its end.
    
    Can either modify self (returning None) or create a new instance.
    
    data_type may be:
    'vertex'
    'arrow'
    'vertex_and_arrow'
    '''
    data_type = data_type.lower()
    # If object is designed as VertexCycle, we cannot proceed
    # (It would cease to be a cycle, create confusion)
    if isinstance(self, VertexCycle):
      raise TypeError('Cannot append vertex/arrow to a cycle.')
    elif self.is_degenerate():
    # The nondegenerate cases always have one vertex more than arrows
    # This is the context of the method. So we exclude degenerate paths
      raise ValueError('Cannot append to degenerate path.')
    # We also want to ensure data and data_type are what promised
    else:
      # We prepare the data: new_vertex and new_arrow
      if data_type == 'vertex_and_arrow':
        if not skip_checks:
          assert hasattr(data, __len__), 'Need data to have length'
          assert len(data) == 2, 'Need data to have two items'
        new_vertice, new_arrow = data
      elif data_type == 'vertex':
        # We strive to have the arrow right here. This will save work in the
        #case of modify_self = False, in which we would need to pass vertex
        #info to __init__ and __init__ would do the job
        new_vertex = data
        if not skip_checks:
          assert new_vertex in self.underlying_digraph, 'Vertex must be from underlying digraph'
        new_arrow = self.underlying_digraph.get_shortest_arrow_between_vertices(
            self.vertices, new_vertex)
      elif data_type == 'arrow':
        new_arrow = data
        new_vertex = new_arrow.target
      else:
        raise ValueError('Option not recognized.')
      # Having new_vertex and new_arrow, do optional checks
      # [Some might be redundant depending on input]
      if not skip_checks:
        assert new_vertex in self.underlying_digraph, 'Vertex must be from underlying digraph'
        assert new_arrow in self.underlying_digraph.get_arrows(), 'Arrow must be from underlying digraph'
        assert new_arrow.source == self.vertices[-1], 'Arrow must fit after path'
        assert new_arrow.target == new_vertex, 'Vertex and arrow information must be consistent'
      # Having new_vertex and new_arrow, do as requested
      if modify_self:
        self.vertices.append(new_vertex)
        self.arrows.append(new_arrow)
      else:
        new_vertices = self.vertices + new_vertex
        new_arrows = self.arrows + new_arrow
        data = (new_vertices, new_arrows)
        data_type = 'vertices_and_arrows'
        verify_validity_on_initialization = not skip_checks
        new_instance = type(self)(data = data, data_type = data_type,
            verify_validity_on_initialization = verify_validity_on_initialization)
        return new_instance
    
  def extend_path(self, data, data_type, modify_self = False, skip_checks = False):
    '''
    Iterable version of appent_to_path.
    
    data_type may be:
    'vertices'
    'arrows'
    'vertices_and_arrows'
    'path'
    '''
    data_type = data_typer.lower()
    # If modify_self, we sent the data (item by item) to append_path
    # Otherwise, we form another_path and call self.__add__(another_path)
    if modify_self:
      if data_type == 'vertices':
        vertices = data
        for vertex in vertices:
          self.append_to_path(data = vertex, data_type = 'vertex',
              modify_self = True, skip_checks = skip_checks)
      elif data_type == 'arrows':
        arrows = data
        for arrow in arrows:
          self.append_to_path(data = arrow, data_type = 'arrow',
              modify_self = True, skip_checks = skip_checks)
      elif data_type == 'vertex_and_arrow':
        vertices, arrows = data
        if skip_checks:
          # There should be equal number of arrows and vertices 
          # This check is so important we can't skip
          # The others can be done inside 'append_to_path
          assert len(vertices) == len(arrows), 'Need same number of vertices and arrows'
          # We use zip to pass arguments
          for vertex_and_arrow in zip(vertices, arrows):
            self.append_to_path(data = vertex_and_arrow,
                data_type = 'vertex_and_arrow',
                modify_self = True, skip_checks = skip_checks)
      elif data_type == 'path':
        another_path = data
        # If degenerate, we won't proceed
        if another_path.is_degenerate:
          raise ValueError('Need nondegenerate path for path addition.')
        # We do a basic check (which can't be done on append_to_path due
        #to the nature of this procedure)
        if not skip_checks:
          assert another_path.vertices[0] == self.vertices[-1], 'First path must segue into second'
        # For data_type == 'path' we take arrows and vertices
        #[except first vertex to avoid repetitions]
        arrows = another_path.arrows
        vertices = another_path.vertices[1:]
        # These should be lists of equal length. We use zip to pass arguments
        for vertex_and_arrow in zip(vertices, arrows):
          self.append_to_path(data = vertex_and_arrow,
              data_type = 'vertex_and_arrow',
              modify_self = True, skip_checks = skip_checks)
    # Enter the realm where modify_self is False (i. e. return new instance)
    else:
      if data_type == 'path':
        another_path = data
      else:
        verify_validity_on_initiation = not skip_checks
        another_path = type(self)(data = data, data_type = data_type,
            verify_validity_on_initiation = verify_validity_on_initiation)
      return self.__add__(another_path = another_path, skip_checks = skip_checks)
    
  def __add__(self, another_path, skip_checks = False):
    '''
    Magic method. Returns the sum of two instances.
    
    Merges two paths into a single one, if the first can segue into the second.
    '''
    # Unless overriden, we ensure another_path is also a path
    # (If it isn't, behavior is unpredictable, and user is responsible)
    # No other checks/verifications will be overriden by skip_checks
    if not skip_checks:
      try:
        assert isinstance(another_path, VertexPath)
      except AssertionError:
        raise TypeError('Can only perform path addition on paths.')
    # We cannot have VertexCycles (it would not make a lot of sense
    #except in very specific cases), so we rule them out
    # Note that is is_cycle is True, the instance might be a VertexPath
    #which coincidentally starts and ends at the same vertex, but it is
    #not really a VertexCycle. Addition is this case is permitted, and very
    #likely is_cycle() will be False for the created instance
    if instance(self, VertexCycle) or isinstance(another_path, VertexCycle):
      raise TypeError('Cannot perform path addition on cycles.')
    # We can only add if paths are from same digraph
    elif self.underlying_digraph != another_path.underlying_digraph:
      raise ValueError('Paths to be added should be from same digraph')
    # We discard the anomalities/degeneracies
    # For Type-I degeneracy, we return the other path
    #(Even if it is also a Type-I or Type-II degeneracy!)
    elif self.is_degenerate_type_i():
      return another_path
    elif another_path.is_degenerate_type_i():
      return self
    # If any degenerate Type-II, we cannot perform a true path addition
    # (Exception if the other was a degeneracy Type-I)
    elif self.is_degenerate_type_ii() or another_path.is_degenerate_type_ii():
      raise ValueError('Cannot perform path addition with Type-II degeneracy.')
    # From here on, no degeneracies. In particular, both self and another_path
    #have exactly one more vertex [as path] than they have arrows
    # We verify one path segues into the next
    # (We don't allow this check to be skipped by skip_checks)
    elif self.vertices[-1] != another_path.vertices[0]:
      raise ValueError('Need first path to segue into the second.')
    else:
      # Here we implement the addition
      # Since we all attributes ready, we can use them for __init__]
      # We obtain the vertices. Note we omit the vertex uniting the paths
      new_vertices = self.vertices + another_path.vertices[1:]
      new_arrows = self.arrows + another_path.arrows
      # We create a new instance (same class) and then return it
      kwargs = {'underlying_digraph': self.underlying_digraph,
          'data': (new_vertices, new_arrows),
          'data_type': 'vertices_and_arrows',
          'verify_validity_on_initiation': False}
      new_instance = type(self)(**kwargs)
      return new_instance

  def __iadd__(self, another_path):
    '''
    Magic method.
    '''
    # We are unsure on how to do this. Should we force instance to modify itself?
    # (Compare with behaviors of __iadd__ on list and on str.)
    raise NotImplemented('Unplanned behavior')

########################################################################
# Class VertexCycle
########################################################################

class VertexCycle(VertexPath):
  '''
  A VertexCycle is a VertexPath which starts and ends on the same vertex.
  '''

  def rebase_cycle(self, base_vertex, modify_self = False):
    '''
    Returns the same cycle but with vertices rotated so requested vertex
    is the first and last of the cycle.
    
    Can either modify self [returning None] or return a new instance.
    '''
    # We get the index of the base_vertex in the cycle
    # In case base_vertex isn't in the cycle, it will raise ValueError
    # Note index returns the first occurrence of the vertex (a VertexPath
    #or VertexCycle potentially contain self intersections)
    base_idx = self.vertices.index(base_vertex)
    # The first arrow will be the one with source base_vertex, that is,
    #the arrow with index base_idx
    # Easiest way is to use moduler arithmetic on the number of arrows
    # (Vertices can be read straight from them during __init__)
    number_of_arrows = len(self.arrows)
    rotated_arrows = [self.arrows[(idx + base_idx) % number_of_arrows]
        for idx in range(number_of_arrows)]
    # To facilitate things, we build a dict for arguments, called kwargs
    kwargs = {'underlying_digraph': self.underlying_digraph,
        data: rotated_arrows,
        data_type: 'arrows',
        verify_validity: True}
    if modify_self:
      self.__init__(**kwargs)
      return None
    else:
      return type(self)(**kwargs)

########################################################################
# Class ImmutableVertexPath
########################################################################

class ImmutableVertexPath(VertexPath):
  pass

  def shorten_path(self, number_to_remove, skip_checks = False):
    '''
    Immutable version of VertexPath.shorten_path
    '''
    return super().shorten_path(number_to_remove = number_to_remove,
        modify_self = False, skip_checks = False)
  
  def append_to_path(self, data, data_type, skip_checks = False):
    '''
    Immutable version of VertexPath.append_to_path
    '''
    return super().append_to_path(data = data, data_type = data_type,
        modify_self = False, skip_checks = False)

  def extend_path(self, data, data_type, skip_checks = False):
    '''
    Immutable version of VertexPath.extend_path
    '''
    return super().extend_path(data = data, data_type = data_type,
        modify_self = False, skip_checks = False)

########################################################################
# Class ImmutableVertexCycle
########################################################################

class ImmutableVertexCycle(VertexCycle, ImmutableVertexPath):
  pass
  
  def rebase_cycle(self, base_vertex):
    '''
    VertexCycle.rebase_cycle version for immutable cycles.
    '''
    # Easiest is to call super(), ensuring modify_self is False
    # Note super, in some sense, takes self as instance of a child class
    #and returns self as instance of a base class
    # That is why there is no need to write self
    # (super() does take arguments but when called inside an instance method
    #they are understood from the context, at least in Python 3)
    return super().rebase_cycle(base_vertex = base_vertex, modify_self = False)

########################################################################
# Class MutableVertexPath
########################################################################

class MutableVertexPath(VertexPath):
  pass
  
  # Since this is mutable (and so will be the classes inheriting from this)
  #we should disable __hash__ which is called from VertexPath
  # Make __hash__ return None to unvalidate it
  # The __mro__ properties guarantee that since this inherits directly
  #from VertexPath, the methods here will always be called earlier.
  def __hash__(self):
    '''
    Magic method. Returns a hash of the instance.
    
    In this particular class, MutableVertexPath, there should be no hashing
    because the instance if mutable. Thus hash() is explicitly set to None
    to avoid inheriting the hash from the parent class VertexPath.
    '''
    return None

########################################################################
# Class MutableVertexCycle
########################################################################

class MutableVertexCycle(VertexCycle, MutableVertexPath):
  pass

########################################################################
# Declaration of Digraph class and of Vertex, Arrow, Edge namedtuples
########################################################################

class Digraph(object):
  '''
  Class which implements digraphs, or directed graphs. A digraph is a set
  of points (called vertices) and a set of arrows (ordered pairs of vertices)
  which may or may not be weighted.
  
  An undirected graph (also called simply graph) is also implemented as
  a diagraph by interpreting its edges (an unordered pair of vertices)
  as two arrows, back and forth within the pair.
  '''
  
  # Before the class per se we create some useful objects as namedtuples
  # Name of vertex should be preferably hashable because they are often keys of dicts
  # On the other hand, we implement no checks for hashableness
  Vertex = collections_namedtuple('Vertex', 'name')
  # Make weight to be None if unweighted (default value)
  Arrow = collections_namedtuple('Arrow', 'source,target,weight', defaults = (None,))
  Edge = collections_namedtuple('Edge', 'first,second,weight', defaults = (None,))

########################################################################
# Static Methods for Vertex, Arrow, Edge namedtuples
########################################################################

  @staticmethod
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

  @staticmethod
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

  @staticmethod
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

  @staticmethod
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

  @staticmethod
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

  @staticmethod
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

  @staticmethod
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

  @staticmethod
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
  
  @staticmethod
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
  
  @staticmethod
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
  
  @staticmethod
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

  @staticmethod
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
  
  @staticmethod
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

  @staticmethod
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

  @staticmethod
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
# Method __init__ (used for initialization and resetting)
########################################################################

  def __init__(self, data, data_type, cast_as_class = None):
    '''
    From some data, builds a digraph object.
    
    The digraph might be weighted, or not, and directed, or not. [An directed
    graph is called Digraph. An undirected graph is simply called Graph.]

    Array is assumed to contain the information for building the digraph.
    
    Data specifications for each data_type:
    
    'all_arrows': self explanatory, a list [or other iterable] of arrows.
    Depending on their characteristics (i. e. the length of important information)
    we will deduce weighted or unweighted
    
    'some_vertices_and_all_arrows': similar to before but a tuple starting with
    a list [or iterable] of vertices, and then the iterable for all arrows
    
    'all_vertices_and_all_arrows': similar but we insist all vertices
    are given; arrows are forced to incide on the known vertices
    
    'all_edges': similar to 'all_arrows' with each edge generating
    two arrows, back and forth
    
    'some_vertices_and_all_edges': similar to 'some_vertices_and_all_arrows'
    
    'all_vertices_and_all_edges': similar to 'all_vertices_and_all_arrows'
    
    'arrows_out_as_dict': gives the arrows out of vertices, each vertex
    being the key of a dict whose value has the arrows or the destinations
    
    'arrows_out_as_list': a list [or iterable] of lists [or iterables],
    and for each of them, the item at position 0 is the vertex and all
    other items correspond to the arrows out of it
    
    'edges_out_as_dict': similar to 'arrows_out_as_dict' but each arrow
    should be interpreted as an edge, giving also rise to its reverse arrow
    
    'edges_out_as_list': similar to 'arrows_out_as_list'
    
    'neighbors_out_as_dict': similar to 'arrows_as_as_dict' but each value
    is not the arrow but only the target of the arrow (and possibly the weight)
    
    'neighbors_out_as_list': similar to 'arrows_out_as_list'
    
    'neighbors_as_dict': similar to 'arrows_as_as_dict', providing neighbors
    and possibly weight, but creating edges instead of arrows
    
    'neighbors_as_list': similar to 'arrows_out_as_list'
    '''
    # We have an optional argument to override the class being called
    # None is the default value (to mean don't interfere with subclassing)
    if cast_as_class is not None:
      # We want to also accept string representations of the 6 possible subclasses
      cast_as_class = Digraph.read_subclass_from_string(cast_as_class, require_string = False)
      try:
        self.__class__ = cast_as_class
      except:
        raise ValueError('Cannot cast as requested class.')
    # Depending on the subclass calling this, it expects a few different things
    # Note that WeightedGraph, for example, inherits from Graph and WeightedDigraph
    # We can test using isinstance()
    if isinstance(self, Graph):
      # Create the needed attributes
      self._edges = []
      self._inciding_edges = {}
      # Sets a flag for use in this __init__ method
      is_initiating_graph = True
    else:
      is_initiating_graph = False
      # In this case we don't want to have self._edges nor self._inciding edges
      # Since this might be a cast from another class (using cast_as_another_class)
      #we prefer to delete, if already existent, those attributes
      if hasattr(self, '_edges'):
        del self._edges
      if hasattr(self, '_inciding_edges'):
        del self._inciding_edges
    if isinstance(self, WeightedDigraph): # Not used anywhere
      expect_weighted_digraph = True
    else:
      expect_weighted_digraph = False
    if isinstance(self, UnweightedDigraph): # Not used anywhere
      expect_unweighted_digraph = True
    else:
      expect_unweighted_digraph = False
    # We create the base attributes: _arrows, _arrows_out, _arrows_in
    self._arrows = []
    self._arrows_out = {}
    self._arrows_in = {}
    # We now do some unpacking
    # Note that even a Graph can be given by arrows, so we won't be fussy
    #in determining the type given. Rather, rely on is_initiating_graph
    #to know how to act regarding the edges
    if 'all_arrows' in data_type.lower():
      if data_type.lower() == 'all_arrows':
        init_vertices, init_arrows = [], data
        require_vertices_in = False
      elif data_type.lower() == 'some_vertices_and_all_arrows':
        init_vertices, init_arrows = data
        require_vertices_in = False
      elif data_type.lower() == 'all_vertices_and_all_arrows':
        init_vertices, init_arrows = data
        require_vertices_in = True
      else:
        raise ValueError('Option not recognized')
      # In this case we have init_vertices and init_arrows]
      # If we are in a Graph, we also need to add information on the edges
      # They can be read from the arrows, with extra work
      # This work is handled by _add_arrows with an extra option
      also_add_formed_edges = is_initiating_graph
      self._add_vertices(init_vertices, require_vertex_not_in = True,
          require_namedtuple = False, skip_checks = False)
      self._add_arrows(init_arrows, require_vertices_in = require_vertices_in,
          also_add_formed_edges = also_add_formed_edges, skip_checks = False)
    elif 'all_edges' in data_type.lower():
      if data_type.lower() == 'all_edges':
        init_vertices, init_edges = [], data
        require_vertices_in = False
      elif data_type.lower() == 'some_vertices_and_all_edges':
        init_vertices, init_edges = data
        require_vertices_in = False
      elif data_type.lower() == 'all_vertices_and_all_edges':
        init_vertices, init_edges = data
        require_vertices_in = True
      else:
        raise ValueError('Option not recognized')
      # In this case we have a graph being formed, but we don't enforce it being a Graph
      # That is, we may or may not add the edges, depending on flags
      add_as_edges = is_initiating_graph
      self._add_vertices(init_vertices, require_vertex_not_in = True,
          require_namedtuple = False, skip_checks = False)
      self._add_edges(init_edges, require_vertices_in = require_vertices_in,
          add_as_edges = add_as_edges, add_as_arrows = True, skip_checks = False)
    elif 'as_dict' in data_type.lower() or 'as_list' in data_type.lower():
      # We need to format the information
      # First, to save time coding, we format a list (in the 'as_list' option)
      # to have the same exact info as the corresponding dict would have
      if 'as_list' in data_type.lower():
        data_as_dict = {}
        for item in data:
          if item[0] in data_as_dict:
            # Repeated information, better to abort
            raise ValueError('Duplicated keys derived from data')
          else:
            # All clear, all first item as key and rest as value
            data_as_dict[item[0]] = item[1:]
      else:
        # Necessarily 'as_dict' in data_type.lower()
        data_as_dict = data
      # Now do things common to both, acting on data_as_dict
      # All of them are dictionaries whose keys are vertices
      vertices = list(data_as_dict)
      # For the operations, the vertices should be in, so we mark so
      require_vertices_in = True
      # We now produce the arrows or edges
      # (Note they will go under further formatting later when being added)
      if 'arrows_out' in data_type.lower():
        # We expect the values of the dict to be lists with the arrows
        init_arrows = get_namedtuples_from_neighbors(data_as_dict, namedtuple_choice = DigraphArrow)
        for vertex in data_as_dict:
          init_arrows.extend(data_as_dict[vertex])
      elif 'edges' in data_type.lower():
        init_edges = []
        for vertex in data_as_dict:
          init_edges.extend(data_as_dict[vertex])
      elif 'neighbors_out' in data_type.lower():
        # There is some complexity and so we defer to another method
        init_arrows = Digraph.get_namedtuples_from_neighbors(data_as_dict,
            output_as = 'list', namedtuple_choice = Digraph.Arrow)
      elif ('neighbors' in data_type_lower()) and (not 'out' in data_type.lower()):
        # There is some complexity and so we defer to another method
        init_edges = Digraph.get_namedtuples_from_neighbors(data_as_dict,
            output_as = 'list', namedtuple_choice = Digraph.Edge)
      else:
        raise ValueError('Option not recognized')
    else:
      raise ValueError('Option not recognized')
      # Now we have either: vertices and edges, or vertices and arrows
      # Either way, we will add those to the digraph
      # (Note that even if they aren't namedtuple Arrows, Edges and Vertex,
      #they will be when added to self._arrows, self._arrows_out and troupe)
      # That is, the info will be sanitized when added
      # The weights will also be sorted (that is, if None is given as weight or
      #if they are omitted altogether, the method still does the right thing)
      self._add_vertices(vertices)
      # Note that we have either only init_arrows or only init_edges available
      # Note that if we are given edges and Graph is not on __mro__
      #(Method Resolution Order) then it won't work because the methods
      #involving edges are under the subclass Graph
      try:
        # First we detect the case where init_arrows may not exist
        init_arrows = init_arrows
        # If they exist, we expect to not have defined init_edges
        try:
          init_edges = init_edges
          # In this case something went wrong in the implementation
          raise RuntimeError('Cannot have both init_arrows and init_edges')
        except NameError:
          # As we wanted! Just pass
          pass
        # Finally, we add all from init_arrows to the Digraph
        # We add them as edges as well if required
        also_add_formed_edges = is_initiating_graph
        self._add_arrows(init_arrows, require_vertices_in = require_vertices_in,
            also_add_formed_edges = also_add_formed_edges)
      except NameError:  
        # We must have init_edges instead
        try:
          init_edges = init_edges
        except NameError:
          # If init_edges doesn't exist something went wrong
          raise RuntimeError('Something went wrong')
          # Now we ensure init_arrows is not defined
          try:
            init_arrows = init_arrows
            # We want to not have it defined, so we raise an exception
            raise RuntimeError('Cannot have both init_arrows and init_edges')
          except NameError:
            # Exactly what we want, init_arrows not defined since init_edges is
            pass
          # Finally, we add the edges
          # We add them as both edges and arrows or only as arrows depending on the case
          add_as_edges = is_initiating_graph
          self._add_edges(init_edges, require_vertices_in = require_vertices_in,
              add_as_edges = add_as_edges, add_as_arrows = True, skip_checks = False)

########################################################################
# Methods for adding vertices, edges, arrows, used in initialization
########################################################################
    
  def _add_vertex(self, vertex, require_vertex_not_in = False,
      require_namedtuple = False):
    '''
    Adds a vertex as Vertex namedtuple to the digraph.
    
    INPUTS:
    self: own instance
    vertex: name or content of the vertex [must be hashable]
    require_vertex_not_in: gives error if vertex to be added is already in
    require_namedtuple: requires Vertex namedtuple as argument
    skip_checks: overrides possible checks
    
    OUTPUTS:
    (None: alters self)
    '''
    # We pass most of the formatting/checking to sanitive_vertex()
    # (That includes the detection of being a Vertex if require_namedtuple is True)
    vertex = Digraph.sanitize_vertex(vertex, require_namedtuple = require_namedtuple)
    # We determine whether the vertex is already in the graph
    if vertex in self:
      # In this case vertex is already present
      if not require_vertex_not_in:
        raise ValueError('Vertex already present')
      else:
        # Vertex present, but not a problem. Leave the method
        pass
    else:
      # In this case all clear, we can add the vertex to the relevant dicts
      self._arrows_in[vertex] = []
      self._arrows_out[vertex] = []
      # If we have Graph on top of Digraph, we need to deal with self._inciding_edges
      if isinstance(self, Graph):
        self._inciding_edges[vertex] = []

  def _add_vertices(self, vertices, require_vertex_not_in = False,
      require_namedtuple = False):
    '''
    Adds an iterable of vertices to self.
    '''
    for vertex in vertices:
      _add_vertex(self, vertex, require_vertex_not_in = require_vertex_not_in,
      require_namedtuple = require_namedtuple)

  def _add_arrow(self, arrow, require_vertices_in = False, require_namedtuple = False):
    '''
    Adds (weighted or unweighted) arrow to self.
    '''
    # We verify it is a valid arrow, putting it into the right format if it makes sense
    # We don't mind if we start with a simple tuple instead of the named tuple Arrow
    # We will put it into a namedtuple Arrow, that is, a sanitized arrow
    arrow = self.sanitize_arrow_or_edge(arrow,
        use_edges_instead_of_arrows = False, require_namedtuple = require_namedtuple)
    # We check whether the vertices are already present
    # If require_vertices_in, we raise an error if the vertices are not
    #already present. Otherwise, we add the vertices too.
    if arrow.source not in self:
      if require_vertices_in:
        raise ValueError('Source of arrow needs to be in digraph.')
      else:
        self._add_vertex(arrow.source)
    if arrow.target not in self:
      if require_vertices_in:
        raise ValueError('Target of arrow needs to be in digraph.')
      else:
        self._add_vertex(arrow.target)    
    # We now work on the arrow
    self._arrows_in[arrow.target].append(arrow)
    self._arrows_out[arrow.source].append(arrow)
    self._arrows.append(arrow)

  def _add_arrows(self, arrows, require_vertices_in = False,
      also_add_formed_edges = False, require_namedtuple = False):
    '''
    Adds an iterable of (weighted or unweighted) arrows to self.
    
    If arrows form edges, they can be requested to be added too.
    '''
    # If we require sanitize arrows, we do this always in this function, at once
    # We do slightly different depending on the edge formation requirement
    if also_add_formed_edges:
      arrows, edges = Digraph.sanitize_arrows_and_return_formed_edges()
    else:
      arrows = Digraph.sanitize_arrows_or_edges(arrows, use_edges_instead_of_arrows = False,
          require_nametuple = require_nametuple)
    # We add the arrows
    for arrow in arrows:
      self._add_arrow(arrow, require_vertices_in = require_vertices_in,
          require_namedtuple = True)
    # Now we add the edges, if requested
    if also_add_formed_edges:
      # Easiest way is to call _add_edges, and of course, ask to not add arrows
      # (Otherwise they would be added in double)
      self._add_edges(edges, require_vertices_in = False, add_as_edges = True,
          add_as_arrows = False, require_namedtuple = True)
    else:
      # Nothing else
      pass
    
  def _add_edge(self, edge, require_vertices_in = False, add_as_edge = False,
      add_as_arrows = False, require_namedtuple = False):
    '''
    Adds an edge to the (di)graph.
    
    Edge appears as as two arrows in self._arrows, among other attributes
    It also appears as one edge in self._edges (in Graph instance only)
    '''
    # We first put the edge into a namedtuple, if not already [sanitize it]
    edge = Digraph.sanitize_arrow_or_edge(edge, require_namedtuple = require_namedtuple)
    # We check whether the vertices are already present
    if edge.first not in self:
      if require_vertices_in:
        raise ValueError('Source of edge needs to be in (di)graph.')
      else:
        self._add_vertex(edge.first)
    if edge.second not in self:
      if require_vertices_in:
        raise ValueError('Target of edge needs to be in (di)graph.')
      else:
        self._add_vertex(edge.second)
    # We care about adding the edges (depend on instance class)
    # Note that if Digraph is Graph we need to deal with more attributes
    # We will trust our flag add_as_edge for the discrimination
    if add_as_edge:
      # That is, we need to add the edge information as an edge:
      self._edges.append(edge)
      self._inciding_edges[edge.first].append(edge)
      self._inciding_edges[edge.second].append(edge)
    # It may or may not be the case that we will add the arrows
    # That is, it depends on the request
    if add_as_arrows:
      # We now work on the arrows: every edge also makes two arrows.
      # We call Digraph._add_arrow, skipping all checks
      # We produce two namedtuples Arrow using get_arrows_from_edge
      two_arrows = self.get_arrows_from_edge(edge)
      for arrow in two_arrows:
        # All checks done already, don't need to put any requirement
        # (Note that edges have been sanitized already.)
        self._add_arrow(arrow, require_vertices_in = False, also_add_formed_edges = False,
            require_namedtuple = True)      
    else:
      # Nothing else
      pass

  def _add_edges(self, edges, require_vertices_in = False, add_as_edges = False,
      add_as_arrows = False, require_namedtuple = False):
    '''
    Adds an iterable of edges to the (di)graph.
    '''
    # Note that if Digraph is Graph this will call Graph method
    # Otherwise, if Digraph is not Graph, this calls Digraph method
    add_as_edge = add_as_edges
    for edge in edges:
      self._add_edge(edge, require_vertices_in = require_vertices_in,
          add_as_edge = add_as_edge, add_as_arrows = add_as_arrows,
          require_namedtuple = require_namedtuple)

########################################################################
# Methods which read simple information from the graph
########################################################################

  def __repr__(self):
    '''
    Returns representation of self.
    '''
    # We take the last part of the class name using split() string method
    # We do this for proper subclassing. Note that Graph instances
    #have their own __repr__ method which has priority over Digraph.__repr__
    class_last_name = self.__class__.__name__.split()[-1]
    about_instance = 'A {} with {} vertices and {} arrows.'.format(
        class_last_name, self.get_number_of_vertices(), self.get_number_of_arrows())
    return about_instance

  def provide_long_representation(self):
    '''
    All information about the graph in a string.
    '''
    raise NotImplementedError('Implement in the future.')
    pass

  def __contains__(self, vertex):
    '''
    Magic method. Returns whether a vertex belongs to the graph.
    '''
    # This is to be used throughout the code
    # So we do a direct attribute lookup and then a dict key lookup
    #(instead of calling get_vertices() which would take longer)
    # Use __contains__ for conditionals: vertex in self
    # For looping over vertices self.get_vertices() is unavoidable
    return vertex in self._arrows_out

  def __bool__(self):
    '''
    Magic method. Returns boolean value of digraph, which is False for the
    empty digraph (no vertices) are True otherwise.
    '''
    return bool(self._arrows_out)

  def is_nonempty_digraph(self):
    '''
    Alias for __bool__.
    
    SEE: __bool__
    '''
    return bool(self)

  def get_vertices(self):
    '''
    Returns all vertices of self.
    '''
    # The vertices are stored as the keys of the dict self._arrows_out
    return list(self._arrows_out)

  def get_number_of_vertices(self):
    '''
    Returns the number of vertices of self.
    '''
    return len(self._arrows_out)
    
  def get_number_of_arrows(self):
    '''
    Returns the number of arrows in self.
    '''
    return len(self._arrows)

  def get_in_degree_of_vertex(self, vertex, skip_checks = False):
    '''
    Returns in-degree of vertex in digraph: the number of arrows in.
    '''
    if not ignore_checks:
      assert vertex in self
    return len(self._arrows_in[vertex])
    
  def get_out_degree_of_vertex(self, vertex, skip_checks = False):
    '''
    Returns out-degree of vertex in digraph: the number of arrows out.
    '''
    if not ignore_checks:
      assert vertex in self
    return len(self._arrows_out[vertex])

  def get_neighbors_in(self, vertex, skip_checks = False):
    '''
    Provides the vertices to which a vertex has arrows from.
    
    In a multigraph, these neighbors are repeated in the output.
    '''
    if not skip_checks:
      assert vertex in self
    return [arrow_in.source for arrow_in in self._arrows_in[vertex]]
    
  def get_neighbors_out(self, vertex, skip_checks = False):
    '''
    Provides the vertices to which a vertex has arrows to.
    
    In a multigraph, these neighbors are repeated in the output.
    '''
    if not skip_checks:
      assert vertex in self
    return [arrow_out.target for arrow_out in self._arrows_out[vertex]]    
    
  def get_arrows(self, output_as = None):
    '''
    Returns all arrows of self.
    '''
    # We make list the default output
    # Originally self._arrows is stored as tuple
    if output_as == None:
      output_as = 'list'
    if output_as.lower() == 'list':
      return list(self._arrows)
    elif output_as.lower() == 'tuple':
      return tuple(self._arrows)
    else:
      raise ValueError('Output option not recognized')

  def get_arrows_out(self, vertex, skip_checks = False):
    '''
    Returns arrows going out of a vertex.
    '''
    if not skip_checks:
      assert vertex in self, 'Vertex must be in digraph'
    return self._arrows_out[vertex]

  def get_arrows_in(self, vertex, skip_checks = False):
    '''
    Returns arrows coming into a vertex.
    '''
    if not skip_checks:
      assert vertex in self, 'Vertex must be in digraph'
    return self._arrows_in[vertex]

########################################################################
# Methods for detecting best subclassing and casting
########################################################################

  @classmethod
  def suggest_properties_from_subclassing(cls):
    '''
    Returns information about the digraph based solely on its subclass.
    '''
    # First we check if it is an undirected graph (called Graph class)
    if issubclass(cls, Graph):
      is_undirected = True
    else:
      is_undirected = False
    # A Digraph with no arrows can theoretically be simultaneously weighted and unweighted
    # That is why is_weighted and is_unweighted are independent
    if issubclass(cls, WeightedDigraph):
      is_weighted = True
    else:
      is_weighted = False
    if issubclass(cls, UnweightedDigraph):
      is_unweighted = True
    else:
      is_unweighted = False
    return (is_undirected, is_weighted, is_unweighted)

  def is_undirected_weighted_unweighted(self):
    '''
    Reports about the digraph: is undirected, is weighted, is unweighted.
    
    Does not look as subclassing. A digraph with no arrows is considered
    undirected, weighted and unweighted.
    '''
    is_undirected = self.is_digraph_undirected()
    is_weighted = self.is_digraph_weighted() 
    is_unweighted = self.is_digraph_unweighted()
    return (is_undirected, is_weighted, is_unweighted)

  def is_digraph_undirected(self):
    '''
    Returns whether the digraph can be made undirected. [For this docstring,
    it is called digraph as long as not proven undirected.]
    
    For that to happen, each arrow (v, w) will correspond to an arrow (w, v).
    
    Does not look as subclassing. A digraph with no arrows is undirected.
    '''
    # Don't want to check the class name, but to verify it is possible
    #to build, with the given arrows, a Graph [an undirected graph]
    # We use the static method get_edges_from_sanitized_arrows, which returns
    #None when the arrows cannot be used to form edges
    new_edges = Digraph.get_edges_from_sanitized_arrows(data = self,
        are_arrows_from_digraph = True, is_multiarrow_free = False)
    # We are explicit on what we're doing
    is_undirected = (new_edges is not None)
    return is_undirected

  def is_digraph_weighted(self):
    '''
    Determines if all arrows of digraph are weighted.
    
    Does not look as subclassing. A digraph with no arrows is weighted.
    '''
    # We verify whether all arrows are weighted.
    # We use self.get_arrows() which reads self._arrows
    # (We could also verify the edges, or also observe self._arrows_out
    #and others, but we assume initialization was correct and therefore
    #self._arrows has all information)
    # Note math.inf (i. e. infinity) counts as float. Since sometimes it
    #is a convention that an infinite distance between two vertices means
    #there is no path between them, we forbid math_inf to be a valid weight
    test = lambda weight: (isinstance(weight, float) and weight != math_inf)
    return all(test(arrow.weight) for arrow in self.get_arrows())

  def is_digraph_unweighted(self):
    '''
    Determines if all arrows of digraph are weighted.
    
    Does not look as subclassing. A digraph with no arrows is unweighted.
    '''
    return all((arrow.weight is None) for arrow in self.get_arrows())

  @staticmethod
  def suggest_subclasses_from_properties(is_undirected, is_weighted, is_unweighted):
    '''
    Given some properties a digraph might have, suggest fitting subclasses.
    
    Returns a list of all possible subclasses it could be a subclass of.
    The best fitting subclass is always the first item.
    '''
    # We provide a few names to help (they disappear outside this method)
    D, WD, UD = Digraph, WeightedDigraph, UnweightedDigraph
    G, WG, UG = Graph, WeightedGraph, UnweightedGraph
    if is_undirected:
      if is_weighted and is_unweighted:
        # Possible for an empty graph
        return [G, UG, WG, D, UD, WD]
      elif is_weighted:
        return [WG, G, WD, D]
      elif is_unweighted:
        return [UG, G, UD, D]
      else:
        return [G, UG, WG, D, UD, WD]
    else:
      if is_weighted and is_unweighted:
        # Possible for an empty digraph
        return [D, UD, WD]
      elif is_weighted:
        return [WD, D]
      elif is_unweighted:
        return [UD, D]
      else:
        return [D, UD, WD]

  @staticmethod
  def read_subclass_from_string(name, require_string = False):
    '''
    Given a string, produces the Digraph subclass which best matches the name.
    
    There are 6 possible classes: Digraph, UnweightedDigraph, WeightedDigraph,
    Graph, UnweightedGraph, WeightedGraph.
    
    Has an option to require a string as name. If False, it accepts the
    class itself as name argument.
    '''
    # If we start with a type, if allowed, we convert it to a string first
    if isinstance(name, type):
      # We have the class already. If that was a valid argument, we return it
      if require_string:
        raise TypeError('Need a string to be converted into a class.')
      elif name not in [Digraph, UnweightedDigraph, WeightedDigraph,
          Graph, WeightedDigraph, UnweightedGraph]:
        raise TypeError('Need one of the 6 (di)graph classes.')
      else:
        name = name.__name__
    # We work on a string (possibly coming from being a type previously)
    if isinstance(name, str):
      if 'digraph' in new_class.lower():
        # For Digraph, UnweightedDigraph, WeightedDigraph
        if 'unweighted' in new_class.lower():
          new_class = UnweightedDigraph
        elif 'weighted' in new_class.lower():
          new_class = WeightedDigraph
        else:
          new_class = Digraph
      elif 'graph' in new_class.lower():
        # For Graph, UnweightedGraph, WeightedGraph
        if 'unweighted' in new_class.lower():
          new_class = UnweightedGraph
        elif 'weighted' in new_class.lower():
          new_class = WeightedGraph
        else:
          new_class = Graph
      else:
        raise ValueError('Can\'t recognize requested subclass.')
    else:
      # Only accept types/classes and strings
      raise TypeError('Can only read class from a string or a type.')

  def cast_as_another_class(self, new_class = None, modify_self = False, accept_data_loss = False):
    '''
    Given an instance of any subclass of Digraph, make it into an instance
    of another subclass of Digraph.
    
    Can either modify object in place (returning None), or create a new,
    equivalent object and return it [keeping original intact].
    
    Has an option to accept data loss, for example, removing weights
    from arrows when they don't all have weight 1. If data loss is not
    accepted but is requested from the other arguments, an error is raised.
    
    Some changes are impossible: for example, a strictly directed graph
    into an undirected graph (i. e. a Graph). In this case, raises error.
    '''
    # First we deal with None; if no class is suggested, we find the best fit.
    if new_class == None:
      properties = self.is_undirected_weighted_unweighted()
      new_class = Digraph.suggest_subclasses_from_properties(properties)[0]
    # We expect new_class to be given as string or as the class itself
    new_class = Digraph.read_subclass_from_string(name, require_string = False)
    # We have new_class well established. We compare with the current class
    # (Note their names are stored in old_class.__name__ and new_class.__name__)
    old_class = self.__class__
    # If they are the same, it's easy
    if old_class == new_class:
      if modify_self:
        # Nothing to change
        return None
      else:
        return copy_copy(self)
    else:
      # From now on it is assumed old and new classes are different
      # We write down properties of the current class and of the desired class
      # We abbreviate undirected, weighted, unweighted for our convenience
      new_und, new_wei, new_unw = Digraph.suggest_properties_from_subclassing(new_class)
      old_und, old_wei, old_unw = Digraph.suggest_properties_from_subclassing(old_class)
      # This is the moment where we look for contradictions
      # That is, impossible subclassing castings
      are_all_weights_one = all(arrow.weight == 1 for arrow in self.get_arrows())
      # We use a flag to tell if casting is possible or if errors should be raised
      is_casting_allowed = True
      if new_und and not old_und:
        is_casting_allowed = False
      elif old_wei and not new_wei and not accept_data_loss and not are_all_weights_one:
        # That is, we are going from weighted to unweighted, not all weights are 1,
        #and we are not accepting data loss
        is_casting_allowed = False
      if is_casting_allowed:
        # Casting is possible
        # We use __init__ to reinitialize the own instance
        # Note an instance has self._edges and self._inciding_edges
        #if and only if it is a Graph (at least that is how __init__ is designed)
        # In the case old_class is undirected, using this information
        #could potentially speed up the formation of another undirected graph
        # (And close to zero effect on digraphs, and the information on
        #edges of that graph won't be kept at all, and that's the design)
        if old_und:
          # Use vertices and edges
          data = (self.get_vertices(), self.get_edges())
          data_type = 'all_vertices_and_all_edges'
        else:
          # Use vertices and arrows
          data = (self.get_vertices(), self.get_arrows())
          data_type = 'all_vertices_and_all_arrows'
        # If we want to change from weighted to unweightd or vice-versa
        #we need to adapt the arrows/edges
        # Note if self has no arrows then old_wei and old_unw are True
        # The following lines take it into consideration (they will have
        #no effect as there will be no arrows or edges to change)
        if new_wei and old_unw:
          # Slightly differently depending on arrows or edges
          if old_und:
            # Note default weight is supposed to be 1
            weighted_edges_or_arrows = Digraph.write_weights_into_arrows_or_edges(
                unweighted_tuples = data[1],
                use_edges_instead_of_arrows = True,
                new_weights = [1 for item in range(len(data[1]))],
                require_namedtuple = True)
          else:
            weighted_edges_or_arrows = Digraph.write_weights_into_arrows_or_edges(
                unweighted_tuples = data[1],
                use_edges_instead_of_arrows = False,
                new_weights = [1 for item in range(len(data[1]))],
                require_namedtuple = True)
          data = (data[0], weighted_edges_or_arrows)
        elif new_unw and old_wei:
          # Possible if accept_data_loss is True, or if all have weight 1
          if old_und:
            unweighted_edges_or_arrows = Digraph.remove_weights_from_arrows_or_edges(
                weighted_tuples = data[1],
                use_edges_instead_of_arrows = True,
                require_namedtuple = True)
          else:
            unweighted_edges_or_arrows = Digraph.remove_weights_from_arrows_or_edges(
                weighted_tuples = data[1],
                use_edges_instead_of_arrows = False,
                require_namedtuple = True)
          data = (data[0], unweighted_edges_or_arrows)
        else:
          # Nothing to adapt on edges/arrows regarding weights
          pass
        # Finally, data and data_type are correct, and we can either
        #modify self or return a new instance with the requested features
        if modify_self:
          self.__init__(data = data, data_type = data_type, cast_as_class = new_class)
          return None
        else:
          # Create a new instance via __init__ and return it
          # (This can me accomplished in multiple ways)
          return new_class(data = data, data_type = data_type, cast_as_class = None)
      else:
        # Casting is not possible
        raise ValueError('Cannot make the requested subclass casting.')

########################################################################
# Uncategorized methods
########################################################################

  def make_multidigraph_simple(self, modify_self = False):
    '''
    From a multidigraph obtains a simple digraph [at most one arrow for each
    pair of vertices].
    
    When there are multiple arrows from the same pair, and the digraph is
    weighted, the shortest [lowest weight] arrow is picked.
    
    Can modify self, or return a new instance with the simple graph.
    '''
    # If we have a Graph, we can more or less cut the work in half
    #by looking at the edges instead of at the vertices
    # Since it is a bit simpler with arrows we do it first
    if not isinstance(self, Graph):
      # Procedure: we copy arrows from self.get_arrows() into shortest_by_pair
      #in corresponding pair of source and target, only one per pair
      # We keep the arrow with smaller weight if two arrows correspond to
      #same pair, or any if arrows are unweighted
      # To keep the work linear on the number of arrows (and not quadratic)
      #and avoid comparisons we use [implicitly, using dict] hashing
      vertices = self.get_vertices()
      pairs_of_vertices = [(u, v) for u in vertices for v in vertices]
      shortest_by_pair = {pair:None for pair in pairs_of_vertices}
      for arrow in self.get_arrows():
        pair_of_arrow = (arrow.source, arrow.target)
        current_shortest = shortest_by_pair[pair_of_arrow]
        if current_shortest is None:
          # First arrow with same source and target, set a flag to add it
          should_write_into_shortest = True
        elif current_shortest.weight is None:
          # This is the case where the digraph is unweighted. Keep first
          should_write_into_shortest = False
        elif arrow.weight < current_shortest.weight:
          # Only possible for weighted
          should_write_into_shortest = True
        else:
          should_write_into_shortest = False
        # If marked as shortest, put it as the value for corresponding pair
        if should_write_into_shortest:
          shortest_by_pair[pair_of_arrow] = arrow
        else:
          pass
      # We now prepare the data for using __init__
      data = (vertices, list(shortest_by_pair.values()))
      data_type = 'all_vertices_and_all_arrows'
    else:
      # We operate very similarly, except that we observe the edges
      # We need to be careful with target and source (or first and second)
      # This is easily done using set
      vertices = self.get_vertices()
      pairs_of_vertices = [set([u, v]) for u in vertices for v in vertices]
      shortest_by_pair = {pair:None for pair in pairs_of_vertices}
      for edge in self.get_edges():
        pair_of_edge = set([edge.first, edge.second])
        current_shortest = shortest_by_pair[pair_of_edge]
        if current_shortest is None:
          should_write_into_shortest = True
        elif current_shortest.weight is None:
          should_write_into_shortest = False
        elif edge.weight < current_shortest.weight:
          should_write_into_shortest = True
        else:
          should_write_into_shortest = False
        if is_this_shortest:
          shortest_by_pair[pair_of_edge] = edge
        else:
          pass
      # We prepare the data for __init__. We use vertices and edges
      data = (vertices, list(shortest_by_pair.values()))
      data_type = 'all_vertices_and_all_edges'
    # Now we modify self or create new instance
    # At this moment, data and data_type are correctly defined
    # Note that we don't change the subclassing (that's __init__ third argument)
    if modify_self:
      self.__init__(data = data, data_type = data_type, cast_as_class = None)
      return None
    else:
      return self.__class__(data = data, data_type = data_type, cast_as_class = None)

  def get_shortest_arrow_between_vertices(self, source, target, skip_checks = False):
    '''
    Gets the shortest arrow between two vertices.
    
    Returns None if there is no direct arrow between the vertices.
    
    If the digraph is unweighted (and there are arrows between them),
    any of those arrows might be returned.
    '''
    if not skip_checks:
      assert source in self, 'Need source to be a vertex'
      assert target in self, 'Need target to be a vertex'
    # We try to make this the most efficient possible
    # Easiest way is by looking at self._arrows_out
    shortest_arrow = None
    for arrow in self.get_arrows_out(source):
      # We only consider the arrows with right source and target
      if arrow.target == target:
        if shortest_arrow is None:
          # Make it the shortest
          should_write_into_shortest = True
        elif shortest_arrow.weight is None:
          # This should not be possible
          # After all, we already tested the current arrow for having
          #None as weight when it was made shortest_arrow
          raise RuntimeError('Internal logic error.')
        elif arrow.weight < shortest_arrow.weight:
          # In this case replace the current arrow
          should_write_into_shortest = True
        else:
          should_write_into_shortest = False
        if should_write_into_shortest:
          # Write into shortest
          shortest_arrow = arrow
          if shortest_arrow.weight is None:
            # If the weight is None, we have an unweighted digraph
            # In this case, we cannot improve the weight, so we break the loop
            break
          else:
            pass
        else:
          # Continue the loop, looking for good arrows
          pass
    # End of loop. Return value in shortest_arrow (either Arrow or None)
    return shortest_arrow

  def is_digraph_simple(self):
    '''
    Returns whether the digraph is simple.
    
    A digraph is simple if for any source and target vertices there are
    0 or 1 arrows between them [in the given direction].
    
    (A digraph which is not simple is called multidigraph.)
    '''
    # Easiest way: observe repetitions on arrows leaving each possible source vertex
    # Use set() and length on self.get_neighbors_out()
    # Let's call this event multiarrows (same source and target, multiple arrows)
    have_multiarrows_been_found = False
    for source in self.get_vertices():
      neighbors_out = self.get_neighbors_out(source, skip_checks = True)
      if len(neighbors_out) != len(set(neighbors_out)):
        # In this case there are multiarrows
        have_multiarrows_been_found = True
        # One is enough
        break
    return (not have_multiarrows_been_found)   

  def is_digraph_multidigraph(self):
    '''
    Returns whether the digraph is a multidigraph, that is, non simple.
    '''
    return (not self.is_digraph_simple())
    
  def get_reversed_graph(self, modify_self = False):
    '''
    Reverses (In the sense of switching direction) all arrows of the digraph.
    
    Can modify self [and return None] or return a new instance.
    '''
    if isinstance(self, Graph):
      # If undirected (i. e. Graph) it doesn't alter itself
      if modify_self:
        # Do nothing
        return None
      else:
        return copy_copy(self)
    else:
      # Easiest and cleanest way is through __init__
      all_reversed_arrows = Digraph.get_reversed_arrows(self.get_arrows(),
          require_namedtuple = True)
      data = (self.get_vertices(), all_reversed_arrows)
      data_type = 'all_vertices_and_all_arrows'
      if modify_self:
        # Use __init__ to reset and re-initialize
        self.__init__(data = data, data_type = data_type, cast_as_class = None)
        return None
      else:
        # Return new instance
        return self.__class__(data = data, data_type = data_type, cast_as_class = None)

########################################################################
# Methods implementing different algorithms in Graph Theory
########################################################################

  def get_sccs(self):
    '''
    Returns the strongly connected components (SCCs) of the digraph.
    
    INPUT:
    self
    
    OUTPUT:
    sccs: a list with lists, each representing a unique connected component
    sccs_lengths: the amount of vertices in each of the output SCCs
    '''
    # We will remove the weights of self since they are not used
    self.make_graph_unweighted()
    # We control everything using a StateDigraphGetSCC instance
    state = StateDigraphGetSCC(self)
    # We also need the inverse/reversed graph
    inverted_graph = self.get_reversed_graph()
    # We do DFS-Loop using the inverted graph to get a new rank
    state.manually_change_graph(inverted_graph)
    new_rank, middle_leaders = state.dfs_outer_loop()
    # We do DFS-Loop using the original graph, self, and the new_rank
    # Note that the leaders don't matter in the first pass
    # (Neither does the final_rank on the second pass.)
    state.manually_change_vertices_ranked(new_rank)
    state.manually_change_graph(self)
    final_rank, final_leaders = state.dfs_outer_loop()
    # Now the leaders give the SCCs. We will store the results in a dict
    # The leader is the key, the value a list of those having it as a leader
    # (That includes itself)
    # Note many will be keys with empty lists as values, which is okay
    state.manually_change_vertices_ranked(final_rank) # Optional
    dict_sccs = {vertex: [] for vertex in state._vertices_ranked}
    for vertex in state._vertices_ranked:
      dict_sccs[final_leaders[vertex]].append(vertex)
    # We don't return the leaders, only the SCCs and the lengths.
    # Note that logically we should only output nonempty lists following leaders
    sccs = [scc for scc in dict_sccs.values() if scc]
    sccs_lengths = [len(scc) for scc in sccs]
    return (sccs, sccs_lengths)

  # Should be run (n**2)*log(n) times with different seeds
  def find_almost_certainly_minimal_cut(self, tries = None):
    '''
    Returns the best cut among many tries, hopefully a minimal cut.
    '''
    # Problem only makes sense with at least two vertices
    n = self.get_number_of_vertices()
    if n <= 1:
      raise ValueError('graph needs at least 2 vertices')
    if tries is None:
      # If no parameter is given, we default to (n**2)*log(n)
      tries = round((n**2)*math_log2(n)) # round() produces an int
    minimal_cut = None
    for try_idx in range(tries):
      # Note find_cut(array) is (crossing_edges, one_side, other_side)
      cut_from_try = self.find_cut()
      if minimal_cut is None or cut_from_try[0] < minimal_cut:
        minimal_cut = cut_from_try[0]
        print('Current try: {}. New minimum reached: {}'.format(try_idx, minimal_cut))
    return minimal_cut

  def k_clustering(self, k):
    '''
    Finds the optimal k-clustering (k >= 1) of the graph.
    
    Requires a complete, weighted graph.
    
    Note: for k = 1 it produces a minimum spanning tree vias Kruskal's algorithm.
    '''
    # We need a weighted, undirected, simple [i.e. non-multigraph] graph
    # Being undirected, we have access to self._edges
    assert self.is_digraph_undirected(), 'Need undirected graph'
    assert self.is_digraph_weighted(), 'Need weighted graph'
    assert self.is_digraph_simple(), 'Need simple graph (cannot be multigraph)'
    # (We don't really require complete. When the graph is not complete,
    #this reduces to the Kruskal algorithm, essentially)
    n = self.get_number_of_vertices()
    assert n >= k, 'Need at least k starting vertices to form k clusters'
    # First we start up the clusters using a union-find structure
    # By cluster we mean: each vertex will have a leader, and vertices
    #of same leader belong to the same cluster
    # But we do lazy union, so we have a parent relation, and we need to
    #transverse it up to finder the leader
    # Also, we do path compression, so we update one's parents to be
    #one's leaders when given the opportunity
    parents = {vertex:vertex for vertex in self.get_vertices()}
    ranks = {vertex:0 for vertex in self.get_vertices()}
    # We put all edges in a heap. We order them by weight, reordering it
    edges_heap = [(edge.weight, edge.first, edge.second) for edge in self._edges]
    heapq_heapify(edges_heap)
    # We need to do n-k union-operations
    # But we later do one special operation, which is part of the main loop
    # So we really do n-k+1, interrupting one
    for idx in range(n-k+1):
      # Locate the smallest edge which is a bridge between two clusters
      while True:
        # If graph is not complete there might be an error in the following
        # Nonetheless, we don't want the try/except overhead for exceptions
        new_edge = heapq_heappop(edges_heap)
        # Call the vertices u and v. Recall the order of the information
        weight, u, v = new_edge
        # Get the leaders of u and v. This is a find-operation
        # (Do path-compression while at it)
        local_leaders = {item:None for item in [u, v]}
        for vertex in [u, v]:
          # We save the path to do path compression
          # Idea is to keep appending the parents until leader is found
          accumulated_path = [vertex]
          # We loop whiel the leader of the root is not found
          while accumulated_path[-1] != parents[accumulated_path[-1]]:
            # We don't have a leader yet, so we append the parent to the path
            accumulated_path.append(parents[accumulated_path[-1]])
          # Ok, now we have a full path to the leader in accumulated_path
          # First we do path-compression
          for item in accumulated_path:
            parents[item] = accumulated_path[-1]       
          # We save the result as local_leaders dict, and break
          local_leaders[vertex] = parents[vertex]
        # Ok, now we have local_leaders[u] and local_leaders[v]
        # If they are in the same cluster, we discard the edge and try again
        # Otherwise we continue with the process
        if local_leaders[u] != local_leaders[v]:
          break
      # We found a good sparating edge.
      # Now we do the union-operation, except in the last operation
      #in which we compute a minimal separation between the clusters
      # And don't proceed, otherwise we would over-cluster the vertices
      if idx == n-k:
        minimal_distance_clusters = weight
      else:
        # ok, we are still in the process of clustering. So we do an union
        # We compare the ranks of the leaders.
        if ranks[local_leaders[u]] == ranks[local_leaders[v]]:
          # If equal, we add one tree to the other in O(1) operations
          # Without loss of generality, let's say local_leaders[u] will lead
          parents[local_leaders[v]] == local_leaders[u]
          # We also adjust the rank of local_leaders[u]
          ranks[local_leaders[u]] += 1
        elif ranks[local_leaders[u]] < ranks[local_leaders[v]]:
          # If different, the smaller/shallower tree is appended to the larger
          parents[local_leaders[u]] = local_leaders[v]
        else:
          parents[local_leaders[v]] = local_leaders[u]
    # Ok. Not the loop has finalized and we have k clusters (given by parents)
    #as well as a last execution which givs the minimal distance
    # We want to output parents (which indirectly give the clusters)
    # But we also output the objective distance, the minimal possible distance
    #between two points in different clusters
    return (parents, minimal_distance_clusters)

########################################################################
# Class WeightedDigraph
########################################################################

class WeightedDigraph(Digraph):
  '''
  A digraph whose arrows are all weighted.
  '''

  def are_weights_nonnegative(self, require_weights_positive = False):
    '''
    For a weighted digraph, returns whether all weights are negative.
    
    Has an option to require weights to be positive and not only nonnegative.
    '''
    # Read straight from arrows
    if require_weights_positive:
      answer = all(arrow.weight > 0 for arrow in self.get_arrows())
    else:
      answer = all(arrow.weight >= 0 for arrow in self.get_arrows())
    return answer

  def make_digraph_unweighted(self, modify_self = False):
    '''
    Creates an unweighted digraph (by removing weights from all arrows).
    
    Can either modify self or return a new instance.
    '''
    # We first determine if we are in an instance of Graph
    # If we are, not only we want our future class to be also a Graph,
    #but we can also greatly shorten __init__ by using edges
    if isinstance(self, Graph):
      selected_class = UnweightedGraph
      data_type = 'all_vertices_and_all_edges'
      use_edges_instead_of_arrows = True
      original_working_data = self.get_edges()
    else:
      selected_class = UnweightedDigraph
      data_type = 'all_vertices_and_all_arrows'
      use_edges_instead_of_arrows = False
      original_working_data = self.get_arrows()
    # We then create the unweighted arrows/edges
    list_new_tuplees = Digraph.remove_weight_from_arrows_or_edges(
        tuplees = original_working_data,
        use_edges_instead_of_arrows = use_edges_instead_of_arrows,
        require_namedtuple = True, output_as_generator = False)
    # We need to keep track of the vertices too, lest they are isolated
    list_new_vertices = self.get_vertices()
    # We prepare for the return. This works the same way whether we had
    #arrows or we had edges
    data = (list_new_vertices, list_new_tuplees)
    # We create a new instance or modify the current
    if modify_self:
      self.__init__(data = data, data_type = data_type,
          cast_as_class = selected_class)
      return None
    else:
      return selected_class(data = data, data_type = data_type,
          cast_as_class = selected_class)

  def get_single_source_shortest_path_via_Dijkstras(self, base_vertex):
    '''
    In a weighted digraph, produces the distances from a fixed source vertex
    to all others transversing the available arrows.
    
    Uses Djikstras's Algorithm to compute the distances.
    
    Weights must be nonnegative for the algorithm to work.
    '''
    # We need all weights to be nonnegative.
    assert self.are_weights_nonnegative(), 'Need weights to be nonnegative'
    # We also need base_vertex to be one of the vertices from the self
    assert base_vertex in self, 'Base vertex must be in the graph'
    # We use the algorithm from class.
    # We implement set_X and set_Y (set_X with vertices whose shortest paths have been computed)
    # set_Y starts as all vertives (no shortest paths computed), and set_X as empty
    # Each time one shortest path is computed, we remove from set_Y and add to set_X
    set_Y = self.get_vertices()
    set_X = []
    # Shortest paths are available as a dictionary (None if not yet computed)
    # Note that these are shortest paths to base_vertex, and thus will not be attributes
    distances = {vertex: None for vertex in self.get_vertices()}
    # Okay, we know shortest path to is base_vertex is 0
    distances[base_vertex] = 0
    set_X.append(base_vertex)
    set_Y.remove(base_vertex)
    # Now it's time to implement algorithm
    # Each "loop of the while" correspond to one vertex being added
    # If they don't connect (there is no path from base vertex to the vertex in question),
    #the distance of that vertex [to the base vertex] will be set to math_inf
    # We could use of the following upper bound for distance: sum of all weights, plus one
    # But we prefer to use inf from built-in math library because it makes it
    #simpler to tell if a vertex cannot be reached from the base_vertex
    upper_bound_distance = math_inf
    # set_Y should contain, right now all vertices except base_vertex
    while set_Y:
      # current_min_distance controls the minimum distance from X to Y this round
      #(This is updated every loop because X and Y keep changing)
      current_min_distance = upper_bound_distance
      # We also control the point in set X and in set Y to realize current_min_distance
      current_min_set_X = None
      current_min_set_Y = None
      # We scan all arrows from X to Y to see if there is a smaller distance
      for vertex in set_Y:
        # We check the arrows. Since we are indexing by set_Y use get_arrows_in
        # (That is, the vertex in set_Y is the target of the relevant arrows)
        for arrow in self.get_arrows_in(vertex):
          # The arrow only matters if the source is in set_X
          if arrow.source in set_X:
            if distances[arrow.source] + arrow.weight < current_min_distance:
              current_min_distance = distances[arrow.source] + arrow.weight
              current_min_set_X = arrow.source
              current_min_set_Y = vertex
      # Now we check if we found anything less than upper_bound_distance
      # In other words, this tests connection between set_X and set_Y
      if current_min_distance < upper_bound_distance:
        # All we need to do is update the information on the current_min_set_Y
        distances[current_min_set_Y] = current_min_distance
        set_Y.remove(current_min_set_Y)
        set_X.append(current_min_set_Y)        
      else:
        # In this case set_Y and set_X are disjoint
        # So we put distance of set_Y to be math_inf, and put them all in set_X
        # (After all, their distances to the base vertex have been calculated!)
        # This will end the while loop
        for vertex in list(set_Y):
          distances[vertex] = upper_bound_distance
          set_Y.remove(vertex)
          set_X.append(vertex)
    # Here the while loop has ended. All vertices have associated distances
    # So we can output the result, the dict distances
    return distances

  #################
  # WORK HERE
  # Use Vertex.Path to streamline this
  #################
  def get_single_source_shortest_paths_via_Bellman_Fords(self, source_vertex,
      output_as = 'lengths_and_vertices_and_arrows'):
    '''
    Computes the length of shortest paths from one single source vertex
    to all other vertices of a directed weighted graph.
    
    Accomodates negative arrows. [By negative arrows, we mean an arrow
    with negative length/weight.] If there are negative cycles [cycles
    whose total length is negative], there are no shortest paths;
    in this case, the algorithm detects and reports a negative cycle.
    
    If n is the number of vertices and m the number of edges, its time
    complexity is O(m*n).
    
    It is slower than Dijkstra's algorithm (which is O(n*log(n))), but it
    has the advantage of handling arrows of negative weight.
    
    INPUT:
    self: the graph
    source_vertex: compute distances starting from this vertex
    output_as: one of the output possibilities for VertexPath.reformat_path()
    
    OUTPUT:
    is_negative_cycle_free: whether the graph is free of negative-weight cycles
    data: if there are no negative cycles, this is a dictionary which gives,
    for each vertex in the path, the shortest path from source to it, given
    in the requested format. If there are negative cyles, this is a negative
    cycle given by its length, its vertices or arrows.
    '''
    # Check if the vertex is indeed in the graph
    assert source_vertex in self, 'Source vertex must be in the graph'
    # At each moment, each vertex will have a number, A, and a pointer, B
    # The dict A carries information on the shortest length from the source
    # The dict B carries info on the last arrow for this shortest path
    n = self.get_number_of_vertices()
    A = {vertex:math_inf for vertex in self.get_vertices()}
    B = {vertex:None for vertex in self.get_vertices()}
    # We set the information on the source vertex
    A[source_vertex] = 0
    # Both will be updated during the algorithm, multiple times
    # If there is an answer, it is certainly reached after n-1 loops
    # But a n-th loop is needed to check for negative cycles
    might_there_be_negative_cycles = True
    for unused_index in range(n):
      # In each iteration, we reset the loop variable
      vertices_updated_this_loop = []
      # We process all arrows
      for arrow in self.get_arrows():
        # Arrows are given as tuples (source, target, weight)
        # We see if the navigating through the arrow [as last arrow]
        #provides a better result than the current one for the target vertex
        if A[arrow.source] + arrow.weight < A[arrow.target]:
          A[arrow.target] = A[arrow.source] + arrow.weight
          B[arrow.target] = arrow
          vertices_updated_this_loop.append(arrow.target)
      # We have processed all arrows. Stop algorithm if A doesn't change
      # In this case we know there are no negative cycles
      if not vertices_updated_this_loop: # I. e. empty
        might_there_be_negative_cycles = False
        break
    # After the n loops we have the answer to the existence of negative cycles
    is_negative_cycle_free = not might_there_be_negative_cycles
    if is_negative_cycle_free:
      # For each vertex, we produce a pathys from source_vertex to vertex
      # A path will be a list [source, ..., vertex]. We record it using information
      #on length (available from A), vertices and arrows (these derived from B)
      shortest_paths_lengths = dict(A) # A copy to be safe
      if ('vertices' in request.lower()) or ('arrows' in request.lower()):
        # We will need this information as long as the request includes either
        #'arrows' or 'vertices' (and both cost about the same to calculate so
        #we do them together)
        shortest_paths_as_vertices = {}
        shortest_paths_as_arrows = {}
        for vertex in self.get_vertices():
          if A[vertex] is math_inf:
            # Note: if graph is not connected some vertices will have math_inf as distance
            # That is not a problem at all, just that there are no paths
            # We put simply shortest_paths[vertex] = None
            shortest_paths_as_vertices[vertex] = None
            shortest_paths_as_arrows[vertex] = None
          else:
            # In this case we give the sequence of edges
            # (For the length of the minimum path, add up their weights)
            # We only record each arrow in each vertex...which is enough
            shortest_paths_as_vertices[vertex] = [vertex] # Start backward
            shortest_paths_as_arrows[vertex] = []
            # The following will follow the shortest path backward
            local_moving_vertex = vertex
            while local_moving_vertex != source_vertex:
              new_arrow_to_add = B[local_moving_vertex]
              new_vertex_to_add = new_arrow_to_add.source # We go backward
              shortest_paths_as_vertices[vertex] = [new_vertex_to_add] + shortest_paths_as_vertices[vertex]
              shortest_paths_as_arrows[vertex] = [new_arrow_to_add] + shortest_paths_as_arrows[vertex]
              local_moving_vertex = new_vertex_to_add # Closer to source
      # Return with variable name requested_data, as specified in request
      # We are concluding case is_negative_cycle_free == True
      # Form tuples when more than one, and return None if request == 'nothing'
      if request.lower() == 'lengths_and_vertices_and_arrows':
        requested_data = (shortest_paths_lengths, shortest_paths_as_vertices, shortest_paths_as_arrows)
      elif request.lower() == 'lengths_and_vertices':
        requested_data = (shortest_paths_lengths, shortest_paths_as_vertices)
      elif request.lower() == 'lengths_and_arrows':
        requested_data = (shortest_paths_lengths, shortest_paths_as_arrows)
      elif request.lower() == 'vertices_and_arrows':
        requested_data = (shortest_paths_as_vertices, shortest_paths_as_arrows)
      elif request.lower() == 'lengths':
        requested_data = shortest_paths_lengths
      elif request.lower() == 'vertices':
        requested_data = shortest_paths_as_vertices
      elif request.lower() == 'arrows':
        requested_data = shortest_paths_as_arrows
      elif request.lower() == 'nothing':
        requested_data = None
      else:
        raise ValueError('request option not recognized')
    else:
      # In this case we return a negative cycle (as a list of its edges)
      # We know that if any vertex updated in the last loop is part of a
      #negative cycle. So we can take vertices_updated_this_loop[0], p. ex.,
      #and then write a full cycle with it
      # (Note: we need to do this iteration outside the loop for the occasion
      #of an arrow of a vertex into itself)
      # (This is only needed once so we do even if not requested)
      negative_cycle_starting_vertex = vertices_updated_this_loop[0]
      negative_cycle_example_length = A[negative_cycle_starting_vertex]
      negative_cycle_example_as_vertices = [negative_cycle_starting_vertex] # Vertices here
      negative_cycle_example_as_arrows = [] # Put arrows here
      first_arrow_to_append = B[negative_cycle_starting_vertex]
      first_vertex_to_append = first_arrow_to_append.source # Go backwards
      negative_cycle_example_as_vertices = [first_vertex_to_append] + negative_cycle_example_as_vertices
      negative_cycle_example_as_arrows = [first_arrow_to_append] + negative_cycle_example_as_arrows
      # We use next_vertice_in_cycle to keep track of the closure of the cycle
      # Note we say next when we really mean previous... just a matter of names
      next_vertex_in_cycle = first_vertex_to_append # Go backward
      #while next_vertice_in_cycle != negative_cycle_starting_vertex:
      while next_vertex_in_cycle not in negative_cycle_example_as_vertices:
        arrow_to_append = B[next_vertice_in_cycle]
        vertex_to_append = arrow_to_append.source # Going backward
        negative_cycle_example_as_vertices = [vertex_to_append] + negative_cycle_example_as_vertices
        negative_cycle_example_as_arrows = [arrow_to_append] + negative_cycle_example_as_arrows # Go backwards
        next_vertex_in_cycle = vertex_to_append # Go backwards
        ########################
        # WORK HERE
        # Infinite recursion here... maybe it fell into a sub-cycle?
        # In this case we should verify the next vertex isn't already in the list
        ########################
      # When done, prepare data. We are in case is_negative_cycle_free == False
      if request.lower() == 'lengths_and_vertices_and_arrows':
        requested_data = (negative_cycle_example_length, negative_cycle_example_as_vertices,
            negative_cycle_example_as_arrows)
      elif request.lower() == 'lengths_and_vertices':
        requested_data = (negative_cycle_example_length, negative_cycle_example_as_vertices)
      elif request.lower() == 'lengths_and_arrows':
        requested_data = (negative_cycle_example_length, negative_cycle_example_as_arrows)
      elif request.lower() == 'vertices_and_arrows':
        requested_data = (shortest_paths_as_vertices, negative_cycle_example_as_arrows)
      elif request.lower() == 'lengths':
        requested_data = negative_cycle_example_length
      elif request.lower() == 'vertices':
        requested_data = negative_cycle_example_as_vertices
      elif request.lower() == 'arrows':
        requested_data = negative_cycle_example_as_arrows
      elif request.lower() == 'nothing':
        requested_data = None
      else:
        raise ValueError('request option not recognized')
    # Now we have formed requested_data for any value of is_negative_cycle_free
    return (is_negative_cycle_free, requested_data)

  def get_all_paths_shortest_paths_via_Floyd_Warshals(self, request = 'lengths'):
    '''
    Computes the shortest paths for every pair of vertices in a directed,
    weighted graph. (Arrow weights might be negative.)
    
    When there is no path, infinity is returned as minimal distance.
    
    If there are negative cycles, there are pairs with no shortest paths,
    but the algorithm detects the situation and outputs a negative cycle.
    
    If n is the number of vertices, the time complexity of the algorithm is O(n^3).
    
    INPUT:
    self: the graph
    request: 'lengths', 'lengths_and_vertices' or 'lengths_and_arrows'
    
    OUTPUT:
    is_negative_cycle_free: whether the graph has a negative cycle
    requested_data:
    '''
    # We define a function to build the optimal path between two vertices
    # This sub-function will be useful towards the end of this method
    # (By default we don't provide the edge as it's costly to search for them)
    def get_shortest_path_using_A_and_B(first, second, request = 'arrows'):
      '''
      Returns the shortest path from a vertex to another, either as a list
      of vertices or as a list of arrows (considerably slower).
      
      Requires A and B (of the method where function is inserted) computed.
      
      INPUT:
      first: the source vertex of the shortest path we want to find
      second: the target vertex of the shortest path we want to find
      request: 'vertices', 'arrows' or 'vertices_and_arrows'
      
      OUTPUT:
      data: depending on request, the shortest path between the vertices
      given given by vertices, given by arrows, or a tuple with it given by
      vertices and arrows
      '''
      assert 'A' in vars(), 'Need variable A available'
      assert 'B' in vars(), 'Need variable B available'
      # First we check if the vertices are not connected at all
      if A[((first, second))] == math_inf:
        # In this case no path, which we return as None (in all request options)
        if request.lower() == 'vertices':
          return None
        elif request.lower() == 'arrows':
          return None
        elif request.lower() == 'vertices_and_arrows':
          return (None, None)
        else:
          raise ValueError('Invalid choice for request')
      elif first == second and B[(first, second)] == None:
        # Now we check if a path is the trivial path (no arrows)
        # Not really a path, but we return a list with the vertex
        #for request == 'vertices', and otherwise an empty list of arrows
        if request.lower() == 'vertices':
          return [first]
        elif request.lower() == 'arrows':
          return []
        elif request.lower() == 'vertices_and_arrows':
          return ([first], [])
        else:
          raise ValueError('Invalid choice for request')
      else:
        # In this case the shortest path is a single or multiple arrows
        # (We can handle all of it in this case)
        current_path = [first, second]
        # Now, at any time in the current_path, if there is no
        #direct edge realizing the shortest distance between two consecutive
        #vertices [possibility detected by B]
        # Perhaps the simples way is using a "while" loop
        # (Note: this case generalizes the previous, but this is not a problem)
        # We use the following variable to know when to exit:
        # When all gaps are 0, there are arrows between any consecutive vertices
        #which can then be found by another method
        inserted_new_vertices_for_examination = True
        # This is also used to control whether there would be repetitions
        # (This can happen for negative cycles... because repeating them
        #always yields paths of shorter and shorted length)
        # We would prefer to have the simples negative cycle in this case,
        #i.e, without repetition of internal vertices
        # Thus, we use inserted_new_vertices_for_examination
        while inserted_new_vertices_for_examination:
          # Unless new vertices are inserted, we turn the variable to False
          inserted_new_vertices_for_examination = False
          # We compute B for every consecutive pair in the list
          current_gaps = []
          for idx in range(len(current_path) - 1):
            current_gaps.append(B[current_path[idx], current_path[idx+1]])
          # If all gaps are 0 then we have all direct edges
          # This will be evaluated during the following loop
          for idx, vertex in enumerate(current_path[:-1]):
            if current_gaps[idx] != 0:
              # Take the lowest occurrence, and replace by the appropriate vertex
              # Add the intermediate vertex to current_path
              # Recall we subtract 1 because with loop_index we consider
              #the vertex whose number is loop_index - 1
              number_to_add = B[current_path[idx], current_path[idx+1]] - 1
              vertex_to_add = number_to_vertex[number_to_add]
              # Due to the fact that we might get an infinite loop for a negative cycle
              #we will forbid vertex_to_add to being already in the path
              # (The source and target might still be the same point.)
              if not vertex_to_add in current_path:
                current_path = current_path[:idx+1] + [vertex_to_add] + current_path[idx+1:]
                inserted_new_vertices_for_examination = True
                # We break of only this for-loop (since we need to redo the list and everything)
                # (Otherwise it would be changing the list while it is being scanned, which is bad)
                # We will, of course, start again
                break
        # Okay. So we have current_path correct; it cannot be improved
        # Let's output as requested: list of vertices or of arrows
        if request.lower() == 'vertices':
          return current_path
        elif 'arrows' in request.lower():
          # For this case we must produce the shortest arrow between
          #eadch consecutive pair of vertices in current_path
          list_of_shortest_arrows = []
          for idx, vertex in enumerate(current_path[:-1]):
            new_arrow = self.get_shortest_arrow_between_vertices(current_path[idx], current_path[idx+1])
            list_of_shortest_arrows.append(new_arrow)
          if request.lower() == 'arrows':
            return list_of_shortest_arrows
          elif request.lower() == 'vertices_and_arrows':
            return (current_path, list_of_shortest_arrows)
          else:
            raise ValueError('Invalid choice for request')
        else:
          raise ValueError('Invalid choice for request')
    # End of inner function
    # Now we start the method code proper
    # We confirm the graph is weighted
    # (An undirected graph is a specific case of a directed graph)
    assert self.is_weighted_graph(), 'Graph must be weighted'
    # To do less searching on the edges, we prefer a simple graph
    # If there are multiple arrows between vertices, we take the shortest ones
    # (In the future, we might implement this removal of redundant edges)
    # For this we need to have a specific numbering of the vertices
    # The best way is a duple correspondence with two dicts
    n = self.get_number_of_vertices()
    vertex_to_number = {}
    number_to_vertex = {}
    # We don't really need the full apparatus but it's a good-to-have
    for number, vertex in enumerate(self.get_vertices):
      number_to_vertex[number] = vertex
      vertex_to_number[vertex] = number
    # We will loop n+1 times. In each time (loop_index from 0 through n)
    #we consider only paths whose interior vertices are among the loop_index first ones
    # We continuously keep track, for each pair of vertices, of dicts A and B
    # Dict A keeps track of the shortest path between from first to second
    # Dict B keeps track of one of the vertices (given by its number)
    all_keys = [(first, second) for first in self.get_vertices() for second in self.get_vertices()]
    A = {key:math_inf for key in all_keys}
    B = {key:None for key in all_keys}
    # We want to stop the iterations the moment we find the first negative cycle
    # (There could be more... we only care about providing an example, if there is one)
    have_found_negative_cycle = False
    for loop_index in range(n+1):
      if have_found_negative_cycle:
        # Stop, no need to keep iterating
        break 
      # Possible interior vertices for this iteration:
      # If loop_index is 0, an empty list. Otherwise, the list has loop_index elements
      possible_interior_numbers = list(range(0, loop_index))
      possible_interior_vertices = [number_to_vertex[number] for number in range(0,loop_index)]
      # We will do loop_index being 0 separately, since it is a bit different
      if loop_index == 0:
        # To control the existence of negative self-arrows
        negative_self_arrow_example = None
        # In this case the possible paths are the existing arrows
        # We also include the "null path", from each vertex to itself, weight 0
        trivial_paths = [(vertex, vertex, 0) for vertex in self.get_vertices()]
        for source, target, weight in list(self._arrows)+trivial_paths:
          # Note that self-arrows are one source of negative cycles
          # (That is, if the self-arrow has negative weight, it forms one)
          if source == target and weight < 0:
            have_found_negative_cycle = True
            is_negative_cycle_self_arrow = True
            negative_self_arrow_example = (source, target, weight)
            # We only need to produce one negative cycle, so we break off this inner loop
            break
          if weight < A[source, target]:
            A[(source, target)] = weight
            # We record the loop_index
            B[(source, target)] = loop_index # In this case 0
      else:
        # This is for loop_index being 1 through n
        # This step is characterized by the fact that a new vertex can
        #now be used as interior vertex.
        for first, second in all_keys:
          # We check if this new vertex provides a shortcut
          # In this case (and only this case) we update A and B for the pair
          newest_vertex = number_to_vertex[loop_index - 1]
          if A[(first, newest_vertex)] + A[(newest_vertex, second)] < A[(first, second)]:
            A[(first, second)] = A[(first, newest_vertex)] + A[(newest_vertex, second)]
            B[(first, second)] = loop_index # Corresponding vertex is loop_index - 1
        # We now want to trigger have_found_negative_cycle the earlier possible
        # To do so we look at the diagonal
        # (This testing does cost CPU time but overall saves time for breaking earlier)
        for vertex in self.get_vertices():
          if A[(vertex, vertex)] < 0:
            # We found a negative cycle, but it is not a self-arrow
            have_found_negative_cycle = True
            is_negative_cycle_self_arrow = False
            # Note that vertex is part of a negative cycle (with more vertices
            #than only itself)
            vertex_in_negative_cycle = vertex
            # One is enough; break out of this for loop
            break        
    # Okay, so now we have A and B ready and correct
    # (If we broke out earlier due to have_found_negative_cycle,
    #they are not optimal but should be enough to correctly output a negative cycle)
    # We heva also determined with 100% certainty whether there is or
    #there isn't a negative cycle. We can express with another variable too
    #(to denote our learning of the property of having or not having such cycles)
    is_negative_cycle_free = not have_found_negative_cycle
    # We first deal with the case when there are negative cycles
    if have_found_negative_cycle: # Equivalent to not is_negative_cycle_free
      if is_negative_cycle_self_arrow:
        # We have recorded a negative-weight self-arrow as example
        # Preparing this data is cheap, and so we prepare everything
        negative_cycle_example_weight = negative_self_arrow_example[2]
        negative_cycle_example_as_arrows = [negative_self_arrow_example]
        involved_vertex = negative_self_arrow_example[0] # Same as in position 1
        negative_cycle_example_as_vertices = [involved_vertex, involved_vertex]
      else:
        # That is, is_negative_cycle_self_arrow is False
        # In this case we have a cycle with more than one vertex
        #(and one of those vertices is given by vertex_in_negative_cycle)
        negative_cycle_example_weight = A[(vertex_in_negative_cycle, vertex_in_negative_cycle)]
        # Since it is a single cycle for the whole function, we don't mind
        #producing arrows and vertices even if they are not requested
        as_vertices_and_arrows = get_shortest_path_using_A_and_B(
              vertex_in_negative_cycle, vertex_in_negative_cycle, request = 'vertices_and_arrows')
        negative_cycle_example_as_vertices = as_vertices_and_arrows[0]
        negative_cycle_example_as_arrows = as_vertices_and_arrows[1]
    else:
      # Now we are in the case without negative cycles
      # We produce the shortest paths for every pair of vertices
      # Paths can be given as their weight/length, their arrows or their vertices
      # This info will be stored in dicts
      # (We only really compute the ones requested as output)
      all_shortest_paths_weights = {}
      all_shortest_paths_as_arrows = {}
      all_shortest_paths_as_vertices = {}
      for pair in all_keys:
        # (We know every path has a 0-long path to itself, any cycle will
        #have positive weight. But, to output a more complete dictionary,
        #we will put default values, exactly as specified by sub-function
        # (I. e. we don't need a separate routine for first == second)
        all_shortest_paths_weights[pair] = A[pair]
        if 'vertices' in request.lower():
          if 'arrows' in request.lower():
            as_vertices_and_arrows = get_shortest_path_using_A_and_B(
                *pair, request = 'vertices_and_arrows')
            all_shortest_paths_as_vertices[pair] = as_vertices_and_arrows[0]
            all_shortest_paths_as_arrows[pair] = as_vertices_and_arrows[1]
          else:
            all_shortest_paths_as_vertices[pair] = get_shortest_path_using_A_and_B(
                *pair, request = 'vertices')
        else:
          if 'arrows' in request.lower():
            all_shortest_paths_as_arrows[pair] = get_shortest_path_using_A_and_B(
                *pair, request = 'arrows')
    # We prepare the data before the final step!!!
    if request.lower() == 'lenghts':
      if is_negative_cycle_free:
        # The dict with all the weights
        requested_data = all_shortest_paths_weights
      else:
        # The weight of the negative cycle example
        requested_data = negative_cycle_example_weight
    elif request.lower() == 'vertices':
      if is_negative_cycle_free:
        # The dict of the vertices forming the shortest paths
        requested_data = all_shortest_paths_as_vertices
      else:
        # The vertices forming the negative cycle example
        requested_data = negative_cycle_example_as_vertices
    elif request.lower() == 'arrows':
      if is_negative_cycle_free:
        # The dict of the arrows forming the shortest paths
        requested_data = all_shortest_paths_as_arrows
      else:
        requested_data = negative_cycle_example_as_arrows
    elif request.lower() == 'lengths_and_vertices':
      if is_negative_cycle_free:
        # Tuple with dict of weights and of vertices
        requested_data = (all_shortest_paths_weights, all_shortest_paths_as_vertices)
      else:
        requested_data = (negative_cycle_example_weight, negative_cycle_example_as_vertices)
    elif request.lower() == 'lengths_and_arrows':
      # Tuple with dict of weights and of arrows
      if is_negative_cycle_free:
        requested_data = (all_shortest_paths_weights, all_shortest_paths_as_arrows)
      else:
        requested_data = (negative_cycle_example_weight, negative_cycle_example_as_arrows)
    elif request.lower() == 'vertices_and_arrows':
      # Tuple with dict of vertices and of arrows
      if is_negative_cycle_free:
        requested_data = (all_shortest_paths_as_vertices, all_shortest_paths_as_arrows)
      else:
        requested_data = (negative_cycle_example_as_vertices, negative_cycle_example_as_arrows)
    elif request.lower() == 'lengths_and_vertices_and_arrows':
      # Tuple with dict of weights, of vertices and of arrows
      if is_negative_cycle_free:
        requested_data = (all_shortest_paths_weights,
            all_shortest_paths_as_vertices, all_shortest_paths_as_arrows)
      else:
        requested_data = (negative_cycle_example_weight,
            negative_cycle_example_as_vertices, negative_cycle_example_as_arrows)
    else:
      raise ValueError('Request option not recognized.')
    # We finally return everything
    ###################
    # WORK HERE
    # Bug: Currently algorithm may fail to produce correct requested_data
    #if is_negative_cycle_free == False. Only remedy is to not output the negative cycle
    #butonly announce its existence
    ###################
    if not is_negative_cycle_free:
      requested_data = 'There is a negative cycle.'
    return (is_negative_cycle_free, requested_data)

  def solve_traveling_salesman_problem(self, compute_path_instead_of_cycle,
        initial_vertex = None, output_as = None):
    '''
    Solves the Traveling Salesman Problem. That is, produces the shortest
    (by sum of weights of traveled arrows) path or cycle going through all
    vertices of the Digraph.
    
    If no such path or cycle exists, returns None.
    '''
    # We use a class
    pass
        
    

########################################################################
# Class UnweightedDigraph
########################################################################

class UnweightedDigraph(Digraph):
  '''
  A digraph whose arrows are all unweighted.
  '''
  
  def make_digraph_weighted(self, modify_self = False):
    '''
    Creates a new digraph, one where all arrows have weight 1.
    
    Can either modify self or return a new instance.
    '''
    # We detect if we are on a Graph. If we are, we use its edges instead
    if isinstance(self, Graph):
      selected_class = UnweightedGraph
      data_type = 'all_vertices_and_all_edges'
      use_edges_instead_of_arrows = True
      original_working_data = self.get_edges()
    else:
      selected_class = UnweightedDigraph
      data_type = 'all_vertices_and_all_arrows'
      use_edges_instead_of_arrows = False
      original_working_data = self.get_arrows()
    # We create the weighted arrows. Note default new weight is 1
    # Easiest way to make them all 1 is no put new_weights = None
    list_new_tuplees = Digraph.write_weights_into_arrows_or_edges(
        tuplees = original_working_data,
        use_edges_instead_of_arrows = use_edges_instead_of_arrows, new_weights = None,
        require_namedtuple = True, output_as_generator = False)
    # We need to keep track of the vertices too, lest they are isolated
    list_new_vertices = self.get_vertices()
    # We prepare for the return
    data = (list_new_vertices, list_new_tuplees)
    # We create a new instance or modify the current
    if modify_self:
      # Easiest way to reset digraph information is with __init__
      self.__init__(data = data, data_type = data_type,
          cast_as_class = selected_class)
      return None
    else:
      return selected_class(data = data, data_type = data_type,
          cast_as_class = selected_class)

  def create_digraph_with_super_source(self):
    '''
    From a weighted digraph, creates a new weighted digraph which has
    all same vertices and arrows but it has one extra vertex, the "super source",
    and arrows of weight 0 from it to all other vertices.
    
    Returns another instance of the class, as well as the new vertex.
    
    This construction is needed for Johnson's algorithm.
    '''
    # First we create a name for the "super source" vertex
    name_super_source = 'super_source'
    # If already, for some reason, a vertex, we add s's to the start
    # This should not be a problem, but it won't hurt
    while name_super_source in [vertex.name for vertex in self.get_vertices()]:
      name_super_source = 's'+name_super_source
      # We consolidate it into a Vertex
      the_super_source = Digraph.sanitize_vertex(name_super_source, require_namedtuple = False)
    # Now we create the new arrows starting from our super source
    new_arrows = []
    for vertex in self.get_vertices():
      new_arrows.append((the_super_source, vertex, 0))
    # We create a new graph with all those new arrows
    # (Note that if self is empty, this won't produce the desired effect.
    # Because of that, we will make it a separate case)
    if not self.is_nonempty_digraph():
      # Best way to avoid name confusion after importing is to use
      #self.__class__ to denote the very class of the instance
      # We create by specifying it the "neighbors way"
      selected_class = type(self)
      new_digraph = selected_class(data = [[the_super_source]], data_type = 'neighbors_out',
          cast_as_class = None)
      return (the_super_source, new_digraph)
    else:
      # In this case we join by the arrows
      all_arrows = new_arrows + self.get_arrows()
      selected_class = type(self)
      new_digraph = selected_class(data = all_arrows, data_type = 'weighted_arrows',
          cast_as_class = None)
      return (the_super_source, new_digraph)
    
  def get_all_paths_shortest_paths_via_Johnsons(self, request = 'lengths'):
    '''
    Computes the shortest paths for every pair of vertices in graph.
    
    The graph is assumed to be a directed, weighted graph. The weights
    of the arrows can be negative.
    
    For a graph with n vertices, consists of one application of Bellman-Fords
    algorithm using an extended graph with a "super source" extra vertex,
    then n applications of Dijkstra's algorithm.
    
    If m is the number of arrows, the time complexity is O(n^2*lon(n) + m*n). 
    '''
    # We first check we have a weighted graph
    assert self.is_weighted_graph(), 'Graph needs to be weighted'
    # According to the algorithm, we need to execute Bellman-Fords on
    #the graph with an extra vertex (often called the "super source")
    # the_super_source is assumed to be a Vertex namedtuple in super_source_graph
    the_super_source, super_source_graph = self.create_digraph_with_super_source()
    ###############
    # WORK HERE
    ###############
    return super_source_graph.get_single_source_shortest_paths_via_Bellman_Fords(the_super_source)

  def get_hamiltonian_cycle(self, source_vertex = None, output_as = None):
    '''
    Returns a Hamiltonian cycle of the unweighted digraph. That is, any
    cycle that travels through all vertices through the available arrows.
    
    If no such cycle exists, returns None.
    
    Accomodates a request for the cycle to start at a specific vertex,
    even if it matter not in a cycle.
    
    SEE ALSO: get_hamiltonian_path
    '''
    # Default output is Cycle (meaning VertexCycle class). Can also output
    #arrows and vertices
    # A Hamiltonian path is a solution to the Traveling Salesman Problem
    #under the condition that all edges have weight/length 1 (under the TSP
    #perspective, all cycles will have the same length; in special, the shortest
    #one output by TSP algorithm will be a Hamiltonian cycle)
    # Thus, we can do Hamiltonian paths with weighing
    weighted_copy = self.make_graph_weighted(modify_self = False)
    return weighted_copy.solve_traveling_salesman_problem(
        compute_path_instead_of_cycle = False,
        source_vertex = source_vertex,
        output_as = output_as)

  def get_hamiltonian_path(self, source_vertex = None, output_as = None):
    '''
    Returns a Hamiltonian path of the unweighted digraph. That is, a path
    through all the vertices traveling through the available arrows.
    
    Accomodates a request for the path to start at a specific vertex.
    
    If no such cycle exists, returns None.
    
    SEE ALSO: get_hamiltonian_cycle
    '''
    # Very similar to get_hamiltonian_cycle, except that we pass the
    #instruction to solv TSP finding a path instead of a cycle
    weighted_copy = self.make_graph_weighted(modify_self = False)
    return weighted_copy.solve_traveling_salesman_problem(
        compute_path_instead_of_cycle = False,
        source_vertex = source_vertex,
        output_as = output_as)

########################################################################
# Class Graph
########################################################################

class Graph(Digraph):
  '''
  A graph is a digraph whose arrows all have "opposites", forming edges.
  
  Edges might or not be weighted, depending on subclassing.
  '''

  def __repr__(self):
    '''
    Returns representation of self.
    '''
    # We take the last part of the class name using split() string method
    class_last_name = self.__class__.__name__.split()[-1]
    about_instance = 'A {} with {} vertices and {} edges.'.format(
        class_last_name, self.get_number_of_vertices(), self.get_number_of_edges())
    return about_instance
    
  def get_edges(self):
    '''
    Returns all edges of the graph.
    '''
    return self._edges
    
  def get_number_of_edges(self):
    '''
    Returns the number of edges of the graph.
    '''
    return len(self._edges)

  def get_inciding_edges(self, vertex, skip_checks = False):
    '''
    Returns the edges inciding on a vertex.
    '''
    if not skip_checks:
      assert vertex in self
    return self._inciding_edges[vertex]
    
  def get_neighbors(self, vertex, skip_checks = False):
    '''
    Return the neighbors of vertex.
    
    In a multigraph, these neighbors are repeated in the output.
    '''
    if not skip_checks:
      assert vertex in self
    # We need to be careful that in an Edge(first, second, weight)
    #the vertex itself might be the first or second
    # For this reason, we use self._arrows_out instead
    # [For self-loops: a vertex will be a neighbor of itself]
    return [arrow.target for arrow in self._arrows_out[vertex]] 
  
  def get_degree_of_vertex(self, vertex, skip_checks = False):
    '''
    Returns degree of vertex in graph: number of edges adjacent to it.
    '''
    # Can do it with self._inciding_edgfes, self._arrows_in or self._arrows_out
    if not skip_checks:
      assert vertex in self
    return len(self._inciding_edges[vertex])

  def find_cut(self):
    '''
    Provides a randomly selected cut of the graph.
    
    Works only for undirected graphs.
    
    INPUTS:
    self
    
    OUTPUT:
    crossing_edges: number of crossing edges between the two subgraphs
    one_side: one of the subgraphs of the cut
    other_side: the other subgraph
    '''
    # Two constants for the call
    n = self.get_number_of_vertices()
    m = self.get_number_of_edges() # Number of edges. Not used in code
    if n <= 1:
      raise ValueError('graph needs at least 2 vertices')
    # We now do contractions at random edges until we are left with two points
    # That means we need to do n - 2 contractions, until there are two leaders
    # We will use [modifiable] lists to control edges and vertices
    edges = list(self._arrows) # Does not change self. Not used in code
    neighbors = dict(self._arrows_out) # Does not change self, to be very safe
    # We will use a structure similar to what is known as "Union-Find"
    # We will divide the vertices into groups. The information will be recorded for each vertex
    # It will be the same (even in the same order) for all in the same group
    # Also, each group will have a single representative, called leader.
    # This will be stored as a dict "full_groups" and as a dict "leaders"
    # We initialize with one group for each vertex, represented by the very vertex
    # Each contraction will merge two groups, with a new leader chosen
    leaders = {vertex: vertex for vertex in neighbors.keys()}
    full_groups = {vertex: [vertex] for vertex in neighbors.keys()}
    for count in range(n - 2):
      # To select edge at random, first we select a vertex at random with weights
      # The weight is the degree (the number of edges that vertex is part of)
      # Note the sum of the weights is actually the double of the sum of the degrees
      # Let's call the first vertex v
      # (Note random_choices produces a list. Thus, we need to pick the first and only element.)
      v = random_choices([vertex for vertex in neighbors], weights = [len(neighbors[vertex]) for vertex in neighbors])[0]
      # Now we select an edge from v, indirectly selecting a vertex called w
      w = random_choices(neighbors[v])[0]
      #print('\nProgress: deleting edge ({}, {}), {} to go'.format(v, w, n-3-count))
      #print('Vertex {} has {} neighbors (with repetitions)'.format(v, len(neighbors[v])))
      #print('Vertex {} has {} neighbors (with repetitions)'.format(w, len(neighbors[w])))
      # We selected (v, w) as edge. We also will merge the groups
      copy_full_group_v = list(full_groups[v])
      copy_full_group_w = list(full_groups[w])
      #print('Group of v = {} is {}'.format(v, copy_full_group_v))
      #print('Group of w = {} is {}\n'.format(w, copy_full_group_w))
      for vertex in copy_full_group_v:
        #print('Examining vertex {} in the group of v = {}'.format(vertex, v))
        full_groups[vertex] = full_groups[vertex] + full_groups[w]
        #for idx in range(len(neighbors[vertex])-1, -1, -1):
          #if leaders[neighbors[vertex][idx]] == leaders[w]:
            #neighbors[vertex].pop(idx)
      for vertex in copy_full_group_w:
        #print('Examining vertex {} in the group of w = {}'.format(vertex, w))
        full_groups[vertex] = full_groups[v] # It should be updated with the union
        #for idx in range(len(neighbors[vertex])-1, -1, -1):
          #if leaders[neighbors[vertex][idx]] == leaders[v]:
            #neighbors[vertex].pop(idx)
      # Now we should have full_groups[v] == full_groups[w], v the leader of all of them
      # The leader of the whole group of v will become the leader of the whole group of w
      for vertex in full_groups[v]:
        leaders[vertex] = v
      # We don't need to change the edges, with one exception: remove the self-edges
      # That is, remove a edge if its two vertices are in the just formed contraction
      for vertex in full_groups[v]:
        neighbors[vertex] = [u for u in neighbors[vertex] if leaders[u] != v]
      #print('')
      #print('Group of v is now {}, with leader {}'.format(full_groups[v], leaders[v]))
      #print('Group of w is now {}, with leader {}'.format(full_groups[w], leaders[w]))
    # When we finish the process, the number of cuts is the number of remaining edges
    # That is, we removed the contracted edges, exactly once, and nothing else
    double_crossing_edges = sum(len(neighbors[vertex]) for vertex in neighbors)
    # We divide by two because we double-counted
    crossing_edges = double_crossing_edges // 2
    # We now identify and return the two groups (by their leaders)
    leader_one_side = None
    leader_other_side = None
    for vertex in neighbors:
      if leader_one_side is None:
        leader_one_side = leaders[vertex]
      elif leader_other_side is None and leaders[vertex] != leader_one_side:
        leader_other_side = leaders[vertex]
    # The two groups are recorded as full_groups under their respective leaders
    one_side = full_groups[leader_one_side]
    other_side = full_groups[leader_other_side]
    return (crossing_edges, one_side, other_side)

  def is_graph_connected(self):
    '''
    Returns whether the graph is connected.
    '''
    # Graph is connected if it has 0 or 1 connected components
    if self.get_ccs()[2] <= 1:
      return True
    else:
      return False

  def get_ccs(self):
    '''
    Returns the connected components of unweighted, undirected graph.
    '''
    # We need an undirected graph. Unfortunately, testing is too costly
    #assert self.is_undirected_graph(), 'Graph needs to be undirected'
    # Graph can be weighted or not, it should work the same way
    # We use a class designed specifically for this method
    # (We need a way to store and control a state, so we use a class)
    state = StateGraphGetCC(self)
    # We use the dfs_outer_loop method to get the results
    vertices_by_label, components, number_components = state.dfs_outer_loop()
    # vertices_by_label is a dict whose keys are number from 0 up to
    #the number of connected components minus one, and whose values are
    #lists of all vertices on the respective connected components
    # components gives the label for each vertex (which is another way
    #of presenting the same information)
    # number_components is the number of connected components
    return (vertices_by_label, components, number_components)
    
  def get_vertex_cover_with_limited_size(self, k):
    '''
    Returns a vertex cover of the graph with at most k vertices, or None
    if such vertex cover doesn't exist.
    
    A vertex cover is a subset of the vertices such that, for any edge,
    one of the two vertices it incides on is on the vertex cover.
    
    If m == self.number_of_edges(), running time is O(m*(2**k)).
    If removed the variable k and tried to find the smallest vertex cover,
    then we would have an NP-complete problem.
    '''
    pass

########################################################################
# Class WeightedGraph
########################################################################

class WeightedGraph(WeightedDigraph, Graph):
  '''
  A graph whose edges are all weighted.
  '''

  def get_minimal_spanning_tree_via_Prims(self):
    '''
    Returns a minimum spanning tree (a list of edges), plus its cost,
    using Prim's algorithm.
    
    Requires a WeightedGraph (undirected, weighted).
    '''
    # We need a weighted, undirected graph. Weights may be negative
    # We expect non-multigraphs, but it is not a problem (only the extra time wasted)
    # Graph needs to be connected. We can check it inside the Prim's algorithm
    assert self.is_nonempty_digraph(), 'Algorithm requires at least one vertex'
    # Principe behind Prim's algorithm is to increase, vertex a vertex, a set X
    # Let Y be the complement.
    X = []
    Y = self.get_vertices()
    # We also have a set of edges forming the tree. Call it E
    E = []
    # We start with a "random" vertex. We'll take the first from Y
    first_vertex = Y[0] # Note this is a pair (vertex, weight)
    X.append(first_vertex)
    Y.remove(first_vertex)
    # Each time, we add the crossing edge with the smallest weight to the tree
    # (Thus adding one extra point to X)
    # We do it via heaps
    # For each vertex, we control the distance by another heap and only its edges
    # First, we do the initialization process
    total_cost = 0
    individual_heaps = {vertex:[] for vertex in Y}
    for vertex in individual_heaps:
      heapq_heapify(individual_heaps[vertex]) # Optional since it's empty
    for arrow in self.get_arrows_out(first_vertex, skip_checks = True): # Equivalently, X[0]
      # We update individual_heaps. The only edges that matter connect to X
      # (We scan through first vertex to scan fewer edges)
      # Note that due to heapq ordering requirements, the edge will be
      #represented as (weight, vertex) instead
      heapq_heappush(individual_heaps[arrow.target], (arrow.weight, first_vertex))
    # We keep one heap for the vertices in Y, each indexed by the least distance to Y
    # We heapify as a tuple (shortest_distance, vertex)
    # By shortest distance we mean the minimum of the corresponding individual_heap
    # We put this cost/distance/score/weight to be math_inf if heap is empty
    vertices_heap = []
    heapq_heapify(vertices_heap)
    for vertex in individual_heaps:
      if not individual_heaps[vertex]:
        # We insert a fake node
        heapq_heappush(individual_heaps[vertex], (math_inf, None))
        # individual_heaps[vertex][0] is (math_inf, None)
        minimal_weight = individual_heaps[vertex][0][0]
      else:
        # In this case individual_heaps[vertex][0] is (minimum weight, other vertex)
        minimal_weight = individual_heaps[vertex][0][0]
      # Note that in case of equality between minimal_weight, heapq resorts to vertex
      # We cross our fingers and hope it will be able to compare them
      # Otherwise we'd need to envelop vertex in a class with __lt__ method
      heapq_heappush(vertices_heap, (minimal_weight, vertex))
    # Now we enter into the main loop
    # In each iteration, we choose the shortest relevant edge
    # And then update the relevant data
    while len(Y) > 0:
      pre_new_vertex = heapq_heappop(vertices_heap)
      # Due to duplicates, we need to check if indeed this result of heappop
      #is not already in X
      # (Otherwise we need to get the next, then the next, and so on)
      while pre_new_vertex[1] in X:
        pre_new_vertex = heapq_heappop(vertices_heap)
      cost_of_new_edge = pre_new_vertex[0]
      new_vertex = pre_new_vertex[1]
      # The vertex is stored as a pair, first the minimum weight to X, then the vertex
      cost_of_new_edge = pre_new_vertex[0]
      new_vertex = pre_new_vertex[1]
      if cost_of_new_edge == math_inf:
        # We open an exception here. If we got +Infinity, it means the graph is not connected
        raise ValueError('We cannot apply Prim\'s algorithm in an unconnected graph.')
      total_cost += cost_of_new_edge
      X.append(new_vertex)
      # Sometimes we insert duplicates of vertices (see later)
      # So we remove as many as possible, doing a complete scan of the list Y
      Y.remove(new_vertex)
      #Y = [vertex for vertex in Y if vertex != new_vertex] # This is costlier
      pre_new_edge_to_add = heapq_heappop(individual_heaps[new_vertex])
      # Recall reverse order in tuple, weight comes first in heap
      # To get the two vertices (first in X, second in Y) and the weight, we do:
      assert pre_new_edge_to_add[0] == cost_of_new_edge, 'Weight of edges does not match'
      connecting_vertex = pre_new_edge_to_add[1]
      new_edge_to_add = Digraph.Edge(connecting_vertex, new_vertex, cost_of_new_edge)
      E.append(new_edge_to_add)
      # Unfortunately deletion is awful using heapq. But it's the best we have
      # We don't need to delete the vertex new_vertex from vertices_heap
      #(Since the deletion is automatic with the heappop)
      # We should delete the entry of new_vertex from the dict individual_heaps
      del individual_heaps[new_vertex]
      # Now we need to update the individual_heaps
      # All we need to do is add a new edges to a few vertices in Y
      # Namely, we need to add the edges inciding on new_vertex to the heaps
      for arrow in self.get_arrows_from(new_vertex, skip_checks = True):
        # adjacency is of the form (new_vertex, some_vertex, weight)
        if arrow.target in Y:
          # Note new_vertex is now in X. (Note weight comes first)
          heapq_heappush(individual_heaps[arrow.target], (arrow.weight, new_vertex))
          # Now we should update vertices_heap with the new weights
          # Since this won't ever increase the minimum_weights,
          #we can add this and keep the old version of adjacency[0] in Y
          # (Not worth to search for it for comparison, simply throw into heap)
          new_minimal_weight = arrow.weight
          # Since this is not a multigraph we don't repeat the adjacency[0]'s
          heapq_heappush(vertices_heap, (new_minimal_weight, arrow.target))
      # Believe it or not, I think we are done for the cycle!!
    # We return the edges as (u, v, weight) from the tree, as well as total cost
    return (E, total_cost)

########################################################################
# Class UnweightedGraph
########################################################################
  
class UnweightedGraph(UnweightedDigraph, Graph):
  '''
  A graph whose edges are all unweighted.
  '''
  pass

########################################################################
# Class StateDiraphGetCC
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
