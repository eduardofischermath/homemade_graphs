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
    Magic method. Returns user-friendly representation of instance.
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
