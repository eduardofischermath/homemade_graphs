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

from setuptools import setup as setuptools_setup
from setuptools import find_packages as setuptools_find_packages

########################################################################
# Setup
########################################################################

# Sets the package up
setuptools_setup(
    name='homemadegraphs',
    version='0.2.0',
    author='Eduardo Fischer',
    author_email='eduardofischermath@gmail.com',
    path='./src',
    packages= setuptools_find_packages(),
    description='Algorithms and classes related to graphs and digraphs.',
    python_requires='>=3.2.0'
)

########################################################################
