""" config.py
#
# Configuration information for the World. These are elements
# to play with as you develop your solution.
#
# Written by: Simon Parsons
# Modified by: Helen Harman
# Last Modified: 25/08/25
"""

import random

# ==========================================
# DIMENSIONSS
WORLD_WIDTH = 40 # BASE SETTING: 40
WORLD_HEIGHT = 20 # BASE SETTING: 20
# ==========================================

# ==========================================
# Number of cites 
NUMBER_OF_CITIES = 8 # BASE SETTING: 8
# ==========================================

# ==========================================
# When performing path planning to calculate fitness,
# You may want to add some random walls to the environment by increasing this value:
NUMBER_OF_WALLS = 0
# ==========================================

# ==========================================
# GENTETIC ALGORITHM PARAMETERS 
POPULATION_SIZE = 300 # BASE SETTING: 30
CROSSOVER_RATE = 1 # BASE SETTING: 0.2
MUTATION_RATE = 0.05 # BASE SETTING: 0.05
# ==========================================

# ==========================================
# STOPPING CRITERIA 
MAX_NUMBER_OF_GENERATIONS = 600 # BASE SETTING: 20
# ==========================================

# ==========================================
# TOURNAMENT SELECTION 
# K = number of random parents selected
TOURNAMENT_K = 3 # BASE SETTING: 3
# ==========================================

# ==========================================
# Early stopping system 
EARLY_STOP = True # BASE SETTING: False
NO_CHANGE_MAX_GENERATIONS = 10 # BASE SETTING: 10
# ==========================================

# ==========================================
# Elitism allows the best fit of the generaton to be preserved
ELITISM = True # BASE SETTING: FALSE
# ==========================================

# ==========================================
# Use this configurable setting to switch between Euclidian and search based
USE_SEARCH_FOR_FITNESS = False # BASE SETTING: TRUE
# ==========================================


# ==========================================
# These are the configurable settings for depth limited depth first search
# You can change the depth limit so that the search algorithm goes deeper before turning around.
DEPTH_LIMIT = 20 # BASE SETTING: 20
# USE_DLS_ONLY set to True allows it to be used instead of euclideian math.
USE_DLS_ONLY = True # BASE SETTING: FALSE
# ==========================================


