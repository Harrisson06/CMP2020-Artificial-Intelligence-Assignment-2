""" abstractGA.py

  The abstract GA.  
  You may want to modify this file when gathering experimental data to produce your results tables and plots.

  Written by: Helen Harman
  Last Modified: 18/08/25
"""

from abc import ABC, abstractmethod
import random

import config
import tsp

""" A class that contains some common GA methods and attributes. """
class AbstractGA(ABC):
    
    def __init__(self, world):
        # The world object contains the list of cities that the agent needs to visit
        self.world = world
        
        # create class attributes:        
        self.population = []                 
        self.fitnesses = []   # the fitness of each individual in the population (in the same order)     
        self.best_fitness = -1
        self.best_individual = None        
        self.number_of_generations = 0
        self.generations_no_improvement = 1
        
    """ Returns the best individual found and the fitness of that individual
       1. reset the GA attributes, 
       2. create the initial population, 
       3. run the GA until the stopping criteria is met and 
       4. return the best found solution 
    """
    def run_GA(self, per_gen_callback=None):
        # reset fitness
        self.fitnesses = []
        self.best_fitness = -1
        self.best_individual = None
        
        # initialise population and calulate the fitness 
        self.initialise_population()      
        self.calculate_fitness_of_population()     
        self.number_of_generations = 1 # initial population = 1st generation
    
        # Checking config if dfs is activated
        limit = getattr(config, "DEPTH_LIMIT", None)
        use_dls_only = getattr(config, "USE_DLS_ONLY", False)

        best_chrom, best_fit = self.depth_limited_search(limit = limit)
        if best_chrom is not None:
            self.best_individual = best_chrom
            self.best_fitness = best_fit

            if use_dls_only:
                # Return instantly with best route found by dfs
                return (self.convert_chromosome_to_city_list(self.best_individual), self.best_fitness)
            
        # Run GA until stopping criteria is met (e.g. number of generations reached) -- you could experiment with alternative stopping criteria
        while not self.finished():
            self.produce_new_generation() 
            self.number_of_generations += 1          
            print ("number of generations =", self.number_of_generations, " best fitness = ", self.best_fitness)

            if per_gen_callback is not None and self.number_of_generations:
                per_gen_callback(self.convert_chromosome_to_city_list(self.best_individual),
                                        self.best_fitness,
                                        self.number_of_generations)
            
        # return the best discovered solution and its fitness
        return (self.convert_chromosome_to_city_list(self.best_individual), self.best_fitness) 
        
    """ Creates the initial population by placing the cities in random orders. """
    def initialise_population(self):
        self.population = []
        for _ in range(config.POPULATION_SIZE):
            # get a copy of the list of cities        
            cities = self.world.get_cities().copy()
            
            # place cities in a random order
            random.shuffle(cities)
            
            # to implement the representation described in the brief, this conversion method does nothing 
            #   but you could experiment with other representations
            chromosome = self.convert_city_list_to_chromosome(cities)
            
            # append to population
            self.population.append( chromosome )    
                            
   
    """ Loops through each individual in the population and calculates its fitness.
        Stores the best individual within the best_individual attribute. 
       You will need to implement self.calculate_fitness(...) within BaselineGA.
    """      
    def calculate_fitness_of_population(self):
        self.fitnesses = [self.calculate_fitness_euclidian(i) for i in self.population]
        
        # check for new best solution		
        for i in range(len(self.population)):
            if ((self.best_individual == None) or (self.fitnesses[i] < self.best_fitness)):
                self.best_fitness = self.fitnesses[i]
                self.best_individual = self.population[i]       
    
    #
    # the abstract methods
    #
    
    @abstractmethod 
    def produce_new_generation(self):
        pass
    
    @abstractmethod   
    def finished(self):
        pass
        
    @abstractmethod 
    def convert_city_list_to_chromosome(self, cities): 
        pass
    
    @abstractmethod 
    def convert_chromosome_to_city_list(self, chromosome):
        pass        
    
    # calculate the fitness of a single chromosome 
    @abstractmethod 
    def calculate_fitness_euclidian(self, chromosome):
        pass
 

# End of abstract GA class    


