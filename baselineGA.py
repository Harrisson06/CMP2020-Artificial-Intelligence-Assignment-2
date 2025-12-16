""" BaselineGA.py
#
# Your GA code. 
#
# This is the file you will modify.  
# The code we have added to this file is to allow the application to run -- you will need to edit the code.
#   (You can modify the other files -- if you do so, tell us about you have modified them in your report).
#
# Modified by XXX
# Last Modified: 18/08/25
"""

from numpy.random import randint # https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html
from numpy.random import rand    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
import random
import math

from abstractGA import AbstractGA
import config
           
""" A GA to solve the TSP
    This class extends the AbstractGA class.
"""
class BaselineGA(AbstractGA):    
        
    """ Creates a new population and returns the best individual found so far.
        EDIT THIS METHOD: you will need to add the code that creates the new population.
        self.population stores the current population 
        self.fitnesses stores the fitness of each member of the population (in the order they appear in self.population). 
    """
    def select_parents(self, population, fitnesses):
        eps = 1e-6
        if not fitnesses:
            return [random.choice(population) for _ in  range(len(population))]
        
        inv = [1.0 / (f + eps) for f in fitnesses]
        total = sum(inv)

        if total == 0:
            return [random.choice(population) for _ in range(len(population))]
        probs = [v / total for v in inv]

        return random.choices(population, weights= probs, k=len(population))

    def choose_two(self, parents):
        return random.sample(parents, 2)

    def crossover(self, parent1, parent2):
        size = len(parent1)
        if size < 2:
            return parent1.copy(), parent2.copy()
        
        a = randint(0, size)
        b = randint(0, size)
        left, right = min(a, b), max(a, b)

        if left == right:
            right = (left + 1) % size
            if right == left:
                return parent1.copy(), parent2.copy()
            if left > right: 
                left, right = right, left

        def ox(p1, p2):
            child = [None] * size
            
            # Copies the segment from p1 into child. 
            for i in range(left, right + 1):
                child[i] = p1[i]
            
            def gene_id(g):
                return getattr(g, "name", g)
            
            # Creating a set of identifiers (city names) already present in "child".
            # Using this set to test membership in O(1) time when filling the remaining positions from parent2.
            existing = set()
            for x in child: 
                if x is not None:
                    existing.add(gene_id(x))

            # Fill the positions from p2 in order.
            fill_pos = (right + 1) % size
            for gene in p2:
                if gene_id(gene) not in existing:
                    child[fill_pos] = gene
                    existing.add(gene_id(gene))
                    fill_pos = (fill_pos + 1) % size
            
            # Checking for any None values in code
            if any(x is None for x in child):
                raise RuntimeError("Crossover produced a child with None Values - check OX()")
            return child
 
        child1 = ox(parent1, parent2)
        child2 = ox(parent2, parent1)
        return child1, child2
        
    def mutation(self, individual):
        ind = individual.copy()
        if rand() < config.MUTATION_RATE:
            size = len(ind)
            if size >= 2:
                i, j = random.sample(range(size), 2)
                ind[i], ind[j] = ind[j], ind[i]
        return ind

    def produce_new_generation(self):

        parents = self.select_parents(self.population, self.fitnesses)
        offspring =[]

        while len(offspring) < len(self.population):
            parent1, parent2 = self.choose_two(parents)

            if rand() < config.CROSSOVER_RATE:
                child1, child2 = self.crossover(parent1.copy(), parent2.copy())
            else: 
                child1, child2 = parent1.copy(), parent2.copy()

            child1 = self.mutation(child1)
            child2 = self.mutation(child2)

            offspring.append(child1)
            if len(offspring) < len(self.population):
                offspring.append(child2)

        offspring = offspring[:len(self.population)]

        self.population = offspring
        
        # calculate the new fitness and return the best individual
        self.calculate_fitness_of_population() # <-- this method is in abstractGA.py

        return (self.best_individual, self.best_fitness)   
    
    
    """ Sum the distance between each of the cities 
        EDIT THIS: you will need to add the code that calculates the fitness of a single individual/chromosome
        (Note, the calculate_fitness_of_population() method in AbstractGA loops through the population.)
    """
    def calculate_fitness(self, chromosome):    
        cities = self.convert_chromosome_to_city_list(chromosome)   
         
        total = 0
        
        if len(cities) > 1:
            for i in range(len(cities)-1):
                dx = cities[i].pose.x - cities[i+1].pose.x
                dy = cities[i].pose.y - cities[i+1].pose.y
                total += math.hypot(dx, dy)
            
            dx = cities[-1].pose.x - cities[0].pose.x
            dy = cities[-1].pose.y - cities[0].pose.y
            total += math.hypot(dx, dy)
        
        return total
    # YOU WILL NEED TO ADD METHODS
           
           
    """ The stopping criteria. When this returns true, the GA will stop producing new generations.
        We have given you one implementation of this -- you could try out other implementations.
    """
    def finished(self):
        return self.number_of_generations >= config.MAX_NUMBER_OF_GENERATIONS
    
       
    #-------------------
    # The below conversion methods do nothing as are chromosome is just a list of cities; however, 
    #  if you decide to experiment with using a different representation, you may want to edit them.
    #   
    
    """ convert a list of cities to a chromosome that can be used by the GA """
    def convert_city_list_to_chromosome(self, cities):  

        world_list = self.world.get_cities()
        name_to_index = {c.name: idx for idx, c in enumerate(world_list)}
        return [name_to_index[c.name] for c in cities]
    
    """ convert a chromosome into a list of cities that can be used by fitness 
         calculation and be returned at the end.
    """
    def convert_chromosome_to_city_list(self, chromosome):
        world_list = self.world.get_cities()
        return [world_list[i] for i in chromosome]
    #-------------------
    
          
    
# End of BaselineGA class    
        
    
       
       
 
