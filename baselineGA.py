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
                        
            # Creating a set of identifiers (city names) already present in "child".
            # Using this set to test membership in O(1) time when filling the remaining positions from parent2.
            existing = set(x for x in child if x is not None)

            # Fill the positions from p2 in order.
            fill_pos = (right + 1) % size
            for gene in p2:
                if gene not in existing:
                    child[fill_pos] = gene
                    existing.add(gene)
                    fill_pos = (fill_pos + 1) % size
            
            # Checking for any None values in code
            if any(x is None for x in child):
                raise RuntimeError("Crossover produced a child with None Values - check OX()")
            return child

        # Returns two offspring 
        return ox(parent1, parent2), ox(parent2, parent1)
        
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


    """ Depth-limited DFS to search for the best complete tour (minimum fitness).
    - limit: maximum depth to explore (number of cities in a partial path). If None, defaults to number of cities.
    Returns: (best_chromosome, best_fitness) where best_chromosome is a list of city indices (same representation as self.population),
         or (None, float('inf')) if no complete tour was found within the limit."
    """
    def depth_limited_search(self, limit=None):

        world_list = self.world.get_cities()
        n = len(world_list)
        if n == 0:
            return (None, float('inf'))
        
        if limit is None:
            return n
        
        best = {"fitness": float("inf"), "chromosome": None}

        def dist(i, j):
            a = world_list[i]
            b = world_list[j]
            dx = a.pose.x - b.pose.x
            dy = a.pose.y - b.pose.y
            return math.hypot(dx, dy)
        
        def dfs(path, used_set, current_parial_len):
            depth = len(path)

            # If a full tour is completed, compute full fitness
            if depth == n:

                fitness = self.calculate_fitness(path)
                if fitness < best["fitness"]:
                    best["fitness"] = fitness
                    best["chromosome"] = path.copy()
                return
            # If depth limit is hit but not a full tour, dont extend further
            if depth >= limit:
                return
            
            # Try extending path for each unused city
            for city_idx in range(n):
                if city_idx in used_set:
                    continue

                # calculate added length if city_idx is extended 
                added = 0.0
                if path:
                    added = dist(path[-1], city_idx)
                new_partial_len = current_parial_len + added

                # Prune if even the partial lenght exceeds current best comlete tour
                if new_partial_len >= best["fitness"]:
                    continue

                # Extend and recurse
                path.append(city_idx)
                used_set.add(city_idx)
                dfs(path, used_set, new_partial_len)
                path.pop()
                used_set.remove(city_idx)

        # Start dfs in every possible start city
        for start in range(n):
            dfs([start], {start}, 0.0)

        if best["chromosome"] is None:
            return (None, float("inf"))
        return (best["chromosome"], best["fitness"])
    

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
        
    
       
       
 
