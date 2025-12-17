"""  tsp.py
#
# Displays the initial environment, runs the GA then displays the GA's result.
#
# run this using:
# python3 tsp.py  OR  python tsp.py  
#
# Written by: Helen Harman based on code by Simon Parsons
# Last Modified: 18/08/25
"""

from world import World
from baselineGA  import *
from environment import Environment

import time

# Handles displaying info for each generation
def per_gen_callback(solution_cities, fitness, gen):
    print(f"Generation {gen} | Best Fitness: {fitness:.4f} | Route: {solution_cities}")


def main():
    world = World()
    # show cities in the random order they were created in
    display = Environment(world, "world -- cities in random order")  
    
    # Run the GA
    ga = BaselineGA(world) # <-- if you write multiple different GAs to compare, you can modify this line to test them out
    solution, fitness = ga.run_GA(per_gen_callback)
    
    # show cities in the order provided by the GA
    world.update_world(solution)
    print("Locations to visit: ", solution, " Fitness:", fitness)
    display_solution = Environment(world, "Best individual")
    

if __name__ == "__main__":
    main()    
    input("Press the Enter key to end program.")
        


