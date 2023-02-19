import heapq
from math import sqrt

from RomaniaMap import get_romania_map


class SimpleProblemSolvingAgent:
    """
    [Figure 3.1]
    Abstract framework for a problem-solving agent.
    """

    def __init__(self, initial_state=None):
        """State is an abstract representation of the state
        of the world, and seq is the list of actions required
        to get to a particular state from the initial state(root)."""
        self.state = initial_state
        self.seq = []

    def __call__(self, percept):
        """[Figure 3.1] Formulate a goal and problem, then
        search for a sequence of actions to solve it."""
        self.state = self.update_state(self.state, percept)
        if not self.seq:
            goal = self.formulate_goal(self.state)
            problem = self.formulate_problem(self.state, goal)
            self.seq = self.search(problem)
            if not self.seq:
                return None
        return self.seq.pop(0)

    def update_state(self, state, percept):
        raise NotImplementedError

    def formulate_goal(self, state):
        raise NotImplementedError

    def formulate_problem(self, state, goal):
        raise NotImplementedError

    def search(self, problem):
        raise NotImplementedError

    # -- Implementation Below -- #

    # Find distance between two cities using the Euclidean distance formula
    # Parameters: self, first city, second city
    # Returns the distance in the path between the two cities
    def euclidean(self, city1, city2):
        lat, long = city1
        lat2, long2 = city2
        return sqrt((lat2 - lat) ** 2 + (long2 - long) ** 2)

    # Creates the search algorithm for the best greedy first search
    # Parameters: self, start city, destination city
    # Prints the path between the two cities and the total cost of the path
    def best_graph_first_search(self, start, destination):
        traveled_cities = []
        city_check = set()
        path = [(0, start)]
        total_cost_so_far = 0

        # Main loop
        while path:
            # Get city with lowest total cost
            (total_cost, intermediate_city) = heapq.heappop(path)
            if intermediate_city == destination:
                traveled_cities.append(intermediate_city)
                total_cost_so_far += total_cost
                # Prints the search results and total cost of travel
                print(f"* Path from {start} - {destination} ")
                print(f"* {traveled_cities}")
                print(f"* Total Cost: {total_cost_so_far} \n")
                return traveled_cities

            # Add city to visited cities
            city_check.add(intermediate_city)
            traveled_cities.append(intermediate_city)

            for adj_city, distance in get_romania_map()[intermediate_city].items():
                if adj_city not in traveled_cities:
                    total_cost = distance + self.euclidean(get_romania_map().locations[adj_city],
                                                           get_romania_map().locations[destination])
                    # Add adjacent city to path
                    heapq.heappush(path, (total_cost, adj_city))

            # Update total cost so far
            total_cost_so_far += total_cost

        # If no path is found, return None
        return None

    def astar_search(self, problem, h):
        total_cost = 1
        intermediate_cities = ('int_city_1', 'int_city_2', '...', 'int_city_n')
        search_results = (total_cost, intermediate_cities)
        return search_results
