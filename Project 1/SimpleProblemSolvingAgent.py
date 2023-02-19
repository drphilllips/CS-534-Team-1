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
        self.map = get_romania_map()

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
            (cost, intermediate_city) = heapq.heappop(path)
            if intermediate_city == destination:
                traveled_cities.append(intermediate_city)
                # Prints the search results and total cost of travel
                print(f"* Path from {start} --> {destination} ")
                for i in range(len(traveled_cities) - 1):
                    # Calculate the cost between consecutive cities using euclidean distance
                    cost_between_cities = self.euclidean(get_romania_map().locations[traveled_cities[i]],
                                                         get_romania_map().locations[traveled_cities[i + 1]])
                    total_cost_so_far += cost_between_cities
                    print(f"* {traveled_cities[i]} --> {traveled_cities[i + 1]}: {cost_between_cities}")
                print(f"* Total Cost: {total_cost_so_far} \n")
                return traveled_cities

            # Add city to visited cities
            city_check.add(intermediate_city)
            traveled_cities.append(intermediate_city)

            for adj_city, distance in get_romania_map()[intermediate_city].items():
                if adj_city not in traveled_cities:
                    total_cost = cost + distance + self.euclidean(get_romania_map().locations[adj_city],
                                                           get_romania_map().locations[destination])
                    # Add adjacent city to path
                    heapq.heappush(path, (total_cost, adj_city))

                    # Update total cost so far
                    total_cost_so_far = cost + distance
                    # print(f"* {adj_city}--> {total_cost_so_far}")

        # If no path is found, return None
        return None

    def astar_search(self, start_end_cities):
        search_results = []  # [total cost, intermediate cities]
        cities_visited = {}
        cities_to_visit = {}
        start = start_end_cities[0]
        end = start_end_cities[1]
        current_city = start
        cities_to_visit[start] = [0, 0, 0]  # g,h,f where f=g+h

        while cities_to_visit:
            least_current_costs = [10000, 10000, 10000]
            for city in cities_to_visit:
                if cities_to_visit[city][2] < least_current_costs[2]:
                    current_city = city
                    least_current_costs = cities_to_visit[city]
            cities_to_visit.pop(current_city)
            cities_visited[current_city] = least_current_costs

            neighbor_cities = self.map.get(current_city)
            for next_city in neighbor_cities:
                next_city_cost = self.map.get(current_city)[next_city]
                next_city_heuristic = self.euclidean(self.map.locations[next_city],
                                                     self.map.locations[end])
                next_city_total_cost = next_city_cost + next_city_heuristic
                cities_to_visit[next_city] = [next_city_cost, next_city_heuristic,
                                              next_city_total_cost]

            least_city = ""
            least_city_costs = [10000, 10000, 10000]
            for visit in cities_to_visit:
                if cities_to_visit[visit][2] < least_city_costs[2]:
                    least_city = visit
                    least_city_costs = cities_to_visit[visit]
            current_city = least_city
            cities_to_visit.pop(current_city)
            cities_visited[current_city] = least_city_costs

            if current_city == end:
                print(cities_visited)
                break

            child_cities = self.map.get(current_city)
            for child_city in child_cities:
                if child_city in cities_visited:
                    continue
                child_city_cost = self.map.get(current_city)[child_city] + \
                                  self.euclidean(self.map.locations[child_city],
                                                 self.map.locations[current_city])
                child_city_heuristic = self.euclidean(self.map.locations[child_city],
                                                      self.map.locations[end])
                child_city_total_cost = child_city_cost + child_city_heuristic
                if child_city in cities_to_visit:
                    prev_max_cost = 0
                    for prev_city in cities_to_visit:
                        if cities_to_visit[prev_city][0] > prev_max_cost:
                            prev_max_cost = cities_to_visit[prev_city][0]
                    if child_city_cost > prev_max_cost:
                        continue
                cities_to_visit[child_city] = [child_city_cost, child_city_heuristic,
                                               child_city_total_cost]
        return search_results
