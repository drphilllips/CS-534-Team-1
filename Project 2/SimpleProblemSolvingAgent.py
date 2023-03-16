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
        #city_check = set()
        possible_moves = {}
        total_cost_so_far = 0
        intermediate_city = start

        # Main loop


        while intermediate_city != destination:


            # Add city to visited cities
            traveled_cities.append(intermediate_city)

            for adj_city, distance in get_romania_map().get(intermediate_city).items():
                if adj_city not in traveled_cities:
                    #total_cost = cost + distance +
                    heuristic = self.euclidean(get_romania_map().locations[adj_city],
                                                           get_romania_map().locations[destination])
                    # Add adjacent city to path
                    #heapq.heappush(possible_moves, (distance, heuristic, adj_city))
                    possible_moves[adj_city] = (heuristic,distance)

                    # Update total cost so far
                    #total_cost_so_far = cost + distance
                    # print(f"* {adj_city}--> {total_cost_so_far}")

            # Get city with lowest total cost
            min = 100000
            for key, (val1,val2) in possible_moves.items():
                if min > val1:
                    min = val1
                    chosen = key
            intermediate_city = chosen
            total_cost_so_far += possible_moves[intermediate_city][1]
            possible_moves = {}

        if intermediate_city == destination:
            traveled_cities.append(intermediate_city)
            # Prints the search results and total cost of travel
            print(f"* Total Cost: {total_cost_so_far}")
            print(f"* Path from {start} --> {destination} ")
            for i in range(len(traveled_cities) - 1):
                # Calculate the cost between consecutive cities using euclidean distance
                # cost_between_cities = self.euclidean(get_romania_map().locations[traveled_cities[i]],
                #                                    get_romania_map().locations[traveled_cities[i + 1]])
                # total_cost_so_far += cost_between_cities
                print(f"* - {traveled_cities[i]} --> {traveled_cities[i + 1]}")

            return traveled_cities

        # If no path is found, return None
        return None

    def astar_search(self, start_end_cities):
        search_results = []  # [total cost, intermediate cities]
        cities_visited = {}
        cities_to_visit = {}
        cities_parents = {}
        start = start_end_cities[0]
        end = start_end_cities[1]
        current_city = start
        cities_to_visit[start] = [0, 0, 0]  # g,h,f where f=g+h

        while cities_to_visit:
            # select current node from cities_to_visit based on lowest f cost
            least_current_costs = [10000, 10000, 10000]
            for city in cities_to_visit:
                if cities_to_visit[city][2] < least_current_costs[2]:
                    current_city = city
                    least_current_costs = cities_to_visit[city]
            cities_to_visit.pop(current_city)
            cities_visited[current_city] = least_current_costs

            # if current_city is end goal, rebuild path and break to return
            if current_city == end:
                path = []
                path_cost = 0
                path_city = current_city
                while cities_parents[path_city] != path_city:
                    path.append(path_city)
                    path_cost = path_cost + cities_visited[path_city][0]
                    path_city = cities_parents[path_city]
                    if path_city == start:
                        break
                path.append(start)
                path.reverse()
                search_results = [path_cost, path]
                break

            # iterate through child cities of current_city
            child_cities = self.map.get(current_city)
            for child_city in child_cities:
                # continue if child_city is already visited
                if child_city in cities_visited:
                    continue
                child_city_cost = self.map.get(current_city)[child_city]
                child_city_heuristic = self.euclidean(self.map.locations[child_city],
                                                      self.map.locations[end])
                child_city_total_cost = child_city_cost + child_city_heuristic
                # if child_city is in cities_to_visit compare g cost to those already
                # in the cities_to_visit, if it's higher than all, don't add
                if child_city in cities_to_visit:
                    prev_max_cost = 0
                    for prev_city in cities_to_visit:
                        if cities_to_visit[prev_city][0] > prev_max_cost:
                            prev_max_cost = cities_to_visit[prev_city][0]
                    if child_city_cost > prev_max_cost:
                        continue
                cities_parents[child_city] = current_city
                cities_to_visit[child_city] = [child_city_cost, child_city_heuristic,
                                               child_city_total_cost]
        return search_results
    
        def exp_schedule(k=20, lam=0.005, limit=100):
            return lambda t: (k * np.exp(-lam * t) if t < limit else 0)

        def probability(self, p):
            return random.random() < p
    
    
           def simulated_annealing_search(self, cities, schedule=exp_schedule()):
        start_city = cities[0]
        end_city = cities[1]
        path = [start_city]
        distances = {start_city: 0, end_city: 0}
        curr_city = start_city
        t=0

        while curr_city != end_city:
            T = schedule(t)
            neighbors = [city for city in cities if city != curr_city]
            if not neighbors:
                break
            next_city = random.choice(neighbors)
            cost = self.euclidean(self.map.locations[curr_city], self.map.locations[next_city])
            delta_e = cost - distances[curr_city]
            if delta_e > 0 or np.random.uniform() < np.exp(delta_e / T):
                curr_city = next_city
                path.append(curr_city)
                distances[curr_city] = distances[curr_city] + cost
        path_str = "\n".join(f"* - {city}" for city in path)
        return {
            'Total Cost: ': distances[end_city],
            'Path: ': path_str + "\n"
        }
