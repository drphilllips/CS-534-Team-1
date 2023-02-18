

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

    def best_graph_first_search(self, problem, f):
        total_cost = 1
        intermediate_cities = ('int_city_1', 'int_city_2', '...', 'int_city_n')
        search_results = (total_cost, intermediate_cities)
        return search_results

    def astar_search(self, problem, h):
        total_cost = 1
        intermediate_cities = ('int_city_1', 'int_city_2', '...', 'int_city_n')
        search_results = (total_cost, intermediate_cities)
        return search_results


