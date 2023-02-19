import os

import RomaniaMap
from SimpleProblemSolvingAgent import SimpleProblemSolvingAgent


def prompt_for_map_file_directory():
    """
    Prompts user for the local directory of the
    Romania map file
    :return: directory of the map file
    """
    map_file_directory = ""
    # prompt user until receiving a valid map file directory
    invalid_directory = True
    while invalid_directory:
        map_file_directory = input("Please enter where your map file is located in your local directory: ")
        # if the path exists, we have a valid directory
        if os.path.exists(map_file_directory):
            invalid_directory = False
        else:
            print("Invalid directory.")
    # return map file directory
    return map_file_directory


def in_romania_map(city):
    """
    Checks if city is in the romania_map file
    :param city: string, city name
    :return: boolean, whether city is in the map
    """
    # import romania map
    romania_map = RomaniaMap.get_romania_map()
    # check if the city is within the map nodes
    city_in_romania_map = city in romania_map.nodes()
    # return result
    return city_in_romania_map


def prompt_for_two_cities():
    """
    Prompts user for the two cities they would
    like to navigate between
    :return: two cities, as a tuple
    """
    # prompt user until receiving a valid pair of cities
    invalid_cities = True
    while invalid_cities:
        city1 = input("Please enter a city from the romania_map: ")
        city2 = input("Please enter another city from the romania_map: ")
        # if two cities are different and in romania_map,
        # we have a valid pair of cities
        if city1 != city2 and \
            in_romania_map(city1) and \
                in_romania_map(city2):
            invalid_cities = False
        else:
            print("Invalid cities.")
    # combine cities into a tuple
    two_cities = (city1, city2)
    return two_cities


def main():
    """
    RomaniaCityApp provides interface for user to
    interact with Greedy and Astar search algorithms
    and find the best path between two cities in the
    Romania Map provided
    """
    # prompt user for continuous run
    greedy_search = False
    astar_search = True
    best_path_again = True
    while best_path_again:
        # prompt user for two cities
        two_cities = prompt_for_two_cities()
        # create simple problem solving agent
        spsa = SimpleProblemSolvingAgent()
        # collect search results for greedy and astar search
        if greedy_search:
            # print greedy search results
            print("Greedy Best-First Search")
            greedy_search_results = spsa.best_graph_first_search(two_cities[0], two_cities[1])
            greedy_total_cost = greedy_search_results[0]
            greedy_intermediate_cities = greedy_search_results[1]
            print("* Total Cost: " + str(greedy_total_cost))
            print("* Intermediate Cities: ")
            for greedy_intermediate_city in greedy_intermediate_cities:
                print("* - " + greedy_intermediate_city)
        # print astar search results
        if astar_search:
            print("Astar Search")
            astar_search_results = spsa.astar_search(two_cities)
            astar_total_cost = astar_search_results[0]
            astar_intermediate_cities = astar_search_results[1]
            print("* Total Cost: " + str(astar_total_cost))
            print("* Intermediate Cities: ")
            for astar_intermediate_city in astar_intermediate_cities:
                print("* - " + astar_intermediate_city)
        # ask user if they would like to find the best path
        # between any two cities again
        invalid_yes_no_response = True
        while invalid_yes_no_response:
            best_path_again = input("Would you like to find the best path between any two cities again? (y/n): ")
            if best_path_again == 'y' or best_path_again == 'Y':
                best_path_again = True
                invalid_yes_no_response = False
            elif best_path_again == 'n' or best_path_again == 'N':
                best_path_again = False
                invalid_yes_no_response = False
            else:
                print("Invalid response, please respond with \'y\' or \'n\'.")
    # terminate program, print thank you
    print("Thank You for Using Our App")


if __name__ == '__main__':
    main()
