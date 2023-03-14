
from TicTacToeClass import random_player, player, alphabeta_search, minimax_search
from TicTacToeClass import TicTacToe, play_game


def main():
    play_again = True
    while (play_again):
        # welcome message
        print("Hello, and welcome to your new TicTacToe Game!")
        # prompt user for size of the game
        print("Please define the size of the board.")
        height = -1
        width = -1
        k = -1
        invalid_size = True
        while invalid_size:
            invalid_size = False
            try:
                height = int(input("Enter the height of the board: "))
                width = int(input("Enter the width of the board: "))
                k = int(input("Enter the number in a row needed to win: "))
            except ValueError:
                invalid_size = True
                print("Invalid size; please enter integers for all arguments.")
        # prompt user for search strategy of X and O, respectively
        # choose from: randomly legal moves, alpha-beta legal moves, and minimax legal moves
        print("Please select your move search strategies from the following choices...")
        print("[1] randomly legal moves")
        print("[2] alpha-beta legal moves")
        print("[3] minimax legal moves")
        strategy_dict = {1: random_player, 2: player(alphabeta_search), 3: player(minimax_search)}
        x_strategy = 0
        o_strategy = 0
        invalid_strategy = True
        while invalid_strategy:
            invalid_strategy = False
            try:
                x_strategy = int(input("Enter the number of your choice of strategy for X: "))
                o_strategy = int(input("Enter the number of your choice of strategy for O: "))
            except ValueError:
                invalid_strategy = True
                print("Invalid strategy; please enter integers for both arguments.")
        x_strategy = strategy_dict[x_strategy]
        o_strategy = strategy_dict[o_strategy]
        # create a TTT object, play game
        play_game(TicTacToe(height, width, k), dict(X=x_strategy, O=o_strategy), verbose=True)
        # prompt user to play again
        play_again = False
        # if yes, repeat a ~ d
        # if no, thank the user for playing
        invalid_play_again = True
        while invalid_play_again:
            invalid_play_again = False
            play_again_str = input("Would you like to play again? (y/n): ").lower()
            if play_again_str == 'y' or play_again_str == 'yes':
                play_again = True
                print("You have chosen to play again. New game starting...\n")
            elif play_again_str == 'n' or play_again_str == 'no':
                play_again = False
            else:
                invalid_play_again = True
                print("Invalid play again; please enter either 'y' for yes, or 'n' for no.")

    print("Thank You for Playing Our Game")


if __name__ == '__main__':
    main()
