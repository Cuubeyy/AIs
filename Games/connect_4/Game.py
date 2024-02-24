import numpy
import torch


class Game:
    def __init__(self):
        self.current_player = None
        self.last_moves = None
        self.board = None
        self.reset()

    def reset(self):
        self.last_moves = []
        self.current_player = "X"
        self.board = [[' ' for _ in range(7)] for _ in range(6)]

    def print_board(self):
        for row in self.board:
            print("|", end="")
            for cell in row:
                print(cell, end="|")
            print()
        print("\n----------------")

    def check_winner(self):
        # Überprüfe horizontale Linien
        for row in self.board:
            for col in range(4):
                if all(row[col + i] == self.current_player for i in range(4)):
                    return True

        # Überprüfe vertikale Linien
        for col in range(7):
            for row in range(3):
                if all(self.board[row + i][col] == self.current_player for i in range(4)):
                    return True

        # Überprüfe diagonale Linien (von links oben nach rechts unten)
        for row in range(3):
            for col in range(4):
                if all(self.board[row + i][col + i] == self.current_player for i in range(4)):
                    return True

        # Überprüfe diagonale Linien (von links unten nach rechts oben)
        for row in range(3, 6):
            for col in range(4):
                if all(self.board[row - i][col + i] == self.current_player for i in range(4)):
                    return True

        return False

    def is_board_full(self):
        return all(cell != ' ' for row in self.board for cell in row)

    def get_state(self):
        states = []
        for c in self.board:
            for cc in c:
                for player in ('X', 'O', ' ') if self.current_player == "X" else ("O", "X", " "):
                    states.append(1 if cc == player else 0)
        return torch.tensor(states, dtype=torch.float)

    def is_move_possible(self, action):
        for row in range(5, -1, -1):
            if self.board[row][action] == ' ':
                self.last_moves.append(action)
                return True
        return False

    def game_step(self, action, verbose=True) -> (int, bool):

        for row in range(5, -1, -1):
            if self.board[row][action] == ' ':
                self.board[row][action] = self.current_player
                break

        if self.check_winner():
            if verbose:
                self.print_board()
                print(f'Spieler {self.current_player} hat gewonnen!')
            return 10, True
        if self.is_board_full():
            if verbose:
                self.print_board()
                print('Das Spiel endet unentschieden!')
            return 0, True

        self.current_player = 'O' if self.current_player == 'X' else 'X'
        if verbose:
            self.print_board()
        return 0, False


if __name__ == '__main__':
    game = Game()
    x = game.get_state()
    while True:
        while not game.is_move_possible(int(input("Spiele deinen Zug: ")) - 1):
            print("Zug ist nicht möglich!")

        if game.game_step(game.last_moves[-1])[1]:
            break
