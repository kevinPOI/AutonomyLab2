EMPTY = "."
HUMAN = "O"
ROBOT = "X"

LINES = [
    [(0, 0), (0, 1), (0, 2)],
    [(1, 0), (1, 1), (1, 2)],
    [(2, 0), (2, 1), (2, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 1), (1, 1), (2, 1)],
    [(0, 2), (1, 2), (2, 2)],
    [(0, 0), (1, 1), (2, 2)],
    [(0, 2), (1, 1), (2, 0)],
]


class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [[EMPTY] * 3 for _ in range(3)]
        self.turn = HUMAN

    def empty_cells(self):
        return [(r, c) for r in range(3) for c in range(3) if self.board[r][c] == EMPTY]

    def place(self, row, col, player):
        if self.board[row][col] != EMPTY:
            return False
        self.board[row][col] = player
        self.turn = ROBOT if player == HUMAN else HUMAN
        return True

    def winner(self):
        for line in LINES:
            a, b, c = (self.board[r][col] for r, col in line)
            if a != EMPTY and a == b == c:
                return a
        return None

    def is_full(self):
        return all(cell != EMPTY for row in self.board for cell in row)

    def is_over(self):
        return self.winner() is not None or self.is_full()

    def best_move(self, player):
        if self.is_over():
            return None
        _, move = self._minimax(player, player, 0)
        return move

    def _minimax(self, to_move, me, depth):
        w = self.winner()
        if w == me:
            return 10 - depth, None
        if w is not None:
            return depth - 10, None
        if self.is_full():
            return 0, None

        other = HUMAN if to_move == ROBOT else ROBOT
        maximize = to_move == me
        best_score = -999 if maximize else 999
        best_move = None

        for r, c in self.empty_cells():
            self.board[r][c] = to_move
            score, _ = self._minimax(other, me, depth + 1)
            self.board[r][c] = EMPTY
            if (maximize and score > best_score) or (not maximize and score < best_score):
                best_score, best_move = score, (r, c)

        return best_score, best_move
