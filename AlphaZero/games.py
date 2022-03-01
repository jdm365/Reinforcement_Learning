import numpy as np

## 1d connect N game
class ConnectN:
    def __init__(self, N=2):
        self.columns = N + 2*(N // 2)
        self.N = N
        self.init_state = np.zeros(self.columns, dtype=int)

    def get_next_state(self, board, action):
        board[action] = 1
        return -board

    def get_valid_moves(self, board):
        valid_moves = np.zeros_like(board, dtype=int)
        moves_available = False
        for idx, space in enumerate(board):
            if space == 0:
                valid_moves[idx] = 1
                moves_available = True
        if moves_available:
            return valid_moves
        return False

    def check_terminal(self, board):
        for idx in range(self.columns-self.N+1):
            chunk = board[idx:idx+self.N]
            loss = [val == -1 for val in chunk]
            loss = False not in loss
            if loss:
                return 'loss'

        if list(board).count(0) == 0:
            if not loss:
                return 'draw'
        return False

    def get_reward(self, board):
        reward = self.check_terminal(board)
        if reward == 'draw':
            return 0
        elif reward == 'loss':
            return -1
        elif reward == False:
            return None