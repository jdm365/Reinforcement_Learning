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
        for idx in range(self.columns-self.N):
            chunk = board[idx:idx+self.N]
            victory = len([None for x in chunk if x == 1]) == self.N
            loss = len([None for x in chunk if x == -1]) == self.N
            draw = list(board).count(0) == 0
            if victory:
                return 1
            if loss:
                return -1
            elif draw:
                ## If 0 is returned, it is seen as "False"
                return None
        return False

    def get_reward(self, board):
        reward = self.check_terminal(board)
        if reward is None:
            return 0
        if reward == False:
            return None
        return reward