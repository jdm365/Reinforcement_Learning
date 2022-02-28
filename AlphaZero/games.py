import numpy as np

## 1d connect N game
class ConnectN:
    def __init__(self, N=2):
        self.columns = N + 2*(N // 2)
        self.N = N
        self.init_state = np.zeros(self.columns, dtype=int)

    def get_next_state(self, board, action, player):
        board_ = np.copy(board)
        board_[action] = player
        player *= -1
        return player * board_

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

    def check_terminal(self, board, player):
        for idx in range(self.columns-self.N):
            chunk = board[idx:idx+self.N]
            victory = len([None for x in chunk if x == player]) == self.N
            loss = len([None for x in chunk if x == -player]) == self.N
            if victory:
                return player
            if loss:
                return -player
        return False

    def get_player_reward(self, board, player):
        reward = self.check_terminal(board, player)
        if reward == False:
            return 0
        return reward
    
    def get_board(self, board, player):
        return player * board