import numpy as np
import pygame
from scipy.signal import convolve2d


## Traditional 6x7 connect 4 game
class Connect4:
    def __init__(self):
        self.rows = 6
        self.columns = 7
        self.init_state = np.zeros((self.rows, self.columns), dtype=np.float32)

        horizontal_kernel = np.array([[ -1, -1, -1, -1]])
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(4, dtype=np.float32)
        diag2_kernel = np.fliplr(diag1_kernel)
        self.detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]


    def get_next_state(self, board, action):
        if board is None and action is None:
            return self.init_state

        ## Board indices are flipped
        row = self.rows - 1
        while board[row, action] != 0:
            row -= 1

        new_board = np.copy(board)
        new_board[row, action] = 1
        return -new_board


    def get_valid_moves(self, board):
        valid_moves = np.zeros(self.columns, dtype=np.float32)
        for col in range(self.columns):
            valid_moves[col] = np.isin(0, board[:, col])

        if any(valid_moves):
            return valid_moves

        return False


    def check_terminal(self, board):
        for kernel in self.detection_kernels:
            if (convolve2d(board, kernel, mode='valid') == 4).any():
                return 'win'
        if not np.isin(0, board).any():
            return 'draw'
        return False


    def get_reward(self, board):
        result = self.check_terminal(board)
        if result == 'draw':
            return 0
        elif result == 'win':
            return 1
        elif result == False:
            return None


    def draw_board(self, board, winner=None):
        board = np.flip(board, 0) 
        blue = (0,0,255)
        black = (0,0,0)
        red = (255,0,0)
        yellow = (255,255,0)

        pygame.init()
        SQUARESIZE = 100
        RADIUS = int(SQUARESIZE / 2 - 5)

        width = self.columns * SQUARESIZE
        height = (self.rows+1) * SQUARESIZE
        size = (width, height)
        screen = pygame.display.set_mode(size)

        for col in range(self.columns):
            for row in range(self.rows):
                pygame.draw.rect(screen, blue, 
                    (col * SQUARESIZE,
                    row * SQUARESIZE + SQUARESIZE, 
                    SQUARESIZE, 
                    SQUARESIZE)
                    )
                pygame.draw.circle(screen, 
                    black, 
                    (int(col * SQUARESIZE + SQUARESIZE / 2), 
                    int(row * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), 
                    RADIUS
                    )
        
        for col in range(self.columns):
            for row in range(self.rows): 
                if board[row][col] == 1:
                    pygame.draw.circle(screen, 
                        red, 
                        (int(col * SQUARESIZE + SQUARESIZE / 2), 
                        height - int(row * SQUARESIZE + SQUARESIZE / 2)), 
                        RADIUS
                    )
                elif board[row][col] == -1:
                    pygame.draw.circle(screen, yellow, 
                        (int(col * SQUARESIZE + SQUARESIZE / 2), 
                        height - int(row * SQUARESIZE + SQUARESIZE / 2)), 
                        RADIUS
                    )
        if winner is not None:
            myfont = pygame.font.SysFont("monospace", 40)
            label = myfont.render(winner, 1, (255,255,255))
            screen.blit(label, (40,10))
            pygame.display.update()
            pygame.time.wait(3000)
        pygame.display.update()


## 1d connect N game
class ConnectN:
    def __init__(self, N=2):
        self.columns = N + 2 * (N // 2)
        self.N = N
        self.init_state = np.zeros(self.columns, dtype=np.float32)

    def get_next_state(self, board=None, action=None):
        if board is None and action is None:
            return self.init_state
        board[action] = 1
        return -board

    def get_valid_moves(self, board):
        valid_moves = np.zeros_like(board, dtype=np.float32)
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
            win = [val == -1 for val in chunk]
            win = False not in win
            if win:
                return 'win'

        if list(board).count(0) == 0:
            if not win:
                return 'draw'
        return False

    def get_reward(self, board):
        result = self.check_terminal(board)
        if result == 'draw':
            return 0
        elif result == 'win':
            return 1
        elif result == False:
            return None


