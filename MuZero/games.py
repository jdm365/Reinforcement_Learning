import numpy as np
import torch as T
import pygame
import gym

## Traditional 6x7 connect 4 game
class Connect4:
    def __init__(self):
        self.rows = 6
        self.columns = 7
        self.input_dims = (self.rows, self.columns)
        self.n_actions = self.columns
        self.init_state = np.zeros((self.rows, self.columns), dtype=int)

    def get_next_state(self, board, action):
        if board is None and action is None:
            return self.init_state
        board_ = np.copy(board)
        column_to_place = board_[:, action]
        rev_idx = [i for i in range(self.rows)]
        for idx, space in enumerate(reversed(column_to_place)):
            val = rev_idx[-(idx+1)]
            if space == 0:
                board_[val, action] = 1
                break
        return -board_

    def get_valid_moves(self, board):
        moves_available = False
        valid_moves = [0 for _ in range(self.columns)]
        for col in range(self.columns):
            if list(board[:, col]).count(0) != 0:
                valid_moves[col] = 1
                moves_available = True
        if moves_available:
            return valid_moves
        return False

    def check_terminal(self, board):
        ## Horizontal check
        for row in range(self.rows):
            board_row = np.copy(board[row, :])
            for idx in range(self.columns-4+1):
                chunk = board_row[idx:idx+4]
                win = [val == -1 for val in chunk]
                win = False not in win
                if win:
                    return 'win'
        ## Vertical check
        for col in range(self.columns):
            board_col = np.copy(board[:, col])
            for idx in range(self.rows-4+1):
                chunk = board_col[idx:idx+4]
                win = [val == -1 for val in chunk]
                win = False not in win
                if win:
                    return 'win'
        
        ## Diagonal check (down right)
        for col in range(self.columns - 3):
            for row in range(self.rows - 3):
                chunk = [board[i+row, i+col] for i in range(4)]
                win = [val == -1 for val in chunk]
                win = False not in win
                if win:
                    return 'win'

        ## Diagonal check (down left); note: just refelct board horizontally
        inverted_board = np.fliplr(np.copy(board))
        for col in range(self.columns - 3):
            for row in range(self.rows - 3):
                chunk = [inverted_board[i+row, i+col] for i in range(4)]
                win = [val == -1 for val in chunk]
                win = False not in win
                if win:
                    return 'win'

        if list(board.flatten()).count(0) == 0:
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
            return 0

    

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
            label = myfont.render(winner, 1, blue)
            screen.blit(label, (40,10))
            pygame.display.update()
            pygame.time.wait(3000)
        pygame.display.update()

        



class GymGames:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.input_dims = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n
        self.init_state = self.env.reset()
        self.n_stacked_frames = 16

    def get_next_state(self, action):
        state = T.zeros((self.input_dims[2:], self.n_stacked_frames*self.input_dims[2]), dtype=T.float)
        rewards = 0
        for i in range(self.n_stacked_frames):
            next_state, reward, done, _ = self.env.step(action)
            rewards += reward
            state[:, :, i:i+3] = next_state
            if done:
                return None
        return state.permute(2, 0, 1).contiguous(), rewards

    def get_valid_moves(self):
        return self.env.action_space.n

    def check_terminal(self, action):
        return self.get_next_state(action)

    def get_reward(self, action):
        return self.get_next_state(action)[1]
    