import numpy as np
import chess
from utils import *

board = chess.Board()
''' 
Get legal moves from python-chess board representation, 
then translate them into the 8x8x73 binary feature plane.

Also get board state from python-chess bitboard representation,
then translate into 8x8xn_pieces board representation.
'''
class Chess:
    def __init__(self):
        self.rows = 8
        self.columns = 8
        self.init_state = chess.Board().reset()

    def get_valid_moves(self, board):
        valid_moves_array = np.zeros((8, 8, 82), dtype=int)

        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            valid_moves_array = encode_move(str(move), valid_moves_array)
        
        return valid_moves_array
            






board = chess.Board()
game = Chess()

game.get_valid_moves(board)