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
        self.n_actions = 8*8*82

    def get_valid_moves(self, board):
        valid_moves_array = np.zeros((8, 8, 82), dtype=int)

        legal_moves = list(board.copy().legal_moves)
        for move in legal_moves:
            valid_moves_array = encode_move(str(move), valid_moves_array)
        return valid_moves_array

    def get_next_state(self, board, action):
        if board is None and action is None:
            board = chess.Board()
            return encode_board(board), board
        coords = np.where(action == 1)
        action = tuple((*coords[0], *coords[1], *coords[2]))
        board = board.copy()
        move = decode_move(action)
        move = chess.Move.from_uci(move)
        board.push(move)
        return -encode_board(board), board

    def check_terminal(self, board):
        if board.is_checkmate():
            return 1
        if board.is_checkmate():
            return 0
        if board.is_insufficient_material():
            return 0
        if board.can_claim_threefold_repetition():
            return 0
        if board.can_claim_fifty_moves():
            return 0
        return False

    def get_reward(self, board):
        if not self.check_terminal(board):
            return None
        return self.check_terminal(board)