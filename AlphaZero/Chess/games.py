import numpy as np

## Traditional 8x8 chess
class Chess:
    def __init__(self):
        self.rows = 8
        self.columns = 8
        self.init_state = ChessRules()
    def get_next_state(self, board, action):
        return

    def get_valid_moves(self, board):
        return

    def check_terminal(self, board):
        return

    def get_reward(self, board):
        return


class ChessRules:
    def __init__(self):
        self.columns = 8
        self.rows = 8
        self.n_possible_moves = 73
        ## 56 queen moves {N NE E SE S SW W NW} 0-7
        ## 8 knight moves
        ## 9 under-promotions
        self.init_board = self.initialize_board()

    def initialize_board(self):
        ## Define representation board. Capital letters indicate white piece.
        ## Use FEN representation -> https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
        board = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        return board

    def interpret_fen(self, fen_string=None):
        if fen_string is None:
            fen_string = self.init_board

        if len(fen_string.split()) == 6:
            fen, to_move, castling_rights, _, \
                halfmove_count, fullmove_count = fen_string.split()
            en_passant = True
        if len(fen_string.split()) == 5:
            fen, to_move, castling_rights, \
                halfmove_count, fullmove_count = fen_string.split()
            en_passant = False

        expanded_fen = ''
        for char in fen:
            if char.isdigit():
                char = int(char) * '.'
            expanded_fen += char
        board_string = expanded_fen.replace('/', '')

        if to_move == 'w':
            to_move = True
        if to_move == 'b':
            to_move = False
        return board_string, to_move, castling_rights, halfmove_count, fullmove_count, en_passant

    def get_possible_moves(self, fen_string=None):
        possible_moves_array = np.zeros((self.rows, self.columns, self.n_possible_moves), dtype=int)

        board_string, to_move = self.interpret_fen(fen_string)[:2]
        board_array = np.char.array(list(board_string)).reshape(self.rows, self.columns)
        for idx, square in enumerate(board_string):
            arr_coords = [(idx+1) // self.rows, idx % self.rows]
            if square != '.':
                pass
            if square.isupper() != to_move:
                pass

            if square == 'r':
                for i in range(7):
                    if (arr_coords[0] + i) > 7:
                        north_range = i - 1
                        break
                    above_square = board_array[arr_coords[0]+i, arr_coords[1]]
                    if above_square.islower():
                        north_range = i - 1
                        break
                    if above_square.isupper():
                        north_range = i
                        break
                    north_range = i

                for i in range(7):
                    if (arr_coords[0] - i) < 0:
                        south_range = i - 1
                        break
                    below_square = board_array[arr_coords[0]-i, arr_coords[1]]
                    if below_square.islower():
                        south_range = i - 1
                        break
                    if below_square.isupper():
                        south_range = i
                        break
                    south_range = i

                for i in range(7):
                    if (arr_coords[1] + i) > 7:
                        east_range = i - 1
                        break
                    right_square = board_array[arr_coords[0], arr_coords[1]+i]
                    if right_square.islower():
                        east_range = i - 1
                        break
                    if right_square.isupper():
                        east_range = i
                        break
                    east_range = i

                for i in range(7):
                    if (arr_coords[1] - i) < 0:
                        west_range = i - 1
                        break
                    left_square = board_array[arr_coords[0], arr_coords[1]-i]
                    if left_square.islower():
                        west_range = i - 1
                        break
                    if left_square.isupper():
                        west_range = i
                        break
                    west_range = i

                possible_moves_array[arr_coords[0], arr_coords[1], 0:north_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 14:14+east_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 28:28+south_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 42:42+west_range] = 1


            if square == 'R':
                for i in range(7):
                    if (arr_coords[0] + i) > 7:
                        north_range = i - 1
                        break
                    above_square = board_array[arr_coords[0]+i, arr_coords[1]]
                    if above_square.isupper():
                        north_range = i - 1
                        break
                    if above_square.islower():
                        north_range = i
                        break
                    north_range = i

                for i in range(7):
                    if (arr_coords[0] - i) < 0:
                        south_range = i - 1
                        break
                    below_square = board_array[arr_coords[0]-i, arr_coords[1]]
                    if above_square.isupper():
                        south_range = i - 1
                        break
                    if above_square.islower():
                        south_range = i
                        break
                    south_range = i

                for i in range(7):
                    if (arr_coords[1] + i) > 7:
                        east_range = i - 1
                        break
                    right_square = board_array[arr_coords[0], arr_coords[1]+i]
                    if right_square.isupper():
                        east_range = i - 1
                        break
                    if right_square.islower():
                        east_range = i
                        break
                    east_range = i

                for i in range(7):
                    if (arr_coords[1] - i) < 0:
                        west_range = i - 1
                        break
                    left_square = board_array[arr_coords[0], arr_coords[1]-i]
                    if left_square.isupper():
                        west_range = i - 1
                        break
                    if left_square.islower():
                        west_range = i
                        break
                    west_range = i

                possible_moves_array[arr_coords[0], arr_coords[1], 0:north_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 14:14+east_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 28:28+south_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 42:42+west_range] = 1

            
            if square == 'b':
                for i in range(7):
                    if (arr_coords[0] + i) > 7 or (arr_coords[1] + i) > 7:
                        north_east_range = i - 1
                        break
                    north_east_square = board_array[arr_coords[0]+i, arr_coords[1]+i]
                    if north_east_square.islower():
                        north_east_range = 0
                        break
                    if north_east_square.isupper():
                        north_east_range = i
                        break
                    north_east_range = i

                for i in range(7):
                    if (arr_coords[0] - i) < 0 or (arr_coords[1] + i) > 7:
                        south_east_range = i - 1
                        break
                    south_east_square = board_array[arr_coords[0]-i, arr_coords[1]+i]
                    if south_east_square.islower():
                        south_east_range = i - 1
                        break
                    if south_east_square.isupper():
                        south_east_range = i
                        break
                    south_east_range = i

                for i in range(7):
                    if (arr_coords[0] - i) < 0 or (arr_coords[1] - i) < 0:
                        south_west_range = i - 1
                        break
                    south_west_square = board_array[arr_coords[0]-i, arr_coords[1]-i]
                    if south_west_square.islower():
                        south_west_range = i - 1
                        break
                    if south_west_square.isupper():
                        south_west_range = i
                        break
                    south_west_range = i

                for i in range(7):
                    if (arr_coords[0] + i) > 7 or (arr_coords[1] - i) < 0:
                        north_west_range = i - 1
                        break
                    north_west_square = board_array[arr_coords[0]+i, arr_coords[1]-i]
                    if north_west_square.islower():
                        north_west_range = i - 1
                        break
                    if north_west_square.isupper():
                        north_west_range = i
                        break
                    north_west_range = i

                possible_moves_array[arr_coords[0], arr_coords[1], 7:7+north_east_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 21:21+south_east_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 35:35+south_west_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 49:49+north_west_range] = 1


            if square == 'B':
                for i in range(7):
                    if (arr_coords[0] + i) > 7 or (arr_coords[1] + i) > 7:
                        north_east_range = i - 1
                        break
                    north_east_square = board_array[arr_coords[0]+i, arr_coords[1]+i]
                    if north_east_square.isupper():
                        north_east_range = 0
                        break
                    if north_east_square.islower():
                        north_east_range = i
                        break
                    north_east_range = i

                for i in range(7):
                    if (arr_coords[0] - i) < 0 or (arr_coords[1] + i) > 7:
                        south_east_range = i - 1
                        break
                    south_east_square = board_array[arr_coords[0]-i, arr_coords[1]+i]
                    if south_east_square.isupper():
                        south_east_range = i - 1
                        break
                    if south_east_square.islower():
                        south_east_range = i
                        break
                    south_east_range = i

                for i in range(7):
                    if (arr_coords[0] - i) < 0 or (arr_coords[1] - i) < 0:
                        south_west_range = i - 1
                        break
                    south_west_square = board_array[arr_coords[0]-i, arr_coords[1]-i]
                    if south_west_square.isupper():
                        south_west_range = i - 1
                        break
                    if south_west_square.islower():
                        south_west_range = i
                        break
                    south_west_range = i

                for i in range(7):
                    if (arr_coords[0] + i) > 7 or (arr_coords[1] - i) < 0:
                        north_west_range = i - 1
                        break
                    north_west_square = board_array[arr_coords[0]+i, arr_coords[1]-i]
                    if north_west_square.isupper():
                        north_west_range = i - 1
                        break
                    if north_west_square.islower():
                        north_west_range = i
                        break
                    north_west_range = i

                possible_moves_array[arr_coords[0], arr_coords[1], 7:7+north_east_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 21:21+south_east_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 35:35+south_west_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 49:49+north_west_range] = 1


            if square == 'q':
                for i in range(7):
                    if (arr_coords[0] + i) > 7:
                        north_range = i - 1
                        break
                    above_square = board_array[arr_coords[0]+i, arr_coords[1]]
                    if above_square.islower():
                        north_range = i - 1
                        break
                    if above_square.isupper():
                        north_range = i
                        break
                    north_range = i

                for i in range(7):
                    if (arr_coords[0] - i) < 0:
                        south_range = i - 1
                        break
                    below_square = board_array[arr_coords[0]-i, arr_coords[1]]
                    if below_square.islower():
                        south_range = i - 1
                        break
                    if below_square.isupper():
                        south_range = i
                        break
                    south_range = i

                for i in range(7):
                    if (arr_coords[1] + i) > 7:
                        east_range = i - 1
                        break
                    right_square = board_array[arr_coords[0], arr_coords[1]+i]
                    if right_square.islower():
                        east_range = i - 1
                        break
                    if right_square.isupper():
                        east_range = i
                        break
                    east_range = i

                for i in range(7):
                    if (arr_coords[1] - i) < 0:
                        west_range = i - 1
                        break
                    left_square = board_array[arr_coords[0], arr_coords[1]-i]
                    if left_square.islower():
                        west_range = i - 1
                        break
                    if left_square.isupper():
                        west_range = i
                        break
                    west_range = i

                for i in range(7):
                    if (arr_coords[0] + i) > 7 or (arr_coords[1] + i) > 7:
                        north_east_range = i - 1
                        break
                    north_east_square = board_array[arr_coords[0]+i, arr_coords[1]+i]
                    if north_east_square.islower():
                        north_east_range = 0
                        break
                    if north_east_square.isupper():
                        north_east_range = i
                        break
                    north_east_range = i

                for i in range(7):
                    if (arr_coords[0] - i) < 0 or (arr_coords[1] + i) > 7:
                        south_east_range = i - 1
                        break
                    south_east_square = board_array[arr_coords[0]-i, arr_coords[1]+i]
                    if south_east_square.islower():
                        south_east_range = i - 1
                        break
                    if south_east_square.isupper():
                        south_east_range = i
                        break
                    south_east_range = i

                for i in range(7):
                    if (arr_coords[0] - i) < 0 or (arr_coords[1] - i) < 0:
                        south_west_range = i - 1
                        break
                    south_west_square = board_array[arr_coords[0]-i, arr_coords[1]-i]
                    if south_west_square.islower():
                        south_west_range = i - 1
                        break
                    if south_west_square.isupper():
                        south_west_range = i
                        break
                    south_west_range = i

                for i in range(7):
                    if (arr_coords[0] + i) > 7 or (arr_coords[1] - i) < 0:
                        north_west_range = i - 1
                        break
                    north_west_square = board_array[arr_coords[0]+i, arr_coords[1]-i]
                    if north_west_square.islower():
                        north_west_range = i - 1
                        break
                    if north_west_square.isupper():
                        north_west_range = i
                        break
                    north_west_range = i

                possible_moves_array[arr_coords[0], arr_coords[1], 0:north_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 14:14+east_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 28:28+south_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 42:42+west_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 7:7+north_east_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 21:21+south_east_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 35:35+south_west_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 49:49+north_west_range] = 1


            if square == 'Q':
                for i in range(7):
                    if (arr_coords[0] + i) > 7:
                        north_range = i - 1
                        break
                    above_square = board_array[arr_coords[0]+i, arr_coords[1]]
                    if above_square.isupper():
                        north_range = i - 1
                        break
                    if above_square.islower():
                        north_range = i
                        break
                    north_range = i

                for i in range(7):
                    if (arr_coords[0] - i) < 0:
                        south_range = i - 1
                        break
                    below_square = board_array[arr_coords[0]-i, arr_coords[1]]
                    if above_square.isupper():
                        south_range = i - 1
                        break
                    if above_square.islower():
                        south_range = i
                        break
                    south_range = i

                for i in range(7):
                    if (arr_coords[1] + i) > 7:
                        east_range = i - 1
                        break
                    right_square = board_array[arr_coords[0], arr_coords[1]+i]
                    if right_square.isupper():
                        east_range = i - 1
                        break
                    if right_square.islower():
                        east_range = i
                        break
                    east_range = i

                for i in range(7):
                    if (arr_coords[1] - i) < 0:
                        west_range = i - 1
                        break
                    left_square = board_array[arr_coords[0], arr_coords[1]-i]
                    if left_square.isupper():
                        west_range = i - 1
                        break
                    if left_square.islower():
                        west_range = i
                        break
                    west_range = i

                for i in range(7):
                    if (arr_coords[0] + i) > 7 or (arr_coords[1] + i) > 7:
                        north_east_range = i - 1
                        break
                    north_east_square = board_array[arr_coords[0]+i, arr_coords[1]+i]
                    if north_east_square.isupper():
                        north_east_range = 0
                        break
                    if north_east_square.islower():
                        north_east_range = i
                        break
                    north_east_range = i

                for i in range(7):
                    if (arr_coords[0] - i) < 0 or (arr_coords[1] + i) > 7:
                        south_east_range = i - 1
                        break
                    south_east_square = board_array[arr_coords[0]-i, arr_coords[1]+i]
                    if south_east_square.isupper():
                        south_east_range = i - 1
                        break
                    if south_east_square.islower():
                        south_east_range = i
                        break
                    south_east_range = i

                for i in range(7):
                    if (arr_coords[0] - i) < 0 or (arr_coords[1] - i) < 0:
                        south_west_range = i - 1
                        break
                    south_west_square = board_array[arr_coords[0]-i, arr_coords[1]-i]
                    if south_west_square.isupper():
                        south_west_range = i - 1
                        break
                    if south_west_square.islower():
                        south_west_range = i
                        break
                    south_west_range = i

                for i in range(7):
                    if (arr_coords[0] + i) > 7 or (arr_coords[1] - i) < 0:
                        north_west_range = i - 1
                        break
                    north_west_square = board_array[arr_coords[0]+i, arr_coords[1]-i]
                    if north_west_square.isupper():
                        north_west_range = i - 1
                        break
                    if north_west_square.islower():
                        north_west_range = i
                        break
                    north_west_range = i

                possible_moves_array[arr_coords[0], arr_coords[1], 0:north_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 14:14+east_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 28:28+south_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 42:42+west_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 7:7+north_east_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 21:21+south_east_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 35:35+south_west_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 49:49+north_west_range] = 1


            if square == 'k':
                if (arr_coords[0] + 1) > 7:
                    north_range = 0
                    break
                above_square = board_array[arr_coords[0]+1, arr_coords[1]]
                north_range = 1
                if above_square.islower():
                    north_range = 0
                    break
                if above_square.isupper():
                    north_range = 1
                    break

                if (arr_coords[0] - i) < 0:
                    south_range = i
                    break
                below_square = board_array[arr_coords[0]-i, arr_coords[1]]
                south_range = 1
                if above_square.islower():
                    north_range = 0
                    break
                if above_square != '.' and above_square.islower() == False:
                    north_range = 1
                    break

                if (arr_coords[1] + 1) > 7:
                    east_range = 0
                    break
                right_square = board_array[arr_coords[0], arr_coords[1]+1]
                east_range = 1
                if right_square.islower():
                    east_range = 0
                    break
                if right_square.isupper():
                    east_range = 1
                    break

                if (arr_coords[1] - 1) < 0:
                    west_range = 0
                    break
                left_square = board_array[arr_coords[0], arr_coords[1]-1]
                west_range = 1
                if left_square.islower():
                    west_range = 0
                    break
                if left_square.isupper():
                    west_range = 1
                    break


                if (arr_coords[0] + 1) > 7 or (arr_coords[1] + 1) > 7:
                    north_east_range = 0
                    break
                north_east_square = board_array[arr_coords[0]+1, arr_coords[1]+1]
                north_east_range = 1
                if north_east_square.islower():
                    north_east_range = 0
                    break
                if north_east_square.isupper():
                    north_east_range = 1
                    break


                if (arr_coords[0] - 1) < 0 or (arr_coords[1] + 1) > 7:
                    south_east_range = 0
                    break
                south_east_square = board_array[arr_coords[0]-1, arr_coords[1]+1]
                south_east_range = 1
                if south_east_square.islower():
                    south_east_range = 0
                    break
                if south_east_square.isupper():
                    south_east_range = 1
                    break


                if (arr_coords[0] - 1) < 0 or (arr_coords[1] - 1) < 0:
                    south_west_range = 0
                    break
                south_west_square = board_array[arr_coords[0]-1, arr_coords[1]-1]
                south_west_range = 1
                if south_west_square.islower():
                    south_west_range = 0
                    break
                if south_west_square.isupper():
                    south_west_range = 1
                    break


                if (arr_coords[0] + 1) > 7 or (arr_coords[1] - 1) < 0:
                    north_west_range = 0
                    break
                north_west_square = board_array[arr_coords[0]+i, arr_coords[1]-i]
                north_west_range = 1
                if north_west_square.islower():
                    north_west_range = 0
                    break
                if north_west_square.isupper():
                    north_west_range = 1
                    break


                possible_moves_array[arr_coords[0], arr_coords[1], 0:north_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 14:14+east_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 28:28+south_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 42:42+west_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 7:7+north_east_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 21:21+south_east_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 35:35+south_west_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 49:49+north_west_range] = 1

            
            if square == 'K':
                if (arr_coords[0] + 1) > 7:
                    north_range = 0
                    break
                above_square = board_array[arr_coords[0]+1, arr_coords[1]]
                north_range = 1
                if above_square.isupper()():
                    north_range = 0
                    break
                if above_square.islower():
                    north_range = 1
                    break

                if (arr_coords[0] - i) < 0:
                    south_range = i
                    break
                below_square = board_array[arr_coords[0]-i, arr_coords[1]]
                south_range = 1
                if above_square.isupper():
                    north_range = 0
                    break
                if above_square != '.' and above_square.islower() == False:
                    north_range = 1
                    break

                if (arr_coords[1] + 1) > 7:
                    east_range = 0
                    break
                right_square = board_array[arr_coords[0], arr_coords[1]+1]
                east_range = 1
                if right_square.isupper():
                    east_range = 0
                    break
                if right_square.islower():
                    east_range = 1
                    break

                if (arr_coords[1] - 1) < 0:
                    west_range = 0
                    break
                left_square = board_array[arr_coords[0], arr_coords[1]-1]
                west_range = 1
                if left_square.isupper():
                    west_range = 0
                    break
                if left_square.islower():
                    west_range = 1
                    break


                if (arr_coords[0] + 1) > 7 or (arr_coords[1] + 1) > 7:
                    north_east_range = 0
                    break
                north_east_square = board_array[arr_coords[0]+1, arr_coords[1]+1]
                north_east_range = 1
                if north_east_square.isupper():
                    north_east_range = 0
                    break
                if north_east_square.islower():
                    north_east_range = 1
                    break


                if (arr_coords[0] - 1) < 0 or (arr_coords[1] + 1) > 7:
                    south_east_range = 0
                    break
                south_east_square = board_array[arr_coords[0]-1, arr_coords[1]+1]
                south_east_range = 1
                if south_east_square.isupper():
                    south_east_range = 0
                    break
                if south_east_square.islower():
                    south_east_range = 1
                    break


                if (arr_coords[0] - 1) < 0 or (arr_coords[1] - 1) < 0:
                    south_west_range = 0
                    break
                south_west_square = board_array[arr_coords[0]-1, arr_coords[1]-1]
                south_west_range = 1
                if south_west_square.isupper():
                    south_west_range = 0
                    break
                if south_west_square.islower():
                    south_west_range = 1
                    break


                if (arr_coords[0] + 1) > 7 or (arr_coords[1] - 1) < 0:
                    north_west_range = 0
                    break
                north_west_square = board_array[arr_coords[0]+i, arr_coords[1]-i]
                north_west_range = 1
                if north_west_square.isupper():
                    north_west_range = 0
                    break
                if north_west_square.islower():
                    north_west_range = 1
                    break


                possible_moves_array[arr_coords[0], arr_coords[1], 0:north_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 14:14+east_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 28:28+south_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 42:42+west_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 7:7+north_east_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 21:21+south_east_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 35:35+south_west_range] = 1
                possible_moves_array[arr_coords[0], arr_coords[1], 49:49+north_west_range] = 1



            if square == 'n':
                if (arr_coords[0] + 2) > 7 or (arr_coords[1] + 1) > 7:
                    north_east_move = 0
                    break
                north_east_square = board_array[arr_coords[0]+2, arr_coords[1]+1]
                north_east_move = 1
                if north_east_square.islower():
                    north_east_move = 0
                    break
                if north_east_square.isupper():
                    north_east_move = 1
                    break


                if (arr_coords[0] + 1) > 7 or (arr_coords[1] + 2) > 7:
                    east_north_move = 0
                    break
                east_north_square = board_array[arr_coords[0]+1, arr_coords[1]+2]
                east_north_move = 1
                if east_north_square.islower():
                    east_north_move = 0
                    break
                if north_east_square.isupper():
                    east_north_move = 1
                    break
                

                if (arr_coords[0] - 1) < 0 or (arr_coords[1] + 2) > 7:
                    east_south_move = 0
                    break
                east_south_square = board_array[arr_coords[0]-1, arr_coords[1]+2]
                east_south_move = 1
                if east_south_square.islower():
                    east_south_move = 0
                    break
                if east_south_square.isupper():
                    east_south_move = 1
                    break


                if (arr_coords[0] - 2) < 0 or (arr_coords[1] + 1) > 7:
                    south_east_move = 0
                    break
                south_east_square = board_array[arr_coords[0]-2, arr_coords[1]+1]
                south_east_move = 1
                if south_east_square.islower():
                    south_east_move = 0
                    break
                if south_east_square.isupper():
                    south_east_move = 1
                    break


                if (arr_coords[0] - 2) < 0 or (arr_coords[1] - 1) < 0:
                    south_west_move = 0
                    break
                south_west_square = board_array[arr_coords[0]-2, arr_coords[1]-1]
                south_west_move = 1
                if south_west_square.islower():
                    south_west_move = 0
                    break
                if south_west_square.isupper():
                    south_west_move = 1
                    break


                if (arr_coords[0] - 1) < 0 or (arr_coords[1] - 2) < 0:
                    west_south_move = 0
                    break
                west_south_square = board_array[arr_coords[0]-2, arr_coords[1]-1]
                west_south_move = 1
                if west_south_square.islower():
                    west_south_move = 0
                    break
                if west_south_square.isupper():
                    west_south_move = 1
                    break


                if (arr_coords[0] + 1) > 7 or (arr_coords[1] - 2) < 0:
                    west_north_move = 0
                    break
                west_north_square = board_array[arr_coords[0]+1, arr_coords[1]-2]
                west_north_move = 1
                if west_north_square.islower():
                    west_north_move = 0
                    break
                if west_north_square.isupper():
                    west_north_move = 1
                    break


                if (arr_coords[0] + 2) > 7 or (arr_coords[1] - 1) < 0:
                    north_west_move = 0
                    break
                north_west_move = board_array[arr_coords[0]+2, arr_coords[1]-1]
                north_west_move = 1
                if north_west_move.islower():
                    north_west_move = 0
                    break
                if north_west_move.isupper():
                    north_west_move = 1
                    break


                possible_moves_array[arr_coords[0], arr_coords[1], 56] = north_east_move
                possible_moves_array[arr_coords[0], arr_coords[1], 57] = east_north_move
                possible_moves_array[arr_coords[0], arr_coords[1], 58] = east_south_move
                possible_moves_array[arr_coords[0], arr_coords[1], 59] = south_east_move
                possible_moves_array[arr_coords[0], arr_coords[1], 60] = south_west_move
                possible_moves_array[arr_coords[0], arr_coords[1], 61] = west_south_move
                possible_moves_array[arr_coords[0], arr_coords[1], 62] = west_north_move
                possible_moves_array[arr_coords[0], arr_coords[1], 63] = north_west_move



            if square == 'N':
                if (arr_coords[0] + 2) > 7 or (arr_coords[1] + 1) > 7:
                    north_east_move = 0
                    break
                north_east_square = board_array[arr_coords[0]+2, arr_coords[1]+1]
                north_east_move = 1
                if north_east_square.isupper():
                    north_east_move = 0
                    break
                if north_east_square.islower():
                    north_east_move = 1
                    break


                if (arr_coords[0] + 1) > 7 or (arr_coords[1] + 2) > 7:
                    east_north_move = 0
                    break
                east_north_square = board_array[arr_coords[0]+1, arr_coords[1]+2]
                east_north_move = 1
                if east_north_square.isupper():
                    east_north_move = 0
                    break
                if north_east_square.islower():
                    east_north_move = 1
                    break
                

                if (arr_coords[0] - 1) < 0 or (arr_coords[1] + 2) > 7:
                    east_south_move = 0
                    break
                east_south_square = board_array[arr_coords[0]-1, arr_coords[1]+2]
                east_south_move = 1
                if east_south_square.isupper():
                    east_south_move = 0
                    break
                if east_south_square.islower():
                    east_south_move = 1
                    break


                if (arr_coords[0] - 2) < 0 or (arr_coords[1] + 1) > 7:
                    south_east_move = 0
                    break
                south_east_square = board_array[arr_coords[0]-2, arr_coords[1]+1]
                south_east_move = 1
                if south_east_square.isupper():
                    south_east_move = 0
                    break
                if south_east_square.islower():
                    south_east_move = 1
                    break


                if (arr_coords[0] - 2) < 0 or (arr_coords[1] - 1) < 0:
                    south_west_move = 0
                    break
                south_west_square = board_array[arr_coords[0]-2, arr_coords[1]-1]
                south_west_move = 1
                if south_west_square.isupper():
                    south_west_move = 0
                    break
                if south_west_square.islower():
                    south_west_move = 1
                    break


                if (arr_coords[0] - 1) < 0 or (arr_coords[1] - 2) < 0:
                    west_south_move = 0
                    break
                west_south_square = board_array[arr_coords[0]-2, arr_coords[1]-1]
                west_south_move = 1
                if west_south_square.isupper():
                    west_south_move = 0
                    break
                if west_south_square.islower():
                    west_south_move = 1
                    break


                if (arr_coords[0] + 1) > 7 or (arr_coords[1] - 2) < 0:
                    west_north_move = 0
                    break
                west_north_square = board_array[arr_coords[0]+1, arr_coords[1]-2]
                west_north_move = 1
                if west_north_square.isupper():
                    west_north_move = 0
                    break
                if west_north_square.islower():
                    west_north_move = 1
                    break


                if (arr_coords[0] + 2) > 7 or (arr_coords[1] - 1) < 0:
                    north_west_move = 0
                    break
                north_west_move = board_array[arr_coords[0]+2, arr_coords[1]-1]
                north_west_move = 1
                if north_west_move.isupper():
                    north_west_move = 0
                    break
                if north_west_move.islower():
                    north_west_move = 1
                    break


                possible_moves_array[arr_coords[0], arr_coords[1], 56] = north_east_move
                possible_moves_array[arr_coords[0], arr_coords[1], 57] = east_north_move
                possible_moves_array[arr_coords[0], arr_coords[1], 58] = east_south_move
                possible_moves_array[arr_coords[0], arr_coords[1], 59] = south_east_move
                possible_moves_array[arr_coords[0], arr_coords[1], 60] = south_west_move
                possible_moves_array[arr_coords[0], arr_coords[1], 61] = west_south_move
                possible_moves_array[arr_coords[0], arr_coords[1], 62] = west_north_move
                possible_moves_array[arr_coords[0], arr_coords[1], 63] = north_west_move


            










#x = ChessRules()
#x.get_possible_moves()