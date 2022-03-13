import numpy as np
import chess

def is_diag(move):
    start_position = move[:2]
    end_position = move[2:]

    start_rank = int(start_position[1])
    end_rank = int(end_position[1])

    start_file = ord(start_position[0]) - 96
    end_file = ord(end_position[0]) - 96

    if start_rank - end_rank != 0 and start_file - end_file != 0:
        if abs(start_rank - end_rank) == abs(start_file - end_file):
            return True
    return False
    

def get_move_type(move):
    if len(move) == 5:
        if move[-1] == 'q' or move[-1] == 'Q':
            move = move[:4]
        else:
            return 'underpromotion move'
    start_position = move[:2]
    end_position = move[2:]

    start_rank = int(start_position[1])
    end_rank = int(end_position[1])

    start_file = ord(start_position[0]) - 96
    end_file = ord(end_position[0]) - 96

    if start_rank - end_rank != 0 and start_file - end_file != 0:
        if abs(start_rank - end_rank) != abs(start_file - end_file):
            return 'knight move'
    return 'queen move'


def encode_queen_move(move, moves_array):
    start_position = move[:2]
    end_position = move[2:]

    start_rank = int(start_position[1])
    end_rank = int(end_position[1])

    start_file = ord(start_position[0]) - 96
    end_file = ord(end_position[0]) - 96

    if is_diag(move):
        north_east_value = end_rank - start_rank if (end_rank - start_rank) > 0 \
            and (end_file - start_file) > 0 else 0
        if north_east_value:
            moves_array[start_rank, start_file, north_east_value + 7 - 1] = 1
            return moves_array

        south_east_value = start_rank - end_rank if (start_rank - end_rank) > 0 \
            and (end_file - start_file) > 0 else 0
        if south_east_value:
            moves_array[start_rank, start_file, south_east_value + 21 - 1] = 1
            return moves_array

        south_west_value = start_rank - end_rank if (start_rank - end_rank) > 0 \
            and (start_file - end_file) > 0 else 0
        if south_west_value:
            moves_array[start_rank, start_file, south_west_value + 35 - 1] = 1
            return moves_array

        north_west_value = end_rank - start_rank if (end_rank - start_rank) > 0 \
            and (start_file - end_file) > 0 else 0
        if north_west_value:
            moves_array[start_rank, start_file, north_west_value + 49 - 1] = 1
            return moves_array

    north_value = end_rank - start_rank if (end_rank - start_rank) > 0 else 0
    if north_value:
        moves_array[start_rank - 1, start_file - 1, north_value - 1] = 1
        return moves_array

    south_value = start_rank - end_rank if (start_rank - end_rank) > 0 else 0
    if south_value:
        moves_array[start_rank - 1, start_file - 1, south_value + 28 - 1] = 1
        return moves_array

    east_value = end_file - start_file if (end_file - start_file) > 0 else 0
    if east_value:
        moves_array[start_rank - 1, start_file - 1, west_value + 14 - 1] = 1
        return moves_array

    west_value = start_file - end_file if (start_file - end_file) > 0 else 0
    if west_value:
        moves_array[start_rank - 1, start_file - 1, east_value + 42 - 1] = 1
        return moves_array


def encode_knight_move(move, moves_array):
    start_position = move[:2]
    end_position = move[2:]

    start_rank = int(start_position[1])
    end_rank = int(end_position[1])

    start_file = ord(start_position[0]) - 96
    end_file = ord(end_position[0]) - 96

    knight_moves = dict({
        '0': (2, 1),
        '1': (1, 2),
        '2': (-1, 2),
        '3': (-2, 1),
        '4': (-2, -1),
        '5': (-1, -2),
        '6': (1, -2),
        '7': (2, -1)
    })
    move_num = ''
    for key, val in knight_moves.items():
        if val == (start_rank-end_rank, start_file-end_file):
            move_num = int(key)
    moves_array[start_rank - 1, start_file - 1, move_num + 56] = 1
    return moves_array

def encode_underpromotion_move(move, moves_array):
    start_position = move[:2]
    end_position = move[2:4]
    underpromotion_piece = move[-1]

    start_rank = int(start_position[1])
    end_rank = int(end_position[1])

    start_file = ord(start_position[0]) - 96
    end_file = ord(end_position[0]) - 96

    ## N NE SE S SW NW; within each [NBR -> 012] - 18 bits per check
    if is_diag(move[:4]):
        north_east_move = True if (end_rank - start_rank) > 0 \
            and (end_file - start_file) > 0 else False
        if north_east_move:
            moves_array[start_rank - 1, start_file - 1, 67:70] = 1
            return moves_array

        south_east_move = True if (start_rank - end_rank) > 0 \
            and (end_file - start_file) > 0 else False
        if south_east_move:
            moves_array[start_rank - 1, start_file - 1, 70:73] = 1
            return moves_array

        south_west_move = True if (start_rank - end_rank) > 0 \
            and (start_file - end_file) > 0 else False
        if south_west_move:
            moves_array[start_rank - 1, start_file - 1, 76:79] = 1
            return moves_array

        north_west_move = True if (end_rank - start_rank) > 0 \
            and (start_file - end_file) > 0 else False
        if north_west_move:
            moves_array[start_rank - 1, start_file - 1, 79:82] = 1
            return moves_array

    north_move = True if (end_rank - start_rank) > 0 else False
    if north_move:
        moves_array[start_rank - 1, start_file - 1, 64:67] = 1
        return moves_array

    south_move = True if (start_rank - end_rank) > 0 else False
    if south_move:
        moves_array[start_rank - 1, start_file - 1, 73:76] = 1
        return moves_array

def encode_move(move, moves_array):
    move_type = get_move_type(move)
    if move_type == 'queen move':
        return encode_queen_move(move, moves_array)
    elif move_type == 'knight move':
        return encode_knight_move(move, moves_array)
    else:
        return encode_underpromotion_move(move, moves_array)


## Queen moves 0-55, knight moves 56-63, underpromotion 64-81
def translate_queen_move(move):
    ## move is coordinates (rank, file, action_num)
    start_rank = move[0]
    start_file = move[1]

    start_square = chr(start_file + 97) + str(start_rank + 1)
    direction = move[2] // 7
    distance = move[2] % 7 + 1

    if direction == 0:
        end_rank = start_rank + distance
        end_file = start_file
        end_square = chr(end_file + 97) + str(end_rank + 1)
        return start_square + end_square
    
    if direction == 1:
        end_rank = start_rank + distance
        end_file = start_file + distance
        end_square = chr(end_file + 97) + str(end_rank + 1)
        return start_square + end_square

    if direction == 2:
        end_rank = start_rank
        end_file = start_file + distance
        end_square = chr(end_file + 97) + str(end_rank + 1)
        return start_square + end_square

    if direction == 3:
        end_rank = start_rank - distance
        end_file = start_file + distance
        end_square = chr(end_file + 97) + str(end_rank + 1)
        return start_square + end_square

    if direction == 4:
        end_rank = start_rank - distance
        end_file = start_file
        end_square = chr(end_file + 97) + str(end_rank + 1)
        return start_square + end_square

    if direction == 5:
        end_rank = start_rank - distance
        end_file = start_file - distance
        end_square = chr(end_file + 97) + str(end_rank + 1)
        return start_square + end_square

    if direction == 6:
        end_rank = start_rank
        end_file = start_file - distance
        end_square = chr(end_file + 97) + str(end_rank + 1)
        return start_square + end_square

    if direction == 7:
        end_rank = start_rank + distance
        end_file = start_file - distance
        end_square = chr(end_file + 97) + str(end_rank + 1)
        return start_square + end_square


def translate_knight_move(move):
    ## move is coordinates (rank, file, action_num)
    start_rank = move[0]
    start_file = move[1]

    start_square = chr(start_file + 97) + str(start_rank + 1)
    direction = move[2] % 7 + 1

    if direction == 0:
        end_rank = start_rank + 2
        end_file = start_file + 1
        end_square = chr(end_file + 97) + str(end_rank + 1)
        return start_square + end_square
    
    if direction == 1:
        end_rank = start_rank + 1
        end_file = start_file + 2
        end_square = chr(end_file + 97) + str(end_rank + 1)
        return start_square + end_square

    if direction == 2:
        end_rank = start_rank - 1
        end_file = start_file + 2
        end_square = chr(end_file + 97) + str(end_rank + 1)
        return start_square + end_square

    if direction == 3:
        end_rank = start_rank - 2
        end_file = start_file + 1
        end_square = chr(end_file + 97) + str(end_rank + 1)
        return start_square + end_square

    if direction == 4:
        end_rank = start_rank - 2
        end_file = start_file - 1
        end_square = chr(end_file + 97) + str(end_rank + 1)
        return start_square + end_square

    if direction == 5:
        end_rank = start_rank - 1
        end_file = start_file - 2
        end_square = chr(end_file + 97) + str(end_rank + 1)
        return start_square + end_square

    if direction == 6:
        end_rank = start_rank + 1
        end_file = start_file - 2
        end_square = chr(end_file + 97) + str(end_rank + 1)
        return start_square + end_square

    if direction == 7:
        end_rank = start_rank + 2
        end_file = start_file - 1
        end_square = chr(end_file + 97) + str(end_rank + 1)
        return start_square + end_square

def translate_underpromotion_move(move):
    ## move is coordinates (rank, file, action_num)
    start_rank = move[0]
    start_file = move[1]

    start_square = chr(start_file + 97) + str(start_rank + 1)

    mapping = dict({0: 'N', 1: 'B', 2: 'R'})
    direction = move[2] // 6
    promoted_piece = mapping(move[2] % 3)

    if direction == 0:
        end_rank = start_rank + 1
        end_file = start_file
        end_square = chr(end_file + 97) + str(end_rank + 1) + promoted_piece
        return start_square + end_square
    
    if direction == 1:
        end_rank = start_rank + 1
        end_file = start_file + 1
        end_square = chr(end_file + 97) + str(end_rank + 1) + promoted_piece
        return start_square + end_square

    if direction == 2:
        end_rank = start_rank - 1
        end_file = start_file + 1
        end_square = chr(end_file + 97) + str(end_rank + 1) + promoted_piece.lower()
        return start_square + end_square

    if direction == 3:
        end_rank = start_rank - 1
        end_file = start_file
        end_square = chr(end_file + 97) + str(end_rank + 1) + promoted_piece.lower()
        return start_square + end_square

    if direction == 4:
        end_rank = start_rank - 1
        end_file = start_file - 1
        end_square = chr(end_file + 97) + str(end_rank + 1) + promoted_piece.lower()
        return start_square + end_square

    if direction == 5:
        end_rank = start_rank + 1
        end_file = start_file - 1
        end_square = chr(end_file + 97) + str(end_rank + 1) + promoted_piece
        return start_square + end_square


def decode_move(move):
    if move[2] < 56:
        return translate_queen_move(move)
    elif move[2] < 64:
        return translate_knight_move(move)
    else:
        return translate_underpromotion_move(move)



def encode_board(board):
    ## Input board is bitboard from python-chess
    piece_map = board.piece_map()

    pawns_array = np.zeros((8, 8), dtype=int)
    for idx, val in piece_map.items():
        val = str(val)
        if val == 'P':
            rank = idx // 8
            file = idx % 8
            pawns_array[rank, file] = 1
        if val == 'p':
            rank = idx // 8
            file = idx % 8
            pawns_array[rank, file] = -1

    knights_array = np.zeros((8, 8), dtype=int)
    for idx, val in piece_map.items():
        val = str(val)
        if val == 'N':
            rank = idx // 8
            file = idx % 8
            knights_array[rank, file] = 1
        if val == 'n':
            rank = idx // 8
            file = idx % 8
            knights_array[rank, file] = -1
    
    bishops_array = np.zeros((8, 8), dtype=int)
    for idx, val in piece_map.items():
        val = str(val)
        if val == 'B':
            rank = idx // 8
            file = idx % 8
            bishops_array[rank, file] = 1
        if val == 'b':
            rank = idx // 8
            file = idx % 8
            bishops_array[rank, file] = -1

    rooks_array = np.zeros((8, 8), dtype=int)
    for idx, val in piece_map.items():
        val = str(val)
        if val == 'R':
            rank = idx // 8
            file = idx % 8
            rooks_array[rank, file] = 1
        if val == 'r':
            rank = idx // 8
            file = idx % 8
            rooks_array[rank, file] = -1

    queens_array = np.zeros((8, 8), dtype=int)
    for idx, val in piece_map.items():
        val = str(val)
        if val == 'Q':
            rank = idx // 8
            file = idx % 8
            queens_array[rank, file] = 1
        if val == 'q':
            rank = idx // 8
            file = idx % 8
            queens_array[rank, file] = -1

    kings_array = np.zeros((8, 8), dtype=int)
    for idx, val in piece_map.items():
        val = str(val)
        if val == 'K':
            rank = idx // 8
            file = idx % 8
            kings_array[rank, file] = 1
        if val == 'k':
            rank = idx // 8
            file = idx % 8
            kings_array[rank, file] = -1

    board_array = np.stack((pawns_array, knights_array, bishops_array, \
        rooks_array, queens_array, kings_array))
    ## Shape (6, 8, 8)
    return board_array
