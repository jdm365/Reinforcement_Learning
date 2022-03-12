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









## Add decode move functionality



