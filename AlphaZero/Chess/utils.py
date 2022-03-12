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
            moves_array[start_rank, start_file, north_east_value + 7] = 1
            return moves_array

        south_east_value = start_rank - end_rank if (start_rank - end_rank) > 0 \
            and (end_file - start_file) > 0 else 0
        if south_east_value:
            moves_array[start_rank, start_file, south_east_value + 21] = 1
            return moves_array

        south_west_value = start_rank - end_rank if (start_rank - end_rank) > 0 \
            and (start_file - end_file) > 0 else 0
        if south_west_value:
            moves_array[start_rank, start_file, south_west_value + 35] = 1
            return moves_array

        north_west_value = end_rank - start_rank if (end_rank - start_rank) > 0 \
            and (start_file - end_file) > 0 else 0
        if north_west_value:
            moves_array[start_rank, start_file, north_west_value + 49] = 1
            return moves_array

    north_value = end_rank - start_rank if (end_rank - start_rank) > 0 else 0
    if north_value:
        moves_array[start_rank, start_file, north_value] = 1
        return moves_array

    south_value = start_rank - end_rank if (start_rank - end_rank) > 0 else 0
    if south_value:
        moves_array[start_rank, start_file, south_value + 28] = 1
        return moves_array

    east_value = end_file - start_file if (end_file - start_file) > 0 else 0
    if east_value:
        moves_array[start_rank, start_file, west_value + 14] = 1
        return moves_array

    west_value = start_file - end_file if (start_file - end_file) > 0 else 0
    if west_value:
        moves_array[start_rank, start_file, east_value + 42] = 1
        return moves_array


def encode_knight_move(move, moves_array):
    return

def encode_underpromotion_move(move, moves_array):
    return

def encode_move(move, moves_array):
    move_type = get_move_type(move)
    if move_type == 'queen move':
        return encode_queen_move(move, moves_array)
    elif move_type == 'knight move':
        return encode_knight_move(move, moves_array)
    else:
        return encode_underpromotion_move(move, moves_array)



