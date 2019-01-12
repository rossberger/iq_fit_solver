"""
iq_fit_solver.py - A brute-force solution finder for the IQ Fit game

https://www.smartgames.eu/uk/one-player-games/iq-fit
https://www.amazon.com/SmartGames-SG423US-IQ-Fit/dp/B0084ZJ9RS

The game consists of 10 3D puzzle pieces and a 5x10 2D playing board.  The
game begins with some number of pieces already placed on the board (the
instruction booklet gives 120 such initial starting configurations
corresponding to a range of difficulties -- the fewer pieces initially placed
the more challenging the puzzle). The challenge is then to place all remaining
pieces on the board in an arrangement such that there are no unfilled spaces
on the board. Due to their 3D shape, each piece can be rotated about its long
axis allowing it to be placed in one of two orientations on the 2D board space
(the third dimension of each piece is hidden under the board and pieces must
lie flat on the board, i.e., no parts may stick up out of the board).

This solver works by iterating over the board and attempting to place pieces
in the available spaces on the board.  All orientations of each piece at all
positions on the board are checked.  You must define the initial conditions
(which pieces are available and the initial board configuration) before
running the solver.  The solver performs a depth first search.  The board and
pieces are represented using 2D numpy arrays.

Solver output consists of both numeric and a text-based color approximation of
the game board.

Requirements:
    - numpy
    - colorama

Usage:
    - Define a board in this file.  See examples below.
    - Update main to use the board you defined.
      E.g., if you defined BOARD_100, main should have "board = BOARD_100"
    - Run:
      $ python iq_fit_solver.py

"""
import datetime
import numpy as np
from colorama import init, Fore, Back, Style
init(autoreset=True)

# An empty 5x10 playing board
BLANK_BOARD = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# Board #120 from the instruction booklet, with pieces 1 and 4 already placed.
BOARD_120 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 4, 4, 0, 0, 0],
                      [0, 0, 1, 0, 0, 4, 4, 4, 4, 0],
                      [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# Define your own boards here and then use them in main() below.
# BOARD_XXX = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


# Dictionary of colors to use when displaying the board and pieces.
COLORS = {}
COLORS[0] = Back.WHITE + Fore.BLACK   # color of empty spaces
# The remaining colors are configured with the pieces below.  Note that the
# colors are an approximation to the board colors since colorama doesn't have
# quite so many color options.

# Dictionary of pieces -- there are 10 included with the game.  You shouldn't
# need to edit these unless you want to experiment with new shapes.
PIECES = {}

# Each piece is itself a two-element list.  Each element is a 2D array
# corresponding to the shape of the piece.  The 'x' variable is used as
# shorthand to make the shape of the piece stand out more.

# Light green
x = 1
COLORS[x] = Style.BRIGHT + Fore.GREEN
PIECES[x] = [np.array([[x, 0, x],
                       [x, x, x]]),
             np.array([[x, x, x],
                       [x, 0, 0]])]

# Dark green
x = 2
COLORS[x] = Fore.GREEN
PIECES[x] = [np.array([[0, x, 0],
                       [x, x, x]]),
             np.array([[x, x, x],
                       [x, x, 0]])]

# Yellow
x = 3
COLORS[x] = Style.BRIGHT + Fore.YELLOW
PIECES[x] = [np.array([[x, x, x, x],
                       [0, 0, 0, x]]),
             np.array([[x, x, x, x],
                       [0, 0, x, x]])]

# Orange
x = 4
COLORS[x] = Back.RED + Style.BRIGHT + Fore.YELLOW
PIECES[x] = [np.array([[0, x, 0, x],
                       [x, x, x, x]]),
             np.array([[0, 0, x, 0],
                       [x, x, x, x]])]

# Red
x = 5
COLORS[x] = Style.BRIGHT + Fore.RED
PIECES[x] = [np.array([[x, x, x, x],
                       [x, 0, 0, x]]),
             np.array([[0, 0, 0, x],
                       [x, x, x, x]])]

# Blue
x = 6
COLORS[x] = Fore.BLUE
PIECES[x] = [np.array([[0, x, 0],
                       [x, x, x]]),
             np.array([[x, x, x],
                       [x, 0, x]])]

# Purple
x = 7
COLORS[x] = Back.RED + Fore.BLUE
PIECES[x] = [np.array([[x, 0, 0],
                       [x, x, x]]),
             np.array([[x, x, x],
                       [0, x, x]])]

# Light blue
x = 8
COLORS[x] = Style.BRIGHT + Fore.BLUE
PIECES[x] = [np.array([[x, x, x, x],
                       [x, 0, 0, 0]]),
             np.array([[x, 0, x, 0],
                       [x, x, x, x]])]

# Pink
x = 9
COLORS[x] = Back.WHITE + Style.BRIGHT + Fore.RED
PIECES[x] = [np.array([[0, x, 0, 0],
                       [x, x, x, x]]),
             np.array([[x, x, x, x],
                       [x, x, 0, 0]])]

# Cyan
x = 10
COLORS[x] = Style.BRIGHT + Fore.CYAN
PIECES[x] = [np.array([[0, 0, x, 0],
                       [x, x, x, x]]),
             np.array([[x, x, x, x],
                       [0, x, x, 0]])]


def show(item):
    """
    Prints a colored version of item (a piece or board) to console.
    """
    nrows, ncols = item.shape
    for row in range(nrows):
        for col in range(ncols):
            try:
                print(COLORS[item[row][col]] + u'\u25CF ', end='')
            except UnicodeEncodeError:
                # This exception can occur when redirecting stdout.
                # In that case, just print without color
                print(COLORS[item[row][col]] + 'X ', end='')
        print()

def fits(board, piece):
    """
    Returns True if the piece can be placed on the board without overlapping
    any existing pieces.
    NOTE: Board must have same shape as piece.
    e.g.:
    board: 0 2 0
           0 0 0

    piece: 1 0 1
           1 1 1

    fits(board, piece) --> True

    This routine works by taking the bitwise-or of board and piece matrices
    and counting the resulting number of non-zero items.  A piece "fits" when
    the count of non-zero items increases by the number of non-zero items in
    the piece.  I.e., all locations in the piece were OR'd with zero and have
    become non-zero.
    """
    board_count = np.count_nonzero(board)
    piece_count = np.count_nonzero(piece)
    comb_count = np.count_nonzero(np.bitwise_or(board, piece))
    return comb_count == board_count + piece_count

def solve(board, pieces, depth=0):
    """

    Given a board and list of pieces, iterates over the board placing each
    piece in each of its two possible orientations at all possible locations
    on the board.

    Arguments:
        board: a 2D numpy array
        pieces: a list of 2-element arrays.  Each element is a 2D numpy array defining the piece shape.

    """
    retval = 0
    brows, bcols = board.shape

    if (depth == len(pieces)) and (np.count_nonzero(board) == 50):
        # all pieces placed and no empty spaces means done
        print("Found solution:")
        print(board)
        show(board)
        return 1

    for piece in pieces[depth]:
        for _rot_count in range(4):
            prows, pcols = piece.shape

            for row in range((brows - prows) + 1):
                for col in range((bcols - pcols) + 1):
                    sub_board = board[row:row+prows, col:col+pcols]
                    if fits(sub_board, piece):
                        # add piece to board and call solve with new board
                        new_board = np.copy(board)
                        new_board[row:row+prows, col:col+pcols] |= piece

                        if fails(new_board):
                            pass
                        else:
                            retval += solve(new_board, pieces, depth+1)

            # Rotate the piece
            piece = np.rot90(piece)

    return retval

def get_region_size(board, row, col):
    """
    Performs a DFS to find all contiguous empty cells in the region containing
    the cell (row,col).  Returns the size of the region.
    e.g.
    board = 0 1 0 1 0
            0 1 1 1 0
            0 0 0 0 0

    get_region_size(board, 0, 0) --> 9
    get_region_size(board, 0, 2) --> 1

    Note that "contiguous" only includes cardinal directions (up, down, left,
    right) and not diagonals, since the playing pieces have no diagonals.

    """
    retval = 0
    if board[row][col] == 0:
        board[row][col] = '3'
        retval += 1

        brows, bcols = board.shape
        # Up
        if (row - 1) >= 0:
            retval += get_region_size(board, row - 1, col)
        # Down
        if (row + 1) < brows:
            retval += get_region_size(board, row + 1, col)
        # Left
        if (col - 1) >= 0:
            retval += get_region_size(board, row, col - 1)
        # Right
        if (col + 1) < bcols:
            retval += get_region_size(board, row, col + 1)

    return retval


def fails(board):
    """
    Returns True if board can't win because there are contiguous regions of
    1-3 empty spaces.  These spaces can't be filled by any piece.
    e.g.:
        0 1 0 1 0
        0 1 1 1 0
        0 0 0 0 0
        
        Piece 1 has been placed on a small board.
        This board fails because the '0' at row 0 col 2 can never be filled
        by any piece since there are no pieces that can be used to fill only a
        single space.

    This routine is used as a heuristic to eliminate obviously-failing boards
    from the search space in order to speed up the brute force search.
    """
    new_board = np.copy(board)
    brows, bcols = board.shape
    for row in range(brows):
        for col in range(bcols):
            if new_board[row][col] == 0:
                # Found new region of one or more zeroes at (row, col)
                # Determine how large the region is
                region_size = get_region_size(new_board, row, col)
                if region_size < 4:
                    # Since there are no pieces are smaller than 4 we know
                    # this region can't be filled by any board piece, so it
                    # fails.

                    # NOTE: This is not a guarantee that the 4 spaces match a
                    # possible shape.
                    return True
    return False

def main():
    # board = BLANK_BOARD
    board = BOARD_120

    # Remove any pieces that are already placed on the board from the list of
    # available pieces before invoking the solver.
    used_pieces = np.unique(board)
    for piece_id in used_pieces:
        if piece_id != 0:
            del PIECES[piece_id]
            msg = "Removing piece %d from available pieces since it's already on the board"
            print(msg % (piece_id))
    pieces = list(PIECES.values())

    print("Initial board:")
    print(board)
    show(board)

    print("Running solver...")
    start_time = datetime.datetime.now()
    count = solve(BOARD_120, pieces)
    end_time = datetime.datetime.now()
    dt_sec = (end_time - start_time).total_seconds()
    print("Found %d solutions in %f seconds" % (count, dt_sec))


if __name__ == '__main__':
    main()
