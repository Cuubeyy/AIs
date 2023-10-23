from random import sample

GRID_SIZE = 9
NUMBERS = list(range(1, GRID_SIZE + 1))


def create_field():
    field = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    fill_field(field)
    return field


def is_valid_move(field, row, col, num):
    # Check if the number is not already in the current row, column, or 3x3 subgrid
    return (
            num not in field[row] and
            num not in [field[i][col] for i in range(GRID_SIZE)] and
            num not in [field[i][j] for i in range(row - row % 3, row - row % 3 + 3)
                        for j in range(col - col % 3, col - col % 3 + 3)]
    )


def fill_field(field):
    empty_cells = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)]
    sample(empty_cells, len(empty_cells))  # Shuffle the list of empty cells
    if solve_sudoku(field, empty_cells):
        return field
    else:
        raise ValueError("No solution found for the Sudoku puzzle.")


def solve_sudoku(field, empty_cells):
    if not empty_cells:
        return True

    row, col = empty_cells.pop()

    for num in sample(NUMBERS, len(NUMBERS)):
        if is_valid_move(field, row, col, num):
            field[row][col] = num
            if solve_sudoku(field, empty_cells):
                return True
            field[row][col] = 0

    empty_cells.append((row, col))
    return False


if __name__ == "__main__":
    field = create_field()

    for row in field:
        print(row)
