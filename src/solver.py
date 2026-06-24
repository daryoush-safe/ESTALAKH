from copy import deepcopy
from itertools import combinations
from typing import List, Optional, Set, Tuple

SIZE = 9
BOX_SIZE = 3


class SudokuException(Exception):
    pass


def str2grid(s: str) -> List[List[int]]:
    grid = []
    for i in range(SIZE):
        row = []
        for j in range(SIZE):
            ch = s[i * SIZE + j]
            row.append(0 if ch in "0." else int(ch))
        grid.append(row)
    return grid


def grid2str(grid: List[List[int]]) -> str:
    return "".join(str(grid[i][j]) for i in range(SIZE) for j in range(SIZE))


def grid_equal(a: List[List[int]], b: List[List[int]]) -> bool:
    n = len(a)
    if n != len(b):
        return False
    return all(a[i][j] == b[i][j] for i in range(n) for j in range(len(a[i])))


def print_grid(grid: List[List[int]]) -> None:
    out = "-" * SIZE * 2 + "\n"
    for i, row in enumerate(grid):
        out += str(row[0])
        for j in range(1, len(row)):
            out += "," if j % BOX_SIZE != 0 else "|"
            out += str(row[j])
        out += "\n"
        if (i + 1) % BOX_SIZE == 0:
            out += "-" * SIZE * 2 + "\n"
    print(out[:-1])


class Sudoku:
    def __init__(self, grid: List[List[int]]):
        n = len(grid)
        assert len(grid[0]) == n, (
            "Grid is not square. n_rows=%d, n_columns=%d" % (n, len(grid[0]))
        )
        self.grid = grid
        self.n = n
        # create a grid of viable candidates for each position
        candidates = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(self.find_options(i, j) if grid[i][j] == 0 else set())
            candidates.append(row)
        self.candidates = candidates

    def __repr__(self) -> str:
        return "".join(str(row) + "\n" for row in self.grid)

    def get_row(self, r: int) -> List[int]:
        return self.grid[r]

    def get_rows_inds(self) -> List[List[Tuple[int, int]]]:
        return [[(i, j) for j in range(self.n)] for i in range(self.n)]

    def get_col(self, c: int) -> List[int]:
        return [row[c] for row in self.grid]

    def get_cols_inds(self) -> List[List[Tuple[int, int]]]:
        return [[(i, j) for i in range(self.n)] for j in range(self.n)]

    def get_box_inds(self, r: int, c: int) -> List[Tuple[int, int]]:
        inds_box = []
        i0 = (r // BOX_SIZE) * BOX_SIZE  # get first row index
        j0 = (c // BOX_SIZE) * BOX_SIZE  # get first column index
        for i in range(i0, i0 + BOX_SIZE):
            for j in range(j0, j0 + BOX_SIZE):
                inds_box.append((i, j))
        return inds_box

    def get_boxes_inds(self) -> List[List[Tuple[int, int]]]:
        inds_box = []
        for i0 in range(0, self.n, BOX_SIZE):
            for j0 in range(0, self.n, BOX_SIZE):
                inds_box.append(self.get_box_inds(i0, j0))
        return inds_box

    def get_box(self, r: int, c: int) -> List[int]:
        return [self.grid[i][j] for i, j in self.get_box_inds(r, c)]

    def get_neighbour_blocks(self, r: int, c: int) -> List[List[Tuple[int, int]]]:
        inds_row = [(r, j) for j in range(self.n)]
        inds_col = [(i, c) for i in range(self.n)]
        inds_box = self.get_box_inds(r, c)
        return [inds_row, inds_col, inds_box]

    def get_neighbour_inds(self, r: int, c: int) -> List[Tuple[int, int]]:
        blocks = self.get_neighbour_blocks(r, c)
        return list(set().union(*blocks))

    def find_options(self, r: int, c: int) -> Set[int]:
        nums = set(range(1, SIZE + 1))
        used = set(self.get_row(r)) | set(self.get_col(c)) | set(self.get_box(r, c))
        return nums.difference(used)

    @staticmethod
    def counting(arr: List[int], m: int = SIZE) -> List[int]:
        count = [0] * (m + 1)
        for x in arr:
            count[x] += 1
        return count

    @staticmethod
    def all_unique(arr: List[int], m: int = SIZE) -> bool:
        count = Sudoku.counting(arr, m=m)
        return all(c == 1 for c in count[1:])  # ignore 0

    @staticmethod
    def no_duplicates(arr: List[int]) -> bool:
        count = Sudoku.counting(arr)
        return all(c <= 1 for c in count[1:])  # exclude 0

    @staticmethod
    def all_exist(arr: List[int]) -> Tuple[bool, Optional[int]]:
        count = Sudoku.counting(arr)
        for num, c in enumerate(count[1:]):  # exclude 0
            if c == 0:
                return False, num + 1
        return True, None

    def check_done(self) -> bool:
        for i in range(self.n):
            if not Sudoku.all_unique(self.get_row(i)):
                return False
        for j in range(self.n):
            if not Sudoku.all_unique(self.get_col(j)):
                return False
        for i0 in range(0, self.n, BOX_SIZE):
            for j0 in range(0, self.n, BOX_SIZE):
                if not Sudoku.all_unique(self.get_box(i0, j0)):
                    return False
        return True

    def get_candidates(self, indices: List[Tuple[int, int]]) -> Set[int]:
        candidates: Set[int] = set()
        for i, j in indices:
            candidates |= self.candidates[i][j]
        return candidates

    def check_possible(self) -> bool:
        row_inds = self.get_rows_inds()
        cols_inds = self.get_cols_inds()
        type_ = ["row", "column"]
        for t, inds_set in enumerate([row_inds, cols_inds]):
            for k, inds in enumerate(inds_set):
                self.assert_possible(inds, type_=type_[t], k=k)
        self.assert_possible_boxes()
        return True

    def assert_possible(self, indices: List[Tuple[int, int]], type_="indices", k=0) -> bool:
        arr = [self.grid[i][j] for i, j in indices]
        if not Sudoku.no_duplicates(arr):
            raise SudokuException("Duplicate values in %s %d" % (type_, k))
        arr += list(self.get_candidates(indices))
        possible, missing_num = Sudoku.all_exist(arr)
        if not possible:
            raise SudokuException("%d not placeable in %s %d" % (missing_num, type_, k))
        return True

    def assert_possible_boxes(self) -> bool:
        for i0 in range(0, self.n, BOX_SIZE):
            for j0 in range(0, self.n, BOX_SIZE):
                arr = self.get_box(i0, j0)[:]
                if not Sudoku.no_duplicates(arr):
                    raise SudokuException("Duplicate values in box (%d, %d)" % (i0, j0))
                for i in range(i0, i0 + BOX_SIZE):
                    for j in range(j0, j0 + BOX_SIZE):
                        arr += list(self.candidates[i][j])
                possible, missing_num = Sudoku.all_exist(arr)
                if not possible:
                    raise SudokuException(
                        "%d not placeable in box (%d, %d)" % (missing_num, i0, j0)
                    )
        return True

    def place_and_erase(self, r: int, c: int, x: int, constraint_prop: bool = True) -> None:
        self.grid[r][c] = x
        self.candidates[r][c] = set()
        inds_neighbours = self.get_neighbour_inds(r, c)
        erased = [(r, c)]
        erased += self.erase([x], inds_neighbours, [])
        while erased and constraint_prop:
            i, j = erased.pop()
            erased += self.apply_strategies(i, j)

    def erase(
        self,
        nums: List[int],
        indices: List[Tuple[int, int]],
        keep: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        erased = []
        for i, j in indices:
            if (i, j) in keep:
                continue
            edited = False
            for x in nums:
                if x in self.candidates[i][j]:
                    self.candidates[i][j].remove(x)
                    edited = True
            if edited:
                erased.append((i, j))
        return erased

    def apply_strategies(self, i: int, j: int) -> List[Tuple[int, int]]:
        erased = []
        for inds in self.get_neighbour_blocks(i, j):
            uniques = self.get_unique(inds, type=[1, 2, 3])
            for inds_combo, combo in uniques:
                self.set_candidates(combo, inds_combo)
                erased += self.erase(combo, inds, inds_combo)
        inds_box = self.get_box_inds(i, j)
        for line, inds_pointer, num in self.pointing_combos(inds_box):
            erased += self.erase(num, line, inds_pointer)
        return erased

    def set_candidates(
        self, nums: List[int], indices: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        erased = []
        for i, j in indices:
            # beware triples where the whole triple is not in each box
            old = self.candidates[i][j].intersection(nums)
            if self.candidates[i][j] != old:
                self.candidates[i][j] = old.copy()
                erased.append((i, j))
        return erased

    def count_candidates(self, indices: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        count: List[List[Tuple[int, int]]] = [[] for _ in range(self.n + 1)]
        for i, j in indices:
            for num in self.candidates[i][j]:
                count[num].append((i, j))
        return count

    def get_unique(self, indices: List[Tuple[int, int]], type=(0, 1, 2)):
        groups = self.count_candidates(indices)
        uniques = []
        uniques_temp = {2: [], 3: []}
        for num, group_inds in enumerate(groups):
            c = len(group_inds)
            if c == 1 and (1 in type):
                uniques.append((group_inds, [num]))
            if c == 2 and ((2 in type) or (3 in type)):
                uniques_temp[2].append(num)
            if c == 3 and (3 in type):
                uniques_temp[3].append(num)
        uniques_temp[3] += uniques_temp[2]
        for c in [2, 3]:
            if c not in type:
                continue
            for combo in list(combinations(uniques_temp[c], c)):
                group_inds = set(groups[combo[0]])
                for k in range(1, c):
                    group_inds = group_inds | set(groups[combo[k]])
                if len(group_inds) == c:
                    uniques.append((list(group_inds), combo))
        return uniques

    def pointing_combos(self, inds_box: List[Tuple[int, int]]):
        groups = self.count_candidates(inds_box)
        pointers = []
        for num, indices in enumerate(groups):
            if len(indices) in (2, 3):
                row_same, col_same = True, True
                i0, j0 = indices[0]
                for i, j in indices[1:]:
                    row_same = row_same and (i == i0)
                    col_same = col_same and (j == j0)
                if row_same:
                    line = [(i0, j) for j in range(self.n)]
                    pointers.append((line, indices, [num]))
                if col_same:
                    line = [(i, j0) for i in range(self.n)]
                    pointers.append((line, indices, [num]))
        return pointers

    def flush_candidates(self) -> None:
        inds_box = self.get_boxes_inds()
        inds_row = self.get_rows_inds()
        inds_col = self.get_cols_inds()
        inds_set = inds_box + inds_row + inds_col
        for inds in inds_set:
            uniques = self.get_unique(inds, type=[1, 2])
            for inds_combo, combo in uniques:
                self.erase(combo, inds, inds_combo)
                self.set_candidates(combo, inds_combo)
        for inds in inds_box:
            for line, inds_pointer, num in self.pointing_combos(inds):
                self.erase(num, line, inds_pointer)


def get_least_candidates(game: Sudoku) -> Tuple[int, Tuple[int, int]]:
    least = (game.n + 1, (0, 0))
    for i in range(game.n):
        for j in range(game.n):
            if game.grid[i][j] == 0:
                c = len(game.candidates[i][j])
                if c < least[0]:
                    least = (c, (i, j))
    return least


def solve_sudoku(
    grid: List[List[int]], verbose: bool = False, all_solutions: bool = False
):
    def solve(game: Sudoku, depth: int = 0) -> bool:
        nonlocal calls, depth_max
        calls += 1
        depth_max = max(depth, depth_max)
        solved = False
        while not solved:
            solved = True
            edited = False
            for i in range(game.n):
                for j in range(game.n):
                    if game.grid[i][j] == 0:
                        solved = False
                        options = game.candidates[i][j]
                        if len(options) == 0:
                            return False
                        elif len(options) == 1:
                            game.place_and_erase(i, j, next(iter(options)))
                            edited = True
            if not edited:
                if solved:
                    solution_set.append(deepcopy(game.grid))
                    return True
                _, (i, j) = get_least_candidates(game)
                options = game.candidates[i][j]
                for guess in options:
                    game_next = deepcopy(game)
                    game_next.place_and_erase(i, j, guess)
                    solved = solve(game_next, depth=depth + 1)
                    if solved and not all_solutions:
                        break
                return solved
        return solved

    calls, depth_max = 0, 0
    solution_set: List[List[List[int]]] = []

    game = Sudoku(deepcopy(grid))
    game.flush_candidates()
    game.check_possible()

    if verbose:
        print("solving...")
    solve(game, depth=0)
    solved = len(solution_set) >= 1

    info = {
        "calls": calls,
        "max depth": depth_max,
        "nsolutions": len(solution_set),
    }
    return solution_set, solved, info


def solve(grid: List[List[int]]) -> Optional[List[List[int]]]:
    try:
        solution_set, solved, _ = solve_sudoku(grid, verbose=False)
    except SudokuException:
        return None
    return solution_set[0] if solved else None


if __name__ == "__main__":
    import time

    puzzle = "400009200000010080005400006004200001050030060700005300500007600090060000002800007"
    grid = str2grid(puzzle)

    print_grid(grid)
    t0 = time.time()
    solution_set, done, info = solve_sudoku(grid, verbose=True)
    dt = time.time() - t0

    print("total time: %.5fs" % dt)
    for key in ("calls", "max depth", "nsolutions"):
        print("%-14s: %d" % (key, info[key]))
    if done:
        print_grid(solution_set[0])
