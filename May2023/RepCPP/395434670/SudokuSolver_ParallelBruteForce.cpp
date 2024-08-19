#include "SudokuSolver_ParallelBruteForce.hpp"
#include "SudokuSolver_SequentialBruteForce.hpp"
#include "termcolor.hpp"
#include <iostream>
#include <vector>
#include <omp.h>


SudokuSolver_ParallelBruteForce::SudokuSolver_ParallelBruteForce(SudokuBoard& board, bool print_message )
: SudokuSolver(board)
{
_mode = MODES::PARALLEL_BRUTEFORCE;
if (print_message)
{
std::cout << "\n" << "Parallel Sudoku solver using brute force algorithm starts, please wait..." << "\n";
}
}

void SudokuSolver_ParallelBruteForce::bootstrap()
{
if (_board_deque.size() == 0) { return; }

SudokuBoard board = _board_deque.front();

if (checkIfAllFilled(board)) { return; }

Position empty_cell_pos = find_empty(board);

int row = empty_cell_pos.first;
int col = empty_cell_pos.second;

for (int num = board.get_min_value(); num <= board.get_max_value(); ++num)
{
if (isValid(board, num, empty_cell_pos))
{
board.set_board_data(row, col, num);
_board_deque.push_back(board);
}
}

_board_deque.pop_front();
}

void SudokuSolver_ParallelBruteForce::bootstrap(SudokuBoardDeque& boardDeque, int indexOfRows)
{
if (boardDeque.size() == 0) { return; }

while (!checkIfRowFilled(boardDeque.front(), indexOfRows))
{
SudokuBoard board = boardDeque.front();

int empty_cell_col_index = find_empty_from_row(board, indexOfRows);

for (int num = board.get_min_value(); num <= board.get_max_value(); ++num)
{
Position empty_cell_pos = std::make_pair(indexOfRows, empty_cell_col_index);

if (isValid(board, num, empty_cell_pos))
{
board.set_board_data(indexOfRows, empty_cell_col_index, num);
boardDeque.push_back(board);
}
}

boardDeque.pop_front();
}
}

void SudokuSolver_ParallelBruteForce::solve_kernel_1()
{
_board_deque.push_back(_board);

int num_bootstraps = omp_get_num_threads();
#pragma omp parallel for schedule(static) default(none) shared(num_bootstraps)
for (int i = 0; i < num_bootstraps; ++i)
{
bootstrap();
}

int numberOfBoards = _board_deque.size();


std::vector<SudokuSolver_SequentialBruteForce> solvers;

#pragma omp parallel for schedule(static) default(none) shared(numberOfBoards, solvers)
for (int indexOfBoard = 0; indexOfBoard < numberOfBoards; ++indexOfBoard)
{
solvers.push_back(SudokuSolver_SequentialBruteForce(_board_deque[indexOfBoard], false));

if (_solved) { continue; }

solvers[indexOfBoard].set_mode(MODES::PARALLEL_BRUTEFORCE);

solvers[indexOfBoard].solve();

if (solvers[indexOfBoard].get_status() == true)
{
_solved = true;
_solution = solvers[indexOfBoard].get_solution();
}
}
}

void SudokuSolver_ParallelBruteForce::solve_kernel_2()
{
std::vector<SudokuBoardDeque> groupOfBoardDeques(_board.get_board_size(), SudokuBoardDeque(_board));
#pragma omp parallel default(none) shared(groupOfBoardDeques)
{	
int SIZE = groupOfBoardDeques.size();

#pragma omp for nowait schedule(static)
for (int i = 0; i < SIZE; ++i)
{
bootstrap(groupOfBoardDeques[i], i);
_board_deque.boardDeque.insert(_board_deque.boardDeque.end(),
groupOfBoardDeques[i].boardDeque.begin(),
groupOfBoardDeques[i].boardDeque.end());
}
}

int numberOfBoards = _board_deque.size();


std::vector<SudokuSolver_SequentialBruteForce> solvers;

#pragma omp parallel for schedule(static) default(none) shared(numberOfBoards, solvers)
for (int indexOfBoard = 0; indexOfBoard < numberOfBoards; ++indexOfBoard)
{	
solvers.push_back(SudokuSolver_SequentialBruteForce(_board_deque[indexOfBoard], false));

if (_solved) { continue; }

solvers[indexOfBoard].solve();

if (solvers[indexOfBoard].get_status() == true)
{
_solved = true;
_solution = solvers[indexOfBoard].get_solution();
}
}
}

void SudokuSolver_ParallelBruteForce::solve_bruteforce_seq(SudokuBoard& board, int row, int col)
{
if (_solved) { return; }

int BOARD_SIZE = board.get_board_size();

int abs_index = row * BOARD_SIZE + col;

if (abs_index >= board.get_num_total_cells())
{
_solved = true;
_solution = board;
return;
}

int row_next = (abs_index + 1) / BOARD_SIZE;
int col_next = (abs_index + 1) % BOARD_SIZE;

if (!isEmpty(board, row, col))
{   
solve_bruteforce_seq(board, row_next, col_next);
}
else
{
for (int num = board.get_min_value(); num <= board.get_max_value(); ++num)
{
Position pos = std::make_pair(row, col);

if (isValid(board, num, pos))
{
board.set_board_data(row, col, num);

if (isUnique(board, num, pos)) { num = BOARD_SIZE + 1; }   

solve_bruteforce_seq(board, row_next, col_next);

board.set_board_data(row, col, board.get_empty_cell_value());
}
}
}

_recursionDepth++;
}

void SudokuSolver_ParallelBruteForce::solve_bruteforce_par(SudokuBoard& board, int row, int col)
{
if (_solved) { return; }

int BOARD_SIZE = board.get_board_size();

int abs_index = row * BOARD_SIZE + col;

if (abs_index >= board.get_num_total_cells())
{
_solved = true;
_solution = board;
return;
}

int row_next = (abs_index + 1) / BOARD_SIZE;
int col_next = (abs_index + 1) % BOARD_SIZE;

if (!isEmpty(board, row, col))
{   
solve_bruteforce_par(board, row_next, col_next);
}
else
{
for (int num = board.get_min_value(); num <= board.get_max_value(); ++num)
{
Position pos = std::make_pair(row, col);

if (isValid(board, num, pos)) 
{
if (_recursionDepth > BOARD_SIZE)
{
board.set_board_data(row, col, num);

if (isUnique(board, num, pos)) { num = BOARD_SIZE + 1; }   

solve_bruteforce_seq(board, row_next, col_next);
}
else
{
SudokuBoard local_board(board);
local_board.set_board_data(row, col, num);

if (isUnique(board, num, pos)) { num = BOARD_SIZE + 1; }   

#pragma omp task default(none) firstprivate(local_board, row_next, col_next) 
solve_bruteforce_par(local_board, row_next, col_next);

}
}
}
}

_recursionDepth++;
}
