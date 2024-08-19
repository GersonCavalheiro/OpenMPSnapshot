
#pragma once
#ifndef PUZZLE_HPP
#define PUZZLE_HPP
#include <vector>
#include <map>
#include "Shape.hpp"
#include "Polyomino.hpp"
#include "GridType.hpp"
#include "DlxCell.hpp"
#include "Timing.hpp"

class Puzzle {
public:
Puzzle();
Shape board;
std::vector<Polyomino> polyominoes;
GridType *grid;

struct DLX {
std::vector<DlxRow> rows;
std::vector<DlxColumn> columns;
std::vector<DlxCell> cells;

DLX() = default;

DLX(const DLX &other);
DLX& operator=(const DLX &other);
void swap(DLX &other) noexcept;
} dlx;

std::map<Coord, int> coordMap;

void buildDlxRows();
void buildDlxColumns();
void buildDlxCells();

DlxColumn *minfit();
void dlxSolve();
int enterBranch(int row);
void leaveBranch(int removedRowCount);

void reduce();
int targetLevel;
int numSolution;
std::vector<std::vector<int>> solutions;
int numRows;
long long attempts;
long long dlxCounter;
bool saveSolution;

private:
std::vector<DlxCell *> removedRows;
std::vector<int> solutionStack;
void dlxSolveRecursive(int lv);
};

#endif
