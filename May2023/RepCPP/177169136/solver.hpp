#pragma once

#include <string>

using namespace std;

struct LCSSolver {
int Solve(size_t, const string&, const string&);
int SolveLessMemory(size_t, const string&, const string&);
int SolveParallel(size_t, const string&, const string&);
int SolveParallelLessMemory(size_t, const string&, const string&);
};
