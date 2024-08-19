#include <chrono>
#include <omp.h>

#ifndef LOF__GOLOPENMP_H_
#define LOF__GOLOPENMP_H_

class GoLOpenMP {
private:
bool *states;
bool *statesTmp;
const int nStep;
const int nCol;
const int nRow;
const int nWorker;

public:
~GoLOpenMP() {
delete (states);
delete (statesTmp);
}
GoLOpenMP(const int userStep, const int userRow, const int userCol, const int seed, const int nw)
: nCol(userCol),
nRow(userRow),
states(new bool[userRow * userCol]{}),
statesTmp(new bool[userRow * userCol]{}),
nStep(userStep),
nWorker(nw) {
std::srand(seed);

for (int row = 1; row < nRow - 1; ++row) {
for (int col = 1; col < nCol - 1; ++col) {
states[nCol * row + col] = std::rand() % 2;
}
}
}
void PrintStates() {
for (int row = 0; row < nRow; ++row) {
for (int col = 0; col < nCol; ++col) {
std::cout << states[row * nCol + col] << " ";
}
std::cout << std::endl;
}
std::cout << std::endl;
}

void Run() {

for (int k = 0; k < nStep; k++) {
#pragma omp parallel for num_threads(nWorker)
for (int row = 1; row < nRow - 1; row++) {
#pragma omp parallel for num_threads(nWorker)
for (int col = 1; col < nCol - 1; col++) {
const int sumNeighbours = states[(row - 1) * nCol + col - 1] +
states[(row - 1) * nCol + col] +
states[(row - 1) * nCol + col + 1] +
states[row * nCol + col - 1] +
states[row * nCol + col + 1] +
states[(row + 1) * nCol + col - 1] +
states[(row + 1) * nCol + col] +
states[(row + 1) * nCol + col + 1];

statesTmp[row * nCol + col] = (sumNeighbours == 3) || (sumNeighbours == 2 && states[row * nCol + col]);

}

}
std::swap(states, statesTmp);
PrintStates();
}
}

void RunWithTime() {
auto tstart = std::chrono::high_resolution_clock::now();
Run();
auto elapsed = std::chrono::high_resolution_clock::now() - tstart;
auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
std::cout << "OpenMP parallel implementation time " << msec << " msec number columns " << nCol << " number rows"
<< nRow << " nWorker" << nWorker << std::endl;
}

};

#endif 
