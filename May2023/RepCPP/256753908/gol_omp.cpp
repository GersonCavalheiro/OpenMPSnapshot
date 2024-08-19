#include <iostream>
#include <vector>
#include <ctime>
#include <chrono>
#include <thread>
#include <omp.h>
#include "gol.hpp"

using namespace std;


class GameOfLifeOmp : public virtual GameOfLife {

private:
int nw = 1;

public:
GameOfLifeOmp(int rows, int cols, int workers) : GameOfLife(rows, cols) {
nw = workers;
}

void step() {
vector<vector<bool>> tmp(n_rows, vector<bool>(n_cols));
std::copy(matrix.begin(), matrix.end(), tmp.begin());
size_t i, j = 0;
#pragma omp parallel for num_threads(nw) private(i, j)
for (i = 1; i < n_rows-1; i++) {
#pragma omp simd
for (j = 1; j < n_cols-1; j++) {
int neighbours_alive = matrix[i-1][j-1] + matrix[i-1][j] + matrix[i-1][j+1] +
matrix[i][j-1]                    + matrix[i][j+1] +
matrix[i+1][j-1] + matrix[i+1][j] + matrix[i+1][j+1];
tmp[i][j] = (neighbours_alive==3) || (neighbours_alive==2 && matrix[i][j]==1);
}
}
matrix = tmp;
tmp.clear();
}

};


int main(int argc, char* argv[]) {
if(argc == 1) cerr<<"Usage: ./gol_omp 'rows' 'columns' 'number of iterations' 'nw (number of workers)'"<<endl;

int rows = atoi(argv[1]);
int cols = atoi(argv[2]);
int iters = atoi(argv[3]);
int nw = atoi(argv[4]);
if (nw == -1) nw = std::thread::hardware_concurrency();
cout<<rows<<"x"<<cols<<" matrix, "
<<(rows * cols)<<" cells, "
<<iters<<" iterations, "
<<nw<<" workers."<<endl;

GameOfLifeOmp* gol = new GameOfLifeOmp(rows, cols, nw);
gol->toString();

double begin = omp_get_wtime();
for (size_t i = 0; i < iters; i++) gol->step();
double duration = omp_get_wtime() - begin;

gol->toString();
cout<<"Total execution time: "<<duration*1000000<<" usec"<<endl;

return 0;
}
