#include <iostream>
#include <random>
#include <functional>
#include <vector>
#include <ctime>
#include <chrono>
#include <thread>
#include <mutex>
#include "gol.hpp"
#include "barrier.hpp"

using namespace std;
using namespace std::chrono;


class GameOfLifePar : public virtual GameOfLife {

private:
int nw = 1;

public:
GameOfLifePar(int rows, int cols, int workers) : GameOfLife(rows, cols) {
nw = workers;
}

void step() {
vector<thread> workers;

vector<Barrier> barriers;
for(auto &b : barriers) b.set_t(nw);

vector<vector<bool>> tmp(n_rows, vector<bool>(n_cols));
std::copy(matrix.begin(), matrix.end(), tmp.begin());
int chunk_size = (n_rows-2) / nw;
int rest = (n_rows-2) % nw;
for (int k = 1; k < n_rows-2; k += chunk_size) {
workers.push_back(thread([&](vector<vector<bool>> M, int idx) {
int chunk_idx = idx + chunk_size - 1;
int start_idx = idx;
int end_idx = (idx != n_rows-1) ? (chunk_idx) : (chunk_idx + rest);
cout<<"Thread [start_idx="<<start_idx<<", end_idx="<<end_idx<<"]"<<endl;

for (size_t i = start_idx; i <= end_idx; i++) {
#pragma GCC ivdep
for (size_t j = 1; j < n_cols-1; j++) {
int neighbours_alive = M[i-1][j-1] + M[i-1][j] + M[i-1][j+1] +
M[i  ][j-1]             + M[i  ][j+1] +
M[i+1][j-1] + M[i+1][j] + M[i+1][j+1];
tmp[i][j] = (neighbours_alive==3) || (neighbours_alive==2 && M[i][j]==1);
}
barriers[i].wait();
}
}, matrix, k));
}

for (thread &t : workers) t.join();
matrix = tmp;
tmp.clear();
workers.clear();
}

};


int main(int argc, char* argv[]) {
if(argc == 1) cerr<<"Usage: ./gol_par 'rows' 'columns' 'number of iterations' 'nw (number of workers)'"<<endl;

int rows = atoi(argv[1]);
int cols = atoi(argv[2]);
int iters = atoi(argv[3]);
int nw = atoi(argv[4]);
if (nw == -1) nw = std::thread::hardware_concurrency();
cout<<rows<<"x"<<cols<<" matrix, "
<<(rows * cols)<<" cells, "
<<iters<<" iterations, "
<<nw<<" workers."<<endl;

GameOfLifePar* gol = new GameOfLifePar(rows, cols, nw);
gol->toString();

auto begin = high_resolution_clock::now();
for (size_t i = 0; i < iters; i++) gol->step();
auto end = high_resolution_clock::now();
auto duration = duration_cast<microseconds>(end - begin);

gol->toString();
cout<<"Total execution time: "<<duration.count()<<" usec"<<endl;

return 0;
}
