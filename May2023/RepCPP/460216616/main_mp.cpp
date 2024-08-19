#include <iostream>
#include "kd_tree.hpp"
#include "kd_tree_mp.cpp"
#include <random>
#include <array>
#include <omp.h>
#include <chrono>

#define MAX 10000000

template<typename T>
std::vector<struct kpoint<T>> generatePoints(const int ndim, const int npoints){

srand((unsigned) time(0));

std::vector<struct kpoint<T>> points;
struct kpoint<T> temp;


for(auto i=0; i<npoints; ++i){
for(auto j=0; j<ndim; j++){
temp.set_point(j, (T)rand() / MAX);
}
points.push_back(temp);
}

return points;
}

int main(int argc, char **argv){

int n = std::stoi(argv[1]);

int ndim = N_DIM;

auto points = generatePoints<int>(ndim, n);

std::chrono::time_point<std::chrono::high_resolution_clock> start;
std::chrono::time_point<std::chrono::high_resolution_clock> end;
struct kdnode<int>* kdtree;
double mp_start, mp_end;

mp_start = omp_get_wtime();

#pragma omp parallel shared(points) private(start, end)
{
#pragma omp single
{

kdtree = build_kdtree<int>(points, ndim, 1);

}
}

mp_end = omp_get_wtime();


std::cout<<mp_end-mp_start<<std::endl;



return 0;
}

