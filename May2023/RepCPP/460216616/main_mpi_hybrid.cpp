#include <iostream>
#include "kd_tree.hpp"
#include "kd_tree_mpi_hybrid.cpp"
#include <random>
#include <array>
#include <chrono>
#include <mpi.h>
#include <omp.h>

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

std::string arg = argv[1];
std::size_t pos;
int n = std::stoi(arg, &pos);

int irank, size;
MPI_Comm comm; 

MPI_Init( &argc, &argv );


MPI_Comm_size( MPI_COMM_WORLD, &size );
MPI_Comm_rank( MPI_COMM_WORLD, &irank );



int ndim = N_DIM; 
int start_axis = 1;

auto points = generatePoints<int>(ndim, n);

std::chrono::time_point<std::chrono::high_resolution_clock> start;
std::chrono::time_point<std::chrono::high_resolution_clock> end;
double mpi_start, mpi_end;
struct kdnode<int>* kdtree;
int c_tag = 0;


#ifdef DEBUG
if(irank == 3){
kdtree = build_serial_kdtree<int>(points, ndim, start_axis);
std::string kd_str = serialize_node(kdtree);
std::cout<<"\nSERIAL KDTREE: "<<kd_str;
}
#endif



start = std::chrono::high_resolution_clock::now();


#pragma omp parallel shared(points) private(start, end)
{
#pragma omp single
{
kdtree = build_parallel_kdtree4<int>(points, ndim, start_axis, size, 0, MPI_COMM_WORLD, 1);
}
}


end = std::chrono::high_resolution_clock::now();


if( irank == 0){
std::chrono::duration<double> diff = end - start;

std::cout<<diff.count()<<std::endl;
}









#ifdef DEBUG
if(irank == 0){
std::string kd_str = serialize_node(kdtree);
std::cout<<"\nPARALLEL KDTREE: "<<kd_str;
}
#endif




MPI_Finalize();


return 0;
}
