#pragma once
#include <string>
#include "matrix.hpp"
#include <mpi.h>

struct ParallelData {
int size;            
int rank;
int nup, ndown;      

ParallelData() {      

MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

nup = rank - 1;
ndown = rank + 1;

if (nup < 0) {
nup = MPI_PROC_NULL;
}
if (ndown > size - 1) {
ndown = MPI_PROC_NULL;
}
};

};

struct Field {
int nx;                     
int ny;
int nx_full;                
int ny_full;                
double dx = 0.01;           
double dy = 0.01;

Matrix<double> temperature;

void setup(int nx_in, int ny_in, ParallelData parallel);

void generate(ParallelData parallel);

double& operator()(int i, int j) {return temperature(i, j);}

const double& operator()(int i, int j) const {return temperature(i, j);}

};

void initialize(int argc, char *argv[], Field& current,
Field& previous, int& nsteps, ParallelData parallel);

void exchange(Field& field, const ParallelData parallel);

void evolve(Field& curr, const Field& prev, const double a, const double dt);

void write_field(const Field& field, const int iter, const ParallelData parallel);

void read_field(Field& field, std::string filename,
const ParallelData parallel);

double average(const Field& field, const ParallelData parallel);
