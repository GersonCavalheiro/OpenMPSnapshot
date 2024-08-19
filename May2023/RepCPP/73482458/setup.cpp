#include <string>
#include <cstdlib>
#include <iostream>
#include "heat.hpp"


void initialize(int argc, char *argv[], Field& current,
Field& previous, int& nsteps, ParallelData parallel)
{



int rows = 2000;             
int cols = 2000;

std::string input_file;        

bool read_file = 0;

nsteps = 500;

switch (argc) {
case 1:

break;
case 2:

input_file = argv[1];
read_file = true;
break;
case 3:

input_file = argv[1];
read_file = true;


nsteps = std::atoi(argv[2]);
break;
case 4:

rows = std::atoi(argv[1]);
cols = std::atoi(argv[2]);

nsteps = std::atoi(argv[3]);
break;
default:
std::cout << "Unsupported number of command line arguments" << std::endl;
exit(-1);
}

if (read_file) {
#pragma omp single
{
if (0 == parallel.rank)
std::cout << "Reading input from " + input_file << std::endl;
read_field(current, input_file, parallel);
}
} else {
#pragma omp single
current.setup(rows, cols, parallel);

current.generate(parallel);
}

#pragma omp single
previous = current;

}
