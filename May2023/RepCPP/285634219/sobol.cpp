

#include <iostream>
#include <stdexcept>
#include <math.h>
#include "sobol.h"
#include "sobol_gold.h"
#include "sobol_gpu.h"

#define L1ERROR_TOLERANCE (1e-6)

void printHelp(int argc, char *argv[])
{
if (argc > 0)
{
std::cout << "\nUsage: " << argv[0] << " <options>\n\n";
}
else
{
std::cout << "\nUsage: <program name> <options>\n\n";
}

std::cout << "\t--vectors=M     specify number of vectors    (required)\n";
std::cout << "\t                The generator will output M vectors\n\n";
std::cout << "\t--dimensions=N  specify number of dimensions (required)\n";
std::cout << "\t                Each vector will consist of N components\n\n";
std::cout << std::endl;
}

int main(int argc, char *argv[])
{
if (argc != 4) {
printf("Usage: %s <number of vectors> <number of dimensions> <repeat>\n", argv[0]);
return 1;
}
int n_vectors = atoi(argv[1]);
int n_dimensions = atoi(argv[2]);
int repeat = atoi(argv[3]); 

std::cout << "Allocating CPU memory..." << std::endl;
unsigned int *h_directions = 0;
float        *h_outputCPU  = 0;
float        *h_outputGPU  = 0;

try
{
h_directions = new unsigned int [n_dimensions * n_directions];
h_outputCPU  = new float [n_vectors * n_dimensions];
h_outputGPU  = new float [n_vectors * n_dimensions];
}
catch (const std::exception &e)
{
std::cerr << "Caught exception: " << e.what() << std::endl;
std::cerr << "Unable to allocate CPU memory (try running with fewer vectors/dimensions)" << std::endl;
exit(EXIT_FAILURE);
}

std::cout << "Initializing direction numbers..." << std::endl;
initSobolDirectionVectors(n_dimensions, h_directions);

std::cout << "Executing QRNG on GPU..." << std::endl;

#pragma omp target data map(to: h_directions[0:n_dimensions * n_directions]) \
map(from: h_outputGPU[0:n_dimensions * n_vectors])
{
double ktime = sobolGPU(repeat, n_vectors, n_dimensions, h_directions, h_outputGPU);

std::cout << "Average kernel execution time: " << (ktime * 1e-9f) / repeat << " (s)\n";
}

std::cout << std::endl;
std::cout << "Executing QRNG on CPU..." << std::endl;
sobolCPU(n_vectors, n_dimensions, h_directions, h_outputCPU);

std::cout << "Checking results..." << std::endl;
float l1norm_diff = 0.0F;
float l1norm_ref  = 0.0F;
float l1error;

if (n_vectors == 1)
{
for (int d = 0, v = 0 ; d < n_dimensions ; d++)
{
float ref = h_outputCPU[d * n_vectors + v];
l1norm_diff += fabs(h_outputGPU[d * n_vectors + v] - ref);
l1norm_ref  += fabs(ref);
}

l1error = l1norm_diff;

if (l1norm_ref != 0)
{
std::cerr << "Error: L1-Norm of the reference is not zero (for single vector), golden generator appears broken\n";
}
else
{
std::cout << "L1-Error: " << l1error << std::endl;
}
}
else
{
for (int d = 0 ; d < n_dimensions ; d++)
{
for (int v = 0 ; v < n_vectors ; v++)
{
float ref = h_outputCPU[d * n_vectors + v];
l1norm_diff += fabs(h_outputGPU[d * n_vectors + v] - ref);
l1norm_ref  += fabs(ref);
}
}

l1error = l1norm_diff / l1norm_ref;

if (l1norm_ref == 0)
{
std::cerr << "Error: L1-Norm of the reference is zero, golden generator appears broken\n";
}
else
{
std::cout << "L1-Error: " << l1error << std::endl;
}
}

std::cout << "Shutting down..." << std::endl;
delete h_directions;
delete h_outputCPU;
delete h_outputGPU;

if (l1error < L1ERROR_TOLERANCE)
std::cout << "PASS" << std::endl;
else 
std::cout << "FAIL" << std::endl;

return 0;
}
