#include <stdio.h>
#include <getopt.h>
#include <math.h>
#include <string>
#include <iostream>

#include "adf.h"
#include <omp.h>

using namespace std;

typedef char mychar;

bool     print_results;
char    *input_file_name;
int      num_threads;
int      matrix_size = 512;
double   epsilon = 1.0;
int      num_iterations;


#define MAX(A,B) A>B ? A : B

#define OPTIMAL






class Solver
{
public:
Solver(int _size, double _epsilon);
~Solver();
void Solve();
void PrintSummery();
void Finish();

int        size;                
long       iterations;			
double     epsilon;         	
double   **grid;          		
double   **tempGrid;      		
double   **result;        		

double    *grid_matrix;
double    *tempGrid_matrix;

};


Solver :: Solver(int _size, double _epsilon)
{
grid = NULL;
tempGrid = NULL;
result = NULL;

size = _size;
epsilon = _epsilon;
iterations = 0;

grid_matrix = new double[size*size];
tempGrid_matrix = new double[size*size];

grid = new double*[size];
tempGrid = new double*[size];
for (int j = 0; j < size; j ++) {
grid[j] = & grid_matrix[j*size];
tempGrid[j] = & tempGrid_matrix[j*size];
}

int val;

for (int i = 0; i < size; i ++) {
for (int j = 0; j < size; j ++) {
val = rand()%1000;
grid[i][j] = (double) val;
tempGrid[i][j]  = (double) val;
}
}
}

Solver :: ~Solver()
{
delete [] grid_matrix;
delete [] tempGrid_matrix;

delete [] grid;
delete [] tempGrid;
}


void Solver::Solve()
{
static double gerror = 0.0;
bool converged = false;

int i;

while(!converged) {

iterations++;

#pragma omp parallel shared(converged, gerror)
{
#ifndef OPTIMAL
TRANSACTION_BEGIN(1,RW)
#endif
double lerror = 0.0;
int j;

#pragma omp for schedule(static)
for (i = 1; i < size - 1; i ++) {
if (iterations % num_iterations != 0) {
for (j = 1; j < size - 1; j ++) {
tempGrid[i][j] = 0.25 * (grid[i-1][j] + grid[i+1][j] + grid[i][j-1] + grid[i][j+1]);
}
}
else {
for (j = 1; j < size - 1; j ++) {
tempGrid[i][j] = 0.25 * (grid[i-1][j] + grid[i+1][j] + grid[i][j-1] + grid[i][j+1]);
lerror = MAX (lerror, fabs(tempGrid[i][j] - grid[i][j]));
}
}
} 

if (iterations % num_iterations == 0) {
#ifdef OPTIMAL
TRANSACTION_BEGIN(1,RW)
#endif
gerror = MAX(gerror, lerror);
#ifdef OPTIMAL
TRANSACTION_END
#endif
}
#ifndef OPTIMAL
TRANSACTION_END
#endif
} 

if (iterations % num_iterations == 0) {
if(gerror <= epsilon)
converged = true;
gerror = 0.0;
}

double ** t = grid;
grid = tempGrid;
tempGrid = t;
}

result = grid;
}


void Solver::PrintSummery()
{
printf("\nIterations: %ld\n", iterations);
}


void Solver::Finish()
{
if (print_results == true) {
FILE *result_file = fopen("result_omp.txt", "wb");
fprintf(result_file, "#Iterations: %ld #Result domain:\r\n", iterations);
for (int i = 0; i < size; i ++)
{
for (int j = 0; j < size; j ++)
{
fprintf(result_file, "%0.8f ", result[i][j]);
}
fprintf(result_file, "\r\n");
}
fclose(result_file);
}
}




void ParseCommandLine(int argc, char **argv)
{
char c;

input_file_name = (char*) "mediumcontent.txt";
num_threads = 1;
num_iterations = 50;
print_results = false;

while ((c = getopt (argc, argv, "hf:n:s:i:e:r")) != -1)
switch (c) {
case 'h':
printf("\nStructured Grid - ADF benchmark application\n"
"\n"
"Usage:\n"
"   structured_grid_omp [options ...]\n"
"\n"
"Options:\n"
"   -h\n"
"        Print this help message.\n"
"   -f <filename>\n"
"        Input file name.\n"
"   -s <int>\n"
"        Size of matrix.\n"
"   -e <double>\n"
"        Epsilon value.\n"
"   -i <int>\n"
"        Number of iterations between error check.\n"
"   -n <long>\n"
"        Number of worker threads. (default 1)\n"
"   -r \n"
"        print results to a result file\n"
"\n"
);
exit (0);
case 'f':
input_file_name = optarg;
break;
case 's':
matrix_size = atoi(optarg);
break;
case 'e':
epsilon = atof(optarg);
break;
case 'i':
num_iterations = atoi(optarg);
break;
case 'n':
num_threads = strtol(optarg, NULL, 10);
break;
case 'r':
print_results = true;
break;
case '?':
if (optopt == 'c')
fprintf (stderr, "Option -%c requires an argument.\n", optopt);
else if (isprint (optopt))
fprintf (stderr, "Unknown option `-%c'.\n", optopt);
else
fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
exit(1);
default:
exit(1);
}

if(input_file_name == NULL) {
printf("missing input file!\n");
exit(1);
}

printf ("\nStarting openmp Structured Grid.\n");
printf ("Matrix size: %d\tepsilon: %g\n", matrix_size, epsilon);
printf ("Running with %d threads.\n", num_threads);
printf ("=====================================================\n\n");
}









int main(int argc, char** argv)
{
ParseCommandLine(argc, argv);

Solver solver(matrix_size, epsilon);				

NonADF_init(num_threads);

omp_set_num_threads(num_threads);
omp_set_nested(1);

solver.Solve();										

solver.PrintSummery();

NonADF_terminate();

solver.Finish();									

return 0;
}


