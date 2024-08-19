#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#ifdef LIFE3D_MPI
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#endif
#include "io.h"
#include "life3d.h"
int main(int argc, char *argv[]) {
#ifdef LIFE3D_MPI
MPI_Init(&argc, &argv);
double start, end;
#pragma omp master
{
start = omp_get_wtime();
}
#endif
if (argc != 3) {
fprintf(stderr, "Usage: %s FILENAME GENERATIONS\n", argv[0]);
return EXIT_FAILURE;
}
FILE *file = fopen(argv[1], "r");
if (file == NULL) {
fprintf(stderr, "%s: %s: No such file or directory\n", argv[0], argv[1]);
return EXIT_FAILURE;
}
unsigned long generations = strtoul(argv[2], NULL, 10);
if (generations == 0 || generations > LONG_MAX) {
fprintf(stderr, "%s: %s: Invalid number of generations\n", argv[0], argv[2]);
return EXIT_FAILURE;
}
unsigned int size = read_size(file);
if (size == 0 || size > 10000) {
fprintf(stderr, "%s: %u: Invalid size\n", argv[0], size);
return EXIT_FAILURE;
}
hashtable_t *state;
unsigned int num_cells = read_file(file, size, &state);
fclose(file);
if (state == NULL) {
fprintf(stderr, "%s: Failed to read initial configuration from file\n", argv[0]);
return EXIT_FAILURE;
}
life3d_run(size, state, num_cells, generations);
#ifndef LIFE3D_MPI
print_cells(state);
#else
#pragma omp master
{
end = omp_get_wtime();
fprintf(stderr, "Time: %f\n", end - start);
}
MPI_Finalize();
#endif
return EXIT_SUCCESS;
}
