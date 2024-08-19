#include <nanos.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#define ARRAY_SIZE 23
#define TEST_SLICER(chunksize) \
for (i=0; i<ARRAY_SIZE; i++) A[i] = nanos_get_wd_id(nanos_current_wd()); \
if ( !check_chunksize( A, ARRAY_SIZE, chunksize ) )  error++;
bool check_chunksize( int* A, int N, int cs )
{
int i, j;
for (i=0; i<N; i+=j) {
int current_id = A[i];
for (j=1; j<cs && i+j<N; j++) {
if ( A[i+j] != current_id )
return false;
}
}
return true;
}
int main( int argc, char *argv[] )
{
int i;
int error = 0;
int A[ARRAY_SIZE];
#pragma omp for schedule(static,2)
TEST_SLICER( 2 );
#pragma omp for schedule(dynamic,2)
TEST_SLICER( 2 );
#pragma omp for schedule(static,13)
TEST_SLICER( 13 );
#pragma omp for schedule(dynamic,13)
TEST_SLICER( 13 );
if ( error ) {
return EXIT_FAILURE;
}
else {
return EXIT_SUCCESS;
}
}
