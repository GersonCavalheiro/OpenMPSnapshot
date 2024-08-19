#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"

int main(int argc,char** argv)
{
int rank, size;
int rc = MPI_Init(&argc, &argv);
if(rc!= MPI_SUCCESS) {
printf("Error starting MPI program. Terminating.\n");
MPI_Abort(MPI_COMM_WORLD, rc);
}

MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

double slices = (double)atoi(argv[1]);
double partial_sum = 0;

double count = (rank+1)/slices;
for(int i=rank*slices;i<(rank+1)*slices;i++){
double temp = (double)i/slices;
partial_sum+=sqrt(1-temp*temp)/slices;
}

if(rank==0){
double total_sum = partial_sum;
for(int i=1;i<size;i++){
MPI_Recv(&partial_sum, 1, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
total_sum+=partial_sum;
}
printf("%f\n",total_sum*4);
}
else{
MPI_Send(&partial_sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
}

MPI_Finalize();
return 0;
}

