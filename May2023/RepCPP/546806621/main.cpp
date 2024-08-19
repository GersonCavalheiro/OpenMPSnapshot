#include "io.h"
#include "init.h"
#include "jacobi.h"
#include "mpi_module.h"

int main(int argc, char *argv[]){


int size, my_rank;

start_MPI(&my_rank,&size);

#pragma omp parallel
{
printf("Thread %d from rank %d\n", omp_get_thread_num(), my_rank);
}

const char* file_name="params.txt";

params p;
readParameters(file_name, &p);

mpi_get_domain(p.nx, p.ny, my_rank, size, &p.xmin, &p.xmax, &p.ymin, &p.ymax);

double **f, **u_old, **u_new;

f = allocateGrid(p.xmax - p.xmin, p.ymax - p.ymin, f);
u_old = allocateGrid(p.xmax - p.xmin, p.ymax - p.ymin, u_old);
u_new = allocateGrid(p.xmax - p.xmin, p.ymax - p.ymin, u_new);
init_variables(p, f, u_old, u_new);

output_source(p, f, my_rank);


jacobi_step(p, u_new, u_old, f, my_rank, size);

double diff = norm_diff(p, u_new, u_old);

int nstep=1;

while (diff>p.tol && nstep<p.nstep_max){
if (my_rank==0) printf("Step %d, Diff=%g\n", nstep, diff);
jacobi_step(p, u_new, u_old, f, my_rank, size);
diff = norm_diff(p, u_new, u_old);
nstep++;

if (nstep%p.foutput==0) output(p, nstep, u_new, my_rank);
}
if (my_rank==0) printf("Step %d, Diff=%g\n", nstep, diff);
output(p, nstep, u_new, my_rank);

close_MPI();
return 0;
}
