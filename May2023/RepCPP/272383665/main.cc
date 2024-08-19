#include"AC.h"
#include<mpi.h>

int main(int argc, char const *argv[]) {

int key[3][3] = {
{1, 2, 3},
{4, 5, 6},
{7, 8, 9}
};

int value_original[10][10] = {
{0, 0, 1, 1, 0, 2, 2, 0, 3, 3},
{1, 1, 1, 1, 0, 0, 0, 0, 0, 0},
{1, 1, 1, 1, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{2, 2, 2, 2, 2, 2, 2, 0, 0, 0},
{2, 2, 2, 2, 2, 0, 0, 0, 0, 0},
{0, 2, 2, 2, 2, 0, 0, 0, 0, 0},
{3, 3, 3, 3, 3, 3, 3, 3, 3, 3},
{3, 3, 3, 3, 3, 3, 3, 3, 0, 0},
{0, 3, 3, 3, 3, 3, 3, 3, 0, 0}
};

if(argc < 3) {
fprintf(stderr, "Usage: matrix-dimension iterations-number\n");
}

int dim = atoi(argv[1]);
int iter = atoi(argv[2]);
int nthreads = atoi(argv[3]);

if(nthreads > 1) {
int provided = 0;
MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
if(provided != MPI_THREAD_MULTIPLE) {
fprintf(stderr, "not enough multithread support\n");
MPI_Abort(MPI_COMM_WORLD,0);
}
} else{
MPI_Init(NULL, NULL);
}
int world_rank, world_size;
MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
MPI_Comm_size(MPI_COMM_WORLD, &world_size);

double start;
if(world_rank == 0)
start = MPI_Wtime();

unsigned seed = (int)MPI_Wtime() * (world_rank + 1);

Cell **mat = NULL;
Cell **tam = NULL;
Cell *matdata, *tamdata;
Cell *sendbufp = NULL;

int p = dim/world_size;

MPI_Datatype vec_t;
MPI_Type_contiguous(dim, MPI_INT, &vec_t);
MPI_Type_commit(&vec_t); 

int extra = 0;
allocateMatrix(&mat, dim, &matdata, dim, seed);
if(world_rank == 0) {
extra = dim - p*world_size;
allocateMatrix(&tam, p + extra, &tamdata, dim, seed);
sendbufp = &( mat[ extra ][ 0 ] );
} else {
allocateMatrix(&tam, p, &tamdata, dim, seed);
}

MPI_Barrier(MPI_COMM_WORLD);


#pragma omp parallel shared(mat, tam, iter, vec_t, sendbufp, value_original, p, extra, world_rank, dim) if(nthreads > 1)
{
int pos[8] = {-4, -3, -2, -1, 1, 2, 3, 4};
int jmap[3][2] = {{8,15}, {3,5}, {2,7}};
int sx, cid, nid;
int pivot, temp, ni, nj, ti, tj;
int chunk_size = (p + extra)/nthreads + 1;
unsigned seed2 = (int)omp_get_wtime() * (omp_get_thread_num() + 1);

for(int it = 0; it < iter; ++it) {
#pragma omp barrier
#pragma omp master
MPI_Bcast(&(mat[0][0]), dim, vec_t, 0, MPI_COMM_WORLD);
#pragma omp barrier

#pragma omp for nowait schedule(guided, chunk_size) 
for(int i = 0; i < p + extra; ++i) {
sx = (p+extra)*world_rank + i;
for(int j = 0; j < dim; ++j) {

pivot = pos[my_rand_r(&seed2)%8];
temp = (sx*dim + j);
ni = (sx + j/dim + pivot/3 + dim) % dim;
nj = (temp%dim + pivot%3 + dim) % dim;

Cell& neighbor = mat[ni][nj];
Cell& cell = mat[sx][j];
Cell& newcell = tam[i][j];

cid = cell%100;
nid = neighbor%100;
ti = (cell/100);
tj = (neighbor/100);	

cid = (cid == jmap[ti-1][1]) ? 2 : (cid > jmap[ti-1][0]) ? 1 : 0;	
nid = (nid == jmap[tj-1][1]) ? 2 : (nid > jmap[tj-1][0]) ? 1 : 0;

ti = (ti == 0) ? 0 : key[ti-1][cid];
tj = (tj == 0) ? 0 : key[tj-1][nid];

newcell = value_original[ti][tj]*100;
newcell += (newcell/100 == 0) ? 0 : cell%100 + 1;

}
}

#pragma omp barrier
#pragma omp master
MPI_Gather(&(tam[extra][0]), p, vec_t, sendbufp, p, vec_t, 0, MPI_COMM_WORLD);
}}

freeMatrix(mat, matdata);
freeMatrix(tam, tamdata);

if(world_rank == 0) {
double end = MPI_Wtime();
printf("Dimension: %d Iterations: %d MPI processes: %d OpenMP threads: %d execution time: %f\n", 
dim, iter, world_size, nthreads, end-start);
}    

MPI_Finalize();
return 0;
}