
#include <iostream>
#include <unistd.h>
#include "adf.h"


long BSIZE = 100;
long DIM = 10;
long NB, N, NN;

float **A;
float **B;
float **C;





static void convert_to_blocks(float *Alin, float **A)
{
long i, j;

for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) {
A[i/NB*DIM + j/NB][(i%NB)*NB+j%NB] = Alin[j*N+i];
}
}

}

void fill_random(float *Alin)
{
long i;
for (i = 0; i < NN; i++) {
Alin[i]=((float)rand())/((float)RAND_MAX);
}
}


void InitMatrix()
{
N=BSIZE*DIM;
NN=N*N;
NB = BSIZE;

float *Alin = (float *) malloc(NN * sizeof(float));
float *Blin = (float *) malloc(NN * sizeof(float));
float *Clin = (float *) malloc(NN * sizeof(float));

long i;
srand(0);
fill_random(Alin);
fill_random(Blin);
for (i=0; i < NN; i++)
Clin[i]=0.0;

A = (float **) malloc(DIM*DIM*sizeof(float *));
B = (float **) malloc(DIM*DIM*sizeof(float *));
C = (float **) malloc(DIM*DIM*sizeof(float *));

for ( i = 0; i < DIM*DIM; i++)
{
A[i] = (float *) malloc(BSIZE*BSIZE*sizeof(float));
B[i] = (float *) malloc(BSIZE*BSIZE*sizeof(float));
C[i] = (float *) malloc(BSIZE*BSIZE*sizeof(float));
}
convert_to_blocks(Alin, A);
convert_to_blocks(Blin, B);
convert_to_blocks(Clin, C);

free(Alin);
free(Blin);
free(Clin);

}



void FreeMatrix()
{
long i;
for ( i = 0; i < DIM*DIM; i++) {
free(A[i]);
free(B[i]);
free(C[i]);
}
free(A);
free(B);
free(C);
}


void matmul(float  *A, float *B, float *C)
{
long i, j, k, I;
float tmp;
for (i = 0; i < NB; i++) {
I=i*NB;
for (j = 0; j < NB; j++) {
tmp=C[I+j];
for (k = 0; k < NB; k++) {
tmp+=A[I+k]*B[k*NB+j];
}
C[I+j]=tmp;
}
}
}



void SequentialWorker()
{
long i, j, k;

for (i = 0; i < DIM; i++)
for (j = 0; j < DIM; j++)
for (k = 0; k < DIM; k++){
matmul (A[i*DIM + k], B[k*DIM + j], C[i*DIM + j]);
}
}



void ParseCommandLine(int argc, char **argv)
{
char c;

while ((c = getopt (argc, argv, "hd:b:")) != -1)
switch (c) {
case 'h':
printf("\nMatrix multiplication - ADF benchmark application\n"
"\n"
"Usage:\n"
"   matmul_seq [options ...]\n"
"\n"
"Options:\n"
"   -h\n"
"        Print this help message.\n"
"   -d <long>\n"
"        number of blocks in one dimension. (default 10)\n"
"   -b <long>\n"
"        Block size. (default 100)\n"
"\n"
);
exit (0);
case 'd':
DIM = atol(optarg);
break;
case 'b':
BSIZE = atol(optarg);
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

if (BSIZE < 1) BSIZE = 1;
if (DIM < 1) DIM = 1;

printf ("\nMatrix mulitplication\nMatrix size is %ldx%ld, devided into %ldx%ld blocks of size %ld.\n",
DIM*BSIZE, DIM*BSIZE, DIM, DIM, BSIZE);
printf ("=====================================================\n\n");
}



int main (int argc, char **argv)
{
ParseCommandLine(argc, argv);

InitMatrix();

sequential_init();

SequentialWorker();

sequential_terminate();

FreeMatrix();

return 0;
}
