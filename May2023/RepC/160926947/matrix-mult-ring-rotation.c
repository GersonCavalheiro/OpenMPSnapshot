#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
struct matrix {
int height;
int width;
int * tab;
};
struct matrix * allocateMatrix(int height, int width) {
struct matrix * tmp = malloc(sizeof(struct matrix));
tmp->height = height;
tmp->width = width;
tmp->tab = (int *)calloc(height * width, sizeof(int));
return tmp;
}
int getValue(struct matrix * m, int i, int j){
return m->tab[i * m->width + j];
}
void setValue(struct matrix * m, int i, int j, int value){
m->tab[i * m->width + j] = value;
}
void printMatrix(struct matrix * m){
for(int i = 0; i < m->height; i++){
for(int y = 0; y < m->width; y++){
printf("%d ", getValue(m, i, y));
}
printf("\n");
}
}
struct matrix * generateMatrixFromFile(FILE * fp){
int x;
int N = 0;
while(fscanf(fp, "%d", &x) != EOF){
N++;
if(fgetc(fp) == 10){
break;
}
}
rewind(fp); 
struct matrix * tmp;
tmp = allocateMatrix(N, N);
int numberOfInt = N * N;
for(int i = 0; i < numberOfInt; i++){
fscanf(fp, "%d", &tmp->tab[i]);
}
return tmp;
}
void multMatrix(struct matrix * a, struct matrix * b, struct matrix * c, int startingLineIndex){
int N = a->width; 
#pragma omp parallel for
for(int i = 0; i < a->height; i++){
for(int j = 0; j < b->width; j++){
for(int k = 0; k < N; k++){
int tmp = getValue(c, i + startingLineIndex, j) + getValue(a, i, k) * getValue(b, k, j);
setValue(c, i + startingLineIndex, j, tmp);
}
}
}
}
void initForScatterv(int * countsA, int * displsA, int * countsB, int * displsB, int N, int num_procs){
int mod = N % num_procs; 
int startingIndice = num_procs - mod;
int numberOfLine = N / num_procs; 
int dispA = 0;
int dispB = 0;
for(int i = 0; i < num_procs; i++){
countsA[i] = numberOfLine;
displsA[i] = dispA;
countsB[i] = numberOfLine;
displsB[i] = dispB;
if(i >= startingIndice){
countsA[i] += 1;
countsB[i] += 1;
}
dispB += countsA[i];
countsA[i] *= N;
dispA += countsA[i];
}
}
int main(int argc, char *argv[]) {
int num_procs, rank;
MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
int suivant = (rank + 1) % num_procs;
int precedent = (rank - 1 + num_procs) % num_procs;
int N;
struct matrix * a = malloc(sizeof(struct matrix));
struct matrix * b = malloc(sizeof(struct matrix));
if(rank == 0){
FILE* fp = NULL;
fp = fopen(argv[1], "r");
free(a);
a = generateMatrixFromFile(fp);
fclose(fp);
fp = fopen(argv[2], "r");
free(b);
b = generateMatrixFromFile(fp);
fclose(fp);
N = a->height;
}
MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD); 
int countsA[num_procs];
int displsA[num_procs];
int countsB[num_procs];
int displsB[num_procs];
initForScatterv(countsA, displsA, countsB, displsB, N, num_procs);
int numberOfItem = countsA[rank];
int * a_tab = malloc(numberOfItem * sizeof(int));
int * b_tab = malloc(numberOfItem * sizeof(int));
MPI_Scatterv(a->tab, countsA, displsA, MPI_INT, a_tab, numberOfItem, MPI_INT, 0, MPI_COMM_WORLD);
struct matrix * sub_a = allocateMatrix(numberOfItem / N, N);
free(sub_a->tab);
sub_a->tab = a_tab;
MPI_Datatype type, column_t;
MPI_Type_vector(N, 1, N, MPI_INT, &type);
MPI_Type_commit(&type);
MPI_Type_create_resized(type, 0, sizeof(int), &column_t);
MPI_Type_commit(&column_t);
MPI_Datatype local_type, local_column_t;
MPI_Type_vector(N, 1, countsB[rank], MPI_INT, &local_type);
MPI_Type_commit(&local_type);
MPI_Type_create_resized(local_type, 0, sizeof(int), &local_column_t);
MPI_Type_commit(&local_column_t);
MPI_Scatterv(b->tab, countsB, displsB, column_t, b_tab, countsB[rank], local_column_t, 0, MPI_COMM_WORLD);
struct matrix * sub_b = allocateMatrix(N, countsB[rank]);
free(sub_b->tab);
sub_b->tab = b_tab;
int currentIndice = displsB[rank]; 
struct matrix * c = allocateMatrix(N, countsB[rank]); 
multMatrix(sub_a, sub_b, c, currentIndice); 
for(int i = 0; i < num_procs - 1; ++i) {
int nextNumberOfItem = countsA[(rank - (i + 1) + num_procs) % num_procs]; 
if(rank % 2 == 0){
MPI_Send(sub_a->tab, numberOfItem, MPI_INT, suivant, 0, MPI_COMM_WORLD);
if(numberOfItem != nextNumberOfItem){
free(sub_a->tab);
free(sub_a);
sub_a = allocateMatrix(nextNumberOfItem / N, N); 
}
MPI_Recv(sub_a->tab, nextNumberOfItem, MPI_INT, precedent, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
else {
int * value_to_receive = malloc(nextNumberOfItem * sizeof(int));
MPI_Recv(value_to_receive, nextNumberOfItem, MPI_INT, precedent, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Send(sub_a->tab, numberOfItem, MPI_INT, suivant, 0, MPI_COMM_WORLD);
if(numberOfItem != nextNumberOfItem){
free(sub_a);
sub_a = allocateMatrix(nextNumberOfItem / N, N);
}
free(sub_a->tab);
sub_a->tab = value_to_receive;
}
currentIndice = displsB[(rank - (i + 1) + num_procs) % num_procs];
numberOfItem = nextNumberOfItem;
multMatrix(sub_a, sub_b, c, currentIndice);
}
int * res = NULL;
if(rank == 0){
res = malloc(N * N * sizeof(int));
}
MPI_Gatherv(c->tab, countsB[rank], local_column_t, res, countsB, displsB, column_t, 0, MPI_COMM_WORLD);
if(rank == 0){
struct matrix * r = allocateMatrix(N, N);
free(r->tab);
r->tab = res;
printMatrix(r);
}
free(sub_a->tab);
free(sub_a);
MPI_Finalize();
return 0;
}
