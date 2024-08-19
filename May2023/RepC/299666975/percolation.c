#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<stdbool.h>
#include<omp.h>
#include "/usr/include/mpi/mpi.h"
#include<string.h>
#include<limits.h>
#define L 46    
#define N (L*L)
#define originProcess 0
#define amountOfProcesses 2
#define amountOfThreads 2
#define split (N/amountOfProcesses)
typedef struct coord{
int row;
int col;
} COORD;
typedef struct cluster{
int id;
COORD pos[N];
int length;
} CLUSTER;
void printGrid(int[L][L]);
void printClusters(CLUSTER[], int);
void printCoords(COORD[], int);
void printArr(int[], int);
void percolate(double, int);
void dfs(int, int, int, int, int[L][L], int[L][L], CLUSTER[], int);
CLUSTER* getCluster(int, CLUSTER[], int);
void mergeClusters(CLUSTER*, CLUSTER*, CLUSTER[], int*, int[L][L]);
void doesPercolate(CLUSTER[], int, bool*, bool*);
int maxCluster(CLUSTER[], int);
bool contains(int[N], int, int);
bool containsAll(int[], int);
int mod (int, int);
double randomProb();
int main() {
double prob;
MPI_Init(NULL, NULL);
int world_rank;
MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
if(world_rank == originProcess){
printf("Enter a number from 0 to 1 for the seeding probability:\n");
scanf("%lf", &prob);
MPI_Send(&prob, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
percolate(prob, world_rank);
}
else{
MPI_Recv(&prob, 1, MPI_DOUBLE, originProcess, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
percolate(prob, world_rank);
}
MPI_Finalize();
return 0;
}
void percolate(double prob, int world_rank) {
bool row_perco = false;
bool col_perco = false; 
void dfs(int row, int col, int minEdge, int maxEdge, int lattice[L][L], int seen[L][L], CLUSTER cList[N], int idx){
if(row < minEdge || row >= maxEdge || seen[row][col] > 0 || lattice[row][col] == 0){
return;
}
CLUSTER* currCluster = &cList[idx];
seen[row][col] = currCluster->id;
int nxt = currCluster->length;
currCluster->pos[nxt] = (COORD){
.row = row,
.col = col
};
currCluster->length++;
dfs(mod(row+1, L), col, minEdge, maxEdge, lattice, seen, cList, idx);
dfs(mod(row-1, L), col, minEdge, maxEdge, lattice, seen, cList, idx);
dfs(row, mod(col+1, L), minEdge, maxEdge, lattice, seen, cList, idx);
dfs(row, mod(col-1, L), minEdge, maxEdge, lattice, seen, cList, idx);
}
omp_set_num_threads(amountOfThreads);
if(world_rank == originProcess){
int (*lattice)[L] = malloc(sizeof(int[L][L]));
double seeding_prob = prob;
srand(time(NULL));
for(int i=0; i < L; ++i){
for(int j=0; j < L; ++j){
if (randomProb() < seeding_prob){
lattice[i][j] = 1;
}
else{
lattice[i][j] = 0;
}       
}
}
for(int i=1; i<amountOfProcesses; i++){
printf("\n%dx%d lattice.\n ",L, L);
MPI_Send(lattice, N, MPI_INT, i, 0, MPI_COMM_WORLD);
int begin = 0;
int end = N;
int vals[2] = {begin, end};
MPI_Send(vals, 2, MPI_INT, i, 0, MPI_COMM_WORLD);
}
free(lattice);
}
else{
int (*lattice)[L] = malloc(sizeof(int[L][L]));
MPI_Recv(lattice, N, MPI_INT, originProcess, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
int vals[2];
MPI_Recv(vals, 2, MPI_INT, originProcess, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
int begin = vals[0];
int end = vals[1];
CLUSTER (*clusterList) = calloc(N, sizeof(int[N]));
int (*seen)[L] = calloc(N, sizeof(int[L][L]));
int amountOfClusters = 1;
int clTop = 0;
int nthreads = 2;
#pragma omp parallel shared(seen, amountOfClusters, clusterList, clTop)
{
int thr = omp_get_thread_num();
int beginRow = thr*(end/nthreads)/L;
int endRow = (thr+1)*(end/nthreads)/L;
#pragma omp for schedule(static, N/nthreads)
for(int i=begin; i<end; i++) {
int row = i/L;
int col = i%L;
if(lattice[row][col] == 1 && seen[row][col] == 0){
#pragma omp critical
{
CLUSTER fst = { .id = amountOfClusters++, .length = 0};
clusterList[clTop] = fst;
}
dfs(row, col, beginRow, endRow, lattice, seen, clusterList, clTop++);
}
}
int idList[N];
int idTop = 0;
#pragma omp critical
{
for(int i=0; i<L; i++){
if(lattice[beginRow][i] != 0 && lattice[mod(beginRow-1, L)][i] != 0 && (seen[beginRow][i] != seen[mod(beginRow-1, L)][i])
&& (!contains(idList, idTop, seen[beginRow][i]) || !contains(idList, idTop, seen[mod(beginRow-1, L)][i]))) {
CLUSTER* c1 = getCluster(seen[mod(beginRow-1, L)][i], clusterList, clTop);
CLUSTER* c2 = getCluster(seen[beginRow][i], clusterList, clTop);
idList[idTop++] = c1->id;
mergeClusters(c1, c2, clusterList, &clTop, seen);
}
}
}
}
printGrid(lattice);
free(lattice);
free(seen);
doesPercolate(clusterList, clTop, &row_perco, &col_perco);
int maxSize = maxCluster(clusterList, clTop);
free(clusterList);
if(maxSize < 0){
fprintf(stderr, "ERROR: No max cluster found");
}
printf("\nNumber of clusters: %d\n", clTop);
printf("Number of sites in biggest cluster: %d\n", maxSize);
printf("Row percolation: %s\n", row_perco ? "true" : "false");
printf("Col percolation: %s\n", col_perco ? "true" : "false");
}
}
void doesPercolate(CLUSTER list[], int top, bool* row_perco, bool* col_perco){
for(int i=0; i<top; i++){
int rowp[L] = {0};
int colp[L] = {0};
for(int j=0; j<list[i].length; j++){
if(rowp[list[i].pos[j].row] == 0){
rowp[list[i].pos[j].row] = 1;
}
if(colp[list[i].pos[j].col] == 0){
colp[list[i].pos[j].col] = 1;
}
}
(*row_perco) = containsAll(rowp,1);
(*col_perco) = containsAll(colp,1);
}
}
bool containsAll(int arr[L], int num){
for(int i=0; i<L; i++){
if(arr[i] != num){
return false;
}
}
return true;
}
int maxCluster(CLUSTER list[N], int top){
int currMax = -1;
for(int i=0; i<top; i++){
if(list[i].length > currMax){
currMax = list[i].length;
}
}
return currMax;
}
bool contains(int list[N], int top, int val){
for(int i=0; i<top; i++){
if(list[i] == val){
return true;
}
}
return false;
}
CLUSTER* getCluster(int id, CLUSTER list[], int size){
for(int i=0; i<size; i++){
if(list[i].id == id){
return &list[i];
}
}
fprintf(stderr, "ERROR: No cluster exists.\n");
exit(EXIT_FAILURE);
}
void mergeClusters(CLUSTER* c1, CLUSTER* c2, CLUSTER list[], int* size, int seen[L][L]){
for(int i=0; i<c2->length; i++){
seen[c2->pos[i].row][c2->pos[i].col] = c1->id;
c1->pos[c1->length++] = c2->pos[i];
}
int to_remove = -1;
for(int i=0; i<(*size); i++){
if(list[i].id == c2->id){
to_remove = i;
}
}
if(to_remove >= 0 && to_remove < (*size)){
list[to_remove] = list[(*size)-1];
(*size)--;
}
else{
fprintf(stderr, "ERROR: index out of range.\n");
exit(EXIT_FAILURE);
}
}
int mod (int a, int b)
{
if(b < 0)
return -mod(-a, -b);   
int ret = a % b;
if(ret < 0)
ret+=b;
return ret;
}
double randomProb(){
return (double)rand()/(double)RAND_MAX;
}
void printGrid(int lattice[L][L]){
for(int i = 0; i < L; ++i){
printf("\n");
for(int j = 0; j < L; ++j){
printf("%d ", lattice[i][j]);
}
}
printf("\n");
}
void printClusters(CLUSTER list[N], int top){
for(int i=0; i<top; i++){
printf("cluster_id: %d\n", list[i].id);
printCoords(list[i].pos, list[i].length);
printf("\n");
}
}
void printCoords(COORD pos[N], int length){
for(int i=0; i<length; i++){
printf("(%d,%d) | ", pos[i].row, pos[i].col);
}
printf("\n");
}
void printArr(int arr[], int top){
printf("[");
for(int i=0; i<top; i++){
printf("%d", arr[i]);
if(i+1 != top){
printf(", ");
}
}
printf("]: %d \n", top);
}
