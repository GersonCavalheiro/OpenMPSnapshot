#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "stdbool.h"
#include "time.h"
#include "omp.h"
#define N 200        
#define GENES 600    
#define TARGET_THREADS 12   
int RandomInteger(int mini, int maxi) {
int gap = maxi-mini;
int randomInGap = (int) (gap * ((float)rand())/((float)RAND_MAX) ); 
return mini + randomInGap; 
}
void GeneInitialization(int genes[GENES][N]) {
for (int i=0; i<GENES; i++) {
for (int j=0; j<N; j++) {
genes[i][j] = RandomInteger(0,N-1);
}
}
}
void Map3(int posY[N], int M) {
for (int i=0; i<N; i++) printf("==="); printf("===\n---");
for (int i=0; i<N/3; i++) printf("---"); printf("  FITTEST GENE  ");
for (int i=0; i<N/3; i++) printf("---"); printf("---\n===");
for (int i=0; i<N; i++) printf("==="); printf("\n");
for (int i=0; i<N; i++) printf("---"); printf("---\n##|");
for (int i=0; i<N; i++) printf("%2d ", i+1); printf("\n---");
for (int i=0; i<N; i++) printf("---"); printf("\n");
for (int y=0; y<N; y++) {
printf("%2d| ", y+1);
for (int x=0; x<N; x++) {
bool flag = false;
for (int i=0; i<M; i++) {
if (i==x && posY[i]==y) {
flag = true;
}
}
if (flag) printf("Q");
else printf("~");
printf("  ");
}
printf("\n");
}
for (int i=0; i<N; i++) printf("---"); printf("---\n");
}
bool isSafeFromPrevious(int posY[N], int x, int y) {
int currentQueen = x;
for (int oldQueen=0; oldQueen<currentQueen; oldQueen++) {
if (oldQueen==x || posY[oldQueen]==y) return false; 
else if (y==posY[oldQueen]+(currentQueen-oldQueen) || y==posY[oldQueen]-(currentQueen-oldQueen)) return false; 
}
return true;
}
int UtilityFunction(int posY[N]) {
int collisions = 0;
for (int crnt=1; crnt<N; crnt++) {
for (int old=0; old<crnt; old++) {
if (old==crnt || posY[old]==posY[crnt]) collisions++; 
else if (posY[crnt]==posY[old]+(crnt-old) || posY[crnt]==posY[old]-(crnt-old)) collisions++; 
}
}
return collisions;
}
void CrossoverFunction(int gene1[N], int gene2[N]) {
for (int i=1; i<N; i++) {
if (abs(gene1[i-1]-gene1[i])<2 || abs(gene2[i-1]-gene2[i])<2) {
int temp = gene1[i];
gene1[i] = gene2[i];
gene2[i] = temp;
}
}
}
void MutationFunction(int gene[N]) {
int inGene[N] = {0};
for (int i=0; i<N; i++) {
inGene[gene[i]] = 1;
}
for (int i=1; i<N; i++) {
for (int j=0; j<i; j++) {
if (gene[i]==gene[j]) {
for (int k=0; k<N; k++){
if (inGene[k]==0) {
gene[i] = k;
inGene[k] = 1;
k = N;
}
}
}
}
}
int barrier = RandomInteger(1,N-3); 
int swapA = RandomInteger(0,barrier);   
int swapB = RandomInteger(barrier+1,N-1); 
int temp = gene[swapA];
gene[swapA] = gene[swapB];
gene[swapB] = temp;
}
void BreedGeneration(int genes[GENES][N], int utilityValues[GENES]) {
int genesNew[GENES][N] = {-1};
for (int i=0; i<GENES-1; i+=2) {
int index1 = -1, index2 = -1;
float limit_value = INFINITY;
float value1 = limit_value, value2 = limit_value;
for (int j=0; j<GENES; j++) {
float value = (float) (10 + RandomInteger(10,20)*utilityValues[j] );
if (value<=value1) {
value2 = value1;
index2 = index1;
value1 = value;
index1 = j;
} else if (value<value2) {
value2 = value;
index2 = j;
}
}
for (int k=0; k<N; k++) {
genesNew[i][k]   = genes[index1][k];
genesNew[i+1][k] = genes[index2][k];
}
CrossoverFunction(genesNew[i], genesNew[i+1]);
MutationFunction(genesNew[i]);
MutationFunction(genesNew[i+1]);
}
for (int i=0; i<GENES; i++) {
for (int j=0; j<N; j++) {
genes[i][j] = genesNew[i][j];
}
}    
}
unsigned CalculateAllUtilityValues(int genes[GENES][N], int utilityValues[GENES]) {
int bestUtilityValueFoundAt = 0;
for (int i=0; i<GENES; i++) {
utilityValues[i] = UtilityFunction(genes[i]);
if (utilityValues[i] < utilityValues[bestUtilityValueFoundAt]) {
bestUtilityValueFoundAt = i;
}
}
return bestUtilityValueFoundAt;
}
long int Solve(int fittestGene[N], unsigned threadID, int *whoHasFinished, unsigned *solversGenerations) {
srand(threadID);
int genes[GENES][N];
int utilityValues[GENES] = {1};
GeneInitialization(genes);
long int generation = 0;
unsigned bestGene = 0;
while(utilityValues[bestGene]!=0 && *whoHasFinished<0) {
generation++; 
BreedGeneration(genes, utilityValues);
bestGene = CalculateAllUtilityValues(genes, utilityValues);
}
#pragma omp critical
{
*whoHasFinished = threadID;
*solversGenerations = generation;
for (int i=0; i<N; i++) fittestGene[i] = genes[bestGene][i];
}
return generation;
}
int main() {
printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
printf("This program implements my Genetic Algorithm method of solving the \"N-Queens Problem\".\n");
printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");
int fittestGene[N] = {0};
int numberOfThreads = 1, whoHasFinished = -1;
unsigned solversGenerations = 0;
long int totalGenerations = 0;
printf("Queens set at: %d   Genes set at: %d\n", N, GENES);
printf("Now solving the problem. Please wait...\n");
#pragma omp parallel num_threads(TARGET_THREADS) reduction(+:totalGenerations)
{
#pragma omp single
numberOfThreads = omp_get_num_threads(); 
totalGenerations = Solve(fittestGene, omp_get_thread_num(), &whoHasFinished, &solversGenerations);
}
printf("Algorithm completed. Number of threads used: %d   Total generations: %ld\n", numberOfThreads, totalGenerations);
printf("Solution found by thread #%d in #%u generations.\n", whoHasFinished, solversGenerations);
printf("The solution found is:\n");
return 0;
}
