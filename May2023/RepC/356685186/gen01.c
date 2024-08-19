#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "stdbool.h"
#include "time.h"
#define N 5        
#define GENES 4    
void GeneInitialization(int genes[GENES][N]) {
for (int i=0; i<GENES; i++) {
for (int j=0; j<N; j++) {
genes[i][j] = N*((float)rand())/((float)RAND_MAX);
}
}
}
void Map(int posY[N], int M, int g) {
printf("\n========================\n"); 
printf("------- GENE %d -------\n", g);
printf("========================\n");
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
return;
}
int RandomInteger(int mini, int maxi) {
int gap = maxi-mini;
int randomInGap = (int) (gap * ((float)rand())/((float)RAND_MAX) ); 
return mini + randomInGap; 
}
void MutationFunction(int gene[N]) {
bool missingFromGene[N] = {true};
for (int i=0; i<N; i++) {
missingFromGene[gene[i]] = false;
}
for (int i=0; i<N; i++) {
for (int j=0; j<i-1; j++) {
if (gene[i]==gene[j]) {
for (int k=0; k<N; k++){
if (missingFromGene[k]) {
gene[i] = k;
missingFromGene[k] = false;
}
}
}
}
}
int barrier = RandomInteger(1,N-3); 
int swapA = RandomInteger(0,barrier);   
int swapB = RandomInteger(barrier+1,N-1); 
if ((swapA<0 || swapA>=N)||(swapB<0 || swapB>=N)||(swapA>=swapB)) {printf("\nSWAP ERROR: barrier=%d, swapA=%d, swapB=%d\n", barrier, swapA, swapB);exit(1);    }
if (RandomInteger(1,10)!=1) {
int temp = gene[swapA];
gene[swapA] = gene[swapB];
gene[swapB] = temp;
} else {
int temp = gene[0];
gene[0] = gene[N-1];
gene[N-1] = temp;
}
return;
}
void BreedGeneration(int genes[GENES][N], int utilityValues[GENES]) {
int index1 = -1, index2 = -1;
float maximum_value = INFINITY;
float value1 = maximum_value, value2 = maximum_value;
int genesNew[GENES][N];
for (int i=0; i<GENES; i+=2) {
for (int j=0; j<GENES; j++) {
float value = (float) (10000 + rand()*utilityValues[j] );
if (value<value1) {
value2 = value1;
index2 = index1;
value1 = value;
index1 = j;
} else if (value<value2) {
value2 = value;
index2 = j;
}
}
if (value1>=maximum_value || value2>=maximum_value) {
printf("\nVALUE ERROR: value1=%f, value2=%f\n", value1, value2);
printf("index1=%d, index2=%d\n", index1, index2);
for (int kappa=0; kappa<GENES; kappa++) printf(">> Utility value %d: %d\n",kappa, utilityValues[kappa]);
exit(1);
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
int CalculateAllUtilityValues(int genes[GENES][N], int utilityValues[GENES]) {
int bestUtilityValueFoundAt = 0;
for (int i=0; i<GENES; i++) {
if ((utilityValues[i]=UtilityFunction(genes[i])) < utilityValues[bestUtilityValueFoundAt]) {
bestUtilityValueFoundAt = i;
}
}
return bestUtilityValueFoundAt;
}
int main() {
printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
printf("This program implements my Genetic Algorithm method of solving the \"N-Queens Problem\".\n");
printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");
__time_t now = time(0);
srand(now);
int genes[GENES][N];
int utilityValues[GENES];
GeneInitialization(genes);
printf("Now solving the problem. Please wait...\n");
int generation = 0, bestGene = -1;
while(++generation) {
bestGene = CalculateAllUtilityValues(genes, utilityValues);
if (generation%100==0) printf("\rGeneration: %3d  Best value: %2d", generation, utilityValues[bestGene]);
if (utilityValues[bestGene]==0) break;
BreedGeneration(genes, utilityValues);
}
Map(genes[bestGene], N, bestGene); 
bool allSafe = true;
for (int i=0; i<N-1; i++) {
if (!isSafeFromPrevious(genes[bestGene], i, genes[bestGene][i])) {
printf("WRONG SOLUTION!!\n");
allSafe = false;
}
}
if (allSafe) printf("The solution was validated and is accepted.\n");
}
