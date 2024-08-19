#include "ompFunctions.h"
void calculateResultsWithOpenMP(Seq2Result *seq2Results, InputData *inputData, int numOfSeq2) {
int i;
#pragma omp parallel num_threads(numOfSeq2*2)
{
#pragma omp for
for (i = 0; i < numOfSeq2; i++) {
seq2Results[i] = calculateScoringMatrix(inputData->seq1, inputData->seq2Array[i], inputData->weightsArray);
if (seq2Results[i].n == ERROR) {
printf("Error during dynamic allocating. Exiting!\n");
exit(1);
}
}
}
}
Seq2Result calculateScoringMatrix(char *seq1, char *seq2, float weights[NUM_WEIGHTS]) {
int i, j, len1 = strlen(seq1) + 1, len2 = strlen(seq2) + 1, maxN = len1 - len2;
float **scoringMatrix, **sumMatrix, minW = weights[1], penalty = PENALTY * weights[0], temp;
Seq2Result seq2Result = { ERROR };
scoringMatrix = (float **) calloc(len2, sizeof(float *));
sumMatrix = (float **) calloc(len2, sizeof(float *));
if (!scoringMatrix || !sumMatrix) {
printf("ERROR during scoringMatrix/sumMatrix memory allocating!\n");
return seq2Result;
}
for (i = 0; i < len2; i++) {
scoringMatrix[i] = (float *) calloc(len1, sizeof(float));
sumMatrix[i] = (float *) calloc(len1, sizeof(float));
if (scoringMatrix[i] == NULL || sumMatrix[i] == NULL) {
printf("ERROR during scoringMatrix[%d]/sumMatrix[%d] memory allocating!\n", i, i);
return seq2Result;
}
#pragma omp parallel for num_threads(i)
for (j = 1; j < i; j++)
scoringMatrix[i][j] = penalty;
#pragma omp parallel for num_threads(maxN)
for (j = i; j <= (maxN + i); j++)
scoringMatrix[i][j] = computePairwiseComparison(seq1[j - 1], seq2[i - 1], weights);
}
scoringMatrix[1][maxN + 1] = penalty;
for (i = 2; i < NUM_WEIGHTS; i++) {
temp = weights[i];
if (minW < temp)
minW = temp;
}
computeSumMatrix(sumMatrix, scoringMatrix, len1, len2, minW, penalty);
computeBestScore(&seq2Result, scoringMatrix, sumMatrix, len1, len2, minW);
freeMatrix(scoringMatrix, len2);
freeMatrix(sumMatrix, len2);
return seq2Result;
}
float computePairwiseComparison(char c1, char c2, float weights[NUM_WEIGHTS]) {
const char *CONSERV_GROUPS[NUM_CONSERV_GROUPS] = { "NDEQ", "NEQK", "STA", "MILV", "QHRK", "NHQK", "FYW", "HY", "MILF" }; 
const char *SEMI_CONSERV_GROUPS[NUM_SEMI_CONSERV_GROUPS] = { "SAG", "ATV", "CSA", "SGND", "STPA", "STNK", "NEQHRK", "NDEQHK", "SNDEQK", "HFY",
"FVLIM" }; 
if (c1 == c2)
return weights[0];
else if (checkIfInGroup(CONSERV_GROUPS, NUM_CONSERV_GROUPS, c1, c2))
return weights[1];
else if (checkIfInGroup(SEMI_CONSERV_GROUPS, NUM_SEMI_CONSERV_GROUPS, c1, c2))
return weights[2];
else
return weights[3];
}
int checkIfInGroup(const char *group[MAX_LEN_GROUP_SEQ], int groupLen, char c1, char c2) {
int i;
for (i = 0; i < groupLen; i++) {
if (strchr(group[i], c1) && strchr(group[i], c2))
return TRUE;
}
return FALSE;
}
int checkIfInGroupOMP(const char *group[MAX_LEN_GROUP_SEQ], int groupLen, char c1, char c2) {
int i, has_zero = 0;
#pragma omp parallel num_threads(groupLen)
{
#pragma omp for
for (i = 0; i < groupLen; i++) {
if (strchr(group[i], c1) && strchr(group[i], c2)) {
#pragma omp critical
{
has_zero = 1;
}
#pragma omp cancel for
}
#pragma omp cancellation point for
}
}
return has_zero;
}
