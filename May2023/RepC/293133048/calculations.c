#include "calculations.h"
void computeSumMatrix(float **sumMatrix, float **scoringMatrix, int len1, int len2, float minW, float penalty) {
int n, k, bestK = 0, i, j, maxN = len1 - len2;
float bestScore = minW * len2, kBestScore = minW * len2, temp = 0.0, maxS, regularSum, iPenalty, jPenalty;
for (i = 1; i < len2; i++) {
for (j = i; j <= maxN + i; j++) {
maxS = penalty;
regularSum = sumMatrix[i - 1][j - 1] + scoringMatrix[i][j];
if (regularSum > maxS)
maxS = regularSum;
iPenalty = sumMatrix[i - 1][j] + scoringMatrix[i][j] + minW;
if (iPenalty > maxS)
maxS = iPenalty;
jPenalty = sumMatrix[i][j - 1] + scoringMatrix[i][j] + minW;
if (jPenalty > maxS)
maxS = jPenalty;
sumMatrix[i][j] = maxS;
}
}
}
void computeBestScore(Seq2Result *seq2Res, float **scoringMatrix, float **sumMatrix, int len1, int len2, float minW) {
int i, j, count, from, n, k = len1 - 2, bestN = 0, bestK = k, maxN = len1 - len2, end = len1 - 1, gap = GAP, tempN = 0;
float bestScore = minW * len2, kBestScore = bestScore, maxOpening = bestScore, temp = 0.0;
i = len2 - 1;
for (j = i; j < len1; j++) {
if (sumMatrix[i][j] >= maxOpening) {
maxOpening = sumMatrix[i][j];
end = j;
}
}
from = end + gap < len1 ? end + gap : len1 - 1;	
for (count = 0; count <= 2 * gap; count++) {
j = from - count, i = len2 - 1, tempN = j - len2;
while (j > 0 && i > 0) {
if (sumMatrix[i][j - 1] >= (sumMatrix[i - 1][j - 1])) {
temp = computeMutantSequenceKScore(i + 1, tempN, scoringMatrix, len1, len2);
if (temp >= kBestScore) {
kBestScore = temp;
bestK = i;
bestN = tempN;
}
}
i -= 1;
j -= 1;
}
if (kBestScore >= bestScore) {
bestScore = kBestScore;
k = bestK;
n = bestN;
}
}
seq2Res->k = k;
seq2Res->n = n;
}
float computeMutantSequenceKScore(int k, int n, float **scoringMatrix, int len1, int len2) {
int i;
float score = 0.0;
#pragma omp parallel
{
#pragma omp for nowait
for (i = 1; i < k; i++)
score = score + scoringMatrix[i][i + n];
#pragma omp for
for (i = k; i < len2; i++)
score = score + scoringMatrix[i][i + n + 1];
}
return score;
}
void freeMatrix(float **scoringMatrix, int matrixLength) {
int i;
for (i = 0; i < matrixLength; i++) {
free(scoringMatrix[i]);
}
free(scoringMatrix);
}
