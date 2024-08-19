#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#define PATH -1
#define NONE 0
#define UP 1
#define LEFT 2
#define DIAGONAL 3
#define min(x, y) (((x) < (y)) ? (x) : (y))
#define max(a,b) ((a) > (b) ? a : b)
long long int m ; 
long long int n ;  
int matchScore = 5;
int missmatchScore = -3;
int gapScore = -4;
char *a, *b;
void generate() {
srand(time(NULL));
long long int i;
for (i = 0; i < m; i++) {
int aux = rand() % 4;
if (aux == 0)
a[i] = 'A';
else if (aux == 2)
a[i] = 'C';
else if (aux == 3)
a[i] = 'G';
else
a[i] = 'T';
}
for (i = 0; i < n; i++) {
int aux = rand() % 4;
if (aux == 0)
b[i] = 'A';
else if (aux == 2)
b[i] = 'C';
else if (aux == 3)
b[i] = 'G';
else
b[i] = 'T';
}
} 
long long int nElement(long long int i) {
if (i < m && i < n) {
return i;
}
else if (i < max(m, n)) {
long long int min = min(m, n);
return min - 1;
}
else {
long long int min = min(m, n);
return 2 * min - i + labs(m - n) - 2;
}
}
int matchMissmatchScore(long long int i, long long int j) {
if (a[j - 1] == b[i - 1])
return matchScore;
else
return missmatchScore;
}  
void similarityScore(long long int i, long long int j, int* H, int* P, long long int* maxPos) {
int up, left, diag;
long long int index = m * i + j;
up = H[index - m] + gapScore;
left = H[index - 1] + gapScore;
diag = H[index - m - 1] + matchMissmatchScore(i, j);
int max = NONE;
int pred = NONE;
if (diag > max) { 
max = diag;
pred = DIAGONAL;
}
if (up > max) { 
max = up;
pred = UP;
}
if (left > max) { 
max = left;
pred = LEFT;
}
H[index] = max;
P[index] = pred;
if (max > H[*maxPos]) {        
#pragma omp critical
*maxPos = index;
}
}  
void calcFirstDiagElement(long long int *i, long long int *si, long long int *sj) {
if (*i < n) {
*si = *i;
*sj = 1;
} else {
*si = n - 1;
*sj = *i - n + 2;
}
}
int main(int argc, char* argv[]) {
m = 2048;
n = 2048;
#ifdef DEBUG
printf("\nMatrix[%lld][%lld]\n", n, m);
#endif
a = malloc(m * sizeof(char));
b = malloc(n * sizeof(char));
m++;
n++;
int *H;
H = calloc(m * n, sizeof(int));
int *P;
P = calloc(m * n, sizeof(int));
generate();
long long int maxPos = 0;
long long int i, j;
double initialTime = omp_get_wtime();
long long int si, sj, ai, aj;
long long int nDiag = m + n - 3;
long long int nEle;
#pragma omp parallel default(none) shared(H, P, maxPos, nDiag) private(nEle, i, si, sj, ai, aj)
{
for (i = 1; i <= nDiag; ++i)
{
nEle = nElement(i);
calcFirstDiagElement(&i, &si, &sj);
#pragma omp for
for (j = 1; j <= nEle; ++j)
{
ai = si - j + 1;
aj = sj + j - 1;
similarityScore(ai, aj, H, P, &maxPos);
}
}
}
}
