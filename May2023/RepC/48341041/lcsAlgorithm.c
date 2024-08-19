#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "structs.h"
#include "utils.h"
short cost(int x)
{
int i, n_iter = 20;
double dcost = 0;
for(i = 0; i < n_iter; i++)
dcost += pow(sin((double) x),2) + pow(cos((double) x),2);
return (short) (dcost / n_iter + 0.1);
}
void funct_parallel(LcsMatrix *lcsMtx, int i, int j, int counter) {
int **matrix = lcsMtx->mtx;
int control;
int ii=i,jj=j;
#pragma omp for schedule(dynamic) private(control)  
for(control = 0; control < counter; control++) {
i = ii + control;
j = jj - control;
if(lcsMtx->seqLine[i] == lcsMtx->seqColumn[j]) {	
matrix[i][j] = matrix[i-1][j-1] + cost(control);
} else {
matrix[i][j] = matrix[i][j-1] >= matrix[i-1][j] ? matrix[i][j-1] : matrix[i-1][j];	
}
}
return;
}
void fillMatrix(LcsMatrix *lcsMtx) {
int i=1;
int j, maxDiag, maxAbs;
int counter=1;
int incCounterVertical=1;
int aux;
maxDiag = (lcsMtx->cols >= lcsMtx->lines) ? lcsMtx->lines : lcsMtx->cols;
maxAbs = (lcsMtx->cols > lcsMtx->lines) ? lcsMtx->cols : lcsMtx->lines;
#pragma omp parallel private(j, aux) firstprivate(i, counter, incCounterVertical) 
{
for(j=1; j <= lcsMtx->cols; j++) {
#pragma omp barrier 
funct_parallel(lcsMtx, i, j, counter);
counter++;	
if(counter > maxDiag)
counter=maxDiag;
}
j=lcsMtx->cols;
aux = j;
#pragma omp barrier 
for(i=2; i <= lcsMtx->lines; i++) {
if (aux == maxAbs) {	
incCounterVertical = -1;
} else {
aux++;
}
counter += incCounterVertical;
if (counter > maxDiag)
counter = maxDiag;
#pragma omp barrier
funct_parallel(lcsMtx, i, j, counter);	
}
}
return;
}
LcsResult findLongestCommonSubsequence(LcsMatrix *lcsMtx) {
LcsResult result;
int i = lcsMtx->lines;
int j = lcsMtx->cols;
char *lcsStringInverted = (char *)calloc(i<j ? i+1 : j+1, sizeof(char));
char *lcsString;
int counter = 0;
while (i!=0 && j!=0) {
if (lcsMtx->seqLine[i] == lcsMtx->seqColumn[j]) {
lcsStringInverted[counter] = lcsMtx->seqLine[i];
counter = counter + 1;
i--;
j--;
} else {
if (lcsMtx->mtx[i][j-1] >= lcsMtx->mtx[i-1][j]) {
j--;
} else {
i--;
}
}
}
lcsStringInverted[counter] = '\0';
lcsString = (char *)calloc(counter, sizeof(char));
for (i=counter-1; i>=0; i--) {
lcsString[counter-1-i] = lcsStringInverted[i];
}
free(lcsStringInverted);
result.counter = counter;
result.sequence = lcsString;
return result;
}
