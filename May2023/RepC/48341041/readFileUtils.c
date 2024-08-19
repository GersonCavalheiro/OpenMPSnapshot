#include <stdio.h>
#include <stdlib.h>
#include "structs.h"
#include "utils.h"
LcsMatrix* readFile(FILE *fp) {
int i;
LcsMatrix *lcs = (LcsMatrix *)calloc(1, sizeof(LcsMatrix));
checkNullPointer(lcs);
if (fscanf(fp, "%d %d", &(lcs->lines), &(lcs->cols)) != 2) {
puts("Error reading file");
exit(-1);
}
lcs->mtx = (int **)calloc(lcs->lines+1, sizeof(int **));
checkNullPointer(lcs->mtx);
#pragma omp parallel for schedule(dynamic) private(i)	
for(i=0; i<lcs->lines+1; i++) {
lcs->mtx[i] = (int *)calloc(lcs->cols+1, sizeof(int *));
checkNullPointer(lcs->mtx[i]);
}
lcs->seqLine = (char *)calloc(lcs->lines+2, sizeof(char));
checkNullPointer(lcs->seqLine);
lcs->seqColumn = (char *)calloc(lcs->cols+2, sizeof(char));
checkNullPointer(lcs->seqColumn);
if (fscanf(fp, "%s", &lcs->seqLine[1]) != 1) {
puts("Error reading file");
exit(-1);
}
lcs->seqLine[0] = ' ';
lcs->seqLine[lcs->lines+1]='\0';
if (fscanf(fp, "%s", &lcs->seqColumn[1]) != 1) {
puts("Error reading file");
exit(-1);
}
lcs->seqColumn[0] = ' ';
lcs->seqColumn[lcs->cols+1]='\0';
return lcs;
}
