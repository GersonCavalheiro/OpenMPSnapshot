#include "SparseMatrix.h"
void spMatrixInit(SparseMatrix &sp, int size, int rows) {
sp._size = size;
sp._rows = rows;
sp.values = new double[size];
sp.columns = new int[size];
sp.pointerB = new int[rows+1];
}
void multiplicateVector(SparseMatrix &sp, double *&vect, double *&result, int size) {
omp_set_num_threads(4);
#pragma omp parallel for if (ENABLE_PARALLEL)
for (int i = 0; i < size; i++){  
double local_result = 0;
for (int j = sp.pointerB[i]; j < sp.pointerB[i+1]; j++) {
local_result += sp.values[j] * vect[sp.columns[j]];
}
result[i] = local_result;
}
}
void fillMatrix2Expr(SparseMatrix &sp, int size, double expr1, double expr2) {
int index = 0;
int pIndex = 0;
sp.values[index] = 1;
sp.columns[index] = 0;
sp.pointerB[pIndex++] = 0;
++index;
for (int i = 1; i < size - 1; ++i) {
sp.values[index] = expr1;
sp.columns[index] = i - 1;
sp.pointerB[pIndex++] = index;
++index;
sp.values[index] = expr2;
sp.columns[index] = i;
++index;
sp.values[index] = expr1;
sp.columns[index] = i + 1;
++index;
}
sp.values[index] = 1;
sp.columns[index] = size - 1;
sp.pointerB[pIndex++] = index;
sp.pointerB[pIndex] = index + 1;   
}
void fillMatrix3d6Expr(SparseMatrix &sp, MatrixValue &taskexpr, int sizeX, int sizeY, int sizeZ) {
int realSizeX = sizeX + 2;
int realSizeY = realSizeX;
int realSizeZ = realSizeY * sizeY;
int index = 0;
int pIndex = 0;
int sectionStart = 0;
for (int z = 0; z < sizeZ; ++z) {
for (int y = 0; y < sizeY ; ++y) {
sectionStart = z * realSizeZ + y * realSizeY;
int fixBounds = 0;
for (int x = 0; x < realSizeX; ++x) {
if (x == 0 ) {
fixBounds = 1;
} else if ((x + 1) == realSizeX) {
fixBounds = -1;
}
sp.values[index] = taskexpr.z1;
sp.columns[index] = z == 0 ?
fixBounds + x + sectionStart + realSizeZ * (sizeZ - 1) :
fixBounds + x + sectionStart - realSizeZ;
sp.pointerB[pIndex++] = index;
++index;
sp.values[index] = taskexpr.y1;
sp.columns[index] = y == 0 ?
fixBounds + x + sectionStart + realSizeY * (sizeY - 1) :
fixBounds + x + sectionStart - realSizeY;
++index;
sp.values[index] = taskexpr.x1;
sp.columns[index] = fixBounds + x - 1;
++index;
sp.values[index] = taskexpr.x2Comp;
sp.columns[index] = fixBounds + x;
++index;
sp.values[index] = taskexpr.x1;
sp.columns[index] = fixBounds + x + 1;
++index;
sp.values[index] = taskexpr.y1;
sp.columns[index] = y == sizeY - 1?
fixBounds + x + sectionStart - realSizeY * (sizeY - 1) :
fixBounds + x + sectionStart + realSizeY;
++index;
sp.values[index] = taskexpr.z1;
sp.columns[index] = z == sizeZ - 1 ?
fixBounds + x + sectionStart - realSizeZ * (sizeZ - 1) :
fixBounds + x + sectionStart + realSizeZ;
++index;
fixBounds = 0;
}
}
}
sp.pointerB[pIndex] = index + 1;   
}
void fillMatrix3d6Expr_wo_boundaries(SparseMatrix &sp, MatrixValue &taskexpr, int sizeX, int sizeY, int sizeZ) {
int realSizeX = sizeX + 2;
int realSizeY = realSizeX;
int realSizeZ = realSizeY * sizeY;
int index = 0;
int pIndex = 0;
int sectionStart = 0;
for (int z = 0; z < sizeZ; ++z) {
for (int y = 0; y < sizeY ; ++y) {
sectionStart = z * realSizeZ + y * realSizeY;
int fixBounds = 0;
for (int x = 0; x < realSizeX; ++x) {
if (x == 0 ) {
fixBounds = 1;
} else if ((x + 1) == realSizeX) {
fixBounds = -1;
}
sp.values[index] = taskexpr.z1;
sp.columns[index] = z == 0 ?
fixBounds + x + sectionStart + realSizeZ * (sizeZ - 1) :
fixBounds + x + sectionStart - realSizeZ;
sp.pointerB[pIndex++] = index;
++index;
sp.values[index] = taskexpr.y1;
sp.columns[index] = y == 0 ?
fixBounds + x + sectionStart + realSizeY * (sizeY - 1) :
fixBounds + x + sectionStart - realSizeY;
++index;
sp.values[index] = taskexpr.x1;
sp.columns[index] = fixBounds + x - 1;
++index;
sp.values[index] = taskexpr.x2Comp;
sp.columns[index] = fixBounds + x;
++index;
sp.values[index] = taskexpr.x1;
sp.columns[index] = fixBounds + x + 1;
++index;
sp.values[index] = taskexpr.y1;
sp.columns[index] = y == sizeY - 1?
fixBounds + x + sectionStart - realSizeY * (sizeY - 1) :
fixBounds + x + sectionStart + realSizeY;
++index;
sp.values[index] = taskexpr.z1;
sp.columns[index] = z == sizeZ - 1 ?
fixBounds + x + sectionStart - realSizeZ * (sizeZ - 1) :
fixBounds + x + sectionStart + realSizeZ;
++index;
fixBounds = 0;
}
}
}
sp.pointerB[pIndex] = index + 1;   
}
void printVectors(SparseMatrix &sp) {
printf("values\n");
for (int i = 0; i < sp._size; ++i) {
printf("%lf ", sp.values[i]);
}
printf("\n");
}
