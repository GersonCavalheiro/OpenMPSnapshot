#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "app-desc.h"
#include "bots.h"
#include "strassen.h"
void matrixmul(int n, REAL *A, int an, REAL *B, int bn, REAL *C, int cn)
{
int i, j, k;
REAL s;
for (i = 0; i < n; ++i)
{ 
for (j = 0; j < n; ++j)
{
s = 0.0;
for (k = 0; k < n; ++k) s += ELEM(A, an, i, k) * ELEM(B, bn, k, j);
ELEM(C, cn, i, j) = s;
}
}
}
void FastNaiveMatrixMultiply(REAL *C, REAL *A, REAL *B, unsigned MatrixSize,
unsigned RowWidthC, unsigned RowWidthA, unsigned RowWidthB)
{ 
PTR RowWidthBInBytes = RowWidthB  << 3;
PTR RowWidthAInBytes = RowWidthA << 3;
PTR MatrixWidthInBytes = MatrixSize << 3;
PTR RowIncrementC = ( RowWidthC - MatrixSize) << 3;
unsigned Horizontal, Vertical;
REAL *ARowStart = A;
for (Vertical = 0; Vertical < MatrixSize; Vertical++) {
for (Horizontal = 0; Horizontal < MatrixSize; Horizontal += 8) {
REAL *BColumnStart = B + Horizontal;
REAL FirstARowValue = *ARowStart++;
REAL Sum0 = FirstARowValue * (*BColumnStart);
REAL Sum1 = FirstARowValue * (*(BColumnStart+1));
REAL Sum2 = FirstARowValue * (*(BColumnStart+2));
REAL Sum3 = FirstARowValue * (*(BColumnStart+3));
REAL Sum4 = FirstARowValue * (*(BColumnStart+4));
REAL Sum5 = FirstARowValue * (*(BColumnStart+5));
REAL Sum6 = FirstARowValue * (*(BColumnStart+6));
REAL Sum7 = FirstARowValue * (*(BColumnStart+7));	
unsigned Products;
for (Products = 1; Products < MatrixSize; Products++) {
REAL ARowValue = *ARowStart++;
BColumnStart = (REAL*) (((PTR) BColumnStart) + RowWidthBInBytes);
Sum0 += ARowValue * (*BColumnStart);
Sum1 += ARowValue * (*(BColumnStart+1));
Sum2 += ARowValue * (*(BColumnStart+2));
Sum3 += ARowValue * (*(BColumnStart+3));
Sum4 += ARowValue * (*(BColumnStart+4));
Sum5 += ARowValue * (*(BColumnStart+5));
Sum6 += ARowValue * (*(BColumnStart+6));
Sum7 += ARowValue * (*(BColumnStart+7));	
}
ARowStart = (REAL*) ( ((PTR) ARowStart) - MatrixWidthInBytes);
*(C) = Sum0;
*(C+1) = Sum1;
*(C+2) = Sum2;
*(C+3) = Sum3;
*(C+4) = Sum4;
*(C+5) = Sum5;
*(C+6) = Sum6;
*(C+7) = Sum7;
C+=8;
}
ARowStart = (REAL*) ( ((PTR) ARowStart) + RowWidthAInBytes );
C = (REAL*) ( ((PTR) C) + RowIncrementC );
}
}
void FastAdditiveNaiveMatrixMultiply(REAL *C, REAL *A, REAL *B, unsigned MatrixSize,
unsigned RowWidthC, unsigned RowWidthA, unsigned RowWidthB)
{ 
PTR RowWidthBInBytes = RowWidthB  << 3;
PTR RowWidthAInBytes = RowWidthA << 3;
PTR MatrixWidthInBytes = MatrixSize << 3;
PTR RowIncrementC = ( RowWidthC - MatrixSize) << 3;
unsigned Horizontal, Vertical;
REAL *ARowStart = A;
for (Vertical = 0; Vertical < MatrixSize; Vertical++) {
for (Horizontal = 0; Horizontal < MatrixSize; Horizontal += 8) {
REAL *BColumnStart = B + Horizontal;
REAL Sum0 = *C;
REAL Sum1 = *(C+1);
REAL Sum2 = *(C+2);
REAL Sum3 = *(C+3);
REAL Sum4 = *(C+4);
REAL Sum5 = *(C+5);
REAL Sum6 = *(C+6);
REAL Sum7 = *(C+7);	
unsigned Products;
for (Products = 0; Products < MatrixSize; Products++) {
REAL ARowValue = *ARowStart++;
Sum0 += ARowValue * (*BColumnStart);
Sum1 += ARowValue * (*(BColumnStart+1));
Sum2 += ARowValue * (*(BColumnStart+2));
Sum3 += ARowValue * (*(BColumnStart+3));
Sum4 += ARowValue * (*(BColumnStart+4));
Sum5 += ARowValue * (*(BColumnStart+5));
Sum6 += ARowValue * (*(BColumnStart+6));
Sum7 += ARowValue * (*(BColumnStart+7));
BColumnStart = (REAL*) (((PTR) BColumnStart) + RowWidthBInBytes);
}
ARowStart = (REAL*) ( ((PTR) ARowStart) - MatrixWidthInBytes);
*(C) = Sum0;
*(C+1) = Sum1;
*(C+2) = Sum2;
*(C+3) = Sum3;
*(C+4) = Sum4;
*(C+5) = Sum5;
*(C+6) = Sum6;
*(C+7) = Sum7;
C+=8;
}
ARowStart = (REAL*) ( ((PTR) ARowStart) + RowWidthAInBytes );
C = (REAL*) ( ((PTR) C) + RowIncrementC );
}
}
void MultiplyByDivideAndConquer(REAL *C, REAL *A, REAL *B,
unsigned MatrixSize,
unsigned RowWidthC,
unsigned RowWidthA,
unsigned RowWidthB,
int AdditiveMode
)
{
#define A00 A
#define B00 B
#define C00 C
REAL  *A01, *A10, *A11, *B01, *B10, *B11, *C01, *C10, *C11;
unsigned QuadrantSize = MatrixSize >> 1;
A01 = A00 + QuadrantSize;
A10 = A00 + RowWidthA * QuadrantSize;
A11 = A10 + QuadrantSize;
B01 = B00 + QuadrantSize;
B10 = B00 + RowWidthB * QuadrantSize;
B11 = B10 + QuadrantSize;
C01 = C00 + QuadrantSize;
C10 = C00 + RowWidthC * QuadrantSize;
C11 = C10 + QuadrantSize;
if (QuadrantSize > SizeAtWhichNaiveAlgorithmIsMoreEfficient) {
MultiplyByDivideAndConquer(C00, A00, B00, QuadrantSize,
RowWidthC, RowWidthA, RowWidthB,
AdditiveMode);
MultiplyByDivideAndConquer(C01, A00, B01, QuadrantSize,
RowWidthC, RowWidthA, RowWidthB,
AdditiveMode);
MultiplyByDivideAndConquer(C11, A10, B01, QuadrantSize,
RowWidthC, RowWidthA, RowWidthB,
AdditiveMode);
MultiplyByDivideAndConquer(C10, A10, B00, QuadrantSize,
RowWidthC, RowWidthA, RowWidthB,
AdditiveMode);
MultiplyByDivideAndConquer(C00, A01, B10, QuadrantSize,
RowWidthC, RowWidthA, RowWidthB,
1);
MultiplyByDivideAndConquer(C01, A01, B11, QuadrantSize,
RowWidthC, RowWidthA, RowWidthB,
1);
MultiplyByDivideAndConquer(C11, A11, B11, QuadrantSize,
RowWidthC, RowWidthA, RowWidthB,
1);
MultiplyByDivideAndConquer(C10, A11, B10, QuadrantSize,
RowWidthC, RowWidthA, RowWidthB,
1);
} else {
if (AdditiveMode) {
FastAdditiveNaiveMatrixMultiply(C00, A00, B00, QuadrantSize,
RowWidthC, RowWidthA, RowWidthB);
FastAdditiveNaiveMatrixMultiply(C01, A00, B01, QuadrantSize,
RowWidthC, RowWidthA, RowWidthB);
FastAdditiveNaiveMatrixMultiply(C11, A10, B01, QuadrantSize,
RowWidthC, RowWidthA, RowWidthB);
FastAdditiveNaiveMatrixMultiply(C10, A10, B00, QuadrantSize,
RowWidthC, RowWidthA, RowWidthB);
} else {
FastNaiveMatrixMultiply(C00, A00, B00, QuadrantSize,
RowWidthC, RowWidthA, RowWidthB);
FastNaiveMatrixMultiply(C01, A00, B01, QuadrantSize,
RowWidthC, RowWidthA, RowWidthB);
FastNaiveMatrixMultiply(C11, A10, B01, QuadrantSize,
RowWidthC, RowWidthA, RowWidthB);
FastNaiveMatrixMultiply(C10, A10, B00, QuadrantSize,
RowWidthC, RowWidthA, RowWidthB);
}
FastAdditiveNaiveMatrixMultiply(C00, A01, B10, QuadrantSize,
RowWidthC, RowWidthA, RowWidthB);
FastAdditiveNaiveMatrixMultiply(C01, A01, B11, QuadrantSize,
RowWidthC, RowWidthA, RowWidthB);
FastAdditiveNaiveMatrixMultiply(C11, A11, B11, QuadrantSize,
RowWidthC, RowWidthA, RowWidthB);
FastAdditiveNaiveMatrixMultiply(C10, A11, B10, QuadrantSize,
RowWidthC, RowWidthA, RowWidthB);
}
return;
}
void OptimizedStrassenMultiply_seq(REAL *C, REAL *A, REAL *B, unsigned MatrixSize,
unsigned RowWidthC, unsigned RowWidthA, unsigned RowWidthB, int Depth)
{
unsigned QuadrantSize = MatrixSize >> 1; 
unsigned QuadrantSizeInBytes = sizeof(REAL) * QuadrantSize * QuadrantSize
+ 32;
unsigned Column, Row;
REAL  *A12, *B12, *C12,
*A21, *B21, *C21, *A22, *B22, *C22;
REAL *S1,*S2,*S3,*S4,*S5,*S6,*S7,*S8,*M2,*M5,*T1sMULT;
#define T2sMULT C22
#define NumberOfVariables 11
PTR TempMatrixOffset = 0;
PTR MatrixOffsetA = 0;
PTR MatrixOffsetB = 0;
char *Heap;
void *StartHeap;
PTR RowIncrementA = ( RowWidthA - QuadrantSize ) << 3;
PTR RowIncrementB = ( RowWidthB - QuadrantSize ) << 3;
PTR RowIncrementC = ( RowWidthC - QuadrantSize ) << 3;
if (MatrixSize <= bots_app_cutoff_value) {
MultiplyByDivideAndConquer(C, A, B, MatrixSize, RowWidthC, RowWidthA, RowWidthB, 0);
return;
}
#define A11 A
#define B11 B
#define C11 C
A12 = A11 + QuadrantSize;
B12 = B11 + QuadrantSize;
C12 = C11 + QuadrantSize;
A21 = A + (RowWidthA * QuadrantSize);
B21 = B + (RowWidthB * QuadrantSize);
C21 = C + (RowWidthC * QuadrantSize);
A22 = A21 + QuadrantSize;
B22 = B21 + QuadrantSize;
C22 = C21 + QuadrantSize;
StartHeap = Heap = (char *)malloc(QuadrantSizeInBytes * NumberOfVariables);
if ( ((PTR) Heap) & 31)
Heap = (char*) ( ((PTR) Heap) + 32 - ( ((PTR) Heap) & 31) );
S1 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S2 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S3 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S4 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S5 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S6 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S7 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S8 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
M2 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
M5 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
T1sMULT = (REAL*) Heap; Heap += QuadrantSizeInBytes;
for (Row = 0; Row < QuadrantSize; Row++) {
for (Column = 0; Column < QuadrantSize; Column++) {
#define E(Matrix)   (* (REAL*) ( ((PTR) Matrix) + TempMatrixOffset ) )
#define EA(Matrix)  (* (REAL*) ( ((PTR) Matrix) + MatrixOffsetA ) )
#define EB(Matrix)  (* (REAL*) ( ((PTR) Matrix) + MatrixOffsetB ) )
E(S4) = EA(A12) - ( E(S2) = ( E(S1) = EA(A21) + EA(A22) ) - EA(A11) );
E(S8) = ( E(S6) = EB(B22) - ( E(S5) = EB(B12) - EB(B11) ) ) - EB(B21);
E(S3) = EA(A11) - EA(A21);
E(S7) = EB(B22) - EB(B12);
TempMatrixOffset += sizeof(REAL);
MatrixOffsetA += sizeof(REAL);
MatrixOffsetB += sizeof(REAL);
} 
MatrixOffsetA += RowIncrementA;
MatrixOffsetB += RowIncrementB;
} 
OptimizedStrassenMultiply_seq(M2, A11, B11, QuadrantSize, QuadrantSize, RowWidthA, RowWidthB, Depth+1);
OptimizedStrassenMultiply_seq(M5, S1, S5, QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth+1);
OptimizedStrassenMultiply_seq(T1sMULT, S2, S6,  QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth+1);
OptimizedStrassenMultiply_seq(C22, S3, S7, QuadrantSize, RowWidthC , QuadrantSize, QuadrantSize, Depth+1);
OptimizedStrassenMultiply_seq(C11, A12, B21, QuadrantSize, RowWidthC, RowWidthA, RowWidthB, Depth+1);
OptimizedStrassenMultiply_seq(C12, S4, B22, QuadrantSize, RowWidthC, QuadrantSize, RowWidthB, Depth+1);
OptimizedStrassenMultiply_seq(C21, A22, S8, QuadrantSize, RowWidthC, RowWidthA, QuadrantSize, Depth+1);
for (Row = 0; Row < QuadrantSize; Row++) {
for (Column = 0; Column < QuadrantSize; Column += 4) {
REAL LocalM5_0 = *(M5);
REAL LocalM5_1 = *(M5+1);
REAL LocalM5_2 = *(M5+2);
REAL LocalM5_3 = *(M5+3);
REAL LocalM2_0 = *(M2);
REAL LocalM2_1 = *(M2+1);
REAL LocalM2_2 = *(M2+2);
REAL LocalM2_3 = *(M2+3);
REAL T1_0 = *(T1sMULT) + LocalM2_0;
REAL T1_1 = *(T1sMULT+1) + LocalM2_1;
REAL T1_2 = *(T1sMULT+2) + LocalM2_2;
REAL T1_3 = *(T1sMULT+3) + LocalM2_3;
REAL T2_0 = *(C22) + T1_0;
REAL T2_1 = *(C22+1) + T1_1;
REAL T2_2 = *(C22+2) + T1_2;
REAL T2_3 = *(C22+3) + T1_3;
(*(C11))   += LocalM2_0;
(*(C11+1)) += LocalM2_1;
(*(C11+2)) += LocalM2_2;
(*(C11+3)) += LocalM2_3;
(*(C12))   += LocalM5_0 + T1_0;
(*(C12+1)) += LocalM5_1 + T1_1;
(*(C12+2)) += LocalM5_2 + T1_2;
(*(C12+3)) += LocalM5_3 + T1_3;
(*(C22))   = LocalM5_0 + T2_0;
(*(C22+1)) = LocalM5_1 + T2_1;
(*(C22+2)) = LocalM5_2 + T2_2;
(*(C22+3)) = LocalM5_3 + T2_3;
(*(C21  )) = (- *(C21  )) + T2_0;
(*(C21+1)) = (- *(C21+1)) + T2_1;
(*(C21+2)) = (- *(C21+2)) + T2_2;
(*(C21+3)) = (- *(C21+3)) + T2_3;
M5 += 4;
M2 += 4;
T1sMULT += 4;
C11 += 4;
C12 += 4;
C21 += 4;
C22 += 4;
}
C11 = (REAL*) ( ((PTR) C11 ) + RowIncrementC);
C12 = (REAL*) ( ((PTR) C12 ) + RowIncrementC);
C21 = (REAL*) ( ((PTR) C21 ) + RowIncrementC);
C22 = (REAL*) ( ((PTR) C22 ) + RowIncrementC);
}
free(StartHeap);
}
#if defined(IF_CUTOFF)
void OptimizedStrassenMultiply_par(REAL *C, REAL *A, REAL *B, unsigned MatrixSize,
unsigned RowWidthC, unsigned RowWidthA, unsigned RowWidthB, int Depth)
{
unsigned QuadrantSize = MatrixSize >> 1; 
unsigned QuadrantSizeInBytes = sizeof(REAL) * QuadrantSize * QuadrantSize
+ 32;
unsigned Column, Row;
REAL  *A12, *B12, *C12,
*A21, *B21, *C21, *A22, *B22, *C22;
REAL *S1,*S2,*S3,*S4,*S5,*S6,*S7,*S8,*M2,*M5,*T1sMULT;
#define T2sMULT C22
#define NumberOfVariables 11
PTR TempMatrixOffset = 0;
PTR MatrixOffsetA = 0;
PTR MatrixOffsetB = 0;
char *Heap;
void *StartHeap;
PTR RowIncrementA = ( RowWidthA - QuadrantSize ) << 3;
PTR RowIncrementB = ( RowWidthB - QuadrantSize ) << 3;
PTR RowIncrementC = ( RowWidthC - QuadrantSize ) << 3;
if (MatrixSize <= bots_app_cutoff_value) {
MultiplyByDivideAndConquer(C, A, B, MatrixSize, RowWidthC, RowWidthA, RowWidthB, 0);
return;
}
#define A11 A
#define B11 B
#define C11 C
A12 = A11 + QuadrantSize;
B12 = B11 + QuadrantSize;
C12 = C11 + QuadrantSize;
A21 = A + (RowWidthA * QuadrantSize);
B21 = B + (RowWidthB * QuadrantSize);
C21 = C + (RowWidthC * QuadrantSize);
A22 = A21 + QuadrantSize;
B22 = B21 + QuadrantSize;
C22 = C21 + QuadrantSize;
StartHeap = Heap = (char *)malloc(QuadrantSizeInBytes * NumberOfVariables);
if ( ((PTR) Heap) & 31)
Heap = (char*) ( ((PTR) Heap) + 32 - ( ((PTR) Heap) & 31) );
S1 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S2 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S3 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S4 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S5 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S6 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S7 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S8 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
M2 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
M5 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
T1sMULT = (REAL*) Heap; Heap += QuadrantSizeInBytes;
for (Row = 0; Row < QuadrantSize; Row++) {
for (Column = 0; Column < QuadrantSize; Column++) {
#define E(Matrix)   (* (REAL*) ( ((PTR) Matrix) + TempMatrixOffset ) )
#define EA(Matrix)  (* (REAL*) ( ((PTR) Matrix) + MatrixOffsetA ) )
#define EB(Matrix)  (* (REAL*) ( ((PTR) Matrix) + MatrixOffsetB ) )
E(S4) = EA(A12) - ( E(S2) = ( E(S1) = EA(A21) + EA(A22) ) - EA(A11) );
E(S8) = ( E(S6) = EB(B22) - ( E(S5) = EB(B12) - EB(B11) ) ) - EB(B21);
E(S3) = EA(A11) - EA(A21);
E(S7) = EB(B22) - EB(B12);
TempMatrixOffset += sizeof(REAL);
MatrixOffsetA += sizeof(REAL);
MatrixOffsetB += sizeof(REAL);
} 
MatrixOffsetA += RowIncrementA;
MatrixOffsetB += RowIncrementB;
} 
#pragma omp task untied if (Depth < bots_cutoff_value)
OptimizedStrassenMultiply_par(M2, A11, B11, QuadrantSize, QuadrantSize, RowWidthA, RowWidthB, Depth+1);
#pragma omp task untied if (Depth < bots_cutoff_value)
OptimizedStrassenMultiply_par(M5, S1, S5, QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth+1);
#pragma omp task untied if (Depth < bots_cutoff_value)
OptimizedStrassenMultiply_par(T1sMULT, S2, S6,  QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth+1);
#pragma omp task untied if (Depth < bots_cutoff_value)
OptimizedStrassenMultiply_par(C22, S3, S7, QuadrantSize, RowWidthC , QuadrantSize, QuadrantSize, Depth+1);
#pragma omp task untied if (Depth < bots_cutoff_value)
OptimizedStrassenMultiply_par(C11, A12, B21, QuadrantSize, RowWidthC, RowWidthA, RowWidthB, Depth+1);
#pragma omp task untied if (Depth < bots_cutoff_value)
OptimizedStrassenMultiply_par(C12, S4, B22, QuadrantSize, RowWidthC, QuadrantSize, RowWidthB, Depth+1);
#pragma omp task untied if (Depth < bots_cutoff_value)
OptimizedStrassenMultiply_par(C21, A22, S8, QuadrantSize, RowWidthC, RowWidthA, QuadrantSize, Depth+1);
#pragma omp taskwait
for (Row = 0; Row < QuadrantSize; Row++) {
for (Column = 0; Column < QuadrantSize; Column += 4) {
REAL LocalM5_0 = *(M5);
REAL LocalM5_1 = *(M5+1);
REAL LocalM5_2 = *(M5+2);
REAL LocalM5_3 = *(M5+3);
REAL LocalM2_0 = *(M2);
REAL LocalM2_1 = *(M2+1);
REAL LocalM2_2 = *(M2+2);
REAL LocalM2_3 = *(M2+3);
REAL T1_0 = *(T1sMULT) + LocalM2_0;
REAL T1_1 = *(T1sMULT+1) + LocalM2_1;
REAL T1_2 = *(T1sMULT+2) + LocalM2_2;
REAL T1_3 = *(T1sMULT+3) + LocalM2_3;
REAL T2_0 = *(C22) + T1_0;
REAL T2_1 = *(C22+1) + T1_1;
REAL T2_2 = *(C22+2) + T1_2;
REAL T2_3 = *(C22+3) + T1_3;
(*(C11))   += LocalM2_0;
(*(C11+1)) += LocalM2_1;
(*(C11+2)) += LocalM2_2;
(*(C11+3)) += LocalM2_3;
(*(C12))   += LocalM5_0 + T1_0;
(*(C12+1)) += LocalM5_1 + T1_1;
(*(C12+2)) += LocalM5_2 + T1_2;
(*(C12+3)) += LocalM5_3 + T1_3;
(*(C22))   = LocalM5_0 + T2_0;
(*(C22+1)) = LocalM5_1 + T2_1;
(*(C22+2)) = LocalM5_2 + T2_2;
(*(C22+3)) = LocalM5_3 + T2_3;
(*(C21  )) = (- *(C21  )) + T2_0;
(*(C21+1)) = (- *(C21+1)) + T2_1;
(*(C21+2)) = (- *(C21+2)) + T2_2;
(*(C21+3)) = (- *(C21+3)) + T2_3;
M5 += 4;
M2 += 4;
T1sMULT += 4;
C11 += 4;
C12 += 4;
C21 += 4;
C22 += 4;
}
C11 = (REAL*) ( ((PTR) C11 ) + RowIncrementC);
C12 = (REAL*) ( ((PTR) C12 ) + RowIncrementC);
C21 = (REAL*) ( ((PTR) C21 ) + RowIncrementC);
C22 = (REAL*) ( ((PTR) C22 ) + RowIncrementC);
}
free(StartHeap);
}
#elif defined(MANUAL_CUTOFF)
void OptimizedStrassenMultiply_par(REAL *C, REAL *A, REAL *B, unsigned MatrixSize,
unsigned RowWidthC, unsigned RowWidthA, unsigned RowWidthB, int Depth)
{
unsigned QuadrantSize = MatrixSize >> 1; 
unsigned QuadrantSizeInBytes = sizeof(REAL) * QuadrantSize * QuadrantSize
+ 32;
unsigned Column, Row;
REAL  *A12, *B12, *C12,
*A21, *B21, *C21, *A22, *B22, *C22;
REAL *S1,*S2,*S3,*S4,*S5,*S6,*S7,*S8,*M2,*M5,*T1sMULT;
#define T2sMULT C22
#define NumberOfVariables 11
PTR TempMatrixOffset = 0;
PTR MatrixOffsetA = 0;
PTR MatrixOffsetB = 0;
char *Heap;
void *StartHeap;
PTR RowIncrementA = ( RowWidthA - QuadrantSize ) << 3;
PTR RowIncrementB = ( RowWidthB - QuadrantSize ) << 3;
PTR RowIncrementC = ( RowWidthC - QuadrantSize ) << 3;
if (MatrixSize <= bots_app_cutoff_value) {
MultiplyByDivideAndConquer(C, A, B, MatrixSize, RowWidthC, RowWidthA, RowWidthB, 0);
return;
}
#define A11 A
#define B11 B
#define C11 C
A12 = A11 + QuadrantSize;
B12 = B11 + QuadrantSize;
C12 = C11 + QuadrantSize;
A21 = A + (RowWidthA * QuadrantSize);
B21 = B + (RowWidthB * QuadrantSize);
C21 = C + (RowWidthC * QuadrantSize);
A22 = A21 + QuadrantSize;
B22 = B21 + QuadrantSize;
C22 = C21 + QuadrantSize;
StartHeap = Heap = (char *)malloc(QuadrantSizeInBytes * NumberOfVariables);
if ( ((PTR) Heap) & 31)
Heap = (char*) ( ((PTR) Heap) + 32 - ( ((PTR) Heap) & 31) );
S1 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S2 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S3 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S4 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S5 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S6 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S7 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S8 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
M2 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
M5 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
T1sMULT = (REAL*) Heap; Heap += QuadrantSizeInBytes;
for (Row = 0; Row < QuadrantSize; Row++) {
for (Column = 0; Column < QuadrantSize; Column++) {
#define E(Matrix)   (* (REAL*) ( ((PTR) Matrix) + TempMatrixOffset ) )
#define EA(Matrix)  (* (REAL*) ( ((PTR) Matrix) + MatrixOffsetA ) )
#define EB(Matrix)  (* (REAL*) ( ((PTR) Matrix) + MatrixOffsetB ) )
E(S4) = EA(A12) - ( E(S2) = ( E(S1) = EA(A21) + EA(A22) ) - EA(A11) );
E(S8) = ( E(S6) = EB(B22) - ( E(S5) = EB(B12) - EB(B11) ) ) - EB(B21);
E(S3) = EA(A11) - EA(A21);
E(S7) = EB(B22) - EB(B12);
TempMatrixOffset += sizeof(REAL);
MatrixOffsetA += sizeof(REAL);
MatrixOffsetB += sizeof(REAL);
} 
MatrixOffsetA += RowIncrementA;
MatrixOffsetB += RowIncrementB;
} 
if (Depth < bots_cutoff_value)
{
#pragma omp task untied
OptimizedStrassenMultiply_par(M2, A11, B11, QuadrantSize, QuadrantSize, RowWidthA, RowWidthB, Depth+1);
#pragma omp task untied
OptimizedStrassenMultiply_par(M5, S1, S5, QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth+1);
#pragma omp task untied
OptimizedStrassenMultiply_par(T1sMULT, S2, S6,  QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth+1);
#pragma omp task untied
OptimizedStrassenMultiply_par(C22, S3, S7, QuadrantSize, RowWidthC , QuadrantSize, QuadrantSize, Depth+1);
#pragma omp task untied
OptimizedStrassenMultiply_par(C11, A12, B21, QuadrantSize, RowWidthC, RowWidthA, RowWidthB, Depth+1);
#pragma omp task untied
OptimizedStrassenMultiply_par(C12, S4, B22, QuadrantSize, RowWidthC, QuadrantSize, RowWidthB, Depth+1);
#pragma omp task untied
OptimizedStrassenMultiply_par(C21, A22, S8, QuadrantSize, RowWidthC, RowWidthA, QuadrantSize, Depth+1);
#pragma omp taskwait
}
else
{
OptimizedStrassenMultiply_par(M2, A11, B11, QuadrantSize, QuadrantSize, RowWidthA, RowWidthB, Depth+1);
OptimizedStrassenMultiply_par(M5, S1, S5, QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth+1);
OptimizedStrassenMultiply_par(T1sMULT, S2, S6,  QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth+1);
OptimizedStrassenMultiply_par(C22, S3, S7, QuadrantSize, RowWidthC , QuadrantSize, QuadrantSize, Depth+1);
OptimizedStrassenMultiply_par(C11, A12, B21, QuadrantSize, RowWidthC, RowWidthA, RowWidthB, Depth+1);
OptimizedStrassenMultiply_par(C12, S4, B22, QuadrantSize, RowWidthC, QuadrantSize, RowWidthB, Depth+1);
OptimizedStrassenMultiply_par(C21, A22, S8, QuadrantSize, RowWidthC, RowWidthA, QuadrantSize, Depth+1);
}
for (Row = 0; Row < QuadrantSize; Row++) {
for (Column = 0; Column < QuadrantSize; Column += 4) {
REAL LocalM5_0 = *(M5);
REAL LocalM5_1 = *(M5+1);
REAL LocalM5_2 = *(M5+2);
REAL LocalM5_3 = *(M5+3);
REAL LocalM2_0 = *(M2);
REAL LocalM2_1 = *(M2+1);
REAL LocalM2_2 = *(M2+2);
REAL LocalM2_3 = *(M2+3);
REAL T1_0 = *(T1sMULT) + LocalM2_0;
REAL T1_1 = *(T1sMULT+1) + LocalM2_1;
REAL T1_2 = *(T1sMULT+2) + LocalM2_2;
REAL T1_3 = *(T1sMULT+3) + LocalM2_3;
REAL T2_0 = *(C22) + T1_0;
REAL T2_1 = *(C22+1) + T1_1;
REAL T2_2 = *(C22+2) + T1_2;
REAL T2_3 = *(C22+3) + T1_3;
(*(C11))   += LocalM2_0;
(*(C11+1)) += LocalM2_1;
(*(C11+2)) += LocalM2_2;
(*(C11+3)) += LocalM2_3;
(*(C12))   += LocalM5_0 + T1_0;
(*(C12+1)) += LocalM5_1 + T1_1;
(*(C12+2)) += LocalM5_2 + T1_2;
(*(C12+3)) += LocalM5_3 + T1_3;
(*(C22))   = LocalM5_0 + T2_0;
(*(C22+1)) = LocalM5_1 + T2_1;
(*(C22+2)) = LocalM5_2 + T2_2;
(*(C22+3)) = LocalM5_3 + T2_3;
(*(C21  )) = (- *(C21  )) + T2_0;
(*(C21+1)) = (- *(C21+1)) + T2_1;
(*(C21+2)) = (- *(C21+2)) + T2_2;
(*(C21+3)) = (- *(C21+3)) + T2_3;
M5 += 4;
M2 += 4;
T1sMULT += 4;
C11 += 4;
C12 += 4;
C21 += 4;
C22 += 4;
}
C11 = (REAL*) ( ((PTR) C11 ) + RowIncrementC);
C12 = (REAL*) ( ((PTR) C12 ) + RowIncrementC);
C21 = (REAL*) ( ((PTR) C21 ) + RowIncrementC);
C22 = (REAL*) ( ((PTR) C22 ) + RowIncrementC);
}
free(StartHeap);
}
#else
void OptimizedStrassenMultiply_par(REAL *C, REAL *A, REAL *B, unsigned MatrixSize,
unsigned RowWidthC, unsigned RowWidthA, unsigned RowWidthB, int Depth)
{
unsigned QuadrantSize = MatrixSize >> 1; 
unsigned QuadrantSizeInBytes = sizeof(REAL) * QuadrantSize * QuadrantSize
+ 32;
unsigned Column, Row;
REAL  *A12, *B12, *C12,
*A21, *B21, *C21, *A22, *B22, *C22;
REAL *S1,*S2,*S3,*S4,*S5,*S6,*S7,*S8,*M2,*M5,*T1sMULT;
#define T2sMULT C22
#define NumberOfVariables 11
PTR TempMatrixOffset = 0;
PTR MatrixOffsetA = 0;
PTR MatrixOffsetB = 0;
char *Heap;
void *StartHeap;
PTR RowIncrementA = ( RowWidthA - QuadrantSize ) << 3;
PTR RowIncrementB = ( RowWidthB - QuadrantSize ) << 3;
PTR RowIncrementC = ( RowWidthC - QuadrantSize ) << 3;
if (MatrixSize <= bots_app_cutoff_value) {
MultiplyByDivideAndConquer(C, A, B, MatrixSize, RowWidthC, RowWidthA, RowWidthB, 0);
return;
}
#define A11 A
#define B11 B
#define C11 C
A12 = A11 + QuadrantSize;
B12 = B11 + QuadrantSize;
C12 = C11 + QuadrantSize;
A21 = A + (RowWidthA * QuadrantSize);
B21 = B + (RowWidthB * QuadrantSize);
C21 = C + (RowWidthC * QuadrantSize);
A22 = A21 + QuadrantSize;
B22 = B21 + QuadrantSize;
C22 = C21 + QuadrantSize;
StartHeap = Heap = (char *)malloc(QuadrantSizeInBytes * NumberOfVariables);
if ( ((PTR) Heap) & 31)
Heap = (char*) ( ((PTR) Heap) + 32 - ( ((PTR) Heap) & 31) );
S1 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S2 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S3 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S4 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S5 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S6 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S7 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
S8 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
M2 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
M5 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
T1sMULT = (REAL*) Heap; Heap += QuadrantSizeInBytes;
for (Row = 0; Row < QuadrantSize; Row++) {
for (Column = 0; Column < QuadrantSize; Column++) {
#define E(Matrix)   (* (REAL*) ( ((PTR) Matrix) + TempMatrixOffset ) )
#define EA(Matrix)  (* (REAL*) ( ((PTR) Matrix) + MatrixOffsetA ) )
#define EB(Matrix)  (* (REAL*) ( ((PTR) Matrix) + MatrixOffsetB ) )
E(S4) = EA(A12) - ( E(S2) = ( E(S1) = EA(A21) + EA(A22) ) - EA(A11) );
E(S8) = ( E(S6) = EB(B22) - ( E(S5) = EB(B12) - EB(B11) ) ) - EB(B21);
E(S3) = EA(A11) - EA(A21);
E(S7) = EB(B22) - EB(B12);
TempMatrixOffset += sizeof(REAL);
MatrixOffsetA += sizeof(REAL);
MatrixOffsetB += sizeof(REAL);
} 
MatrixOffsetA += RowIncrementA;
MatrixOffsetB += RowIncrementB;
} 
#pragma omp task untied
OptimizedStrassenMultiply_par(M2, A11, B11, QuadrantSize, QuadrantSize, RowWidthA, RowWidthB, Depth+1);
#pragma omp task untied
OptimizedStrassenMultiply_par(M5, S1, S5, QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth+1);
#pragma omp task untied
OptimizedStrassenMultiply_par(T1sMULT, S2, S6,  QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth+1);
#pragma omp task untied
OptimizedStrassenMultiply_par(C22, S3, S7, QuadrantSize, RowWidthC , QuadrantSize, QuadrantSize, Depth+1);
#pragma omp task untied
OptimizedStrassenMultiply_par(C11, A12, B21, QuadrantSize, RowWidthC, RowWidthA, RowWidthB, Depth+1);
#pragma omp task untied
OptimizedStrassenMultiply_par(C12, S4, B22, QuadrantSize, RowWidthC, QuadrantSize, RowWidthB, Depth+1);
#pragma omp task untied
OptimizedStrassenMultiply_par(C21, A22, S8, QuadrantSize, RowWidthC, RowWidthA, QuadrantSize, Depth+1);
#pragma omp taskwait
for (Row = 0; Row < QuadrantSize; Row++) {
for (Column = 0; Column < QuadrantSize; Column += 4) {
REAL LocalM5_0 = *(M5);
REAL LocalM5_1 = *(M5+1);
REAL LocalM5_2 = *(M5+2);
REAL LocalM5_3 = *(M5+3);
REAL LocalM2_0 = *(M2);
REAL LocalM2_1 = *(M2+1);
REAL LocalM2_2 = *(M2+2);
REAL LocalM2_3 = *(M2+3);
REAL T1_0 = *(T1sMULT) + LocalM2_0;
REAL T1_1 = *(T1sMULT+1) + LocalM2_1;
REAL T1_2 = *(T1sMULT+2) + LocalM2_2;
REAL T1_3 = *(T1sMULT+3) + LocalM2_3;
REAL T2_0 = *(C22) + T1_0;
REAL T2_1 = *(C22+1) + T1_1;
REAL T2_2 = *(C22+2) + T1_2;
REAL T2_3 = *(C22+3) + T1_3;
(*(C11))   += LocalM2_0;
(*(C11+1)) += LocalM2_1;
(*(C11+2)) += LocalM2_2;
(*(C11+3)) += LocalM2_3;
(*(C12))   += LocalM5_0 + T1_0;
(*(C12+1)) += LocalM5_1 + T1_1;
(*(C12+2)) += LocalM5_2 + T1_2;
(*(C12+3)) += LocalM5_3 + T1_3;
(*(C22))   = LocalM5_0 + T2_0;
(*(C22+1)) = LocalM5_1 + T2_1;
(*(C22+2)) = LocalM5_2 + T2_2;
(*(C22+3)) = LocalM5_3 + T2_3;
(*(C21  )) = (- *(C21  )) + T2_0;
(*(C21+1)) = (- *(C21+1)) + T2_1;
(*(C21+2)) = (- *(C21+2)) + T2_2;
(*(C21+3)) = (- *(C21+3)) + T2_3;
M5 += 4;
M2 += 4;
T1sMULT += 4;
C11 += 4;
C12 += 4;
C21 += 4;
C22 += 4;
}
C11 = (REAL*) ( ((PTR) C11 ) + RowIncrementC);
C12 = (REAL*) ( ((PTR) C12 ) + RowIncrementC);
C21 = (REAL*) ( ((PTR) C21 ) + RowIncrementC);
C22 = (REAL*) ( ((PTR) C22 ) + RowIncrementC);
}
free(StartHeap);
}
#endif
void init_matrix(int n, REAL *A, int an)
{
int i, j;
for (i = 0; i < n; ++i)
for (j = 0; j < n; ++j) 
ELEM(A, an, i, j) = ((double) rand()) / (double) RAND_MAX; 
}
int compare_matrix(int n, REAL *A, int an, REAL *B, int bn)
{
int i, j;
REAL c;
for (i = 0; i < n; ++i)
for (j = 0; j < n; ++j) {
c = ELEM(A, an, i, j) - ELEM(B, bn, i, j);
if (c < 0.0) 
c = -c;
c = c / ELEM(A, an, i, j);
if (c > EPSILON) {
bots_message("Strassen: Wrong answer!\n");
return BOTS_RESULT_UNSUCCESSFUL;
}
}
return BOTS_RESULT_SUCCESSFUL;
}
REAL *alloc_matrix(int n) 
{
return (REAL *)malloc(n * n * sizeof(REAL));
}
void strassen_main_par(REAL *A, REAL *B, REAL *C, int n)
{
bots_message("Computing parallel Strassen algorithm (n=%d) ", n);
#pragma omp parallel
#pragma omp single
#pragma omp task untied     
OptimizedStrassenMultiply_par(C, A, B, n, n, n, n, 1);
bots_message(" completed!\n");
}
void strassen_main_seq(REAL *A, REAL *B, REAL *C, int n)
{
bots_message("Computing sequential Strassen algorithm (n=%d) ", n);
OptimizedStrassenMultiply_seq(C, A, B, n, n, n, n, 1);
bots_message(" completed!\n");
}
