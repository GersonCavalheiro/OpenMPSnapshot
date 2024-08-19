#pragma once
#ifndef CALCULATIONS_H_
#define CALCULATIONS_H_
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "structs.h"
void calculateScoringMatrixOnGPU(Seq2Result *seq2Results, InputData *inputData, int totalNumOfSeq2, int cudaNumSeq2);
void computeSumMatrix(float **sumMatrix, float **scoringMatrix, int len1, int len2, float minW, float penalty);
void computeBestScore(Seq2Result *seq2Res, float **scoringMatrix, float **sumMatrix, int len1, int len2, float minW);
float computeMutantSequenceKScore(int k, int n, float **scoringMatrix, int len1, int len2);
void freeMatrix(float **scoringMatrix, int matrixLength);
#endif 
