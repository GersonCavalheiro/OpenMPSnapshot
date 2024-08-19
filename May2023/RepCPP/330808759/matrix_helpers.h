
#pragma once

#include <string>
#include <chrono>


void fillRandomMatrix(long *matrix, int height, int width, int range);


void resetMatrix(long *matrix, int height, int width);


void printMatrix(long *matrix, int height, int width, std::string name);


float calcAvg(long *array, int length);


void printTimeDiff(std::chrono::duration<long int, std::ratio<1, 1000000> > duration, std::string algorithm);

long checkCorrectness(long *matrix, long *reference, int height, int width);

void printAverages(std::string *alg_names, float *alg_averages, int limit, int linepoint);


float doTheMeasuring(void func(long *result, long *a, long *b, int heightR, int widthR, int widthA),
long *result, long *a, long *b, int heightR, int widthR, int widthA,
int loops, bool print, long *refMatrix, std::string description);
float doIncreasingMeasuring(void func(long *result, long *a, long *b, int heightR, int widthR, int widthA, int threads),
long *result, long *a, long *b, int size,
int loops, long *refMatrix, std::string description, int threads);