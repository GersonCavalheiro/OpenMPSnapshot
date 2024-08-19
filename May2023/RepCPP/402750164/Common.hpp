#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "mpi.h"
#include "../init_conds/InitialConditions.hpp"

void printLog(int numberOfProcesses,
int sumSize,
double epsilon,
bool isDebugMode);

void readParameters(InitialConditions &initialConditions);

void divideResponsibilities(std::vector<double> &y,
std::vector<double> &yLocal,
std::vector<double> &yLocalPrevious,
int numberOfProcesses,
int processId,
int &localSize,
int &localDisplacement,
int &localRows,
int &localOffsetInRows,
std::vector<int> &processesLocationSizes,
std::vector<int> &processesDisplacement,
InitialConditions initialConditions);

void printProcessLocations(int numberOfProcesses,
std::vector<int> processesDisplacement,
std::vector<int> processesLocalRows,
std::vector<int> &processesOffsetInRows,
bool isDebugMode);

void fillVectorWithZeros(std::vector<double> &y);

void init(int processId,
std::vector<double> &y,
std::vector<double> &yLocal,
std::vector<double> &yLocalPrevious,
std::vector<double> &yLocalHighBorder,
std::vector<double> &yLocalLowBorder,
std::vector<double> &buf1,
std::vector<double> &buf2,
InitialConditions initialConditions,
int localSize);

void setUpLocations(std::vector<int> &processesLocationSizes,
std::vector<int> &processesDisplacement,
std::vector<int> &processesLocalRows,
std::vector<int> &processesOffsetInRows,
int numberOfProcesses,
InitialConditions initialConditions);

void printProcessData(std::vector<double> yLocal,
std::vector<double> yLocalPrevious,
int processId,
int localSize,
int localDisplacement,
int localRows,
int localOffsetInRows,
bool isDebugMode);

void printMethodStatistic(std::string methodName,
int finalNorm,
int iterationsNumber,
double timeStart,
double timeEnd,
bool isDebugMode);

double infiniteNorm(std::vector<double> &x, std::vector<double> &y);

void copyFirstRow(std::vector<double> &yLocal,
std::vector<double> &localHighBorder,
InitialConditions initialConditions);

void copyLastRow(std::vector<double> &yLocal,
std::vector<double> &localLowBorder,
InitialConditions initialConditions);

void exchangeDataV1(std::vector<double> &yLocalPrevious,
std::vector<double> &yLocalPreviousUpHighBorder,
std::vector<double> &yLocalPreviousDownLowBorder,
std::vector<double> &buf1,
std::vector<double> &buf2,
int numberOfProcesses,
int processId,
InitialConditions initialConditions);

void exchangeDataV2(std::vector<double> &yLocal,
std::vector<double> &yLocalPreviousUpHighBorder,
std::vector<double> &yLocalPreviousDownLowBorder,
std::vector<double> &buf1,
std::vector<double> &buf2,
int numberOfProcesses,
int processId,
InitialConditions initialConditions);

void setSourceAndDestination(const int numberOfProcesses,
const int processId,
int &higherRankProcess,
int &lowerRankProcess);