#pragma once
#ifndef PROGRAMDATA_H_
#define PROGRAMDATA_H_
#include <mpi.h>
#include "mpi.h"
#include "calculations.h"
#include "ompFunctions.h"
#define INPUT_FILE_NAME "input.txt"
#define OUTPUT_FILE_NAME "output.txt"
#define NUM_PROCESSES 2
InputData readInputFile(const char *filename);
void updateWeightsAccordingToScoreFunction(float weights[NUM_WEIGHTS]);
void sendInputDataToOtherProcess(InputData *inputData, int numSeq2ToSend, int dest, int tag);
void receiveInputDataFromMainProcess(InputData *inputData, int *numSeq2ToReceive, int source, int tag);
void writeOutputFile(const char *filename, int numOfSeq2, Seq2Result *seq2Results);
void printData(InputData inputData, int arraysLen);
void freeData(InputData inputData, int arraysLen);
#endif 
