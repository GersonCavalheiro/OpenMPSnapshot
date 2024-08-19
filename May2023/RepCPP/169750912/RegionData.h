#pragma once

#include "WorkerInfo.h"


struct RegionData {



WorkerInfo workerInfo;


unsigned long mpiCommunicationTime;


unsigned short int *data;
int data_length;

};