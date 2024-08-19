#pragma once

#include <iostream>

#include <mpi.h>

#include <climits>

#include "constants.h"

class ConfigWeight {
public:
ConfigWeight(int configLength, long weight, short* config);

explicit ConfigWeight(int configLength);

virtual ~ConfigWeight();

virtual void send(int destination, int tag = TAG_RESULT);

virtual MPI_Status receive(int destination = MPI_ANY_SOURCE, int tag = TAG_RESULT);

virtual long size();

long getWeight();

short* getConfig();

void setWeightAndConfig(long weight, short* config);

protected:
long weight;       
int configLength;  
short* config;     
};
