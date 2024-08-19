#pragma once

#include "ConfigWeight.h"

class ConfigWeightTask : public ConfigWeight {
public:
ConfigWeightTask(int configLength, long weight, short* config, short* task);

ConfigWeightTask(int configLength, long weight, short* config);

explicit ConfigWeightTask(int configLength);

~ConfigWeightTask() override;

void send(int destination, int tag = TAG_RESULT) override;

MPI_Status receive(int destination = MPI_ANY_SOURCE, int tag = TAG_RESULT) override;

long size() override;
short* getTask();

protected:
short* task;  
};