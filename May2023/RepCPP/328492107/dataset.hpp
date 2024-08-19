

#pragma once

#include "interface.hpp"


class dataset
{
public:
int samples, dimensions, classes;
double** X, ** Y;

ssize_t getline(char** lineptr, size_t* n, FILE* stream);
void read_csv(const char* filename, int dataset_flag, double x_max);
int get_label(int sample);
void print_dataset(void);

dataset(int classes, int samples) :
classes{ classes },
samples{ samples }
{
dimensions = 0;
}

~dataset()
{
for (int i = 0; i < samples; i += 1)
{
delete[] X[i];
delete[] Y[i];
}
delete[] X;
delete[] Y;
}
};
