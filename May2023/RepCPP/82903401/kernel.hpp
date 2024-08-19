#pragma once

class HybridCuda {
public:
HybridCuda(size_t divisionPoint, size_t pitch, int deviceId);

~HybridCuda();

void launchCompute(float* temperature_in);
void finalizeCompute(float* temperature_out);

void copyInitial(float* temperature_in);
void copyFinal(float* temperature_out);

private:
const size_t DIVISION_POINT;

size_t allocNumRows;
const size_t pitch;
size_t d_pitch;
float *d_temperature_in, *d_temperature_out;
int deviceId;

enum Part { TOP, BOTTOM };
Part part;

void setDevice();
};