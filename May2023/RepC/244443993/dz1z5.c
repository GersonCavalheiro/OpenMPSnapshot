#include "dz1z5.h"
float kernel_value_CPU(float v)
{
float rValue = 0;
const float z = v * v;
float num = (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z *
(z * 0.210580722890567e-22f + 0.380715242345326e-19f) +
0.479440257548300e-16f) +
0.435125971262668e-13f) +
0.300931127112960e-10f) +
0.160224679395361e-7f) +
0.654858370096785e-5f) +
0.202591084143397e-2f) +
0.463076284721000e0f) +
0.754337328948189e2f) +
0.830792541809429e4f) +
0.571661130563785e6f) +
0.216415572361227e8f) +
0.356644482244025e9f) +
0.144048298227235e10f);
float den = (z * (z * (z - 0.307646912682801e4f) + 0.347626332405882e7f) - 0.144048298227235e10f);
rValue = -num / den;
return rValue;
}
void calculateLUT(float beta, float width, float **LUT, unsigned int *sizeLUT)
{
float v;
float cutoff2 = (width * width) / 4.0;
unsigned int size;
if (width > 0)
{
size = (unsigned int)(10000 * width);
(*LUT) = (float *)malloc(size * sizeof(float));
unsigned int k;
for (k = 0; k < size; ++k)
{
v = (((float)k) / ((float)size)) * cutoff2;
(*LUT)[k] = kernel_value_CPU(beta * sqrt(1.0 - (v / cutoff2)));
}
(*sizeLUT) = size;
}
}
float kernel_value_LUT(float v, float *LUT, int sizeLUT, float _1overCutoff2)
{
unsigned int k0;
float v0;
v *= (float)sizeLUT;
k0 = (unsigned int)(v * _1overCutoff2);
v0 = ((float)k0) / _1overCutoff2;
return LUT[k0] + ((v - v0) * (LUT[k0 + 1] - LUT[k0]) / _1overCutoff2);
}
int gridding_Gold(unsigned int n, parameters params, ReconstructionSample *sample, float *LUT, unsigned int sizeLUT, cmplx *gridData, float *sampleDensity)
{
unsigned int NxL, NxH;
unsigned int NyL, NyH;
unsigned int NzL, NzH;
int nx;
int ny;
int nz;
float w;
unsigned int idx;
unsigned int idx0;
unsigned int idxZ;
unsigned int idxY;
float Dx2[100];
float Dy2[100];
float Dz2[100];
float *dx2 = NULL;
float *dy2 = NULL;
float *dz2 = NULL;
float dy2dz2;
float v;
unsigned int size_x = params.gridSize[0];
unsigned int size_y = params.gridSize[1];
unsigned int size_z = params.gridSize[2];
float cutoff = ((float)(params.kernelWidth)) / 2.0;
float cutoff2 = cutoff * cutoff;
float _1overCutoff2 = 1 / cutoff2;
float beta = PI * sqrt(4 * params.kernelWidth * params.kernelWidth / (params.oversample * params.oversample) * (params.oversample - .5) * (params.oversample - .5) - .8);
int i;
for (i = 0; i < n; i++)
{
ReconstructionSample pt = sample[i];
float kx = pt.kX;
float ky = pt.kY;
float kz = pt.kZ;
NxL = max((kx - cutoff), 0.0);
NxH = min((kx + cutoff), size_x - 1.0);
NyL = max((ky - cutoff), 0.0);
NyH = min((ky + cutoff), size_y - 1.0);
NzL = max((kz - cutoff), 0.0);
NzH = min((kz + cutoff), size_z - 1.0);
if ((pt.real != 0.0 || pt.imag != 0.0) && pt.sdc != 0.0)
{
for (dz2 = Dz2, nz = NzL; nz <= NzH; ++nz, ++dz2)
{
*dz2 = ((kz - nz) * (kz - nz));
}
for (dx2 = Dx2, nx = NxL; nx <= NxH; ++nx, ++dx2)
{
*dx2 = ((kx - nx) * (kx - nx));
}
for (dy2 = Dy2, ny = NyL; ny <= NyH; ++ny, ++dy2)
{
*dy2 = ((ky - ny) * (ky - ny));
}
idxZ = (NzL - 1) * size_x * size_y;
for (dz2 = Dz2, nz = NzL; nz <= NzH; ++nz, ++dz2)
{
idxZ += size_x * size_y;
idxY = (NyL - 1) * size_x;
if ((*dz2) < cutoff2)
{
for (dy2 = Dy2, ny = NyL; ny <= NyH; ++ny, ++dy2)
{
idxY += size_x;
dy2dz2 = (*dz2) + (*dy2);
idx0 = idxY + idxZ;
if (dy2dz2 < cutoff2)
{
for (dx2 = Dx2, nx = NxL; nx <= NxH; ++nx, ++dx2)
{
v = dy2dz2 + (*dx2);
if (v < cutoff2)
{
idx = nx + idx0;
if (params.useLUT)
{
w = kernel_value_LUT(v, LUT, sizeLUT, _1overCutoff2) * pt.sdc;
}
else
{
w = kernel_value_CPU(beta * sqrt(1.0 - (v * _1overCutoff2))) * pt.sdc;
}
gridData[idx].real += (w * pt.real);
gridData[idx].imag += (w * pt.imag);
sampleDensity[idx] += 1.0;
}
}
}
}
}
}
}
}
}
int gridding_Gold_Parallel(unsigned int n, parameters params, ReconstructionSample *sample, float *LUT, unsigned int sizeLUT, cmplx *gridData, float *sampleDensity)
{
unsigned int size_x = params.gridSize[0];
unsigned int size_y = params.gridSize[1];
unsigned int size_z = params.gridSize[2];
float cutoff = ((float)(params.kernelWidth)) / 2.0;
float cutoff2 = cutoff * cutoff;
float _1overCutoff2 = 1 / cutoff2;
float beta = PI * sqrt(4 * params.kernelWidth * params.kernelWidth / (params.oversample * params.oversample) * (params.oversample - .5) * (params.oversample - .5) - .8);
#pragma omp parallel for schedule(dynamic, 5000) default(none) shared(gridData, sampleDensity, beta, LUT, sizeLUT, params, n, cutoff, cutoff2, _1overCutoff2, size_x, size_y, size_z, sample)
for (int i = 0; i < n; i++)
{
ReconstructionSample pt = sample[i];
unsigned int NxL, NxH;
unsigned int NyL, NyH;
unsigned int NzL, NzH;
int nx;
int ny;
int nz;
float w;
unsigned int idx;
unsigned int idx0;
unsigned int idxZ;
unsigned int idxY;
float Dx2[100];
float Dy2[100];
float Dz2[100];
float *dx2 = NULL;
float *dy2 = NULL;
float *dz2 = NULL;
float dy2dz2;
float v;
float kx = pt.kX;
float ky = pt.kY;
float kz = pt.kZ;
NxL = max((kx - cutoff), 0.0);
NxH = min((kx + cutoff), size_x - 1.0);
NyL = max((ky - cutoff), 0.0);
NyH = min((ky + cutoff), size_y - 1.0);
NzL = max((kz - cutoff), 0.0);
NzH = min((kz + cutoff), size_z - 1.0);
if ((pt.real != 0.0 || pt.imag != 0.0) && pt.sdc != 0.0)
{
for (dz2 = Dz2, nz = NzL; nz <= NzH; ++nz, ++dz2)
{
*dz2 = ((kz - nz) * (kz - nz));
}
for (dx2 = Dx2, nx = NxL; nx <= NxH; ++nx, ++dx2)
{
*dx2 = ((kx - nx) * (kx - nx));
}
for (dy2 = Dy2, ny = NyL; ny <= NyH; ++ny, ++dy2)
{
*dy2 = ((ky - ny) * (ky - ny));
}
idxZ = (NzL - 1) * size_x * size_y;
for (dz2 = Dz2, nz = NzL; nz <= NzH; ++nz, ++dz2)
{
idxZ += size_x * size_y;
idxY = (NyL - 1) * size_x;
if ((*dz2) < cutoff2)
{
for (dy2 = Dy2, ny = NyL; ny <= NyH; ++ny, ++dy2)
{
idxY += size_x;
dy2dz2 = (*dz2) + (*dy2);
idx0 = idxY + idxZ;
if (dy2dz2 < cutoff2)
{
for (dx2 = Dx2, nx = NxL; nx <= NxH; ++nx, ++dx2)
{
v = dy2dz2 + (*dx2);
if (v < cutoff2)
{
idx = nx + idx0;
if (params.useLUT)
{
w = kernel_value_LUT(v, LUT, sizeLUT, _1overCutoff2) * pt.sdc;
}
else
{
w = kernel_value_CPU(beta * sqrt(1.0 - (v * _1overCutoff2))) * pt.sdc;
}
#pragma omp atomic
gridData[idx].real += (w * pt.real);
#pragma omp atomic
gridData[idx].imag += (w * pt.imag);
#pragma omp atomic
sampleDensity[idx] += 1.0;
}
}
}
}
}
}
}
}
}
void setParameters(FILE *file, parameters *p)
{
fscanf(file, "aquisition.numsamples=%d\n", &(p->numSamples));
fscanf(file, "aquisition.kmax=%f %f %f\n", &(p->kMax[0]), &(p->kMax[1]), &(p->kMax[2]));
fscanf(file, "aquisition.matrixSize=%d %d %d\n", &(p->aquisitionMatrixSize[0]), &(p->aquisitionMatrixSize[1]), &(p->aquisitionMatrixSize[2]));
fscanf(file, "reconstruction.matrixSize=%d %d %d\n", &(p->reconstructionMatrixSize[0]), &(p->reconstructionMatrixSize[1]), &(p->reconstructionMatrixSize[2]));
fscanf(file, "gridding.matrixSize=%d %d %d\n", &(p->gridSize[0]), &(p->gridSize[1]), &(p->gridSize[2]));
fscanf(file, "gridding.oversampling=%f\n", &(p->oversample));
fscanf(file, "kernel.width=%f\n", &(p->kernelWidth));
fscanf(file, "kernel.useLUT=%d\n", &(p->useLUT));
printf("	Number of samples = %d\n", p->numSamples);
printf("	Grid Size = %dx%dx%d\n", p->gridSize[0], p->gridSize[1], p->gridSize[2]);
printf("	Input Matrix Size = %dx%dx%d\n", p->aquisitionMatrixSize[0], p->aquisitionMatrixSize[1], p->aquisitionMatrixSize[2]);
printf("	Recon Matrix Size = %dx%dx%d\n", p->reconstructionMatrixSize[0], p->reconstructionMatrixSize[1], p->reconstructionMatrixSize[2]);
printf("	Kernel Width = %f\n", p->kernelWidth);
printf("	KMax = %.2f %.2f %.2f\n", p->kMax[0], p->kMax[1], p->kMax[2]);
printf("	Oversampling = %f\n", p->oversample);
printf("	GPU Binsize = %d\n", p->binsize);
printf("	Use LUT = %s\n", (p->useLUT) ? "Yes" : "No");
}
unsigned int readSampleData(parameters params, FILE *uksdata_f, ReconstructionSample *samples)
{
unsigned int i;
for (i = 0; i < params.numSamples; i++)
{
if (feof(uksdata_f))
{
break;
}
fread((void *)&(samples[i]), sizeof(ReconstructionSample), 1, uksdata_f);
}
float kScale[3];
kScale[0] = (float)(params.aquisitionMatrixSize[0]) / ((float)(params.reconstructionMatrixSize[0]) * (float)(params.kMax[0]));
kScale[1] = (float)(params.aquisitionMatrixSize[1]) / ((float)(params.reconstructionMatrixSize[1]) * (float)(params.kMax[1]));
kScale[2] = (float)(params.aquisitionMatrixSize[2]) / ((float)(params.reconstructionMatrixSize[2]) * (float)(params.kMax[2]));
int size_x = params.gridSize[0];
int size_y = params.gridSize[1];
int size_z = params.gridSize[2];
float ax = (kScale[0] * (size_x - 1)) / 2.0;
float bx = (float)(size_x - 1) / 2.0;
float ay = (kScale[1] * (size_y - 1)) / 2.0;
float by = (float)(size_y - 1) / 2.0;
float az = (kScale[2] * (size_z - 1)) / 2.0;
float bz = (float)(size_z - 1) / 2.0;
int n;
for (n = 0; n < i; n++)
{
samples[n].kX = floor((samples[n].kX * ax) + bx);
samples[n].kY = floor((samples[n].kY * ay) + by);
samples[n].kZ = floor((samples[n].kZ * az) + bz);
}
return i;
}
int main(int argc, char *argv[])
{
char uksfile[256];
char uksdata[256];
parameters params;
FILE *uksfile_f = NULL;
FILE *uksdata_f = NULL;
if (argc != 3)
return 1;
strcpy(uksfile, argv[1]);
strcpy(uksdata, argv[1]);
strcat(uksdata, ".data");
uksfile_f = fopen(uksfile, "r");
if (uksfile_f == NULL)
{
printf("ERROR: Could not open %s\n", uksfile);
exit(1);
}
printf("\nReading parameters\n");
if (argc >= 2)
{
params.binsize = atoi(argv[2]);
}
else
{
params.binsize = 128;
}
setParameters(uksfile_f, &params);
ReconstructionSample *samples = (ReconstructionSample *)malloc(params.numSamples * sizeof(ReconstructionSample));
float *LUT, *LUTParallel;
unsigned int sizeLUT, sizeLUTParallel;
int gridNumElems = params.gridSize[0] * params.gridSize[1] * params.gridSize[2];
cmplx *gridData = (cmplx *)calloc(gridNumElems, sizeof(cmplx));
float *sampleDensity = (float *)calloc(gridNumElems, sizeof(float));
cmplx *gridDataParallel = (cmplx *)calloc(gridNumElems, sizeof(cmplx));
float *sampleDensityParallel = (float *)calloc(gridNumElems, sizeof(float));
if (samples == NULL)
{
printf("ERROR: Unable to allocate memory for input data\n");
exit(1);
}
if (sampleDensity == NULL || gridData == NULL)
{
printf("ERROR: Unable to allocate memory for output data\n");
exit(1);
}
uksdata_f = fopen(uksdata, "rb");
if (uksdata_f == NULL)
{
printf("ERROR: Could not open data file\n");
exit(1);
}
printf("Reading input data from files\n");
unsigned int n = readSampleData(params, uksdata_f, samples);
fclose(uksdata_f);
double timeSequential, timeParallel;
if (params.useLUT)
{
printf("Generating Look-Up Table\n");
float beta = PI * sqrt(4 * params.kernelWidth * params.kernelWidth / (params.oversample * params.oversample) * (params.oversample - .5) * (params.oversample - .5) - .8);
timeSequential = omp_get_wtime();
calculateLUT(beta, params.kernelWidth, &LUT, &sizeLUT);
timeSequential = omp_get_wtime() - timeSequential;
timeParallel = omp_get_wtime();
calculateLUT(beta, params.kernelWidth, &LUTParallel, &sizeLUTParallel);
timeParallel = omp_get_wtime() - timeParallel;
}
timeSequential += omp_get_wtime();
gridding_Gold(n, params, samples, LUT, sizeLUT, gridData, sampleDensity);
timeSequential = omp_get_wtime() - timeSequential;
timeParallel += omp_get_wtime();
gridding_Gold_Parallel(n, params, samples, LUTParallel, sizeLUTParallel, gridDataParallel, sampleDensityParallel);
timeParallel = omp_get_wtime() - timeParallel;
printf("Number of threds: %d\n", omp_get_max_threads());
printf("Sequential execution time: %f\n", timeSequential);
printf("Parallel execution time: %f\n", timeParallel);
int failed = 0;
for (int i = 0; i < n; i++)
{
if (fabs(gridData[i].real - gridDataParallel[i].real) > ACCURACY ||
fabs(gridData[i].imag - gridDataParallel[i].imag) > ACCURACY)
{
failed = 1;
break;
}
}
if (failed == 1)
printf("TEST FAILED - gridData\n");
else
printf("TEST PASSED - gridData\n");
failed = 0;
for (int i = 0; i < n; i++)
{
if (sampleDensity[i] != sampleDensityParallel[i])
{
failed = 1;
break;
}
}
if (failed == 1)
printf("TEST FAILED - sampleDensity\n");
else
printf("TEST PASSED - sampleDensity\n");
if (params.useLUT)
{
free(LUT);
free(LUTParallel);
}
free(samples);
free(gridData);
free(gridDataParallel);
free(sampleDensity);
free(sampleDensityParallel);
printf("\n");
return 0;
}