

#include <malloc.h>
#include <iostream>
#include <hdf5.h>

#ifdef WITH_PAPI
#include "papi_cntr.h"
#endif


void evaluateLayer(
size_t inputSize,
size_t neuronCount,
const float* input,
const float* weight,
const float* bias,
float* output
);


void simpleReadDataset(const char* datasetName, hid_t file, void* buffer);


void transpose2D(float*& data, size_t dimX, size_t dimY);


float* allocateMemory(size_t numberOfElements)
{
return (float*) memalign((size_t) 64, numberOfElements * sizeof(float));
}


int main(int argc, char* argv[])
{
if(argc != 4)
{
throw std::runtime_error("Expecting three arguments. Path to the network file, "
"to the data file and to the output file.");
}

std::string networkDataFile(argv[1]);
std::string testingDataFile(argv[2]);
std::string outputDataFile(argv[3]);


#ifdef WITH_PAPI
printf("Compiled with PAPI\n");
PapiCounterList papi_routines;
papi_routines.AddRoutine("network");
#else
printf("Compiled without PAPI\n");
#endif


const size_t imagePixels = 784;
const size_t layerSize = 512;
const size_t outputSize = 10;

float* weight1 = allocateMemory(imagePixels * layerSize);
float* weight2 = allocateMemory(layerSize * layerSize);
float* weight3 = allocateMemory(layerSize * outputSize);
float* bias1 = allocateMemory(layerSize);
float* bias2 = allocateMemory(layerSize);
float* bias3 = allocateMemory(outputSize);

hid_t networkFile = H5Fopen(networkDataFile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
simpleReadDataset("/model_weights/dense_1/dense_1/kernel:0", networkFile, weight1);
simpleReadDataset("/model_weights/dense_2/dense_2/kernel:0", networkFile, weight2);
simpleReadDataset("/model_weights/dense_3/dense_3/kernel:0", networkFile, weight3);
simpleReadDataset("/model_weights/dense_1/dense_1/bias:0", networkFile, bias1);
simpleReadDataset("/model_weights/dense_2/dense_2/bias:0", networkFile, bias2);
simpleReadDataset("/model_weights/dense_3/dense_3/bias:0", networkFile, bias3);

H5Fclose(networkFile);


size_t imageCount;
hid_t dataFile = H5Fopen(testingDataFile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

hid_t dataset = H5Dopen(dataFile, "/image_count", H5P_DEFAULT);

hid_t status = H5Dread(dataset, H5T_NATIVE_ULONG, H5S_ALL,
H5S_ALL, H5P_DEFAULT, &imageCount);

printf("%zu\n", imageCount);

if (status < 0)
{
throw std::runtime_error("Could not read the image count.");
}
H5Dclose(dataset);


float* input = allocateMemory(imageCount * imagePixels);

simpleReadDataset("/x_test", dataFile, input);

H5Fclose(dataFile);


float* output1 = allocateMemory(imageCount * layerSize);
float* output2 = allocateMemory(imageCount * layerSize);
float* output3 = allocateMemory(imageCount * outputSize);

#ifdef WITH_PAPI
papi_routines["network"].Start();
#endif

transpose2D(weight1, layerSize, imagePixels);
for (size_t i = 0; i < imageCount; i++)
evaluateLayer(imagePixels, layerSize, &input[i * imagePixels], weight1, bias1, &output1[i * layerSize]);

transpose2D(weight2, layerSize, layerSize);
for (size_t i = 0; i < imageCount; i++)
evaluateLayer(layerSize, layerSize, &output1[i * layerSize], weight2, bias2, &output2[i * layerSize]);

transpose2D(weight3, outputSize, layerSize);
for (size_t i = 0; i < imageCount; i++)
evaluateLayer(layerSize, outputSize, &output2[i * layerSize], weight3, bias3, &output3[i*outputSize]);

#ifdef WITH_PAPI
papi_routines["network"].Stop();
papi_routines.PrintScreen();
#endif

for (size_t image = 0; image < 5; image++)
{
printf("Image %zu: ", image);

for (size_t i = 0; i < outputSize; i++)
{
printf("%zu: %f  ", i, output3[image*outputSize+i]);
}
std::cout << std::endl;
}



hsize_t     dims[2] = {imageCount, outputSize};

hid_t file = H5Fcreate(outputDataFile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

hid_t space = H5Screate_simple (2, dims, dims);

hid_t dset = H5Dcreate (file, "/output_data", H5T_NATIVE_FLOAT, space, H5P_DEFAULT,
H5P_DEFAULT, H5P_DEFAULT);

status = H5Dwrite (dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, output3);
if (status < 0)
{
throw std::runtime_error("Could not store the output dataset.");
}

H5Dclose(dset);
H5Sclose(space);
H5Fclose(file);

free(output1);
free(output2);
free(output3);

free(input);

free(bias1);
free(bias2);
free(bias3);

free(weight1);
free(weight2);
free(weight3);

return 0;
}


void simpleReadDataset(const char* datasetName, hid_t file, void* buffer)
{

hid_t dataset = H5Dopen(file, datasetName, H5P_DEFAULT);
hid_t status = H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL,
H5S_ALL, H5P_DEFAULT, buffer);

if (status < 0)
{
throw std::runtime_error("Could not read the dataset");
}
H5Dclose(dataset);
}


void evaluateLayer(
const size_t inputSize,
const size_t neuronCount,
const float* input,
const float* weight,
const float* bias,
float* output
)
{
for (size_t i = 0; i < neuronCount; i++) {
float result = 0.0f;
#pragma omp simd simdlen(8) aligned(weight, input) reduction(+:result)
for(size_t j = 0; j < inputSize; j++) {
result += input[j] * weight[i * inputSize + j];
}
output[i] = (result + bias[i] > 0.0f) ? result + bias[i] : 0.0f;
}
}


void transpose2D(float*& data, const size_t dimX, const size_t dimY)
{
float* tmp = data;
data = allocateMemory(dimX * dimY);

for (size_t x = 0; x < dimX; x++)
{
for (size_t y = 0; y < dimY; y++)
{
data[x * dimY + y] = tmp[y * dimX + x];
}
}
free(tmp);
}

