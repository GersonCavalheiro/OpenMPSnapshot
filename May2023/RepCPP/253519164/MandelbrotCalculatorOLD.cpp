#include "MandelbrotCalculatorOLD.h"

MandelbrotCalculatorOLD::MandelbrotCalculatorOLD(unsigned int* escapeCounts, int width, int height)
{
this->width = width;
this->height = height;
this->escapeCounts = escapeCounts;
sizeOfTheWorld = width*height*sizeof(unsigned int);
temporaryResultSerialAVX = (double*)_aligned_malloc(4*sizeof(double), 32); 
temporaryResultsParallelAVX = (double**)malloc(height*sizeof(double*));
for(int i = 0; i < height; i++)
temporaryResultsParallelAVX[i] = (double*)_aligned_malloc(4*sizeof(double), 32);

GPUInit();
}

MandelbrotCalculatorOLD::~MandelbrotCalculatorOLD()
{
_aligned_free(temporaryResultSerialAVX);
for(unsigned int i = 0; i < height; i++)
_aligned_free(temporaryResultsParallelAVX[i]);
free(temporaryResultsParallelAVX);
}

std::string MandelbrotCalculatorOLD::readKernelSource(const std::string filename)
{
std::ifstream inputFile(filename);

if(!inputFile)
return "";

std::ostringstream buffer;
buffer << inputFile.rdbuf();

std::string sourceCode = buffer.str();

inputFile.close();
return sourceCode;
}

void MandelbrotCalculatorOLD::GPUInit()
{
cl_int err;

err = clGetPlatformIDs(1, &cpPlatform, NULL);
if(err != CL_SUCCESS)
throw TranslateOpenCLError(err);

err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
if(err != CL_SUCCESS)
throw TranslateOpenCLError(err);

context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
if(err != CL_SUCCESS)
throw TranslateOpenCLError(err);

queue = clCreateCommandQueue(context, device_id, 0, &err);
if(err != CL_SUCCESS)
throw TranslateOpenCLError(err);

std::string kernelSource = readKernelSource(mandelbrotKernelFilename);
if(kernelSource.empty())
{
qInfo() << "Couldn't read the kernel source file";
return;
}

const char* tempSource = kernelSource.c_str();

program = clCreateProgramWithSource(context, 1, (const char**)&tempSource, NULL, &err);
if(err != CL_SUCCESS)
throw TranslateOpenCLError(err);

err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

qInfo() << "before first if";
if (err != CL_SUCCESS)
{
size_t log_size;
clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
char* log = new char[log_size];

clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
qInfo() << "build not successful: " << log;

std::string temp(log);
delete[] log;
return; 
}

kernel = clCreateKernel(program, kernelName, &err);
if(err != CL_SUCCESS)
throw TranslateOpenCLError(err);

clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(localSize), &localSize, NULL);


globalSize = std::ceil((double)(width*height) / localSize) * localSize;

escapeCountsGPU = clCreateBuffer(context, CL_MEM_READ_WRITE, width*height*sizeof(unsigned int), NULL, &err);
if(err != CL_SUCCESS)
throw TranslateOpenCLError(err);


err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &escapeCountsGPU);
err |= clSetKernelArg(kernel, 1, sizeof(int), &width);
err |= clSetKernelArg(kernel, 2, sizeof(int), &height);
if(err != CL_SUCCESS)
qInfo() << TranslateOpenCLError(err);
clFinish(queue);
}

void MandelbrotCalculatorOLD::calculateCPUSerial(unsigned int numberOfIterations, double upperLeftX, double upperLeftY, double downRightX, double downRightY)
{
double incrementX = (downRightX - upperLeftX) / (double)width;
double incrementY = (upperLeftY - downRightY) / (double)height;


double imaginaryValue = upperLeftY;
for(unsigned int y = 0; y < height; y++)
{
double realValue = upperLeftX;
for(unsigned int x = 0; x < width; x++)
{
escapeCounts[y*width + x] = isMandelbrotNumber(realValue, imaginaryValue, numberOfIterations);
realValue += incrementX;
}
imaginaryValue -= incrementY;
}


}

void MandelbrotCalculatorOLD::calculateGPU(unsigned int numberOfIterations, double upperLeftX, double upperLeftY, double downRightX, double downRightY)
{

double incrementX = (downRightX - upperLeftX) / (double)width;
double incrementY = (upperLeftY - downRightY) / (double)height;

cl_int err = 0;

err |= clSetKernelArg(kernel, 3, sizeof(double), &upperLeftX);
err |= clSetKernelArg(kernel, 4, sizeof(double), &upperLeftY);
err |= clSetKernelArg(kernel, 5, sizeof(double), &incrementX);
err |= clSetKernelArg(kernel, 6, sizeof(double), &incrementY);
err |= clSetKernelArg(kernel, 7, sizeof(int), &numberOfIterations);
if(err != CL_SUCCESS)
throw TranslateOpenCLError(err);
clFinish(queue);

err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
clFinish(queue);


err = clEnqueueReadBuffer(queue, escapeCountsGPU, CL_TRUE, 0, sizeOfTheWorld, escapeCounts, 0, NULL, NULL);
if(err != CL_SUCCESS)
throw TranslateOpenCLError(err);

clFinish(queue);
}



void MandelbrotCalculatorOLD::calculateCPUParallel(unsigned int numberOfIterations, double upperLeftX, double upperLeftY, double downRightX, double downRightY)
{
double incrementX = (downRightX - upperLeftX) / (double)width;
double incrementY = (upperLeftY - downRightY) / (double)height;

#pragma omp parallel for
for(int y = 0; y < height; y++)
{
double imaginaryValue = upperLeftY - y*incrementY;
double realValue = upperLeftX;
for(unsigned int x = 0; x < width; x++)
{
escapeCounts[y*width + x] = isMandelbrotNumber(realValue, imaginaryValue, numberOfIterations);
realValue += incrementX;
}
}
}

void MandelbrotCalculatorOLD::calculateAVXSerial(unsigned int numberOfIterations, double upperLeftX, double upperLeftY, double downRightX, double downRightY)
{

double incrementX = (downRightX - upperLeftX) / (double)width;
double incrementY = (upperLeftY - downRightY) / (double)height;


__m256d divergenceIterations, groupOfFour, imaginary, _upperLeftX, _four, _two, _incrementX, _secondaryReal, _secondaryImaginary;


_upperLeftX = _mm256_set1_pd(upperLeftX);
_four = _mm256_set1_pd(4.0);
_two = _mm256_set1_pd(2.0);
_incrementX = _mm256_set1_pd(incrementX);


int wholeParts = width / 4; 

for(unsigned int y = 0; y < height; y++)
{
__m256d _incrementor = _mm256_set_pd(3, 2, 1, 0); 

double imaginaryComponent = upperLeftY - y*incrementY;
imaginary = _mm256_set1_pd(imaginaryComponent);

for(int x = 0; x < wholeParts*4; x += 4)
{
divergenceIterations = _mm256_setzero_pd();
__m256d diverged = _mm256_castsi256_pd(_mm256_set1_epi64x(-1)); 
groupOfFour = _mm256_fmadd_pd(_incrementor, _incrementX, _upperLeftX);
_secondaryImaginary = _mm256_setzero_pd();
_secondaryReal = _mm256_setzero_pd();

for(unsigned int i = 0; i < numberOfIterations; i++)
{
__m256d currentIteration = _mm256_castsi256_pd(_mm256_set1_epi64x((long long)i));
__m256d a2 = _mm256_mul_pd(_secondaryReal, _secondaryReal); 
__m256d b2 = _mm256_mul_pd(_secondaryImaginary, _secondaryImaginary); 


__m256d moduloSquare = _mm256_add_pd(a2, b2); 
__m256d comparisonMask = _mm256_cmp_pd(moduloSquare, _four, _CMP_LE_OQ);
groupOfFour = _mm256_and_pd(groupOfFour, comparisonMask);


divergenceIterations =_mm256_or_pd(divergenceIterations, _mm256_and_pd(currentIteration, _mm256_andnot_pd(comparisonMask, diverged))); 
diverged = _mm256_and_pd(diverged, comparisonMask);

if(_mm256_movemask_pd(diverged) == 0)
break;


__m256d tempReal = _mm256_add_pd(_mm256_sub_pd(a2, b2), groupOfFour); 
_secondaryImaginary = _mm256_fmadd_pd(_mm256_mul_pd(_secondaryReal, _secondaryImaginary), _two, imaginary); 
_secondaryReal = tempReal;
}

_mm256_store_pd(temporaryResultSerialAVX, divergenceIterations);


unsigned int first = *((unsigned int*)(temporaryResultSerialAVX));
unsigned int second = *((unsigned int*)(temporaryResultSerialAVX + 1));
unsigned int third = *((unsigned int*)(temporaryResultSerialAVX + 2));
unsigned int fourth = *((unsigned int*)(temporaryResultSerialAVX + 3));

escapeCounts[y*width + x] = first;
escapeCounts[y*width + x+1] = second;
escapeCounts[y*width + x+2] = third;
escapeCounts[y*width + x+3] = fourth;

_incrementor = _mm256_add_pd(_incrementor, _four);


}

if((wholeParts*4) != width)
{
double realValue = upperLeftX + incrementX*(wholeParts*4);
int counter = 0;
for(int x = wholeParts*4; x < width; x++)
escapeCounts[y*width + x] = isMandelbrotNumber(realValue + incrementX*(counter++), imaginaryComponent, numberOfIterations);
}
}

}

void MandelbrotCalculatorOLD::calculateAVXParallel(unsigned int numberOfIterations, double upperLeftX, double upperLeftY, double downRightX, double downRightY)
{


double incrementX = (downRightX - upperLeftX) / (double)width;
double incrementY = (upperLeftY - downRightY) / (double)height;


__m256d _upperLeftX,_four, _two, _incrementX;



_upperLeftX = _mm256_set1_pd(upperLeftX);
_four = _mm256_set1_pd(4.0);
_two = _mm256_set1_pd(2.0);
_incrementX = _mm256_set1_pd(incrementX);


int wholeParts = width / 4; 

#pragma omp parallel for
for(int y = 0; y < height; y++)
{
__m256d divergenceIterations, groupOfFour, imaginary, _secondaryReal, _secondaryImaginary;
__m256d _incrementor = _mm256_set_pd(3, 2, 1, 0); 


double* temporaryResult = temporaryResultsParallelAVX[y];


double imaginaryComponent = upperLeftY - y*incrementY;
imaginary = _mm256_set1_pd(imaginaryComponent);

for(int x = 0; x < wholeParts*4; x += 4)
{
divergenceIterations = _mm256_setzero_pd();
__m256d diverged = _mm256_castsi256_pd(_mm256_set1_epi64x(-1)); 
groupOfFour = _mm256_fmadd_pd(_incrementor, _incrementX, _upperLeftX);
_secondaryImaginary = _mm256_setzero_pd();
_secondaryReal = _mm256_setzero_pd();

for(unsigned int i = 0; i < numberOfIterations; i++)
{
__m256d currentIteration = _mm256_castsi256_pd(_mm256_set1_epi64x((long long)i));
__m256d a2 = _mm256_mul_pd(_secondaryReal, _secondaryReal); 
__m256d b2 = _mm256_mul_pd(_secondaryImaginary, _secondaryImaginary); 


__m256d moduloSquare = _mm256_add_pd(a2, b2); 
__m256d comparisonMask = _mm256_cmp_pd(moduloSquare, _four, _CMP_LE_OQ);
groupOfFour = _mm256_and_pd(groupOfFour, comparisonMask);


divergenceIterations =_mm256_or_pd(divergenceIterations, _mm256_and_pd(currentIteration, _mm256_andnot_pd(comparisonMask, diverged))); 
diverged = _mm256_and_pd(diverged, comparisonMask);

if(_mm256_movemask_pd(diverged) == 0)
break;


__m256d tempReal = _mm256_add_pd(_mm256_sub_pd(a2, b2), groupOfFour); 
_secondaryImaginary = _mm256_fmadd_pd(_mm256_mul_pd(_secondaryReal, _secondaryImaginary), _two, imaginary); 
_secondaryReal = tempReal;
}

_mm256_store_pd(temporaryResult, divergenceIterations);


unsigned int first = *((unsigned int*)(temporaryResult));
unsigned int second = *((unsigned int*)(temporaryResult + 1));
unsigned int third = *((unsigned int*)(temporaryResult + 2));
unsigned int fourth = *((unsigned int*)(temporaryResult + 3));

escapeCounts[y*width + x] = first;
escapeCounts[y*width + x+1] = second;
escapeCounts[y*width + x+2] = third;
escapeCounts[y*width + x+3] = fourth;

_incrementor = _mm256_add_pd(_incrementor, _four);
}

if((wholeParts*4) != width)
{
double realValue = upperLeftX + incrementX*(wholeParts*4);
int counter = 0;
for(int x = wholeParts*4; x < height; x++)
escapeCounts[y*width + x] = isMandelbrotNumber(realValue + incrementX*(counter++), imaginaryComponent, numberOfIterations);
}
}

}

unsigned int MandelbrotCalculatorOLD::isMandelbrotNumber(double real, double imaginary, unsigned int numberOfIterations)
{
double secondaryReal = 0;
double secondaryImaginary = 0;

for (unsigned int i = 0; i < numberOfIterations; i++)
{

double a2 = secondaryReal * secondaryReal; 
double b2 = secondaryImaginary*secondaryImaginary; 

if((a2 + b2) > 4)
return i;

secondaryImaginary = 2*secondaryReal*secondaryImaginary + imaginary;
secondaryReal = a2 - b2 + real;
}
return 0;
}


const char* MandelbrotCalculatorOLD::TranslateOpenCLError(cl_int errorCode)
{
switch (errorCode)
{
case CL_SUCCESS:                            return "CL_SUCCESS";
case CL_DEVICE_NOT_FOUND:                   return "CL_DEVICE_NOT_FOUND";
case CL_DEVICE_NOT_AVAILABLE:               return "CL_DEVICE_NOT_AVAILABLE";
case CL_COMPILER_NOT_AVAILABLE:             return "CL_COMPILER_NOT_AVAILABLE";
case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
case CL_OUT_OF_RESOURCES:                   return "CL_OUT_OF_RESOURCES";
case CL_OUT_OF_HOST_MEMORY:                 return "CL_OUT_OF_HOST_MEMORY";
case CL_PROFILING_INFO_NOT_AVAILABLE:       return "CL_PROFILING_INFO_NOT_AVAILABLE";
case CL_MEM_COPY_OVERLAP:                   return "CL_MEM_COPY_OVERLAP";
case CL_IMAGE_FORMAT_MISMATCH:              return "CL_IMAGE_FORMAT_MISMATCH";
case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
case CL_BUILD_PROGRAM_FAILURE:              return "CL_BUILD_PROGRAM_FAILURE";
case CL_MAP_FAILURE:                        return "CL_MAP_FAILURE";
case CL_MISALIGNED_SUB_BUFFER_OFFSET:       return "CL_MISALIGNED_SUB_BUFFER_OFFSET";                          
case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";   
case CL_COMPILE_PROGRAM_FAILURE:            return "CL_COMPILE_PROGRAM_FAILURE";                               
case CL_LINKER_NOT_AVAILABLE:               return "CL_LINKER_NOT_AVAILABLE";                                  
case CL_LINK_PROGRAM_FAILURE:               return "CL_LINK_PROGRAM_FAILURE";                                  
case CL_DEVICE_PARTITION_FAILED:            return "CL_DEVICE_PARTITION_FAILED";                               
case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";                         
case CL_INVALID_VALUE:                      return "CL_INVALID_VALUE";
case CL_INVALID_DEVICE_TYPE:                return "CL_INVALID_DEVICE_TYPE";
case CL_INVALID_PLATFORM:                   return "CL_INVALID_PLATFORM";
case CL_INVALID_DEVICE:                     return "CL_INVALID_DEVICE";
case CL_INVALID_CONTEXT:                    return "CL_INVALID_CONTEXT";
case CL_INVALID_QUEUE_PROPERTIES:           return "CL_INVALID_QUEUE_PROPERTIES";
case CL_INVALID_COMMAND_QUEUE:              return "CL_INVALID_COMMAND_QUEUE";
case CL_INVALID_HOST_PTR:                   return "CL_INVALID_HOST_PTR";
case CL_INVALID_MEM_OBJECT:                 return "CL_INVALID_MEM_OBJECT";
case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
case CL_INVALID_IMAGE_SIZE:                 return "CL_INVALID_IMAGE_SIZE";
case CL_INVALID_SAMPLER:                    return "CL_INVALID_SAMPLER";
case CL_INVALID_BINARY:                     return "CL_INVALID_BINARY";
case CL_INVALID_BUILD_OPTIONS:              return "CL_INVALID_BUILD_OPTIONS";
case CL_INVALID_PROGRAM:                    return "CL_INVALID_PROGRAM";
case CL_INVALID_PROGRAM_EXECUTABLE:         return "CL_INVALID_PROGRAM_EXECUTABLE";
case CL_INVALID_KERNEL_NAME:                return "CL_INVALID_KERNEL_NAME";
case CL_INVALID_KERNEL_DEFINITION:          return "CL_INVALID_KERNEL_DEFINITION";
case CL_INVALID_KERNEL:                     return "CL_INVALID_KERNEL";
case CL_INVALID_ARG_INDEX:                  return "CL_INVALID_ARG_INDEX";
case CL_INVALID_ARG_VALUE:                  return "CL_INVALID_ARG_VALUE";
case CL_INVALID_ARG_SIZE:                   return "CL_INVALID_ARG_SIZE";
case CL_INVALID_KERNEL_ARGS:                return "CL_INVALID_KERNEL_ARGS";
case CL_INVALID_WORK_DIMENSION:             return "CL_INVALID_WORK_DIMENSION";
case CL_INVALID_WORK_GROUP_SIZE:            return "CL_INVALID_WORK_GROUP_SIZE";
case CL_INVALID_WORK_ITEM_SIZE:             return "CL_INVALID_WORK_ITEM_SIZE";
case CL_INVALID_GLOBAL_OFFSET:              return "CL_INVALID_GLOBAL_OFFSET";
case CL_INVALID_EVENT_WAIT_LIST:            return "CL_INVALID_EVENT_WAIT_LIST";
case CL_INVALID_EVENT:                      return "CL_INVALID_EVENT";
case CL_INVALID_OPERATION:                  return "CL_INVALID_OPERATION";
case CL_INVALID_GL_OBJECT:                  return "CL_INVALID_GL_OBJECT";
case CL_INVALID_BUFFER_SIZE:                return "CL_INVALID_BUFFER_SIZE";
case CL_INVALID_MIP_LEVEL:                  return "CL_INVALID_MIP_LEVEL";
case CL_INVALID_GLOBAL_WORK_SIZE:           return "CL_INVALID_GLOBAL_WORK_SIZE";                           
case CL_INVALID_PROPERTY:                   return "CL_INVALID_PROPERTY";                                   
case CL_INVALID_IMAGE_DESCRIPTOR:           return "CL_INVALID_IMAGE_DESCRIPTOR";                           
case CL_INVALID_COMPILER_OPTIONS:           return "CL_INVALID_COMPILER_OPTIONS";                           
case CL_INVALID_LINKER_OPTIONS:             return "CL_INVALID_LINKER_OPTIONS";                             
case CL_INVALID_DEVICE_PARTITION_COUNT:     return "CL_INVALID_DEVICE_PARTITION_COUNT";                     

default:
return "UNKNOWN ERROR CODE";
}
}
