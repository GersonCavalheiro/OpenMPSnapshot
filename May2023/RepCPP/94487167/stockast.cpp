

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <omp.h>
#include <random>
#include <sstream>
#include <string>

float calculateVolatility(float spotPrice, int32_t timeSteps)
{
std::ifstream filePtr;
filePtr.open("data.csv", std::ifstream::in);
if (!filePtr.is_open())
{
std::cerr << "Cannot open data.csv! Exiting..\n";
exit(EXIT_FAILURE);
}

std::string line;
if (!std::getline(filePtr, line))
{
std::cerr << "Cannot read from data.csv! Exiting..\n";
filePtr.close();
exit(EXIT_FAILURE);
}
filePtr.close();

int32_t i = 0, len = timeSteps - 1;
std::unique_ptr<float[]> priceArr = std::make_unique<float[]>(timeSteps - 1);
std::istringstream iss(line);
std::string token;

while (std::getline(iss, token, ','))
priceArr[i++] = std::stof(token);

float sum = spotPrice;
for (i = 0; i < len; i++)
sum += priceArr[i];
float meanPrice = sum / (len + 1);

sum = std::powf((spotPrice - meanPrice), 2.0f);
for (i = 0; i < len; i++)
sum += std::powf((priceArr[i] - meanPrice), 2.0f);

float stdDev = std::sqrtf(sum);

return stdDev / 100.0f;
}


float* find2dMean(float** matrix, int32_t numLoops, int32_t timeSteps)
{
int32_t j;
float* avg = new float[timeSteps];
float sum = 0.0f;

for (int32_t i = 0; i < timeSteps; i++)
{

#pragma omp parallel for private(j) reduction(+:sum)
for (j = 0; j < numLoops; j++)
{
sum += matrix[j][i];
}

avg[i] = sum / numLoops;
sum = 0.0f;
}

return avg;
}


float genRand(float mean, float stdDev)
{
const auto seed = std::chrono::system_clock::now().time_since_epoch().count();
std::default_random_engine generator(static_cast<uint32_t>(seed));
std::normal_distribution<float> distribution(mean, stdDev);
return distribution(generator);
}

float* runBlackScholesModel(float spotPrice, int32_t timeSteps, float riskRate, float volatility)
{
static constexpr float  mean = 0.0f, stdDev = 1.0f;                                             
float  deltaT = 1.0f / timeSteps;                                              
std::unique_ptr<float[]> normRand = std::make_unique<float[]>(timeSteps - 1); 
float* stockPrice = new float[timeSteps];                                     
stockPrice[0] = spotPrice;                                                    

for (int32_t i = 0; i < timeSteps - 1; i++)
normRand[i] = genRand(mean, stdDev);

for (int32_t i = 0; i < timeSteps - 1; i++)
stockPrice[i + 1] = stockPrice[i] * exp(((riskRate - (std::powf(volatility, 2.0f) / 2.0f)) * deltaT) + (volatility * normRand[i] * std::sqrtf(deltaT)));

return stockPrice;
}

int32_t main(int32_t argc, char** argv)
{
const auto beginTime = std::chrono::system_clock::now();

static constexpr int32_t inLoops = 100;     
static constexpr int32_t outLoops = 10000;  
static constexpr int32_t timeSteps = 180;   

float** stock = new float* [inLoops];
for (int32_t i = 0; i < inLoops; i++)
stock[i] = new float[timeSteps];

float** avgStock = new float* [outLoops];
for (int32_t i = 0; i < outLoops; i++)
avgStock[i] = new float[timeSteps];

static constexpr float spotPrice = 100.0f;  

const float volatility = calculateVolatility(spotPrice, timeSteps);

std::cout << "==============================================\n";
std::cout << "      Stockast - Stock Forecasting Tool\n";
std::cout << "    Copyright (c) 2017-2023 Rajdeep Konwar\n";
std::cout << "==============================================\n\n";
std::cout << "Using market volatility: " << volatility << "\n";

int32_t i;
#pragma omp parallel private(i)
{
#pragma omp single
{
const int32_t numThreads = omp_get_num_threads();   
std::cout << "Using " << numThreads << " thread(s)\n\n";
std::cout << "Have patience! Computing.. ";
omp_set_num_threads(numThreads);
}


#pragma omp for schedule(dynamic)
for (i = 0; i < outLoops; i++)
{

for (int32_t j = 0; j < inLoops; j++)
{
static constexpr float riskRate = 0.001f;   
stock[j] = runBlackScholesModel(spotPrice, timeSteps, riskRate, volatility);
}

avgStock[i] = find2dMean(stock, inLoops, timeSteps);
}
}

float *optStock = new float[timeSteps];     
optStock = find2dMean(avgStock, outLoops, timeSteps);

std::ofstream filePtr;
filePtr.open("opt.csv", std::ofstream::out);
if (!filePtr.is_open())
{
std::cerr << "Couldn't open opt.csv! Exiting..\n";
return EXIT_FAILURE;
}

for (i = 0; i < timeSteps; i++)
filePtr << optStock[i] << "\n";
filePtr.close();

for (i = 0; i < inLoops; i++)
delete[] stock[i];
delete[] stock;

for (i = 0; i < outLoops; i++)
delete[] avgStock[i];
delete[] avgStock;

delete[] optStock;

std::cout << "done!\nTime taken: " << std::to_string(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - beginTime).count()) << "s";
return std::getchar();
}
