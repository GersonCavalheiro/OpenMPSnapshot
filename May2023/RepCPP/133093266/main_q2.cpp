#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include <cstdio>
#include <assert.h>
#include <omp.h>
#include <stdlib.h>

#include "tests_q2.h"

typedef unsigned int uint;

const uint kSizeTestVector = 4000000;
const uint kSizeMask =
16; 
const uint kRandMax = 1 << 31;
const uint kNumBitsUint = 32;


std::vector<uint> computeBlockHistograms(const std::vector<uint>& keys,
uint numBlocks, uint numBuckets,
uint numBits, uint startBit, uint blockSize) {
std::vector<uint> blockHistograms(numBlocks * numBuckets, 0);
uint mask = numBuckets - 1;
#pragma omp parallel for
for(uint i = 0; i < numBlocks; ++i) {
for (auto k = keys.begin() + i*blockSize; k < keys.end() && k < keys.begin() + (i+1)*blockSize; ++k) {
uint key = (*k >> startBit) & mask;
++blockHistograms[i*numBuckets + key];
}
}
return blockHistograms;
}


std::vector<uint> reduceLocalHistoToGlobal(const std::vector<uint>&
blockHistograms, uint numBlocks, uint numBuckets) {
std::vector<uint> globalHisto(numBuckets, 0);
for(uint i = 0; i < blockHistograms.size(); ++i) {
uint ind = i % numBuckets;
globalHisto[ind] += blockHistograms[i];
}
return globalHisto;
}


std::vector<uint> computeBlockExScanFromGlobalHisto(uint numBuckets,
uint numBlocks,
const std::vector<uint>& globalHistoExScan,
const std::vector<uint>& blockHistograms) {
std::vector<uint> blockExScan(numBuckets * numBlocks, 0);
for(uint block = 0; block < numBlocks; ++block) {
for(uint buck = 0; buck < numBuckets; ++buck) {
uint ind = block*numBuckets + buck;
if (block == 0)
blockExScan[ind] = globalHistoExScan[buck];
else {
uint preind = (block - 1)*numBuckets + buck;
blockExScan[ind] = blockExScan[preind] + blockHistograms[preind];
}
}
}
return blockExScan;
}


void populateOutputFromBlockExScan(const std::vector<uint>& blockExScan,
uint numBlocks, uint numBuckets, uint startBit,
uint numBits, uint blockSize, const std::vector<uint>& keys,
std::vector<uint>& sorted) {
uint mask = numBuckets - 1;
#pragma omp parallel for
for(uint i = 0; i < numBlocks; ++i) {
std::vector<uint> offset (numBuckets, 0);
for (auto k = keys.begin() + i*blockSize; k < keys.end() && k < keys.begin() + (i+1)*blockSize; ++k) {
uint key = *k;
uint key_bucket = (key >> startBit) & mask;
uint ind = blockExScan[i*numBuckets + key_bucket];
sorted[ind + offset[key_bucket]] = key;
++offset[key_bucket];

}
}
}


std::vector<uint> scanGlobalHisto(const std::vector<uint>& globalHisto,
uint numBuckets) {
std::vector<uint> globalHistoExScan(numBuckets, 0);
globalHistoExScan[0] = 0;
for(uint i = 1; i < numBuckets; ++i) {
globalHistoExScan[i] = globalHisto[i-1] + globalHistoExScan[i-1];
}
return globalHistoExScan;
}


void radixSortParallelPass(std::vector<uint>& keys, std::vector<uint>& sorted,
uint numBits, uint startBit,
uint blockSize) {
uint numBuckets = 1 << numBits;
uint numBlocks = (keys.size() + blockSize - 1) / blockSize;

std::vector<uint> blockHistograms = computeBlockHistograms(keys, numBlocks,
numBuckets, numBits, startBit, blockSize);

std::vector<uint> globalHisto = reduceLocalHistoToGlobal(blockHistograms,
numBlocks, numBuckets);

std::vector<uint> globalHistoExScan = scanGlobalHisto(globalHisto, numBuckets);

std::vector<uint> blockExScan = computeBlockExScanFromGlobalHisto(numBuckets,
numBlocks, globalHistoExScan, blockHistograms);

populateOutputFromBlockExScan(blockExScan, numBlocks, numBuckets, startBit,
numBits, blockSize, keys, sorted);
}


int radixSortParallel(std::vector<uint>& keys, std::vector<uint>& keys_tmp,
uint numBits, uint numBlocks) {
for(uint startBit = 0; startBit < 32; startBit += 2 * numBits) {
radixSortParallelPass(keys, keys_tmp, numBits, startBit, keys.size() / numBlocks);
radixSortParallelPass(keys_tmp, keys, numBits, startBit + numBits,
keys.size() / numBlocks);
}
return 0;
}



void radixSortSerialPass(std::vector<uint>& keys, std::vector<uint>& keys_radix,
uint startBit, uint numBits) {
uint numBuckets = 1 << numBits;
uint mask = numBuckets - 1;

std::vector<uint> histogramRadixFrequency(numBuckets);

for(uint i = 0; i < keys.size(); ++i) {
uint key = (keys[i] >> startBit) & mask;
++histogramRadixFrequency[key];
}

std::vector<uint> exScanHisto(numBuckets, 0);

for(uint i = 1; i < numBuckets; ++i) {
exScanHisto[i] = exScanHisto[i - 1] + histogramRadixFrequency[i-1];
histogramRadixFrequency[i - 1] = 0;
}

histogramRadixFrequency[numBuckets - 1] = 0;

for(uint i = 0; i < keys.size(); ++i) {
uint key = (keys[i] >> startBit) & mask;

uint localOffset = histogramRadixFrequency[key]++;
uint globalOffset = exScanHisto[key] + localOffset;

keys_radix[globalOffset] = keys[i];
}
}

int radixSortSerial(std::vector<uint>& keys, std::vector<uint>& keys_radix,
uint numBits) {
assert(numBits <= 16);

for(uint startBit = 0; startBit < 32; startBit += 2 * numBits) {
radixSortSerialPass(keys, keys_radix, startBit, numBits);
radixSortSerialPass(keys_radix, keys, startBit+numBits, numBits);
}

return 0;
}

void initializeRandomly(std::vector<uint>& keys) {
std::default_random_engine generator;
std::uniform_int_distribution<uint> distribution(0, kRandMax);

for(uint i = 0; i < keys.size(); ++i) {
keys[i] = distribution(generator);
}
}

int main() {
Test1();
Test2();
Test3();
Test4();
Test5();

std::vector<uint> keys_stl(kSizeTestVector);
initializeRandomly(keys_stl);
std::vector<uint> keys_serial = keys_stl;
std::vector<uint> keys_parallel = keys_stl;
std::vector<uint> temp_keys(kSizeTestVector);

#ifdef QUESTION6
std::vector<uint> keys_orig = keys_stl;

#endif

double startstl = omp_get_wtime();
std::sort(keys_stl.begin(), keys_stl.end());
double endstl = omp_get_wtime();

double startRadixSerial = omp_get_wtime();
radixSortSerial(keys_serial, temp_keys, kSizeMask);
double endRadixSerial = omp_get_wtime();

bool success = true;
EXPECT_VECTOR_EQ(keys_stl, keys_serial, &success);

if(success) {
std::cout << "Serial Radix Sort is correct" << std::endl;
}

double startRadixParallel = omp_get_wtime();
radixSortParallel(keys_parallel, temp_keys, kSizeMask, 8);
double endRadixParallel = omp_get_wtime();

success = true;
EXPECT_VECTOR_EQ(keys_stl, keys_parallel, &success);

if(success) {
std::cout << "Parallel Radix Sort is correct" << std::endl;
}

std::cout << "stl: " << endstl - startstl << std::endl;
std::cout << "serial radix: " << endRadixSerial - startRadixSerial << std::endl;
std::cout << "parallel radix: " << endRadixParallel - startRadixParallel <<
std::endl;

#ifdef QUESTION6
std::vector<uint> jNumBlock = {1, 2, 4, 8, 12, 16, 24, 32, 40, 48};
printf("Threads Blocks / Timing\n  ");
for(auto jNum : jNumBlock) {
printf("%8d", jNum);
}
printf("\n");
success = true;
for(auto n_threads : jNumBlock) {
omp_set_num_threads(n_threads);
printf("%4d ", n_threads);
for(auto jNum : jNumBlock) {
keys_parallel = keys_orig;
double startRadixParallel = omp_get_wtime();
radixSortParallel(keys_parallel, temp_keys, kSizeMask, jNum);
double endRadixParallel = omp_get_wtime();

EXPECT_VECTOR_EQ(keys_stl, keys_parallel, &success);
printf("%8.3f", endRadixParallel - startRadixParallel);
}
printf("\n");
}
if(success) {
std::cout << "Benchmark runs: PASS" << std::endl;
} else {
std::cout << "Benchmark runs: FAIL" << std::endl;
}
#endif

return 0;
}
