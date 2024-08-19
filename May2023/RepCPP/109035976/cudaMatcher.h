#pragma once

#include "brief.h"

class CudaMatcher {
public:
int num_desc1;
int num_desc2;

Descriptor* cudaDesc1;
Descriptor* cudaDesc2;

float* cuda_ratios;
int* cuda_match_indices;

CudaMatcher();
virtual ~CudaMatcher();

void setup(std::vector<Descriptor> desc1, std::vector<Descriptor> desc2);
MatchResult findMatch();

void getMatchResult(float* ratios, int* match_indices);
};