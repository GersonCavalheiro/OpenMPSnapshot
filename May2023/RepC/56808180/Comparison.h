#ifndef GENETICALGORITHM_COMPARISION_H
#define GENETICALGORITHM_COMPARISION_H
#pragma offload_attribute(push, target(mic))
struct Comparison
{
Comparison(int i, double d);
static bool CompareComparison (const Comparison& first, const Comparison& second);
unsigned int index;
double value;
};
#pragma offload_attribute(pop)
#endif 
