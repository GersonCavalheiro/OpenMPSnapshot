#pragma once
#include "Comparison.h"
#pragma offload_attribute(push, target(mic))
Comparison::Comparison(int i, double d) :
index(i), value(d)
{
}
#pragma offload_attribute(pop)
bool Comparison::CompareComparison (const Comparison& first, const Comparison& second)
{
return first.value > second.value;
}