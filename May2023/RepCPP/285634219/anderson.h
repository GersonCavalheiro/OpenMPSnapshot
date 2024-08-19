

#pragma once
#include "lsqt.h"
#include <random>

class Anderson
{
public:
void add_disorder(int N, std::mt19937& generator, real* potential);
bool has_disorder = false;
real disorder_strength;
};
