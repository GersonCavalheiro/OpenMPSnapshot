

#pragma once
#include "common.h"
#include <string>

extern queue q; 

#ifdef USE_SP
typedef float real; 
#else
typedef double real; 
#endif

#ifndef CPU_ONLY
#include <cstdio>
#endif

void lsqt(std::string input_directory);
