

#pragma once

#ifdef USE_SP
typedef float real; 
#else
typedef double real; 
#endif

#ifndef CPU_ONLY
#include <stdio.h>

#endif 
