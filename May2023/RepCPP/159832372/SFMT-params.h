#pragma once
#ifndef SFMT_PARAMS_H
#define SFMT_PARAMS_H

#if !defined(SFMT_MEXP)
#if defined(__GNUC__) && !defined(__ICC)
#warning "SFMT_MEXP is not defined. I assume MEXP is 19937."
#endif
#define SFMT_MEXP 19937
#endif



#define SFMT_N (SFMT_MEXP / 128 + 1)

#define SFMT_N32 (SFMT_N * 4)

#define SFMT_N64 (SFMT_N * 2)
















#if SFMT_MEXP == 607
#include "SFMT-params607.h"
#elif SFMT_MEXP == 1279
#include "SFMT-params1279.h"
#elif SFMT_MEXP == 2281
#include "SFMT-params2281.h"
#elif SFMT_MEXP == 4253
#include "SFMT-params4253.h"
#elif SFMT_MEXP == 11213
#include "SFMT-params11213.h"
#elif SFMT_MEXP == 19937
#include "SFMT-params19937.h"
#elif SFMT_MEXP == 44497
#include "SFMT-params44497.h"
#elif SFMT_MEXP == 86243
#include "SFMT-params86243.h"
#elif SFMT_MEXP == 132049
#include "SFMT-params132049.h"
#elif SFMT_MEXP == 216091
#include "SFMT-params216091.h"
#else
#if defined(__GNUC__) && !defined(__ICC)
#error "SFMT_MEXP is not valid."
#undef SFMT_MEXP
#else
#undef SFMT_MEXP
#endif

#endif

#endif 
