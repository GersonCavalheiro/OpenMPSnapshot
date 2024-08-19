
#pragma once

#include "../math/math.h"


#if defined(__SSE__)
#  include "sse.h"
#endif


#if defined(__AVX__)
#  include "avx.h"
#endif


#if defined (__AVX512F__)
#  include "avx512.h"
#endif

#if defined(__AVX512F__)
#  define AVX_ZERO_UPPER()
#elif defined (__AVX__)
#  define AVX_ZERO_UPPER() _mm256_zeroupper()
#else
#  define AVX_ZERO_UPPER()
#endif

namespace embree
{

template<typename vbool, typename vint, typename Closure>
__forceinline void foreach_unique(const vbool& valid0, const vint& vi, const Closure& closure)
{
vbool valid1 = valid0;
while (any(valid1)) {
const int j = int(__bsf(movemask(valid1)));
const int i = vi[j];
const vbool valid2 = valid1 & (i == vi);
valid1 = valid1 & !valid2;
closure(valid2,i);
}
}


template<typename vbool, typename vint, typename Closure>
__forceinline void foreach_unique_index(const vbool& valid0, const vint& vi, const Closure& closure)
{
vbool valid1 = valid0;
while (any(valid1)) {
const int j = (int) __bsf(movemask(valid1));
const int i = vi[j];
const vbool valid2 = valid1 & (i == vi);
valid1 = valid1 & !valid2;
closure(valid2,i,j);
}
}

template<typename Closure>
__forceinline void foreach2(int x0, int x1, int y0, int y1, const Closure& closure) 
{
__aligned(64) int U[2*VSIZEX];
__aligned(64) int V[2*VSIZEX];
int index = 0;
for (int y=y0; y<y1; y++) {
const bool lasty = y+1>=y1;
const vintx vy = y;
for (int x=x0; x<x1; ) { 
const bool lastx = x+VSIZEX >= x1;
vintx vx = x+vintx(step);
vintx::storeu(&U[index],vx);
vintx::storeu(&V[index],vy);
const int dx = min(x1-x,VSIZEX);
index += dx;
x += dx;
if (index >= VSIZEX || (lastx && lasty)) {
const vboolx valid = vintx(step) < vintx(index);
closure(valid,vintx::load(U),vintx::load(V));
x-= max(0,index-VSIZEX);
index = 0;
}
}
}
}
}
