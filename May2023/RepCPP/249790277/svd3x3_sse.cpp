#ifdef __SSE__
#include "svd3x3_sse.h"

#include <cmath>
#include <algorithm>

#undef USE_SCALAR_IMPLEMENTATION
#define USE_SSE_IMPLEMENTATION
#undef USE_AVX_IMPLEMENTATION
#define COMPUTE_U_AS_MATRIX
#define COMPUTE_V_AS_MATRIX
#include "Singular_Value_Decomposition_Preamble.hpp"

#pragma runtime_checks( "u", off )  
template<typename T>
IGL_INLINE void igl::svd3x3_sse(
const Eigen::Matrix<T, 3*4, 3>& A, 
Eigen::Matrix<T, 3*4, 3> &U, 
Eigen::Matrix<T, 3*4, 1> &S, 
Eigen::Matrix<T, 3*4, 3>&V)
{
float Ashuffle[9][4], Ushuffle[9][4], Vshuffle[9][4], Sshuffle[3][4];
for (int i=0; i<3; i++)
{
for (int j=0; j<3; j++)
{
for (int k=0; k<4; k++)
{
Ashuffle[i + j*3][k] = A(i + 3*k, j);
}
}
}

#include "Singular_Value_Decomposition_Kernel_Declarations.hpp"

ENABLE_SSE_IMPLEMENTATION(Va11=_mm_loadu_ps(Ashuffle[0]);)
ENABLE_SSE_IMPLEMENTATION(Va21=_mm_loadu_ps(Ashuffle[1]);)
ENABLE_SSE_IMPLEMENTATION(Va31=_mm_loadu_ps(Ashuffle[2]);)
ENABLE_SSE_IMPLEMENTATION(Va12=_mm_loadu_ps(Ashuffle[3]);)
ENABLE_SSE_IMPLEMENTATION(Va22=_mm_loadu_ps(Ashuffle[4]);)
ENABLE_SSE_IMPLEMENTATION(Va32=_mm_loadu_ps(Ashuffle[5]);)
ENABLE_SSE_IMPLEMENTATION(Va13=_mm_loadu_ps(Ashuffle[6]);)
ENABLE_SSE_IMPLEMENTATION(Va23=_mm_loadu_ps(Ashuffle[7]);)
ENABLE_SSE_IMPLEMENTATION(Va33=_mm_loadu_ps(Ashuffle[8]);)

#include "Singular_Value_Decomposition_Main_Kernel_Body.hpp"

ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Ushuffle[0],Vu11);)
ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Ushuffle[1],Vu21);)
ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Ushuffle[2],Vu31);)
ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Ushuffle[3],Vu12);)
ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Ushuffle[4],Vu22);)
ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Ushuffle[5],Vu32);)
ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Ushuffle[6],Vu13);)
ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Ushuffle[7],Vu23);)
ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Ushuffle[8],Vu33);)

ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Vshuffle[0],Vv11);)
ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Vshuffle[1],Vv21);)
ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Vshuffle[2],Vv31);)
ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Vshuffle[3],Vv12);)
ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Vshuffle[4],Vv22);)
ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Vshuffle[5],Vv32);)
ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Vshuffle[6],Vv13);)
ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Vshuffle[7],Vv23);)
ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Vshuffle[8],Vv33);)

ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Sshuffle[0],Va11);)
ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Sshuffle[1],Va22);)
ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Sshuffle[2],Va33);)

for (int i=0; i<3; i++)
{
for (int j=0; j<3; j++)
{
for (int k=0; k<4; k++)
{
U(i + 3*k, j) = Ushuffle[i + j*3][k];
V(i + 3*k, j) = Vshuffle[i + j*3][k];
}
}
}

for (int i=0; i<3; i++)
{
for (int k=0; k<4; k++)
{
S(i + 3*k, 0) = Sshuffle[i][k];
}
}
}
#pragma runtime_checks( "u", restore )

template void igl::svd3x3_sse(const Eigen::Matrix<float, 3*4, 3>& A, Eigen::Matrix<float, 3*4, 3> &U, Eigen::Matrix<float, 3*4, 1> &S, Eigen::Matrix<float, 3*4, 3>&V);
#endif
