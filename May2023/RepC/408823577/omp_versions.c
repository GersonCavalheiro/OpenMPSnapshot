#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
int main( int argc, char **argv )
{
#if defined(__GNUC__) && !defined(__ICC) && !defined(__INTEL_COMPILER)
printf("gcc compiler %d.%d patchevel %d\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#elif defined(__clang__)
printf("clang compiler %d.%d\n", __clang__, __clang_minor__);
#elif defined(__INTEL_COMPILER)
printf("clang compiler %d update %d\n", __INTEL_COMPILER, __INTEL_COMPILER_UPDATE);
#elif defined(__ICC)
printf("intel compiler %d\n", __ICC);
#elif defined(__PGIC__)
printf("PGI compiler %d.%d patchlevel %d\n", __PGIC__, __PGIC_MINOR__, __PGIC_PATCHLEVEL__);
#elif defined(__MSC_VER__)
printf("some MSoft compiler is running\n");
#else
printf("An unknown compiler is running\n");
#endif
#if defined(_OPENMP)
#pragma omp parallel               
#pragma omp single
{
#include "omp_versions.h"           
int i;
for( i = 0; i < _OPENMP_KNOWN_VERSIONS; i++ )
if ( strncmp( _OMPv_STR(_OPENMP), omp_versions[i]._openmp_value, 6) == 0 )
break;
if ( i < _OPENMP_KNOWN_VERSIONS )
printf("OpenMP supported version is %d.%d\n",
omp_versions[i].major, omp_versions[i].minor );
else
printf("Oh gawsh!! this (%s) is an unknown OpenMP version!!\n",
_OMPv_STR(_OPENMP) );
}
#endif
return 0;
}
