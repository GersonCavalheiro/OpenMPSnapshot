





#include "vlad.h"
#include "mathop.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif


#ifdef VL_VLAD_INSTANTIATING

static void
VL_XCAT(_vl_vlad_encode_, SFX)
(TYPE * enc,
TYPE const * means, vl_size dimension, vl_size numClusters,
TYPE const * data, vl_size numData,
TYPE const * assignments,
int flags)
{
vl_uindex dim ;
vl_index i_cl, i_d ;

memset(enc, 0, sizeof(TYPE) * dimension * numClusters) ;

#if defined(_OPENMP)
#pragma omp parallel for default(shared) private(i_cl,i_d,dim) num_threads(vl_get_max_threads())
#endif
for (i_cl = 0; i_cl < (signed)numClusters; i_cl++) {
double clusterMass = 0 ;
for (i_d = 0; i_d < (signed)numData; i_d++) {
if (assignments[i_d*numClusters + i_cl] > 0) {
double q = assignments[i_d*numClusters+i_cl] ;
clusterMass +=  q ;
for(dim = 0; dim < dimension; dim++) {
enc [i_cl * dimension + dim] += q * data [i_d  * dimension + dim] ;
}
}
}

if (clusterMass > 0) {
if (flags & VL_VLAD_FLAG_NORMALIZE_MASS) {
for(dim = 0; dim < dimension; dim++) {
enc[i_cl*dimension + dim] /= clusterMass ;
enc[i_cl*dimension + dim] -= means[i_cl*dimension+dim];
}
} else {
for(dim = 0; dim < dimension; dim++) {
enc[i_cl*dimension + dim] -= clusterMass * means[i_cl*dimension+dim];
}
}
}

if (flags & VL_VLAD_FLAG_SQUARE_ROOT) {
for(dim = 0; dim < dimension; dim++) {
TYPE z = enc[i_cl*dimension + dim] ;
if (z >= 0) {
enc[i_cl*dimension + dim] = VL_XCAT(vl_sqrt_, SFX)(z) ;
} else {
enc[i_cl*dimension + dim] = - VL_XCAT(vl_sqrt_, SFX)(- z) ;
}
}
}

if (flags & VL_VLAD_FLAG_NORMALIZE_COMPONENTS) {
TYPE n = 0 ;
dim = 0 ;
for(dim = 0; dim < dimension; dim++) {
TYPE z = enc[i_cl*dimension + dim] ;
n += z * z ;
}
n = VL_XCAT(vl_sqrt_, SFX)(n) ;
n = VL_MAX(n, 1e-12) ;
for(dim = 0; dim < dimension; dim++) {
enc[i_cl*dimension + dim] /= n ;
}
}
}

if (! (flags & VL_VLAD_FLAG_UNNORMALIZED)) {
TYPE n = 0 ;
for(dim = 0 ; dim < dimension * numClusters ; dim++) {
TYPE z = enc [dim] ;
n += z * z ;
}
n = VL_XCAT(vl_sqrt_, SFX)(n) ;
n = VL_MAX(n, 1e-12) ;
for(dim = 0 ; dim < dimension * numClusters ; dim++) {
enc[dim] /= n ;
}
}
}


#else

#ifndef __DOXYGEN__
#define FLT VL_TYPE_FLOAT
#define TYPE float
#define SFX f
#define VL_VLAD_INSTANTIATING
#include "vlad.c"

#define FLT VL_TYPE_DOUBLE
#define TYPE double
#define SFX d
#define VL_VLAD_INSTANTIATING
#include "vlad.c"
#endif


#endif


#ifndef VL_VLAD_INSTANTIATING



void
vl_vlad_encode (void * enc, vl_type dataType,
void const * means, vl_size dimension, vl_size numClusters,
void const * data, vl_size numData,
void const * assignments,
int flags)
{
switch(dataType) {
case VL_TYPE_FLOAT:
_vl_vlad_encode_f ((float *) enc,
(float const *) means, dimension, numClusters,
(float const *) data,  numData,
(float const *) assignments, flags) ;
break;
case VL_TYPE_DOUBLE:
_vl_vlad_encode_d ((double *) enc,
(double const *) means, dimension, numClusters,
(double const *) data, numData,
(double const *) assignments, flags) ;
break;
default:
abort();
}
}


#endif

#undef SFX
#undef TYPE
#undef FLT
#undef VL_VLAD_INSTANTIATING
