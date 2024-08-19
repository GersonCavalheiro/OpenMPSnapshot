





#include "fisher.h"
#include "gmm.h"
#include "mathop.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef VL_FISHER_INSTANTIATING

#endif

#ifdef VL_FISHER_INSTANTIATING

static void
VL_XCAT(_vl_fisher_encode_, SFX)
(TYPE * enc,
TYPE const * means, vl_size dimension, vl_size numClusters,
TYPE const * covariances,
TYPE const * priors,
TYPE const * data, vl_size numData,
int flags)
{
vl_size dim;
vl_index i_cl, i_d;
TYPE * posteriors ;
TYPE * sqrtInvSigma;

posteriors = vl_malloc(sizeof(TYPE) * numClusters * numData);
sqrtInvSigma = vl_malloc(sizeof(TYPE) * dimension * numClusters);

memset(enc, 0, sizeof(TYPE) * 2 * dimension * numClusters) ;

for (i_cl = 0 ; i_cl < (signed)numClusters ; ++i_cl) {
for(dim = 0; dim < dimension; dim++) {
sqrtInvSigma[i_cl*dimension + dim] = sqrt(1.0 / covariances[i_cl*dimension + dim]);
}
}

VL_XCAT(vl_get_gmm_data_posteriors_, SFX)(posteriors, numClusters, numData,
priors,
means, dimension,
covariances,
data) ;

#if defined(_OPENMP)
#pragma omp parallel for default(shared) private(i_cl, i_d, dim) num_threads(vl_get_max_threads())
#endif
for(i_cl = 0; i_cl < (signed)numClusters; ++ i_cl) {
TYPE uprefix;
TYPE vprefix;

TYPE * uk = enc + i_cl*dimension ;
TYPE * vk = enc + i_cl*dimension + numClusters * dimension ;

if (priors[i_cl] < 1e-6) { continue ; }

for(i_d = 0; i_d < (signed)numData; i_d++) {
TYPE p = posteriors[i_cl + i_d * numClusters] ;
if (p == 0) continue ;
for(dim = 0; dim < dimension; dim++) {
TYPE diff = data[i_d*dimension + dim] - means[i_cl*dimension + dim] ;
diff *= sqrtInvSigma[i_cl*dimension + dim] ;
*(uk + dim) += p * diff ;
*(vk + dim) += p * (diff * diff - 1);
}
}

uprefix = 1/(numData*sqrt(priors[i_cl]));
vprefix = 1/(numData*sqrt(2*priors[i_cl]));

for(dim = 0; dim < dimension; dim++) {
*(uk + dim) = *(uk + dim) * uprefix;
*(vk + dim) = *(vk + dim) * vprefix;
}
}

vl_free(posteriors);
vl_free(sqrtInvSigma) ;

if (flags & VL_FISHER_FLAG_SQUARE_ROOT) {
for(dim = 0; dim < 2 * dimension * numClusters ; dim++) {
TYPE z = enc [dim] ;
if (z >= 0) {
enc[dim] = VL_XCAT(vl_sqrt_, SFX)(z) ;
} else {
enc[dim] = - VL_XCAT(vl_sqrt_, SFX)(- z) ;
}
}
}

if (flags & VL_FISHER_FLAG_NORMALIZED) {
TYPE n = 0 ;
for(dim = 0 ; dim < 2 * dimension * numClusters ; dim++) {
TYPE z = enc [dim] ;
n += z * z ;
}
n = VL_XCAT(vl_sqrt_, SFX)(n) ;
n = VL_MAX(n, 1e-12) ;
for(dim = 0 ; dim < 2 * dimension * numClusters ; dim++) {
enc[dim] /= n ;
}
}
}


#else

#ifndef __DOXYGEN__
#define FLT VL_TYPE_FLOAT
#define TYPE float
#define SFX f
#define VL_FISHER_INSTANTIATING
#include "fisher.c"

#define FLT VL_TYPE_DOUBLE
#define TYPE double
#define SFX d
#define VL_FISHER_INSTANTIATING
#include "fisher.c"
#endif

#endif


#ifndef VL_FISHER_INSTANTIATING



VL_EXPORT void
vl_fisher_encode
(void * enc, vl_type dataType,
void const * means, vl_size dimension, vl_size numClusters,
void const * covariances,
void const * priors,
void const * data,  vl_size numData,
int flags
)
{
switch(dataType) {
case VL_TYPE_FLOAT:
_vl_fisher_encode_f
((float *) enc,
(float const *) means, dimension, numClusters,
(float const *) covariances,
(float const *) priors,
(float const *) data, numData,
flags);
break;
case VL_TYPE_DOUBLE:
_vl_fisher_encode_d
((double *) enc,
(double const *) means, dimension, numClusters,
(double const *) covariances,
(double const *) priors,
(double const *) data, numData,
flags);
break;
default:
abort();
}
}

#endif

#undef SFX
#undef TYPE
#undef FLT
#undef VL_FISHER_INSTANTIATING
