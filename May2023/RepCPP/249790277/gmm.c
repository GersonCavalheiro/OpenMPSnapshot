







#include "gmm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef VL_DISABLE_SSE2
#include "mathop_sse2.h"
#endif

#ifndef VL_DISABLE_AVX
#include "mathop_avx.h"
#endif


#ifndef VL_GMM_INSTANTIATING


#define VL_GMM_MIN_VARIANCE 1e-6
#define VL_GMM_MIN_POSTERIOR 1e-2
#define VL_GMM_MIN_PRIOR 1e-6

struct _VlGMM
{
vl_type dataType ;                  
vl_size dimension ;                 
vl_size numClusters ;               
vl_size numData ;                   
vl_size maxNumIterations ;          
vl_size numRepetitions   ;          
int     verbosity ;                 
void *  means;                      
void *  covariances;                
void *  priors;                     
void *  posteriors;                 
double * sigmaLowBound ;            
VlGMMInitialization initialization; 
VlKMeans * kmeansInit;              
double LL ;                         
vl_bool kmeansInitIsOwner; 
} ;





static void
_vl_gmm_prepare_for_data (VlGMM* self, vl_size numData)
{
if (self->numData < numData) {
vl_free(self->posteriors) ;
self->posteriors = vl_malloc(vl_get_type_size(self->dataType) * numData * self->numClusters) ;
}
self->numData = numData ;
}



VlGMM *
vl_gmm_new (vl_type dataType, vl_size dimension, vl_size numComponents)
{
vl_index i ;
vl_size size = vl_get_type_size(dataType) ;
VlGMM * self = vl_calloc(1, sizeof(VlGMM)) ;
self->dataType = dataType;
self->numClusters = numComponents ;
self->numData = 0;
self->dimension = dimension ;
self->initialization = VlGMMRand;
self->verbosity = 0 ;
self->maxNumIterations = 50;
self->numRepetitions = 1;
self->sigmaLowBound =  NULL ;
self->priors = NULL ;
self->covariances = NULL ;
self->means = NULL ;
self->posteriors = NULL ;
self->kmeansInit = NULL ;
self->kmeansInitIsOwner = VL_FALSE;

self->priors = vl_calloc (numComponents, size) ;
self->means = vl_calloc (numComponents * dimension, size) ;
self->covariances = vl_calloc (numComponents * dimension, size) ;
self->sigmaLowBound = vl_calloc (dimension, sizeof(double)) ;

for (i = 0 ; i < (unsigned)self->dimension ; ++i)  { self->sigmaLowBound[i] = 1e-4 ; }
return self ;
}



void
vl_gmm_reset (VlGMM * self)
{
if (self->posteriors) {
vl_free(self->posteriors) ;
self->posteriors = NULL ;
self->numData = 0 ;
}
if (self->kmeansInit && self->kmeansInitIsOwner) {
vl_kmeans_delete(self->kmeansInit) ;
self->kmeansInit = NULL ;
self->kmeansInitIsOwner = VL_FALSE ;
}
}



void
vl_gmm_delete (VlGMM * self)
{
if(self->means) vl_free(self->means);
if(self->covariances) vl_free(self->covariances);
if(self->priors) vl_free(self->priors);
if(self->posteriors) vl_free(self->posteriors);
if(self->kmeansInit && self->kmeansInitIsOwner) {
vl_kmeans_delete(self->kmeansInit);
}
vl_free(self);
}







vl_type
vl_gmm_get_data_type (VlGMM const * self)
{
return self->dataType ;
}



vl_size
vl_gmm_get_num_clusters (VlGMM const * self)
{
return self->numClusters ;
}



vl_size
vl_gmm_get_num_data (VlGMM const * self)
{
return self->numData ;
}



double
vl_gmm_get_loglikelihood (VlGMM const * self)
{
return self->LL ;
}



int
vl_gmm_get_verbosity (VlGMM const * self)
{
return self->verbosity ;
}



void
vl_gmm_set_verbosity (VlGMM * self, int verbosity)
{
self->verbosity = verbosity ;
}



void const *
vl_gmm_get_means (VlGMM const * self)
{
return self->means ;
}



void const *
vl_gmm_get_covariances (VlGMM const * self)
{
return self->covariances ;
}



void const *
vl_gmm_get_priors (VlGMM const * self)
{
return self->priors ;
}



void const *
vl_gmm_get_posteriors (VlGMM const * self)
{
return self->posteriors ;
}



vl_size
vl_gmm_get_max_num_iterations (VlGMM const * self)
{
return self->maxNumIterations ;
}



void
vl_gmm_set_max_num_iterations (VlGMM * self, vl_size maxNumIterations)
{
self->maxNumIterations = maxNumIterations ;
}



vl_size
vl_gmm_get_num_repetitions (VlGMM const * self)
{
return self->numRepetitions ;
}



void
vl_gmm_set_num_repetitions (VlGMM * self, vl_size numRepetitions)
{
assert (numRepetitions >= 1) ;
self->numRepetitions = numRepetitions ;
}



vl_size
vl_gmm_get_dimension (VlGMM const * self)
{
return self->dimension ;
}



VlGMMInitialization
vl_gmm_get_initialization (VlGMM const * self)
{
return self->initialization ;
}


void
vl_gmm_set_initialization (VlGMM * self, VlGMMInitialization init)
{
self->initialization = init;
}


VlKMeans * vl_gmm_get_kmeans_init_object (VlGMM const * self)
{
return self->kmeansInit;
}


void vl_gmm_set_kmeans_init_object (VlGMM * self, VlKMeans * kmeans)
{
if (self->kmeansInit && self->kmeansInitIsOwner) {
vl_kmeans_delete(self->kmeansInit) ;
}
self->kmeansInit = kmeans;
self->kmeansInitIsOwner = VL_FALSE;
}


double const * vl_gmm_get_covariance_lower_bounds (VlGMM const * self)
{
return self->sigmaLowBound;
}


void vl_gmm_set_covariance_lower_bounds (VlGMM * self, double const * bounds)
{
memcpy(self->sigmaLowBound, bounds, sizeof(double) * self->dimension) ;
}


void vl_gmm_set_covariance_lower_bound (VlGMM * self, double bound)
{
int i ;
for (i = 0 ; i < (signed)self->dimension ; ++i) {
self->sigmaLowBound[i] = bound ;
}
}




#define VL_SHUFFLE_type vl_uindex
#define VL_SHUFFLE_prefix _vl_gmm
#include "shuffle-def.h"


#endif


#ifdef VL_GMM_INSTANTIATING








double
VL_XCAT(vl_get_gmm_data_posteriors_, SFX)
(TYPE * posteriors,
vl_size numClusters,
vl_size numData,
TYPE const * priors,
TYPE const * means,
vl_size dimension,
TYPE const * covariances,
TYPE const * data)
{
vl_index i_d, i_cl;
vl_size dim;
double LL = 0;

TYPE halfDimLog2Pi = (dimension / 2.0) * log(2.0*VL_PI);
TYPE * logCovariances ;
TYPE * logWeights ;
TYPE * invCovariances ;

#if (FLT == VL_TYPE_FLOAT)
VlFloatVector3ComparisonFunction distFn = vl_get_vector_3_comparison_function_f(VlDistanceMahalanobis) ;
#else
VlDoubleVector3ComparisonFunction distFn = vl_get_vector_3_comparison_function_d(VlDistanceMahalanobis) ;
#endif

logCovariances = vl_malloc(sizeof(TYPE) * numClusters) ;
invCovariances = vl_malloc(sizeof(TYPE) * numClusters * dimension) ;
logWeights = vl_malloc(numClusters * sizeof(TYPE)) ;

#if defined(_OPENMP)
#pragma omp parallel for private(i_cl,dim) num_threads(vl_get_max_threads())
#endif
for (i_cl = 0 ; i_cl < (signed)numClusters ; ++ i_cl) {
TYPE logSigma = 0 ;
if (priors[i_cl] < VL_GMM_MIN_PRIOR) {
logWeights[i_cl] = - (TYPE) VL_INFINITY_D ;
} else {
logWeights[i_cl] = log(priors[i_cl]);
}
for(dim = 0 ; dim < dimension ; ++ dim) {
logSigma += log(covariances[i_cl*dimension + dim]);
invCovariances [i_cl*dimension + dim] = (TYPE) 1.0 / covariances[i_cl*dimension + dim];
}
logCovariances[i_cl] = logSigma;
} 

#if defined(_OPENMP)
#pragma omp parallel for private(i_cl,i_d) reduction(+:LL) \
num_threads(vl_get_max_threads())
#endif
for (i_d = 0 ; i_d < (signed)numData ; ++ i_d) {
TYPE clusterPosteriorsSum = 0;
TYPE maxPosterior = (TYPE)(-VL_INFINITY_D) ;

for (i_cl = 0 ; i_cl < (signed)numClusters ; ++ i_cl) {
TYPE p =
logWeights[i_cl]
- halfDimLog2Pi
- 0.5 * logCovariances[i_cl]
- 0.5 * distFn (dimension,
data + i_d * dimension,
means + i_cl * dimension,
invCovariances + i_cl * dimension) ;
posteriors[i_cl + i_d * numClusters] = p ;
if (p > maxPosterior) { maxPosterior = p ; }
}

for (i_cl = 0 ; i_cl < (signed)numClusters ; ++i_cl) {
TYPE p = posteriors[i_cl + i_d * numClusters] ;
p =  exp(p - maxPosterior) ;
posteriors[i_cl + i_d * numClusters] = p ;
clusterPosteriorsSum += p ;
}

LL +=  log(clusterPosteriorsSum) + (double) maxPosterior ;

for (i_cl = 0 ; i_cl < (signed)numClusters ; ++i_cl) {
posteriors[i_cl + i_d * numClusters] /= clusterPosteriorsSum ;
}
} 

vl_free(logCovariances);
vl_free(logWeights);
vl_free(invCovariances);

return LL;
}





static void
VL_XCAT(_vl_gmm_maximization_, SFX)
(VlGMM * self,
TYPE * posteriors,
TYPE * priors,
TYPE * covariances,
TYPE * means,
TYPE const * data,
vl_size numData) ;

static vl_size
VL_XCAT(_vl_gmm_restart_empty_modes_, SFX) (VlGMM * self, TYPE const * data)
{
vl_size dimension = self->dimension;
vl_size numClusters = self->numClusters;
vl_index i_cl, j_cl, i_d, d;
vl_size zeroWNum = 0;
TYPE * priors = (TYPE*)self->priors ;
TYPE * means = (TYPE*)self->means ;
TYPE * covariances = (TYPE*)self->covariances ;
TYPE * posteriors = (TYPE*)self->posteriors ;


TYPE * mass = vl_calloc(sizeof(TYPE), self->numClusters) ;

if (numClusters <= 1) { return 0 ; }


{
vl_uindex i, k ;
vl_size numNullAssignments = 0 ;
for (i = 0 ; i < self->numData ; ++i) {
for (k = 0 ; k < self->numClusters ; ++k) {
TYPE p = ((TYPE*)self->posteriors)[k + i * self->numClusters] ;
mass[k] += p ;
if (p < VL_GMM_MIN_POSTERIOR) {
numNullAssignments ++ ;
}
}
}
if (self->verbosity) {
VL_PRINTF("gmm: sparsity of data posterior: %.1f%%\n", (double)numNullAssignments / (self->numData * self->numClusters) * 100) ;
}
}

#if 0

for (i_cl = 0 ; i_cl < numClusters ; ++i_cl) {
if (priors[i_cl] < 0.00001/numClusters) {
double mass = priors[0]  ;
vl_index best = 0 ;

for (j_cl = 1 ; j_cl < numClusters ; ++j_cl) {
if (priors[j_cl] > mass) { mass = priors[j_cl] ; best = j_cl ; }
}

if (j_cl == i_cl) {

continue ;
}

j_cl = best ;
zeroWNum ++ ;

VL_PRINTF("gmm: restarting mode %d by splitting mode %d (with prior %f)\n", i_cl,j_cl,mass) ;

priors[i_cl] = mass/2 ;
priors[j_cl] = mass/2 ;
for (d = 0 ; d < dimension ; ++d) {
TYPE sigma2 =  covariances[j_cl*dimension + d] ;
TYPE sigma = VL_XCAT(vl_sqrt_,SFX)(sigma2) ;
means[i_cl*dimension + d] = means[j_cl*dimension + d] + 0.001 * (vl_rand_real1(rand) - 0.5) * sigma ;
covariances[i_cl*dimension + d] = sigma2 ;
}
}
}
#endif


for (i_cl = 0 ; i_cl < (signed)numClusters ; ++i_cl) {
double size = - VL_INFINITY_D ;
vl_index best = -1 ;

if (mass[i_cl] >= VL_GMM_MIN_POSTERIOR *
VL_MAX(1.0, (double) self->numData / self->numClusters))
{
continue ;
}

if (self->verbosity) {
VL_PRINTF("gmm: mode %d is nearly empty (mass %f)\n", i_cl, mass[i_cl]) ;
}



for (j_cl = 0 ; j_cl < (signed)numClusters ; ++j_cl) {
double size_ ;
if (priors[j_cl] < VL_GMM_MIN_PRIOR) { continue ; }
size_ = - 0.5 * (1.0 + log(2*VL_PI)) ;
for(d = 0 ; d < (signed)dimension ; d++) {
double sigma2 = covariances[j_cl * dimension + d] ;
size_ -= 0.5 * log(sigma2) ;
}
size_ *= priors[j_cl] ;

if (self->verbosity > 2) {
VL_PRINTF("gmm: mode %d: prior %f, mass %f, score %f\n",
j_cl, priors[j_cl], mass[j_cl], size_) ;
}

if (size_ > size) {
size = size_ ;
best = j_cl ;
}
}

j_cl = best ;

if (j_cl == i_cl || j_cl < 0) {
if (self->verbosity) {
VL_PRINTF("gmm: mode %d is empty, "
"but no other mode to split could be found\n", i_cl) ;
}
continue ;
}

if (self->verbosity) {
VL_PRINTF("gmm: reinitializing empty mode %d with mode %d (prior %f, mass %f, score %f)\n",
i_cl, j_cl, priors[j_cl], mass[j_cl], size) ;
}



size = - VL_INFINITY_D ;
best = - 1 ;

for(d = 0; d < (signed)dimension; d++) {
double sigma2 = covariances[j_cl * dimension + d] ;
if (sigma2 > size) {
size = sigma2 ;
best = d ;
}
}


{
TYPE mu = means[best + j_cl * self->dimension] ;
for(i_d = 0 ; i_d < (signed)self->numData ; ++ i_d) {
TYPE p = posteriors[j_cl + self->numClusters * i_d] ;
TYPE q = posteriors[i_cl + self->numClusters * i_d] ; 
if (data[best + i_d * self->dimension] < mu) {

posteriors[i_cl + self->numClusters * i_d] += p ;
posteriors[j_cl + self->numClusters * i_d] = 0 ;
} else {

posteriors[i_cl + self->numClusters * i_d] = 0 ;
posteriors[j_cl + self->numClusters * i_d] += q ;
}
}
}


VL_XCAT(_vl_gmm_maximization_, SFX)
(self,posteriors,priors,covariances,means,data,self->numData) ;
}

return zeroWNum;
}





static void
VL_XCAT(_vl_gmm_apply_bounds_, SFX)(VlGMM * self)
{
vl_uindex dim ;
vl_uindex k ;
vl_size numAdjusted = 0 ;
TYPE * cov = (TYPE*)self->covariances ;
double const * lbs = self->sigmaLowBound ;

for (k = 0 ; k < self->numClusters ; ++k) {
vl_bool adjusted = VL_FALSE ;
for (dim = 0 ; dim < self->dimension ; ++dim) {
if (cov[k * self->dimension + dim] < lbs[dim] ) {
cov[k * self->dimension + dim] = lbs[dim] ;
adjusted = VL_TRUE ;
}
}
if (adjusted) { numAdjusted ++ ; }
}

if (numAdjusted > 0 && self->verbosity > 0) {
VL_PRINT("gmm: detected %d of %d modes with at least one dimension "
"with covariance too small (set to lower bound)\n",
numAdjusted, self->numClusters) ;
}
}





static void
VL_XCAT(_vl_gmm_maximization_, SFX)
(VlGMM * self,
TYPE * posteriors,
TYPE * priors,
TYPE * covariances,
TYPE * means,
TYPE const * data,
vl_size numData)
{
vl_size numClusters = self->numClusters;
vl_index i_d, i_cl;
vl_size dim ;
TYPE * oldMeans ;
double time = 0 ;

if (self->verbosity > 1) {
VL_PRINTF("gmm: em: entering maximization step\n") ;
time = vl_get_cpu_time() ;
}

oldMeans = vl_malloc(sizeof(TYPE) * self->dimension * numClusters) ;
memcpy(oldMeans, means, sizeof(TYPE) * self->dimension * numClusters) ;

memset(priors, 0, sizeof(TYPE) * numClusters) ;
memset(means, 0, sizeof(TYPE) * self->dimension * numClusters) ;
memset(covariances, 0, sizeof(TYPE) * self->dimension * numClusters) ;

#if defined(_OPENMP)
#pragma omp parallel default(shared) private(i_d, i_cl, dim) \
num_threads(vl_get_max_threads())
#endif
{
TYPE * clusterPosteriorSum_, * means_, * covariances_ ;

#if defined(_OPENMP)
#pragma omp critical
#endif
{
clusterPosteriorSum_ = vl_calloc(sizeof(TYPE), numClusters) ;
means_ = vl_calloc(sizeof(TYPE), self->dimension * numClusters) ;
covariances_ = vl_calloc(sizeof(TYPE), self->dimension * numClusters) ;
}



#if defined(_OPENMP)
#pragma omp for
#endif
for (i_d = 0 ; i_d < (signed)numData ; ++i_d) {
for (i_cl = 0 ; i_cl < (signed)numClusters ; ++i_cl) {
TYPE p = posteriors[i_cl + i_d * self->numClusters] ;
vl_bool calculated = VL_FALSE ;


if (p < VL_GMM_MIN_POSTERIOR / numClusters) { continue ; }

clusterPosteriorSum_ [i_cl] += p ;

#ifndef VL_DISABLE_AVX
if (vl_get_simd_enabled() && vl_cpu_has_avx()) {
VL_XCAT(_vl_weighted_mean_sse2_, SFX)
(self->dimension,
means_+ i_cl * self->dimension,
data + i_d * self->dimension,
p) ;

VL_XCAT(_vl_weighted_sigma_sse2_, SFX)
(self->dimension,
covariances_ + i_cl * self->dimension,
data + i_d * self->dimension,
oldMeans + i_cl * self->dimension,
p) ;

calculated = VL_TRUE;
}
#endif
#ifndef VL_DISABLE_SSE2
if (vl_get_simd_enabled() && vl_cpu_has_sse2() && !calculated) {
VL_XCAT(_vl_weighted_mean_sse2_, SFX)
(self->dimension,
means_+ i_cl * self->dimension,
data + i_d * self->dimension,
p) ;

VL_XCAT(_vl_weighted_sigma_sse2_, SFX)
(self->dimension,
covariances_ + i_cl * self->dimension,
data + i_d * self->dimension,
oldMeans + i_cl * self->dimension,
p) ;

calculated = VL_TRUE;
}
#endif
if(!calculated) {
for (dim = 0 ; dim < self->dimension ; ++dim) {
TYPE x = data[i_d * self->dimension + dim] ;
TYPE mu = oldMeans[i_cl * self->dimension + dim] ;
TYPE diff = x - mu ;
means_ [i_cl * self->dimension + dim] += p * x ;
covariances_ [i_cl * self->dimension + dim] += p * (diff*diff) ;
}
}
}
}


#if defined(_OPENMP)
#pragma omp critical
#endif
{
for (i_cl = 0 ; i_cl < (signed)numClusters ; ++i_cl) {
priors [i_cl] += clusterPosteriorSum_ [i_cl];
for (dim = 0 ; dim < self->dimension ; ++dim) {
means [i_cl * self->dimension + dim] += means_ [i_cl * self->dimension + dim] ;
covariances [i_cl * self->dimension + dim] += covariances_ [i_cl * self->dimension + dim] ;
}
}
vl_free(means_);
vl_free(covariances_);
vl_free(clusterPosteriorSum_);
}
} 


for (i_cl = 0 ; i_cl < (signed)numClusters ; ++ i_cl) {
TYPE mass = priors[i_cl] ;

if (mass >= 1e-6 / numClusters) {
for (dim = 0 ; dim < self->dimension ; ++dim) {
means[i_cl * self->dimension + dim] /= mass ;
covariances[i_cl * self->dimension + dim] /= mass ;
}
}
}


for (i_cl = 0 ; i_cl < (signed)numClusters ; ++ i_cl) {
TYPE mass = priors[i_cl] ;
if (mass >= 1e-6 / numClusters) {
for (dim = 0 ; dim < self->dimension ; ++dim) {
TYPE mu = means[i_cl * self->dimension + dim] ;
TYPE oldMu = oldMeans[i_cl * self->dimension + dim] ;
TYPE diff = mu - oldMu ;
covariances[i_cl * self->dimension + dim] -= diff * diff ;
}
}
}

VL_XCAT(_vl_gmm_apply_bounds_,SFX)(self) ;

{
TYPE sum = 0;
for (i_cl = 0 ; i_cl < (signed)numClusters ; ++i_cl) {
sum += priors[i_cl] ;
}
sum = VL_MAX(sum, 1e-12) ;
for (i_cl = 0 ; i_cl < (signed)numClusters ; ++i_cl) {
priors[i_cl] /= sum ;
}
}

if (self->verbosity > 1) {
VL_PRINTF("gmm: em: maximization step completed in %.2f s\n",
vl_get_cpu_time() - time) ;
}

vl_free(oldMeans);
}






static double
VL_XCAT(_vl_gmm_em_, SFX)
(VlGMM * self,
TYPE const * data,
vl_size numData)
{
vl_size iteration, restarted ;
double previousLL = (TYPE)(-VL_INFINITY_D) ;
double LL = (TYPE)(-VL_INFINITY_D) ;
double time = 0 ;

_vl_gmm_prepare_for_data (self, numData) ;

VL_XCAT(_vl_gmm_apply_bounds_,SFX)(self) ;

for (iteration = 0 ; 1 ; ++ iteration) {
double eps ;



if (self->verbosity > 1) {
VL_PRINTF("gmm: em: entering expectation step\n") ;
time = vl_get_cpu_time() ;
}

LL = VL_XCAT(vl_get_gmm_data_posteriors_,SFX)
(self->posteriors,
self->numClusters,
numData,
self->priors,
self->means,
self->dimension,
self->covariances,
data) ;

if (self->verbosity > 1) {
VL_PRINTF("gmm: em: expectation step completed in %.2f s\n",
vl_get_cpu_time() - time) ;
}


if (self->verbosity) {
VL_PRINTF("gmm: em: iteration %d: loglikelihood = %f (variation = %f)\n",
iteration, LL, LL - previousLL) ;
}
if (iteration >= self->maxNumIterations) {
if (self->verbosity) {
VL_PRINTF("gmm: em: terminating because "
"the maximum number of iterations "
"(%d) has been reached.\n", self->maxNumIterations) ;
}
break ;
}

eps = vl_abs_d ((LL - previousLL) / (LL));
if ((iteration > 0) && (eps < 0.00001)) {
if (self->verbosity) {
VL_PRINTF("gmm: em: terminating because the algorithm "
"fully converged (log-likelihood variation = %f).\n", eps) ;
}
break ;
}
previousLL = LL ;


if (iteration > 1) {
restarted = VL_XCAT(_vl_gmm_restart_empty_modes_, SFX)
(self, data);
if ((restarted > 0) & (self->verbosity > 0)) {
VL_PRINTF("gmm: em: %d Gaussian modes restarted because "
"they had become empty.\n", restarted);
}
}


VL_XCAT(_vl_gmm_maximization_, SFX)
(self,self->posteriors,self->priors,self->covariances,self->means,data,numData) ;
}
return LL;
}






static void
VL_XCAT(_vl_gmm_init_with_kmeans_, SFX)
(VlGMM * self,
TYPE const * data,
vl_size numData,
VlKMeans * kmeansInit)
{
vl_size i_d ;
vl_uint32 * assignments = vl_malloc(sizeof(vl_uint32) * numData);

_vl_gmm_prepare_for_data (self, numData) ;

memset(self->means,0,sizeof(TYPE) * self->numClusters * self->dimension) ;
memset(self->priors,0,sizeof(TYPE) * self->numClusters) ;
memset(self->covariances,0,sizeof(TYPE) * self->numClusters * self->dimension) ;
memset(self->posteriors,0,sizeof(TYPE) * self->numClusters * numData) ;


if (kmeansInit) { vl_gmm_set_kmeans_init_object (self, kmeansInit) ; }


if(self->kmeansInit == NULL) {
vl_size ncomparisons = VL_MAX(numData / 4, 10) ;
vl_size niter = 5 ;
vl_size ntrees = 1 ;
vl_size nrepetitions = 1 ;
VlKMeansAlgorithm algorithm = VlKMeansANN ;
VlKMeansInitialization initialization = VlKMeansRandomSelection ;

VlKMeans * kmeansInitDefault = vl_kmeans_new(self->dataType,VlDistanceL2) ;
vl_kmeans_set_initialization(kmeansInitDefault, initialization);
vl_kmeans_set_max_num_iterations (kmeansInitDefault, niter) ;
vl_kmeans_set_max_num_comparisons (kmeansInitDefault, ncomparisons) ;
vl_kmeans_set_num_trees (kmeansInitDefault, ntrees);
vl_kmeans_set_algorithm (kmeansInitDefault, algorithm);
vl_kmeans_set_num_repetitions(kmeansInitDefault, nrepetitions);
vl_kmeans_set_verbosity (kmeansInitDefault, self->verbosity);

self->kmeansInit = kmeansInitDefault;
self->kmeansInitIsOwner = VL_TRUE ;
}


vl_kmeans_cluster (self->kmeansInit, data, self->dimension, numData, self->numClusters);
vl_kmeans_quantize (self->kmeansInit, assignments, NULL, data, numData) ;


for(i_d = 0; i_d < numData; i_d++) {
((TYPE*)self->posteriors)[assignments[i_d] + i_d * self->numClusters] = (TYPE) 1.0 ;
}


VL_XCAT(_vl_gmm_maximization_, SFX)
(self,self->posteriors,self->priors,self->covariances,self->means,data,numData);
vl_free(assignments) ;
}





static void
VL_XCAT(_vl_gmm_compute_init_sigma_, SFX)
(VlGMM * self,
TYPE const * data,
TYPE * initSigma,
vl_size dimension,
vl_size numData)
{
vl_size dim;
vl_uindex i;

TYPE * dataMean ;

memset(initSigma,0,sizeof(TYPE)*dimension) ;
if (numData <= 1) return ;

dataMean = vl_malloc(sizeof(TYPE)*dimension);
memset(dataMean,0,sizeof(TYPE)*dimension) ;


for(dim = 0 ; dim < dimension ; dim++) {
for(i = 0 ; i < numData ; i++) {
dataMean[dim] += data[i*dimension + dim];
}
dataMean[dim] /= numData;
}


for(dim = 0; dim < dimension; dim++) {
for(i = 0; i < numData; i++) {
TYPE diff = (data[i*self->dimension + dim] - dataMean[dim]) ;
initSigma[dim] += diff*diff ;
}
initSigma[dim] /= numData - 1 ;
}

vl_free(dataMean) ;
}

static void
VL_XCAT(_vl_gmm_init_with_rand_data_, SFX)
(VlGMM * self,
TYPE const * data,
vl_size numData)
{
vl_uindex i, k, dim ;
VlKMeans * kmeans ;

_vl_gmm_prepare_for_data(self, numData) ;


for (i = 0 ; i < self->numClusters ; ++i) { ((TYPE*)self->priors)[i] = (TYPE) (1.0 / self->numClusters) ; }


VL_XCAT(_vl_gmm_compute_init_sigma_, SFX) (self, data, self->covariances, self->dimension, numData);
for (k = 1 ; k < self->numClusters ; ++ k) {
for(dim = 0; dim < self->dimension; dim++) {
*((TYPE*)self->covariances + k * self->dimension + dim) =
*((TYPE*)self->covariances + dim) ;
}
}


kmeans = vl_kmeans_new(self->dataType,VlDistanceL2) ;
vl_kmeans_init_centers_plus_plus(kmeans, data, self->dimension, numData, self->numClusters) ;
memcpy(self->means, vl_kmeans_get_centers(kmeans), sizeof(TYPE) * self->dimension * self->numClusters) ;
vl_kmeans_delete(kmeans) ;
}


#else 


#ifndef __DOXYGEN__
#define FLT VL_TYPE_FLOAT
#define TYPE float
#define SFX f
#define VL_GMM_INSTANTIATING
#include "gmm.c"

#define FLT VL_TYPE_DOUBLE
#define TYPE double
#define SFX d
#define VL_GMM_INSTANTIATING
#include "gmm.c"
#endif


#endif


#ifndef VL_GMM_INSTANTIATING




VlGMM *
vl_gmm_new_copy (VlGMM const * self)
{
vl_size size = vl_get_type_size(self->dataType) ;
VlGMM * gmm = vl_gmm_new(self->dataType, self->dimension, self->numClusters);
gmm->initialization = self->initialization;
gmm->maxNumIterations = self->maxNumIterations;
gmm->numRepetitions = self->numRepetitions;
gmm->verbosity = self->verbosity;
gmm->LL = self->LL;

memcpy(gmm->means, self->means, size*self->numClusters*self->dimension);
memcpy(gmm->covariances, self->covariances, size*self->numClusters*self->dimension);
memcpy(gmm->priors, self->priors, size*self->numClusters);
return gmm ;
}



void
vl_gmm_init_with_rand_data
(VlGMM * self,
void const * data,
vl_size numData)
{
vl_gmm_reset (self) ;
switch (self->dataType) {
case VL_TYPE_FLOAT : _vl_gmm_init_with_rand_data_f (self, (float const *)data, numData) ; break ;
case VL_TYPE_DOUBLE : _vl_gmm_init_with_rand_data_d (self, (double const *)data, numData) ; break ;
default:
abort() ;
}
}



void
vl_gmm_init_with_kmeans
(VlGMM * self,
void const * data,
vl_size numData,
VlKMeans * kmeansInit)
{
vl_gmm_reset (self) ;
switch (self->dataType) {
case VL_TYPE_FLOAT :
_vl_gmm_init_with_kmeans_f
(self, (float const *)data, numData, kmeansInit) ;
break ;
case VL_TYPE_DOUBLE :
_vl_gmm_init_with_kmeans_d
(self, (double const *)data, numData, kmeansInit) ;
break ;
default:
abort() ;
}
}

#if 0
#include<fenv.h>
#endif



double vl_gmm_cluster (VlGMM * self,
void const * data,
vl_size numData)
{
void * bestPriors = NULL ;
void * bestMeans = NULL;
void * bestCovariances = NULL;
void * bestPosteriors = NULL;
vl_size size = vl_get_type_size(self->dataType) ;
double bestLL = -VL_INFINITY_D;
vl_uindex repetition;

assert(self->numRepetitions >=1) ;

bestPriors = vl_malloc(size * self->numClusters) ;
bestMeans = vl_malloc(size * self->dimension * self->numClusters) ;
bestCovariances = vl_malloc(size * self->dimension * self->numClusters) ;
bestPosteriors = vl_malloc(size * self->numClusters * numData) ;

#if 0
feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif

for (repetition = 0 ; repetition < self->numRepetitions ; ++ repetition) {
double LL ;
double timeRef ;

if (self->verbosity) {
VL_PRINTF("gmm: clustering: starting repetition %d of %d\n", repetition + 1, self->numRepetitions) ;
}


timeRef = vl_get_cpu_time() ;
switch (self->initialization) {
case VlGMMKMeans : vl_gmm_init_with_kmeans (self, data, numData, NULL) ; break ;
case VlGMMRand : vl_gmm_init_with_rand_data (self, data, numData) ; break ;
case VlGMMCustom : break ;
default: abort() ;
}
if (self->verbosity) {
VL_PRINTF("gmm: model initialized in %.2f s\n",
vl_get_cpu_time() - timeRef) ;
}


timeRef = vl_get_cpu_time () ;
LL = vl_gmm_em (self, data, numData) ;
if (self->verbosity) {
VL_PRINTF("gmm: optimization terminated in %.2f s with loglikelihood %f\n",
vl_get_cpu_time() - timeRef, LL) ;
}

if (LL > bestLL || repetition == 0) {
void * temp ;

temp = bestPriors ;
bestPriors = self->priors ;
self->priors = temp ;

temp = bestMeans ;
bestMeans = self->means ;
self->means = temp ;

temp = bestCovariances ;
bestCovariances = self->covariances ;
self->covariances = temp ;

temp = bestPosteriors ;
bestPosteriors = self->posteriors ;
self->posteriors = temp ;

bestLL = LL;
}
}

vl_free (self->priors) ;
vl_free (self->means) ;
vl_free (self->covariances) ;
vl_free (self->posteriors) ;

self->priors = bestPriors ;
self->means = bestMeans ;
self->covariances = bestCovariances ;
self->posteriors = bestPosteriors ;
self->LL = bestLL;

if (self->verbosity) {
VL_PRINTF("gmm: all repetitions terminated with final loglikelihood %f\n", self->LL) ;
}

return bestLL ;
}



double vl_gmm_em (VlGMM * self, void const * data, vl_size numData)
{
switch (self->dataType) {
case VL_TYPE_FLOAT:
return _vl_gmm_em_f (self, (float const *)data, numData) ; break ;
case VL_TYPE_DOUBLE:
return _vl_gmm_em_d (self, (double const *)data, numData) ; break ;
default:
abort() ;
}
return 0 ;
}



void
vl_gmm_set_means (VlGMM * self, void const * means)
{
memcpy(self->means,means,
self->dimension * self->numClusters * vl_get_type_size(self->dataType));
}



void vl_gmm_set_covariances (VlGMM * self, void const * covariances)
{
memcpy(self->covariances,covariances,
self->dimension * self->numClusters * vl_get_type_size(self->dataType));
}



void vl_gmm_set_priors (VlGMM * self, void const * priors)
{
memcpy(self->priors,priors,
self->numClusters * vl_get_type_size(self->dataType));
}


#endif

#undef SFX
#undef TYPE
#undef FLT
#undef VL_GMM_INSTANTIATING
