

#include <omp.h>

#include <stdio.h>
#include <stdlib.h>

#include <nvector/nvector_openmp.h>
#include <sundials/sundials_math.h>

#define ZERO   RCONST(0.0)
#define HALF   RCONST(0.5)
#define ONE    RCONST(1.0)
#define ONEPT5 RCONST(1.5)


static void VCopy_OpenMP(N_Vector x, N_Vector z);                              
static void VSum_OpenMP(N_Vector x, N_Vector y, N_Vector z);                   
static void VDiff_OpenMP(N_Vector x, N_Vector y, N_Vector z);                  
static void VNeg_OpenMP(N_Vector x, N_Vector z);                               
static void VScaleSum_OpenMP(realtype c, N_Vector x, N_Vector y, N_Vector z);  
static void VScaleDiff_OpenMP(realtype c, N_Vector x, N_Vector y, N_Vector z); 
static void VLin1_OpenMP(realtype a, N_Vector x, N_Vector y, N_Vector z);      
static void VLin2_OpenMP(realtype a, N_Vector x, N_Vector y, N_Vector z);      
static void Vaxpy_OpenMP(realtype a, N_Vector x, N_Vector y);                  
static void VScaleBy_OpenMP(realtype a, N_Vector x);                           


static int VSumVectorArray_OpenMP(int nvec, N_Vector* X, N_Vector* Y, N_Vector* Z);                   
static int VDiffVectorArray_OpenMP(int nvec, N_Vector* X, N_Vector* Y, N_Vector* Z);                  
static int VScaleSumVectorArray_OpenMP(int nvec, realtype c, N_Vector* X, N_Vector* Y, N_Vector* Z);  
static int VScaleDiffVectorArray_OpenMP(int nvec, realtype c, N_Vector* X, N_Vector* Y, N_Vector* Z); 
static int VLin1VectorArray_OpenMP(int nvec, realtype a, N_Vector* X, N_Vector* Y, N_Vector* Z);      
static int VLin2VectorArray_OpenMP(int nvec, realtype a, N_Vector* X, N_Vector* Y, N_Vector* Z);      
static int VaxpyVectorArray_OpenMP(int nvec, realtype a, N_Vector* X, N_Vector* Y);                   




N_Vector_ID N_VGetVectorID_OpenMP(N_Vector v)
{
return SUNDIALS_NVEC_OPENMP;
}



N_Vector N_VNewEmpty_OpenMP(sunindextype length, int num_threads, SUNContext sunctx)
{
N_Vector v;
N_VectorContent_OpenMP content;


v = NULL;
v = N_VNewEmpty(sunctx);
if (v == NULL) return(NULL);




v->ops->nvgetvectorid     = N_VGetVectorID_OpenMP;
v->ops->nvclone           = N_VClone_OpenMP;
v->ops->nvcloneempty      = N_VCloneEmpty_OpenMP;
v->ops->nvdestroy         = N_VDestroy_OpenMP;
v->ops->nvspace           = N_VSpace_OpenMP;
v->ops->nvgetarraypointer = N_VGetArrayPointer_OpenMP;
v->ops->nvsetarraypointer = N_VSetArrayPointer_OpenMP;
v->ops->nvgetlength       = N_VGetLength_OpenMP;


v->ops->nvlinearsum    = N_VLinearSum_OpenMP;
v->ops->nvconst        = N_VConst_OpenMP;
v->ops->nvprod         = N_VProd_OpenMP;
v->ops->nvdiv          = N_VDiv_OpenMP;
v->ops->nvscale        = N_VScale_OpenMP;
v->ops->nvabs          = N_VAbs_OpenMP;
v->ops->nvinv          = N_VInv_OpenMP;
v->ops->nvaddconst     = N_VAddConst_OpenMP;
v->ops->nvdotprod      = N_VDotProd_OpenMP;
v->ops->nvmaxnorm      = N_VMaxNorm_OpenMP;
v->ops->nvwrmsnormmask = N_VWrmsNormMask_OpenMP;
v->ops->nvwrmsnorm     = N_VWrmsNorm_OpenMP;
v->ops->nvmin          = N_VMin_OpenMP;
v->ops->nvwl2norm      = N_VWL2Norm_OpenMP;
v->ops->nvl1norm       = N_VL1Norm_OpenMP;
v->ops->nvcompare      = N_VCompare_OpenMP;
v->ops->nvinvtest      = N_VInvTest_OpenMP;
v->ops->nvconstrmask   = N_VConstrMask_OpenMP;
v->ops->nvminquotient  = N_VMinQuotient_OpenMP;




v->ops->nvdotprodlocal     = N_VDotProd_OpenMP;
v->ops->nvmaxnormlocal     = N_VMaxNorm_OpenMP;
v->ops->nvminlocal         = N_VMin_OpenMP;
v->ops->nvl1normlocal      = N_VL1Norm_OpenMP;
v->ops->nvinvtestlocal     = N_VInvTest_OpenMP;
v->ops->nvconstrmasklocal  = N_VConstrMask_OpenMP;
v->ops->nvminquotientlocal = N_VMinQuotient_OpenMP;
v->ops->nvwsqrsumlocal     = N_VWSqrSumLocal_OpenMP;
v->ops->nvwsqrsummasklocal = N_VWSqrSumMaskLocal_OpenMP;


v->ops->nvdotprodmultilocal = N_VDotProdMulti_OpenMP;


v->ops->nvbufsize   = N_VBufSize_OpenMP;
v->ops->nvbufpack   = N_VBufPack_OpenMP;
v->ops->nvbufunpack = N_VBufUnpack_OpenMP;


content = NULL;
content = (N_VectorContent_OpenMP) malloc(sizeof *content);
if (content == NULL) { N_VDestroy(v); return(NULL); }


v->content = content;


content->length      = length;
content->num_threads = num_threads;
content->own_data    = SUNFALSE;
content->data        = NULL;

return(v);
}



N_Vector N_VNew_OpenMP(sunindextype length, int num_threads, SUNContext sunctx)
{
N_Vector v;
realtype *data;

v = NULL;
v = N_VNewEmpty_OpenMP(length, num_threads, sunctx);
if (v == NULL) return(NULL);


if (length > 0) {


data = NULL;
data = (realtype *) malloc(length * sizeof(realtype));
if(data == NULL) { N_VDestroy_OpenMP(v); return(NULL); }


NV_OWN_DATA_OMP(v) = SUNTRUE;
NV_DATA_OMP(v)     = data;

}

return(v);
}



N_Vector N_VMake_OpenMP(sunindextype length, realtype *v_data, int num_threads, SUNContext sunctx)
{
N_Vector v;

v = NULL;
v = N_VNewEmpty_OpenMP(length, num_threads, sunctx);
if (v == NULL) return(NULL);

if (length > 0) {

NV_OWN_DATA_OMP(v) = SUNFALSE;
NV_DATA_OMP(v)     = v_data;
}

return(v);
}



N_Vector* N_VCloneVectorArray_OpenMP(int count, N_Vector w)
{
return(N_VCloneVectorArray(count, w));
}



N_Vector* N_VCloneVectorArrayEmpty_OpenMP(int count, N_Vector w)
{
return(N_VCloneEmptyVectorArray(count, w));
}



void N_VDestroyVectorArray_OpenMP(N_Vector* vs, int count)
{
N_VDestroyVectorArray(vs, count);
return;
}


sunindextype N_VGetLength_OpenMP(N_Vector v)
{
return NV_LENGTH_OMP(v);
}



void N_VPrint_OpenMP(N_Vector x)
{
N_VPrintFile_OpenMP(x, stdout);
}



void N_VPrintFile_OpenMP(N_Vector x, FILE *outfile)
{
sunindextype i, N;
realtype *xd;

xd = NULL;

N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);

for (i = 0; i < N; i++) {
#if defined(SUNDIALS_EXTENDED_PRECISION)
STAN_SUNDIALS_FPRINTF(outfile, "%11.8Lg\n", xd[i]);
#elif defined(SUNDIALS_DOUBLE_PRECISION)
STAN_SUNDIALS_FPRINTF(outfile, "%11.8g\n", xd[i]);
#else
STAN_SUNDIALS_FPRINTF(outfile, "%11.8g\n", xd[i]);
#endif
}
STAN_SUNDIALS_FPRINTF(outfile, "\n");

return;
}





N_Vector N_VCloneEmpty_OpenMP(N_Vector w)
{
N_Vector v;
N_VectorContent_OpenMP content;

if (w == NULL) return(NULL);


v = NULL;
v = N_VNewEmpty(w->sunctx);
if (v == NULL) return(NULL);


if (N_VCopyOps(w, v)) { N_VDestroy(v); return(NULL); }


content = NULL;
content = (N_VectorContent_OpenMP) malloc(sizeof *content);
if (content == NULL) { N_VDestroy(v); return(NULL); }


v->content = content;


content->length      = NV_LENGTH_OMP(w);
content->num_threads = NV_NUM_THREADS_OMP(w);
content->own_data    = SUNFALSE;
content->data        = NULL;

return(v);
}




N_Vector N_VClone_OpenMP(N_Vector w)
{
N_Vector v;
realtype *data;
sunindextype length;

v = NULL;
v = N_VCloneEmpty_OpenMP(w);
if (v == NULL) return(NULL);

length = NV_LENGTH_OMP(w);


if (length > 0) {


data = NULL;
data = (realtype *) malloc(length * sizeof(realtype));
if(data == NULL) { N_VDestroy_OpenMP(v); return(NULL); }


NV_OWN_DATA_OMP(v) = SUNTRUE;
NV_DATA_OMP(v)     = data;

}

return(v);
}




void N_VDestroy_OpenMP(N_Vector v)
{
if (v == NULL) return;


if (v->content != NULL) {

if (NV_OWN_DATA_OMP(v) && NV_DATA_OMP(v) != NULL) {
free(NV_DATA_OMP(v));
NV_DATA_OMP(v) = NULL;
}
free(v->content);
v->content = NULL;
}


if (v->ops != NULL) { free(v->ops); v->ops = NULL; }
free(v); v = NULL;

return;
}




void N_VSpace_OpenMP(N_Vector v, sunindextype *lrw, sunindextype *liw)
{
*lrw = NV_LENGTH_OMP(v);
*liw = 1;

return;
}




realtype *N_VGetArrayPointer_OpenMP(N_Vector v)
{
return((realtype *) NV_DATA_OMP(v));
}




void N_VSetArrayPointer_OpenMP(realtype *v_data, N_Vector v)
{
if (NV_LENGTH_OMP(v) > 0) NV_DATA_OMP(v) = v_data;

return;
}




void N_VLinearSum_OpenMP(realtype a, N_Vector x, realtype b, N_Vector y, N_Vector z)
{
sunindextype i, N;
realtype c, *xd, *yd, *zd;
N_Vector v1, v2;
booleantype test;

i  = 0; 
xd = yd = zd = NULL;

if ((b == ONE) && (z == y)) {    
Vaxpy_OpenMP(a,x,y);
return;
}

if ((a == ONE) && (z == x)) {    
Vaxpy_OpenMP(b,y,x);
return;
}



if ((a == ONE) && (b == ONE)) {
VSum_OpenMP(x, y, z);
return;
}



if ((test = ((a == ONE) && (b == -ONE))) || ((a == -ONE) && (b == ONE))) {
v1 = test ? y : x;
v2 = test ? x : y;
VDiff_OpenMP(v2, v1, z);
return;
}




if ((test = (a == ONE)) || (b == ONE)) {
c  = test ? b : a;
v1 = test ? y : x;
v2 = test ? x : y;
VLin1_OpenMP(c, v1, v2, z);
return;
}



if ((test = (a == -ONE)) || (b == -ONE)) {
c = test ? b : a;
v1 = test ? y : x;
v2 = test ? x : y;
VLin2_OpenMP(c, v1, v2, z);
return;
}




if (a == b) {
VScaleSum_OpenMP(a, x, y, z);
return;
}



if (a == -b) {
VScaleDiff_OpenMP(a, x, y, z);
return;
}



N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);
yd = NV_DATA_OMP(y);
zd = NV_DATA_OMP(z);

#pragma omp parallel for default(none) private(i) shared(N,a,b,xd,yd,zd) schedule(static) \
num_threads(NV_NUM_THREADS_OMP(x))
for (i = 0; i < N; i++)
zd[i] = (a*xd[i])+(b*yd[i]);

return;
}




void N_VConst_OpenMP(realtype c, N_Vector z)
{
sunindextype i, N;
realtype *zd;

i  = 0; 
zd = NULL;

N  = NV_LENGTH_OMP(z);
zd = NV_DATA_OMP(z);

#pragma omp parallel for default(none) private(i) shared(N,c,zd) schedule(static) \
num_threads(NV_NUM_THREADS_OMP(z))
for (i = 0; i < N; i++) zd[i] = c;

return;
}




void N_VProd_OpenMP(N_Vector x, N_Vector y, N_Vector z)
{
sunindextype i, N;
realtype *xd, *yd, *zd;

i  = 0; 
xd = yd = zd = NULL;

N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);
yd = NV_DATA_OMP(y);
zd = NV_DATA_OMP(z);

#pragma omp parallel for default(none) private(i) shared(N,xd,yd,zd) schedule(static) \
num_threads(NV_NUM_THREADS_OMP(x))
for (i = 0; i < N; i++)
zd[i] = xd[i]*yd[i];

return;
}




void N_VDiv_OpenMP(N_Vector x, N_Vector y, N_Vector z)
{
sunindextype i, N;
realtype *xd, *yd, *zd;

i  = 0; 
xd = yd = zd = NULL;

N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);
yd = NV_DATA_OMP(y);
zd = NV_DATA_OMP(z);

#pragma omp parallel for default(none) private(i) shared(N,xd,yd,zd) schedule(static) \
num_threads(NV_NUM_THREADS_OMP(x))
for (i = 0; i < N; i++)
zd[i] = xd[i]/yd[i];

return;
}




void N_VScale_OpenMP(realtype c, N_Vector x, N_Vector z)
{
sunindextype i, N;
realtype *xd, *zd;

i  = 0; 
xd = zd = NULL;

if (z == x) {  
VScaleBy_OpenMP(c, x);
return;
}

if (c == ONE) {
VCopy_OpenMP(x, z);
} else if (c == -ONE) {
VNeg_OpenMP(x, z);
} else {
N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);
zd = NV_DATA_OMP(z);

#pragma omp parallel for default(none) private(i) shared(N,c,xd,zd) schedule(static) \
num_threads(NV_NUM_THREADS_OMP(x))
for (i = 0; i < N; i++)
zd[i] = c*xd[i];
}

return;
}




void N_VAbs_OpenMP(N_Vector x, N_Vector z)
{
sunindextype i, N;
realtype *xd, *zd;

i  = 0; 
xd = zd = NULL;

N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);
zd = NV_DATA_OMP(z);

#pragma omp parallel for schedule(static) num_threads(NV_NUM_THREADS_OMP(x))
for (i = 0; i < N; i++)
zd[i] = SUNRabs(xd[i]);

return;
}




void N_VInv_OpenMP(N_Vector x, N_Vector z)
{
sunindextype i, N;
realtype *xd, *zd;

i  = 0; 
xd = zd = NULL;

N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);
zd = NV_DATA_OMP(z);

#pragma omp parallel for default(none) private(i) shared(N,xd,zd) schedule(static) \
num_threads(NV_NUM_THREADS_OMP(x))
for (i = 0; i < N; i++)
zd[i] = ONE/xd[i];

return;
}




void N_VAddConst_OpenMP(N_Vector x, realtype b, N_Vector z)
{
sunindextype i, N;
realtype *xd, *zd;

i  = 0; 
xd = zd = NULL;

N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);
zd = NV_DATA_OMP(z);

#pragma omp parallel for default(none) private(i) shared(N,b,xd,zd) schedule(static) \
num_threads(NV_NUM_THREADS_OMP(x))
for (i = 0; i < N; i++)
zd[i] = xd[i]+b;

return;
}




realtype N_VDotProd_OpenMP(N_Vector x, N_Vector y)
{
sunindextype i, N;
realtype sum, *xd, *yd;

i   = 0; 
sum = ZERO;
xd  = yd = NULL;

N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);
yd = NV_DATA_OMP(y);

#pragma omp parallel for default(none) private(i) shared(N,xd,yd) \
reduction(+:sum) schedule(static) num_threads(NV_NUM_THREADS_OMP(x))
for (i = 0; i < N; i++) {
sum += xd[i]*yd[i];
}

return(sum);
}




realtype N_VMaxNorm_OpenMP(N_Vector x)
{
sunindextype i, N;
realtype tmax, max, *xd;

i   = 0; 
max = ZERO;
xd  = NULL;

N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);

#pragma omp parallel default(none) private(i,tmax) shared(N,max,xd) \
num_threads(NV_NUM_THREADS_OMP(x))
{
tmax = ZERO;
#pragma omp for schedule(static)
for (i = 0; i < N; i++) {
if (SUNRabs(xd[i]) > tmax) tmax = SUNRabs(xd[i]);
}
#pragma omp critical
{
if (tmax > max)
max = tmax;
}
}
return(max);
}




realtype N_VWrmsNorm_OpenMP(N_Vector x, N_Vector w)
{
return(SUNRsqrt(N_VWSqrSumLocal_OpenMP(x, w)/(NV_LENGTH_OMP(x))));
}




realtype N_VWrmsNormMask_OpenMP(N_Vector x, N_Vector w, N_Vector id)
{
return(SUNRsqrt(N_VWSqrSumMaskLocal_OpenMP(x, w, id)/(NV_LENGTH_OMP(x))));
}




realtype N_VMin_OpenMP(N_Vector x)
{
sunindextype i, N;
realtype min, *xd;
realtype tmin;

i  = 0; 
xd = NULL;

N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);

min = xd[0];

#pragma omp parallel default(none) private(i,tmin) shared(N,min,xd) \
num_threads(NV_NUM_THREADS_OMP(x))
{
tmin = xd[0];
#pragma omp for schedule(static)
for (i = 1; i < N; i++) {
if (xd[i] < tmin) tmin = xd[i];
}
if (tmin < min) {
#pragma omp critical
{
if (tmin < min) min = tmin;
}
}
}

return(min);
}




realtype N_VWL2Norm_OpenMP(N_Vector x, N_Vector w)
{
sunindextype i, N;
realtype sum, *xd, *wd;

i   = 0; 
sum = ZERO;
xd  = wd = NULL;

N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);
wd = NV_DATA_OMP(w);

#pragma omp parallel for default(none) private(i) shared(N,xd,wd) \
reduction(+:sum) schedule(static) num_threads(NV_NUM_THREADS_OMP(x))
for (i = 0; i < N; i++) {
sum += SUNSQR(xd[i]*wd[i]);
}

return(SUNRsqrt(sum));
}




realtype N_VL1Norm_OpenMP(N_Vector x)
{
sunindextype i, N;
realtype sum, *xd;

i   = 0; 
sum = ZERO;
xd  = NULL;

N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);

#pragma omp parallel for default(none) private(i) shared(N,xd) \
reduction(+:sum) schedule(static) num_threads(NV_NUM_THREADS_OMP(x))
for (i = 0; i<N; i++)
sum += SUNRabs(xd[i]);

return(sum);
}




void N_VCompare_OpenMP(realtype c, N_Vector x, N_Vector z)
{
sunindextype i, N;
realtype *xd, *zd;

i  = 0; 
xd = zd = NULL;

N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);
zd = NV_DATA_OMP(z);

#pragma omp parallel for default(none) private(i) shared(N,c,xd,zd) schedule(static) \
num_threads(NV_NUM_THREADS_OMP(x))
for (i = 0; i < N; i++) {
zd[i] = (SUNRabs(xd[i]) >= c) ? ONE : ZERO;
}

return;
}




booleantype N_VInvTest_OpenMP(N_Vector x, N_Vector z)
{
sunindextype i, N;
realtype *xd, *zd, val;

i  = 0; 
xd = zd = NULL;

N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);
zd = NV_DATA_OMP(z);

val = ZERO;

#pragma omp parallel for default(none) private(i) shared(N,val,xd,zd) schedule(static) \
num_threads(NV_NUM_THREADS_OMP(x))
for (i = 0; i < N; i++) {
if (xd[i] == ZERO)
val = ONE;
else
zd[i] = ONE/xd[i];
}

if (val > ZERO)
return (SUNFALSE);
else
return (SUNTRUE);
}




booleantype N_VConstrMask_OpenMP(N_Vector c, N_Vector x, N_Vector m)
{
sunindextype i, N;
realtype temp;
realtype *cd, *xd, *md;
booleantype test;

i  = 0; 
cd = xd = md = NULL;

N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);
cd = NV_DATA_OMP(c);
md = NV_DATA_OMP(m);

temp = ZERO;

#pragma omp parallel for default(none) private(i,test) shared(N,xd,cd,md,temp) \
schedule(static) num_threads(NV_NUM_THREADS_OMP(x))
for (i = 0; i < N; i++) {
md[i] = ZERO;


if (cd[i] == ZERO)
continue;


test = (SUNRabs(cd[i]) > ONEPT5 && xd[i]*cd[i] <= ZERO) ||
(SUNRabs(cd[i]) > HALF   && xd[i]*cd[i] <  ZERO);
if (test) {
temp = md[i] = ONE; 
}
}

return (temp == ONE) ? SUNFALSE : SUNTRUE;
}




realtype N_VMinQuotient_OpenMP(N_Vector num, N_Vector denom)
{
sunindextype i, N;
realtype *nd, *dd, min, tmin, val;

i  = 0; 
nd = dd = NULL;

N  = NV_LENGTH_OMP(num);
nd = NV_DATA_OMP(num);
dd = NV_DATA_OMP(denom);

min = BIG_REAL;

#pragma omp parallel default(none) private(i,tmin,val) shared(N,min,nd,dd) \
num_threads(NV_NUM_THREADS_OMP(num))
{
tmin = BIG_REAL;
#pragma omp for schedule(static)
for (i = 0; i < N; i++) {
if (dd[i] != ZERO) {
val = nd[i]/dd[i];
if (val < tmin) tmin = val;
}
}
if (tmin < min) {
#pragma omp critical
{
if (tmin < min) min = tmin;
}
}
}

return(min);
}




realtype N_VWSqrSumLocal_OpenMP(N_Vector x, N_Vector w)
{
sunindextype i, N;
realtype sum, *xd, *wd;

i   = 0; 
sum = ZERO;
xd  = wd = NULL;

N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);
wd = NV_DATA_OMP(w);

#pragma omp parallel for default(none) private(i) shared(N,xd,wd) \
reduction(+:sum) schedule(static) num_threads(NV_NUM_THREADS_OMP(x))
for (i = 0; i < N; i++) {
sum += SUNSQR(xd[i]*wd[i]);
}

return(sum);
}




realtype N_VWSqrSumMaskLocal_OpenMP(N_Vector x, N_Vector w, N_Vector id)
{
sunindextype i, N;
realtype sum, *xd, *wd, *idd;

i   = 0; 
sum = ZERO;
xd  = wd = idd = NULL;

N   = NV_LENGTH_OMP(x);
xd  = NV_DATA_OMP(x);
wd  = NV_DATA_OMP(w);
idd = NV_DATA_OMP(id);

#pragma omp parallel for default(none) private(i) shared(N,xd,wd,idd) \
reduction(+:sum) schedule(static) num_threads(NV_NUM_THREADS_OMP(x))
for (i = 0; i < N; i++) {
if (idd[i] > ZERO) {
sum += SUNSQR(xd[i]*wd[i]);
}
}

return(sum);
}




int N_VLinearCombination_OpenMP(int nvec, realtype* c, N_Vector* X, N_Vector z)
{
int          i;
sunindextype j, N;
realtype*    zd=NULL;
realtype*    xd=NULL;

i = 0; 
j = 0;


if (nvec < 1) return(-1);


if (nvec == 1) {
N_VScale_OpenMP(c[0], X[0], z);
return(0);
}


if (nvec == 2) {
N_VLinearSum_OpenMP(c[0], X[0], c[1], X[1], z);
return(0);
}


N  = NV_LENGTH_OMP(z);
zd = NV_DATA_OMP(z);



if ((X[0] == z) && (c[0] == ONE)) {
#pragma omp parallel default(none) private(i,j,xd) shared(nvec,X,N,c,zd) \
num_threads(NV_NUM_THREADS_OMP(z))
{
for (i=1; i<nvec; i++) {
xd = NV_DATA_OMP(X[i]);
#pragma omp for schedule(static)
for (j=0; j<N; j++) {
zd[j] += c[i] * xd[j];
}
}
}
return(0);
}


if (X[0] == z) {
#pragma omp parallel default(none) private(i,j,xd) shared(nvec,X,N,c,zd) \
num_threads(NV_NUM_THREADS_OMP(z))
{
#pragma omp for schedule(static)
for (j=0; j<N; j++) {
zd[j] *= c[0];
}

for (i=1; i<nvec; i++) {
xd = NV_DATA_OMP(X[i]);
#pragma omp for schedule(static)
for (j=0; j<N; j++) {
zd[j] += c[i] * xd[j];
}
}
}
return(0);
}


#pragma omp parallel default(none) private(i,j,xd) shared(nvec,X,N,c,zd) \
num_threads(NV_NUM_THREADS_OMP(z))
{
xd = NV_DATA_OMP(X[0]);
#pragma omp for schedule(static)
for (j=0; j<N; j++) {
zd[j] = c[0] * xd[j];
}

for (i=1; i<nvec; i++) {
xd = NV_DATA_OMP(X[i]);
#pragma omp for schedule(static)
for (j=0; j<N; j++) {
zd[j] += c[i] * xd[j];
}
}
}
return(0);
}


int N_VScaleAddMulti_OpenMP(int nvec, realtype* a, N_Vector x, N_Vector* Y, N_Vector* Z)
{
int          i;
sunindextype j, N;
realtype*    xd=NULL;
realtype*    yd=NULL;
realtype*    zd=NULL;

i = 0; 
j = 0;


if (nvec < 1) return(-1);


if (nvec == 1) {
N_VLinearSum_OpenMP(a[0], x, ONE, Y[0], Z[0]);
return(0);
}


N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);


if (Y == Z) {
#pragma omp parallel default(none) private(i,j,yd) shared(nvec,Y,N,a,xd) \
num_threads(NV_NUM_THREADS_OMP(x))
{
for (i=0; i<nvec; i++) {
yd = NV_DATA_OMP(Y[i]);
#pragma omp for schedule(static)
for (j=0; j<N; j++) {
yd[j] += a[i] * xd[j];
}
}
}
return(0);
}


#pragma omp parallel default(none) private(i,j,yd,zd) shared(nvec,Y,Z,N,a,xd) \
num_threads(NV_NUM_THREADS_OMP(x))
{
for (i=0; i<nvec; i++) {
yd = NV_DATA_OMP(Y[i]);
zd = NV_DATA_OMP(Z[i]);
#pragma omp for schedule(static)
for (j=0; j<N; j++) {
zd[j] = a[i] * xd[j] + yd[j];
}
}
}
return(0);
}


int N_VDotProdMulti_OpenMP(int nvec, N_Vector x, N_Vector* Y, realtype* dotprods)
{
int          i;
sunindextype j, N;
realtype     sum;
realtype*    xd=NULL;
realtype*    yd=NULL;

i = 0; 
j = 0;


if (nvec < 1) return(-1);


if (nvec == 1) {
dotprods[0] = N_VDotProd_OpenMP(x, Y[0]);
return(0);
}


N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);


for (i=0; i<nvec; i++) {
dotprods[i] = ZERO;
}


#pragma omp parallel default(none) private(i,j,yd,sum) shared(nvec,Y,N,xd,dotprods) \
num_threads(NV_NUM_THREADS_OMP(x))
{
for (i=0; i<nvec; i++) {
yd = NV_DATA_OMP(Y[i]);
sum = ZERO;
#pragma omp for schedule(static)
for (j=0; j<N; j++) {
sum += xd[j] * yd[j];
}
#pragma omp critical
{
dotprods[i] += sum;
}
}
}

return(0);
}




int N_VLinearSumVectorArray_OpenMP(int nvec,
realtype a, N_Vector* X,
realtype b, N_Vector* Y,
N_Vector* Z)
{
int          i;
sunindextype j, N;
realtype*    xd=NULL;
realtype*    yd=NULL;
realtype*    zd=NULL;
realtype     c;
N_Vector*   V1;
N_Vector*   V2;
booleantype  test;

i = 0; 
j = 0;


if (nvec < 1) return(-1);


if (nvec == 1) {
N_VLinearSum_OpenMP(a, X[0], b, Y[0], Z[0]);
return(0);
}


if ((b == ONE) && (Z == Y))
return(VaxpyVectorArray_OpenMP(nvec, a, X, Y));


if ((a == ONE) && (Z == X))
return(VaxpyVectorArray_OpenMP(nvec, b, Y, X));


if ((a == ONE) && (b == ONE))
return(VSumVectorArray_OpenMP(nvec, X, Y, Z));




if ((test = ((a == ONE) && (b == -ONE))) || ((a == -ONE) && (b == ONE))) {
V1 = test ? Y : X;
V2 = test ? X : Y;
return(VDiffVectorArray_OpenMP(nvec, V2, V1, Z));
}





if ((test = (a == ONE)) || (b == ONE)) {
c  = test ? b : a;
V1 = test ? Y : X;
V2 = test ? X : Y;
return(VLin1VectorArray_OpenMP(nvec, c, V1, V2, Z));
}




if ((test = (a == -ONE)) || (b == -ONE)) {
c  = test ? b : a;
V1 = test ? Y : X;
V2 = test ? X : Y;
return(VLin2VectorArray_OpenMP(nvec, c, V1, V2, Z));
}



if (a == b)
return(VScaleSumVectorArray_OpenMP(nvec, a, X, Y, Z));


if (a == -b)
return(VScaleDiffVectorArray_OpenMP(nvec, a, X, Y, Z));







N = NV_LENGTH_OMP(Z[0]);


#pragma omp parallel default(none) private(i,j,xd,yd,zd) shared(nvec,X,Y,Z,N,a,b) \
num_threads(NV_NUM_THREADS_OMP(Z[0]))
{
for (i=0; i<nvec; i++) {
xd = NV_DATA_OMP(X[i]);
yd = NV_DATA_OMP(Y[i]);
zd = NV_DATA_OMP(Z[i]);
#pragma omp for schedule(static)
for (j=0; j<N; j++) {
zd[j] = a * xd[j] + b * yd[j];
}
}
}

return(0);
}


int N_VScaleVectorArray_OpenMP(int nvec, realtype* c, N_Vector* X, N_Vector* Z)
{
int          i;
sunindextype j, N;
realtype*    xd=NULL;
realtype*    zd=NULL;

i = 0; 
j = 0;


if (nvec < 1) return(-1);


if (nvec == 1) {
N_VScale_OpenMP(c[0], X[0], Z[0]);
return(0);
}


N = NV_LENGTH_OMP(Z[0]);


if (X == Z) {
#pragma omp parallel default(none) private(i,j,xd) shared(nvec,X,N,c) \
num_threads(NV_NUM_THREADS_OMP(Z[0]))
{
for (i=0; i<nvec; i++) {
xd = NV_DATA_OMP(X[i]);
#pragma omp for schedule(static)
for (j=0; j<N; j++) {
xd[j] *= c[i];
}
}
}
return(0);
}


#pragma omp parallel default(none) private(i,j,xd,zd) shared(nvec,X,Z,N,c) \
num_threads(NV_NUM_THREADS_OMP(Z[0]))
{
for (i=0; i<nvec; i++) {
xd = NV_DATA_OMP(X[i]);
zd = NV_DATA_OMP(Z[i]);
#pragma omp for schedule(static)
for (j=0; j<N; j++) {
zd[j] = c[i] * xd[j];
}
}
}
return(0);
}


int N_VConstVectorArray_OpenMP(int nvec, realtype c, N_Vector* Z)
{
int          i;
sunindextype j, N;
realtype*    zd=NULL;

i = 0; 
j = 0;


if (nvec < 1) return(-1);


if (nvec == 1) {
N_VConst_OpenMP(c, Z[0]);
return(0);
}


N = NV_LENGTH_OMP(Z[0]);


#pragma omp parallel default(none) private(i,j,zd) shared(nvec,Z,N,c) \
num_threads(NV_NUM_THREADS_OMP(Z[0]))
{
for (i=0; i<nvec; i++) {
zd = NV_DATA_OMP(Z[i]);
#pragma omp for schedule(static)
for (j=0; j<N; j++) {
zd[j] = c;
}
}
}

return(0);
}


int N_VWrmsNormVectorArray_OpenMP(int nvec, N_Vector* X, N_Vector* W, realtype* nrm)
{
int          i;
sunindextype j, N;
realtype     sum;
realtype*    wd=NULL;
realtype*    xd=NULL;

i = 0; 
j = 0;


if (nvec < 1) return(-1);


if (nvec == 1) {
nrm[0] = N_VWrmsNorm_OpenMP(X[0], W[0]);
return(0);
}


N  = NV_LENGTH_OMP(X[0]);


for (i=0; i<nvec; i++) {
nrm[i] = ZERO;
}


#pragma omp parallel default(none) private(i,j,xd,wd,sum) shared(nvec,X,W,N,nrm) \
num_threads(NV_NUM_THREADS_OMP(X[0]))
{
for (i=0; i<nvec; i++) {
xd = NV_DATA_OMP(X[i]);
wd = NV_DATA_OMP(W[i]);
sum = ZERO;
#pragma omp for schedule(static)
for (j=0; j<N; j++) {
sum += SUNSQR(xd[j] * wd[j]);
}
#pragma omp critical
{
nrm[i] += sum;
}
}
}

for (i=0; i<nvec; i++) {
nrm[i] = SUNRsqrt(nrm[i]/N);
}

return(0);
}


int N_VWrmsNormMaskVectorArray_OpenMP(int nvec, N_Vector* X, N_Vector* W,
N_Vector id, realtype* nrm)
{
int          i;
sunindextype j, N;
realtype     sum;
realtype*    wd=NULL;
realtype*    xd=NULL;
realtype*    idd=NULL;

i = 0; 
j = 0;


if (nvec < 1) return(-1);


if (nvec == 1) {
nrm[0] = N_VWrmsNormMask_OpenMP(X[0], W[0], id);
return(0);
}


N   = NV_LENGTH_OMP(X[0]);
idd = NV_DATA_OMP(id);


for (i=0; i<nvec; i++) {
nrm[i] = ZERO;
}


#pragma omp parallel default(none) private(i,j,xd,wd,sum) shared(nvec,X,W,N,idd,nrm) \
num_threads(NV_NUM_THREADS_OMP(X[0]))
{
for (i=0; i<nvec; i++) {
xd = NV_DATA_OMP(X[i]);
wd = NV_DATA_OMP(W[i]);
sum = ZERO;
#pragma omp for schedule(static)
for (j=0; j<N; j++) {
if (idd[j] > ZERO)
sum += SUNSQR(xd[j] * wd[j]);
}
#pragma omp critical
{
nrm[i] += sum;
}
}
}

for (i=0; i<nvec; i++) {
nrm[i] = SUNRsqrt(nrm[i]/N);
}

return(0);
}


int N_VScaleAddMultiVectorArray_OpenMP(int nvec, int nsum, realtype* a,
N_Vector* X, N_Vector** Y, N_Vector** Z)
{
int          i, j;
sunindextype k, N;
realtype*    xd=NULL;
realtype*    yd=NULL;
realtype*    zd=NULL;

int          retval;
N_Vector*   YY;
N_Vector*   ZZ;

i = 0; 
k = 0;


if (nvec < 1) return(-1);
if (nsum < 1) return(-1);



if (nvec == 1) {


if (nsum == 1) {
N_VLinearSum_OpenMP(a[0], X[0], ONE, Y[0][0], Z[0][0]);
return(0);
}


YY = (N_Vector*) malloc(nsum * sizeof(N_Vector));
ZZ = (N_Vector*) malloc(nsum * sizeof(N_Vector));

for (j=0; j<nsum; j++) {
YY[j] = Y[j][0];
ZZ[j] = Z[j][0];
}

retval = N_VScaleAddMulti_OpenMP(nsum, a, X[0], YY, ZZ);

free(YY);
free(ZZ);
return(retval);
}




if (nsum == 1) {
retval = N_VLinearSumVectorArray_OpenMP(nvec, a[0], X, ONE, Y[0], Z[0]);
return(retval);
}




N  = NV_LENGTH_OMP(X[0]);


if (Y == Z) {
#pragma omp parallel default(none) private(i,j,k,xd,yd) shared(nvec,nsum,X,Y,N,a) \
num_threads(NV_NUM_THREADS_OMP(X[0]))
{
for (i=0; i<nvec; i++) {
xd = NV_DATA_OMP(X[i]);
for (j=0; j<nsum; j++) {
yd = NV_DATA_OMP(Y[j][i]);
#pragma omp for schedule(static)
for (k=0; k<N; k++) {
yd[k] += a[j] * xd[k];
}
}
}
}
return(0);
}


#pragma omp parallel default(none) private(i,j,k,xd,yd,zd) shared(nvec,nsum,X,Y,Z,N,a) \
num_threads(NV_NUM_THREADS_OMP(X[0]))
{
for (i=0; i<nvec; i++) {
xd = NV_DATA_OMP(X[i]);
for (j=0; j<nsum; j++) {
yd = NV_DATA_OMP(Y[j][i]);
zd = NV_DATA_OMP(Z[j][i]);
#pragma omp for schedule(static)
for (k=0; k<N; k++) {
zd[k] = a[j] * xd[k] + yd[k];
}
}
}
}
return(0);
}


int N_VLinearCombinationVectorArray_OpenMP(int nvec, int nsum,
realtype* c,
N_Vector** X,
N_Vector* Z)
{
int          i; 
int          j; 
sunindextype k; 
sunindextype N;
realtype*    zd=NULL;
realtype*    xd=NULL;

realtype*    ctmp;
N_Vector*   Y;

i = 0; 
k = 0;


if (nvec < 1) return(-1);
if (nsum < 1) return(-1);



if (nvec == 1) {


if (nsum == 1) {
N_VScale_OpenMP(c[0], X[0][0], Z[0]);
return(0);
}


if (nsum == 2) {
N_VLinearSum_OpenMP(c[0], X[0][0], c[1], X[1][0], Z[0]);
return(0);
}


Y = (N_Vector*) malloc(nsum * sizeof(N_Vector));

for (i=0; i<nsum; i++) {
Y[i] = X[i][0];
}

N_VLinearCombination_OpenMP(nsum, c, Y, Z[0]);

free(Y);
return(0);
}




if (nsum == 1) {

ctmp = (realtype*) malloc(nvec * sizeof(realtype));

for (j=0; j<nvec; j++) {
ctmp[j] = c[0];
}

N_VScaleVectorArray_OpenMP(nvec, ctmp, X[0], Z);

free(ctmp);
return(0);
}


if (nsum == 2) {
N_VLinearSumVectorArray_OpenMP(nvec, c[0], X[0], c[1], X[1], Z);
return(0);
}




N = NV_LENGTH_OMP(Z[0]);


if ((X[0] == Z) && (c[0] == ONE)) {
#pragma omp parallel default(none) private(i,j,k,xd,zd) shared(nvec,nsum,X,Z,N,c) \
num_threads(NV_NUM_THREADS_OMP(Z[0]))
{
for (j=0; j<nvec; j++) {
zd = NV_DATA_OMP(Z[j]);
for (i=1; i<nsum; i++) {
xd = NV_DATA_OMP(X[i][j]);
#pragma omp for schedule(static)
for (k=0; k<N; k++) {
zd[k] += c[i] * xd[k];
}
}
}
}
return(0);
}


if (X[0] == Z) {
#pragma omp parallel default(none) private(i,j,k,xd,zd) shared(nvec,nsum,X,Z,N,c) \
num_threads(NV_NUM_THREADS_OMP(Z[0]))
{
for (j=0; j<nvec; j++) {
zd = NV_DATA_OMP(Z[j]);
#pragma omp for schedule(static)
for (k=0; k<N; k++) {
zd[k] *= c[0];
}
for (i=1; i<nsum; i++) {
xd = NV_DATA_OMP(X[i][j]);
#pragma omp for schedule(static)
for (k=0; k<N; k++) {
zd[k] += c[i] * xd[k];
}
}
}
}
return(0);
}


#pragma omp parallel default(none) private(i,j,k,xd,zd) shared(nvec,nsum,X,Z,N,c) \
num_threads(NV_NUM_THREADS_OMP(Z[0]))
{
for (j=0; j<nvec; j++) {

xd = NV_DATA_OMP(X[0][j]);
zd = NV_DATA_OMP(Z[j]);
#pragma omp for schedule(static)
for (k=0; k<N; k++) {
zd[k] = c[0] * xd[k];
}

for (i=1; i<nsum; i++) {
xd = NV_DATA_OMP(X[i][j]);
#pragma omp for schedule(static)
for (k=0; k<N; k++) {
zd[k] += c[i] * xd[k];
}
}
}
}
return(0);
}





int N_VBufSize_OpenMP(N_Vector x, sunindextype *size)
{
if (x == NULL) return(-1);
*size = NV_LENGTH_OMP(x) * ((sunindextype)sizeof(realtype));
return(0);
}


int N_VBufPack_OpenMP(N_Vector x, void *buf)
{
sunindextype i, N;
realtype     *xd = NULL;
realtype     *bd = NULL;

if (x == NULL || buf == NULL) return(-1);

N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);
bd = (realtype*) buf;

#pragma omp for schedule(static)
for (i = 0; i < N; i++)
bd[i] = xd[i];

return(0);
}


int N_VBufUnpack_OpenMP(N_Vector x, void *buf)
{
sunindextype i, N;
realtype     *xd = NULL;
realtype     *bd = NULL;

if (x == NULL || buf == NULL) return(-1);

N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);
bd = (realtype*) buf;

#pragma omp for schedule(static)
for (i = 0; i < N; i++)
xd[i] = bd[i];

return(0);
}







static void VCopy_OpenMP(N_Vector x, N_Vector z)
{
sunindextype i, N;
realtype *xd, *zd;

i  = 0; 
xd = zd = NULL;

N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);
zd = NV_DATA_OMP(z);

#pragma omp parallel for default(none) private(i) shared(N,xd,zd) schedule(static) \
num_threads(NV_NUM_THREADS_OMP(x))
for (i = 0; i < N; i++)
zd[i] = xd[i];

return;
}




static void VSum_OpenMP(N_Vector x, N_Vector y, N_Vector z)
{
sunindextype i, N;
realtype *xd, *yd, *zd;

i  = 0; 
xd = yd = zd = NULL;

N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);
yd = NV_DATA_OMP(y);
zd = NV_DATA_OMP(z);

#pragma omp parallel for default(none) private(i) shared(N,xd,yd,zd) schedule(static) \
num_threads(NV_NUM_THREADS_OMP(x))
for (i = 0; i < N; i++)
zd[i] = xd[i]+yd[i];

return;
}




static void VDiff_OpenMP(N_Vector x, N_Vector y, N_Vector z)
{
sunindextype i, N;
realtype *xd, *yd, *zd;

i  = 0; 
xd = yd = zd = NULL;

N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);
yd = NV_DATA_OMP(y);
zd = NV_DATA_OMP(z);

#pragma omp parallel for default(none) private(i) shared(N,xd,yd,zd) schedule(static) \
num_threads(NV_NUM_THREADS_OMP(x))
for (i = 0; i < N; i++)
zd[i] = xd[i]-yd[i];

return;
}




static void VNeg_OpenMP(N_Vector x, N_Vector z)
{
sunindextype i, N;
realtype *xd, *zd;

i  = 0; 
xd = zd = NULL;

N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);
zd = NV_DATA_OMP(z);

#pragma omp parallel for default(none) private(i) shared(N,xd,zd) schedule(static) \
num_threads(NV_NUM_THREADS_OMP(x))
for (i = 0; i < N; i++)
zd[i] = -xd[i];

return;
}




static void VScaleSum_OpenMP(realtype c, N_Vector x, N_Vector y, N_Vector z)
{
sunindextype i, N;
realtype *xd, *yd, *zd;

i  = 0; 
xd = yd = zd = NULL;

N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);
yd = NV_DATA_OMP(y);
zd = NV_DATA_OMP(z);

#pragma omp parallel for default(none) private(i) shared(N,c,xd,yd,zd) schedule(static) \
num_threads(NV_NUM_THREADS_OMP(x))
for (i = 0; i < N; i++)
zd[i] = c*(xd[i]+yd[i]);

return;
}




static void VScaleDiff_OpenMP(realtype c, N_Vector x, N_Vector y, N_Vector z)
{
sunindextype i, N;
realtype *xd, *yd, *zd;

i  = 0; 
xd = yd = zd = NULL;

N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);
yd = NV_DATA_OMP(y);
zd = NV_DATA_OMP(z);

#pragma omp parallel for default(none) private(i) shared(N,c,xd,yd,zd) schedule(static) \
num_threads(NV_NUM_THREADS_OMP(x))
for (i = 0; i < N; i++)
zd[i] = c*(xd[i]-yd[i]);

return;
}




static void VLin1_OpenMP(realtype a, N_Vector x, N_Vector y, N_Vector z)
{
sunindextype i, N;
realtype *xd, *yd, *zd;

i  = 0; 
xd = yd = zd = NULL;

N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);
yd = NV_DATA_OMP(y);
zd = NV_DATA_OMP(z);

#pragma omp parallel for default(none) private(i) shared(N,a,xd,yd,zd) schedule(static) \
num_threads(NV_NUM_THREADS_OMP(x))
for (i = 0; i < N; i++)
zd[i] = (a*xd[i])+yd[i];

return;
}




static void VLin2_OpenMP(realtype a, N_Vector x, N_Vector y, N_Vector z)
{
sunindextype i, N;
realtype *xd, *yd, *zd;

i  = 0; 
xd = yd = zd = NULL;

N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);
yd = NV_DATA_OMP(y);
zd = NV_DATA_OMP(z);

#pragma omp parallel for default(none) private(i) shared(N,a,xd,yd,zd) schedule(static) \
num_threads(NV_NUM_THREADS_OMP(x))
for (i = 0; i < N; i++)
zd[i] = (a*xd[i])-yd[i];

return;
}




static void Vaxpy_OpenMP(realtype a, N_Vector x, N_Vector y)
{
sunindextype i, N;
realtype *xd, *yd;

i  = 0; 
xd = yd = NULL;

N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);
yd = NV_DATA_OMP(y);

if (a == ONE) {
#pragma omp parallel for default(none) private(i) shared(N,xd,yd) schedule(static) \
num_threads(NV_NUM_THREADS_OMP(x))
for (i = 0; i < N; i++)
yd[i] += xd[i];
return;
}

if (a == -ONE) {
#pragma omp parallel for default(none) private(i) shared(N,xd,yd) schedule(static) \
num_threads(NV_NUM_THREADS_OMP(x))
for (i = 0; i < N; i++)
yd[i] -= xd[i];
return;
}

#pragma omp parallel for default(none) private(i) shared(N,a,xd,yd) schedule(static) \
num_threads(NV_NUM_THREADS_OMP(x))
for (i = 0; i < N; i++)
yd[i] += a*xd[i];

return;
}




static void VScaleBy_OpenMP(realtype a, N_Vector x)
{
sunindextype i, N;
realtype *xd;

i  = 0; 
xd = NULL;

N  = NV_LENGTH_OMP(x);
xd = NV_DATA_OMP(x);

#pragma omp parallel for default(none) private(i) shared(N,a,xd) schedule(static) \
num_threads(NV_NUM_THREADS_OMP(x))
for (i = 0; i < N; i++)
xd[i] *= a;

return;
}




static int VSumVectorArray_OpenMP(int nvec, N_Vector* X, N_Vector* Y, N_Vector* Z)
{
int          i;
sunindextype j, N;
realtype*    xd=NULL;
realtype*    yd=NULL;
realtype*    zd=NULL;

i = 0; 
j = 0;

N = NV_LENGTH_OMP(X[0]);

#pragma omp parallel default(none) private(i,j,xd,yd,zd) shared(nvec,X,Y,Z,N) \
num_threads(NV_NUM_THREADS_OMP(X[0]))
{
for (i=0; i<nvec; i++) {
xd = NV_DATA_OMP(X[i]);
yd = NV_DATA_OMP(Y[i]);
zd = NV_DATA_OMP(Z[i]);
#pragma omp for schedule(static)
for (j=0; j<N; j++)
zd[j] = xd[j] + yd[j];
}
}

return(0);
}

static int VDiffVectorArray_OpenMP(int nvec, N_Vector* X, N_Vector* Y, N_Vector* Z)
{
int          i;
sunindextype j, N;
realtype*    xd=NULL;
realtype*    yd=NULL;
realtype*    zd=NULL;

i = 0; 
j = 0;

N = NV_LENGTH_OMP(X[0]);

#pragma omp parallel default(none) private(i,j,xd,yd,zd) shared(nvec,X,Y,Z,N) \
num_threads(NV_NUM_THREADS_OMP(X[0]))
{
for (i=0; i<nvec; i++) {
xd = NV_DATA_OMP(X[i]);
yd = NV_DATA_OMP(Y[i]);
zd = NV_DATA_OMP(Z[i]);
#pragma omp for schedule(static)
for (j=0; j<N; j++)
zd[j] = xd[j] - yd[j];
}
}

return(0);
}

static int VScaleSumVectorArray_OpenMP(int nvec, realtype c, N_Vector* X, N_Vector* Y, N_Vector* Z)
{
int          i;
sunindextype j, N;
realtype*    xd=NULL;
realtype*    yd=NULL;
realtype*    zd=NULL;

i = 0; 
j = 0;

N = NV_LENGTH_OMP(X[0]);

#pragma omp parallel default(none) private(i,j,xd,yd,zd) shared(nvec,X,Y,Z,N,c) \
num_threads(NV_NUM_THREADS_OMP(X[0]))
{
for (i=0; i<nvec; i++) {
xd = NV_DATA_OMP(X[i]);
yd = NV_DATA_OMP(Y[i]);
zd = NV_DATA_OMP(Z[i]);
#pragma omp for schedule(static)
for (j=0; j<N; j++)
zd[j] = c * (xd[j] + yd[j]);
}
}

return(0);
}

static int VScaleDiffVectorArray_OpenMP(int nvec, realtype c, N_Vector* X, N_Vector* Y, N_Vector* Z)
{
int          i;
sunindextype j, N;
realtype*    xd=NULL;
realtype*    yd=NULL;
realtype*    zd=NULL;

i = 0; 
j = 0;

N = NV_LENGTH_OMP(X[0]);

#pragma omp parallel default(none) private(i,j,xd,yd,zd) shared(nvec,X,Y,Z,N,c) \
num_threads(NV_NUM_THREADS_OMP(X[0]))
{
for (i=0; i<nvec; i++) {
xd = NV_DATA_OMP(X[i]);
yd = NV_DATA_OMP(Y[i]);
zd = NV_DATA_OMP(Z[i]);
#pragma omp for schedule(static)
for (j=0; j<N; j++)
zd[j] = c * (xd[j] - yd[j]);
}
}

return(0);
}

static int VLin1VectorArray_OpenMP(int nvec, realtype a, N_Vector* X, N_Vector* Y, N_Vector* Z)
{
int          i;
sunindextype j, N;
realtype*    xd=NULL;
realtype*    yd=NULL;
realtype*    zd=NULL;

i = 0; 
j = 0;

N = NV_LENGTH_OMP(X[0]);

#pragma omp parallel default(none) private(i,j,xd,yd,zd) shared(nvec,X,Y,Z,N,a) \
num_threads(NV_NUM_THREADS_OMP(X[0]))
{
for (i=0; i<nvec; i++) {
xd = NV_DATA_OMP(X[i]);
yd = NV_DATA_OMP(Y[i]);
zd = NV_DATA_OMP(Z[i]);
#pragma omp for schedule(static)
for (j=0; j<N; j++)
zd[j] = (a * xd[j]) + yd[j];
}
}

return(0);
}

static int VLin2VectorArray_OpenMP(int nvec, realtype a, N_Vector* X, N_Vector* Y, N_Vector* Z)
{
int          i;
sunindextype j, N;
realtype*    xd=NULL;
realtype*    yd=NULL;
realtype*    zd=NULL;

i = 0; 
j = 0;

N = NV_LENGTH_OMP(X[0]);

#pragma omp parallel default(none) private(i,j,xd,yd,zd) shared(nvec,X,Y,Z,N,a) \
num_threads(NV_NUM_THREADS_OMP(X[0]))
{
for (i=0; i<nvec; i++) {
xd = NV_DATA_OMP(X[i]);
yd = NV_DATA_OMP(Y[i]);
zd = NV_DATA_OMP(Z[i]);
#pragma omp for schedule(static)
for (j=0; j<N; j++)
zd[j] = (a * xd[j]) - yd[j];
}
}

return(0);
}

static int VaxpyVectorArray_OpenMP(int nvec, realtype a, N_Vector* X, N_Vector* Y)
{
int          i;
sunindextype j, N;
realtype*    xd=NULL;
realtype*    yd=NULL;

i = 0; 
j = 0;

N = NV_LENGTH_OMP(X[0]);

if (a == ONE) {
#pragma omp parallel default(none) private(i,j,xd,yd) shared(nvec,X,Y,N,a) \
num_threads(NV_NUM_THREADS_OMP(X[0]))
{
for (i=0; i<nvec; i++) {
xd = NV_DATA_OMP(X[i]);
yd = NV_DATA_OMP(Y[i]);
#pragma omp for schedule(static)
for (j=0; j<N; j++)
yd[j] += xd[j];
}
}
return(0);
}

if (a == -ONE) {
#pragma omp parallel default(none) private(i,j,xd,yd) shared(nvec,X,Y,N,a) \
num_threads(NV_NUM_THREADS_OMP(X[0]))
{
for (i=0; i<nvec; i++) {
xd = NV_DATA_OMP(X[i]);
yd = NV_DATA_OMP(Y[i]);
#pragma omp for schedule(static)
for (j=0; j<N; j++)
yd[j] -= xd[j];
}
}
return(0);
}

#pragma omp parallel default(none) private(i,j,xd,yd) shared(nvec,X,Y,N,a) \
num_threads(NV_NUM_THREADS_OMP(X[0]))
{
for (i=0; i<nvec; i++) {
xd = NV_DATA_OMP(X[i]);
yd = NV_DATA_OMP(Y[i]);
#pragma omp for schedule(static)
for (j=0; j<N; j++)
yd[j] += a * xd[j];
}
}
return(0);
}




int N_VEnableFusedOps_OpenMP(N_Vector v, booleantype tf)
{

if (v == NULL) return(-1);


if (v->ops == NULL) return(-1);

if (tf) {

v->ops->nvlinearcombination = N_VLinearCombination_OpenMP;
v->ops->nvscaleaddmulti     = N_VScaleAddMulti_OpenMP;
v->ops->nvdotprodmulti      = N_VDotProdMulti_OpenMP;

v->ops->nvlinearsumvectorarray         = N_VLinearSumVectorArray_OpenMP;
v->ops->nvscalevectorarray             = N_VScaleVectorArray_OpenMP;
v->ops->nvconstvectorarray             = N_VConstVectorArray_OpenMP;
v->ops->nvwrmsnormvectorarray          = N_VWrmsNormVectorArray_OpenMP;
v->ops->nvwrmsnormmaskvectorarray      = N_VWrmsNormMaskVectorArray_OpenMP;
v->ops->nvscaleaddmultivectorarray     = N_VScaleAddMultiVectorArray_OpenMP;
v->ops->nvlinearcombinationvectorarray = N_VLinearCombinationVectorArray_OpenMP;

v->ops->nvdotprodmultilocal = N_VDotProdMulti_OpenMP;
} else {

v->ops->nvlinearcombination = NULL;
v->ops->nvscaleaddmulti     = NULL;
v->ops->nvdotprodmulti      = NULL;

v->ops->nvlinearsumvectorarray         = NULL;
v->ops->nvscalevectorarray             = NULL;
v->ops->nvconstvectorarray             = NULL;
v->ops->nvwrmsnormvectorarray          = NULL;
v->ops->nvwrmsnormmaskvectorarray      = NULL;
v->ops->nvscaleaddmultivectorarray     = NULL;
v->ops->nvlinearcombinationvectorarray = NULL;

v->ops->nvdotprodmultilocal = NULL;
}


return(0);
}


int N_VEnableLinearCombination_OpenMP(N_Vector v, booleantype tf)
{

if (v == NULL) return(-1);


if (v->ops == NULL) return(-1);


if (tf)
v->ops->nvlinearcombination = N_VLinearCombination_OpenMP;
else
v->ops->nvlinearcombination = NULL;


return(0);
}

int N_VEnableScaleAddMulti_OpenMP(N_Vector v, booleantype tf)
{

if (v == NULL) return(-1);


if (v->ops == NULL) return(-1);


if (tf)
v->ops->nvscaleaddmulti = N_VScaleAddMulti_OpenMP;
else
v->ops->nvscaleaddmulti = NULL;


return(0);
}

int N_VEnableDotProdMulti_OpenMP(N_Vector v, booleantype tf)
{

if (v == NULL) return(-1);


if (v->ops == NULL) return(-1);


if (tf) {
v->ops->nvdotprodmulti      = N_VDotProdMulti_OpenMP;
v->ops->nvdotprodmultilocal = N_VDotProdMulti_OpenMP;
} else {
v->ops->nvdotprodmulti      = NULL;
v->ops->nvdotprodmultilocal = NULL;
}


return(0);
}

int N_VEnableLinearSumVectorArray_OpenMP(N_Vector v, booleantype tf)
{

if (v == NULL) return(-1);


if (v->ops == NULL) return(-1);


if (tf)
v->ops->nvlinearsumvectorarray = N_VLinearSumVectorArray_OpenMP;
else
v->ops->nvlinearsumvectorarray = NULL;


return(0);
}

int N_VEnableScaleVectorArray_OpenMP(N_Vector v, booleantype tf)
{

if (v == NULL) return(-1);


if (v->ops == NULL) return(-1);


if (tf)
v->ops->nvscalevectorarray = N_VScaleVectorArray_OpenMP;
else
v->ops->nvscalevectorarray = NULL;


return(0);
}

int N_VEnableConstVectorArray_OpenMP(N_Vector v, booleantype tf)
{

if (v == NULL) return(-1);


if (v->ops == NULL) return(-1);


if (tf)
v->ops->nvconstvectorarray = N_VConstVectorArray_OpenMP;
else
v->ops->nvconstvectorarray = NULL;


return(0);
}

int N_VEnableWrmsNormVectorArray_OpenMP(N_Vector v, booleantype tf)
{

if (v == NULL) return(-1);


if (v->ops == NULL) return(-1);


if (tf)
v->ops->nvwrmsnormvectorarray = N_VWrmsNormVectorArray_OpenMP;
else
v->ops->nvwrmsnormvectorarray = NULL;


return(0);
}

int N_VEnableWrmsNormMaskVectorArray_OpenMP(N_Vector v, booleantype tf)
{

if (v == NULL) return(-1);


if (v->ops == NULL) return(-1);


if (tf)
v->ops->nvwrmsnormmaskvectorarray = N_VWrmsNormMaskVectorArray_OpenMP;
else
v->ops->nvwrmsnormmaskvectorarray = NULL;


return(0);
}

int N_VEnableScaleAddMultiVectorArray_OpenMP(N_Vector v, booleantype tf)
{

if (v == NULL) return(-1);


if (v->ops == NULL) return(-1);


if (tf)
v->ops->nvscaleaddmultivectorarray = N_VScaleAddMultiVectorArray_OpenMP;
else
v->ops->nvscaleaddmultivectorarray = NULL;


return(0);
}

int N_VEnableLinearCombinationVectorArray_OpenMP(N_Vector v, booleantype tf)
{

if (v == NULL) return(-1);


if (v->ops == NULL) return(-1);


if (tf)
v->ops->nvlinearcombinationvectorarray = N_VLinearCombinationVectorArray_OpenMP;
else
v->ops->nvlinearcombinationvectorarray = NULL;


return(0);
}
