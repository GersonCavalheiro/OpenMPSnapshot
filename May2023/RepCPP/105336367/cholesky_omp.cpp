
#include <math.h>
#include <stdio.h>
#include <ctype.h>
#include <omp.h>

#include "adf.h"


#define MIN(a,b) (((a)<=(b)) ? a : b)
#define MAX(a,b) (((a)>=(b)) ? a : b)

typedef float real;
typedef float doublereal;
typedef long integer;
typedef bool logical;


float  ***A;
float    *Alin;
long      MatrixSize;


long      BlockSize;
long      NumBlocks;
int       num_threads = 1;




long dlarnv_ (long *idist, long *iseed, long *n, float *x);
int  sgemm_  (char *transa, char *transb, integer *m, integer *n, integer *k, real *alpha,
real *a, integer *lda, real *b, integer *ldb, real *beta, real *c__, integer *ldc);
int  strsm_  (char *side, char *uplo, char *transa, char *diag, integer *m, integer *n,
real *alpha, real *a, integer *lda, real *b, integer *ldb);
int  ssyrk_  (char *uplo, char *trans, integer *n, integer *k, real *alpha, real *a,
integer *lda, real *beta, real *c__, integer *ldc);
int  spotf2_ (char *uplo, integer *n, real *a, integer *lda, integer *info);




void spotrf_tile(float *A,long BSize,long j)
{
long INFO;
char LO='L';

spotf2_(&LO,
&BSize,
A, &BSize,
&INFO);

}

void sgemm_tile(float  *A, float *B, float *C, long BSize, long i, long j)
{
char TR='T', NT='N';
float DONE=1.0, DMONE=-1.0;

sgemm_(&NT, &TR,                  
&BSize, &BSize, &BSize,   
&DMONE,                   
A, &BSize,                
B, &BSize,                
&DONE,                    
C, &BSize);               
}

void strsm_tile(float *T, float *B, long BSize, long i, long j)
{
char LO='L', TR='T', NU='N', RI='R';
float DONE=1.0;

strsm_ (&RI, &LO, &TR, &NU,       
&BSize, &BSize,           
&DONE,                    
T, &BSize,                
B, &BSize);               
}

void ssyrk_tile( float *A, float *C, long BSize, long j)
{
char LO='L', NT='N';
float DONE=1.0, DMONE=-1.0;

ssyrk_ (&LO, &NT,                 
&BSize, &BSize,           
&DMONE,                   
A, &BSize,                
&DONE,                    
C, &BSize);               
}






void PrintMatrix(long size, float *A) {
printf("\n");
for(long i = 0; i < size; i++) {
for(long j = 0; j < size; j++) {
printf("%g ", A[i * size + j]);
}
printf("\n");
}
printf("\n\n");
}




void InitMatrix()
{
long ISEED[4] = {0,0,0,1};
long IONE=1;

MatrixSize = BlockSize * NumBlocks;
long NN = MatrixSize * MatrixSize;

Alin = (float *) malloc(NN * sizeof(float));

dlarnv_(&IONE, ISEED, &NN, Alin);

for(long i = 0; i < MatrixSize; i++) {
for(long j = i+1; j < MatrixSize; j++) {
Alin[i * MatrixSize + j] = Alin[j * MatrixSize + i];
}
}

for(long i = 0; i < MatrixSize; i++) {
Alin[i * MatrixSize + i] += MatrixSize;
}

A = (float***) malloc(NumBlocks * sizeof(float **));
for (long i = 0; i < NumBlocks; i++)
A[i] = (float**) malloc(NumBlocks * sizeof(float *));

for (long i = 0; i < NumBlocks; i++) {
for (long j = 0; j < NumBlocks; j++) {
A[i][j] = (float *) malloc(BlockSize * BlockSize * sizeof(float));
}
}

for (long i = 0; i < MatrixSize; i++) {
for (long j = 0; j < MatrixSize; j++) {
A[i/BlockSize][j/BlockSize][(i%BlockSize) * BlockSize + j%BlockSize] = Alin[i*MatrixSize + j];
}
}
}


void FreeMatrix() {
for (long i = 0; i < NumBlocks; i++) {
for (long j = 0; j < NumBlocks; j++) {
free(A[i][j]);
}
}

for (long i = 0; i < NumBlocks; i++)
free(A[i]);

free(A);
free(Alin);
}



void Solve()
{
for (long j = 0; j < NumBlocks; j++) {

#pragma omp parallel
{
#pragma omp single
{
for (long k= 0; k< j; k++) {

for (long i = j+1; i < NumBlocks; i++)
{
#pragma omp task firstprivate(i, j, k)
{
sgemm_tile( A[i][k], A[j][k], A[i][j], BlockSize,i,j);
}
}

}
}
}

#pragma omp parallel for firstprivate(j)
for (long i = 0; i < j; i++) {
ssyrk_tile( A[j][i], A[j][j], BlockSize,j);
}

spotrf_tile( A[j][j], BlockSize,j);


#pragma omp parallel for firstprivate(j)
for (long i = j+1; i < NumBlocks; i++) {
strsm_tile( A[j][j], A[i][j], BlockSize,i,j);
}

}
}




void ParseCommandLine(int argc, char **argv)
{
char c;

while ((c = getopt (argc, argv, "hd:b:n:")) != -1)
switch (c) {
case 'h':
printf("\nCholesky Decomposition - ADF benchmark application\n"
"\n"
"Usage:\n"
"   cholseky_omp [options ...]\n"
"\n"
"Options:\n"
"   -h\n"
"        Print this help message.\n"
"   -d <long>\n"
"        number of blocks in one dimension. (default 10)\n"
"   -b <long>\n"
"        Block size. (default 100)\n"
"   -n <int>\n"
"        Number of ADF worker threads. (default 1)\n"
"\n"
);
exit (0);
case 'd':
NumBlocks = atol(optarg);
break;
case 'b':
BlockSize = atol(optarg);
break;
case 'n':
num_threads = atoi(optarg);
break;
case '?':
if (optopt == 'c')
fprintf (stderr, "Option -%c requires an argument.\n", optopt);
else if (isprint (optopt))
fprintf (stderr, "Unknown option `-%c'.\n", optopt);
else
fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
exit(1);
default:
exit(1);
}

if (BlockSize < 10) BlockSize = 10;
if (NumBlocks < 1) NumBlocks = 1;
if (num_threads < 1) num_threads = 1;

printf ("\nCholesky Decomposition\nMatrix size is %ldx%ld, devided into %ldx%ld blocks of size %ld.\n",
NumBlocks*BlockSize, NumBlocks*BlockSize, NumBlocks, NumBlocks, BlockSize);
printf ("Running with %d threads.\n", num_threads);
printf ("=====================================================\n\n");
}




int
main(int argc, char *argv[])
{
ParseCommandLine(argc, argv);

InitMatrix();

NonADF_init(num_threads);

omp_set_num_threads(num_threads);
omp_set_nested(1);

Solve();

NonADF_terminate();

FreeMatrix();

return 0;
}























int dlaruv_(long *iseed, long *n, float *x)
{


static long mm[512]	 = { 494,2637,255,2008,1253,
3344,4084,1739,3143,3468,688,1657,1238,3166,1292,3422,1270,2016,
154,2862,697,1706,491,931,1444,444,3577,3944,2184,1661,3482,657,
3023,3618,1267,1828,164,3798,3087,2400,2870,3876,1905,1593,1797,
1234,3460,328,2861,1950,617,2070,3331,769,1558,2412,2800,189,287,
2045,1227,2838,209,2770,3654,3993,192,2253,3491,2889,2857,2094,
1818,688,1407,634,3231,815,3524,1914,516,164,303,2144,3480,119,
3357,837,2826,2332,2089,3780,1700,3712,150,2000,3375,1621,3090,
3765,1149,3146,33,3082,2741,359,3316,1749,185,2784,2202,2199,1364,
1244,2020,3160,2785,2772,1217,1822,1245,2252,3904,2774,997,2573,
1148,545,322,789,1440,752,2859,123,1848,643,2405,2638,2344,46,
3814,913,3649,339,3808,822,2832,3078,3633,2970,637,2249,2081,4019,
1478,242,481,2075,4058,622,3376,812,234,641,4005,1122,3135,2640,
2302,40,1832,2247,2034,2637,1287,1691,496,1597,2394,2584,1843,336,
1472,2407,433,2096,1761,2810,566,442,41,1238,1086,603,840,3168,
1499,1084,3438,2408,1589,2391,288,26,512,1456,171,1677,2657,2270,
2587,2961,1970,1817,676,1410,3723,2803,3185,184,663,499,3784,1631,
1925,3912,1398,1349,1441,2224,2411,1907,3192,2786,382,37,759,2948,
1862,3802,2423,2051,2295,1332,1832,2405,3638,3661,327,3660,716,
1842,3987,1368,1848,2366,2508,3754,1766,3572,2893,307,1297,3966,
758,2598,3406,2922,1038,2934,2091,2451,1580,1958,2055,1507,1078,
3273,17,854,2916,3971,2889,3831,2621,1541,893,736,3992,787,2125,
2364,2460,257,1574,3912,1216,3248,3401,2124,2762,149,2245,166,466,
4018,1399,190,2879,153,2320,18,712,2159,2318,2091,3443,1510,449,
1956,2201,3137,3399,1321,2271,3667,2703,629,2365,2431,1113,3922,
2554,184,2099,3228,4012,1921,3452,3901,572,3309,3171,817,3039,
1696,1256,3715,2077,3019,1497,1101,717,51,981,1978,1813,3881,76,
3846,3694,1682,124,1660,3997,479,1141,886,3514,1301,3604,1888,
1836,1990,2058,692,1194,20,3285,2046,2107,3508,3525,3801,2549,
1145,2253,305,3301,1065,3133,2913,3285,1241,1197,3729,2501,1673,
541,2753,949,2361,1165,4081,2725,3305,3069,3617,3733,409,2157,
1361,3973,1865,2525,1409,3445,3577,77,3761,2149,1449,3005,225,85,
3673,3117,3089,1349,2057,413,65,1845,697,3085,3441,1573,3689,2941,
929,533,2841,4077,721,2821,2249,2397,2817,245,1913,1997,3121,997,
1833,2877,1633,981,2009,941,2449,197,2441,285,1473,2741,3129,909,
2801,421,4073,2813,2337,1429,1177,1901,81,1669,2633,2269,129,1141,
249,3917,2481,3941,2217,2749,3041,1877,345,2861,1809,3141,2825,
157,2881,3637,1465,2829,2161,3365,361,2685,3745,2325,3609,3821,
3537,517,3017,2141,1537 };

long jota;
static long ind, i1, i2, i3, i4, it1, it2, it3, it4;


--iseed;
--x;



i1 = iseed[1];
i2 = iseed[2];
i3 = iseed[3];
i4 = iseed[4];

jota = MIN(*n,128);
for (ind = 1; ind <= jota; ++ind) {

L20:



it4 = i4 * mm[ind + 383];
it3 = it4 / 4096;
it4 -= it3 << 12;
it3 = it3 + i3 * mm[ind + 383] + i4 * mm[ind + 255];
it2 = it3 / 4096;
it3 -= it2 << 12;
it2 = it2 + i2 * mm[ind + 383] + i3 * mm[ind + 255] + i4 * mm[ind +
127];
it1 = it2 / 4096;
it2 -= it1 << 12;
it1 = it1 + i1 * mm[ind + 383] + i2 * mm[ind + 255] + i3 * mm[ind +
127] + i4 * mm[ind - 1];
it1 %= 4096;



x[ind] = ((float) it1 + ((float) it2 + ((float) it3 + (
float) it4 * 2.44140625e-4) * 2.44140625e-4) *
2.44140625e-4) * 2.44140625e-4;

if (x[ind] == 1.) {

i1 += 2;
i2 += 2;
i3 += 2;
i4 += 2;
goto L20;
}
}



iseed[1] = it1;
iseed[2] = it2;
iseed[3] = it3;
iseed[4] = it4;

return 0;
} 



long dlarnv_(long *idist, long *iseed, long *n, float *x)
{

long alpha, beta, gama;
static long val;
static float u[128];
static long il, iv, il2;


--x;
--iseed;


alpha = *n;
for (iv = 1; iv <= alpha; iv += 64) {

beta = 64, gama = *n - iv + 1;
il = MIN(beta,gama);
if (*idist == 3) {
il2 = il << 1;
} else {
il2 = il;
}


dlaruv_(&iseed[1], &il2, u);

if (*idist == 1) {

beta = il;
for (val = 1; val <= beta; ++val) {
x[iv + val - 1] = u[val - 1];
}
} else if (*idist == 2) {

beta = il;
for (val = 1; val <= beta; ++val) {
x[iv + val - 1] = u[val - 1] * 2. - 1.;
}
} else if (*idist == 3) {

beta = il;
for (val = 1; val <= beta; ++val) {
x[iv + val - 1] = sqrt(log(u[(val << 1) - 2]) * -2.) *
cos(u[(val << 1) - 1] * 6.2831853071795864769252867663);
}
}
}

return 0;
} 









logical lsame_(char *a, char*b) {
return (toupper(*a) == toupper(*b));
}

void xerbla_(char *funcName, integer *val) {
printf("Error in function %s\tvalue = %d", funcName, (int) *val);
}



int ssyrk_(char *uplo, char *trans, integer *n, integer *k, real *alpha, real *a, integer *lda, real *beta, real *c__, integer *ldc)
{

integer a_dim1, a_offset, c_dim1, c_offset, i__1, i__2, i__3;

static integer i__, j, l, info;
static real temp;
static integer nrowa;
static logical upper;


a_dim1 = *lda;
a_offset = 1 + a_dim1;
a -= a_offset;
c_dim1 = *ldc;
c_offset = 1 + c_dim1;
c__ -= c_offset;


if (lsame_(trans, (char *) "N")) {
nrowa = *n;
} else {
nrowa = *k;
}
upper = lsame_(uplo, (char *) "U");
info = 0;
if (! upper && ! lsame_(uplo, (char *) "L")) {
info = 1;
} else if (! lsame_(trans, (char *) "N") && ! lsame_(trans, (char *) "T") && ! lsame_(trans, (char *) "C")) {
info = 2;
} else if (*n < 0) {
info = 3;
} else if (*k < 0) {
info = 4;
} else if (*lda < MAX(1,nrowa)) {
info = 7;
} else if (*ldc < MAX(1,*n)) {
info = 10;
}
if (info != 0) {
xerbla_((char *) "SSYRK ", &info);
return 0;
}

if (*n == 0 || ((*alpha == 0.f || *k == 0) && *beta == 1.f)) {
return 0;
}

if (*alpha == 0.f) {
if (upper) {
if (*beta == 0.f) {
i__1 = *n;
for (j = 1; j <= i__1; ++j) {
i__2 = j;
for (i__ = 1; i__ <= i__2; ++i__) {
c__[i__ + j * c_dim1] = 0.f;
}
}
} else {
i__1 = *n;
for (j = 1; j <= i__1; ++j) {
i__2 = j;
for (i__ = 1; i__ <= i__2; ++i__) {
c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
}
}
}
} else {
if (*beta == 0.f) {
i__1 = *n;
for (j = 1; j <= i__1; ++j) {
i__2 = *n;
for (i__ = j; i__ <= i__2; ++i__) {
c__[i__ + j * c_dim1] = 0.f;
}
}
} else {
i__1 = *n;
for (j = 1; j <= i__1; ++j) {
i__2 = *n;
for (i__ = j; i__ <= i__2; ++i__) {
c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
}
}
}
}
return 0;
}

if (lsame_(trans, (char *) "N")) {

if (upper) {
i__1 = *n;
for (j = 1; j <= i__1; ++j) {
if (*beta == 0.f) {
i__2 = j;
for (i__ = 1; i__ <= i__2; ++i__) {
c__[i__ + j * c_dim1] = 0.f;
}
} else if (*beta != 1.f) {
i__2 = j;
for (i__ = 1; i__ <= i__2; ++i__) {
c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
}
}
i__2 = *k;
for (l = 1; l <= i__2; ++l) {
if (a[j + l * a_dim1] != 0.f) {
temp = *alpha * a[j + l * a_dim1];
i__3 = j;
for (i__ = 1; i__ <= i__3; ++i__) {
c__[i__ + j * c_dim1] += temp * a[i__ + l * a_dim1];

}
}

}

}
} else {
i__1 = *n;
for (j = 1; j <= i__1; ++j) {
if (*beta == 0.f) {
i__2 = *n;
for (i__ = j; i__ <= i__2; ++i__) {
c__[i__ + j * c_dim1] = 0.f;

}
} else if (*beta != 1.f) {
i__2 = *n;
for (i__ = j; i__ <= i__2; ++i__) {
c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];

}
}
i__2 = *k;
for (l = 1; l <= i__2; ++l) {
if (a[j + l * a_dim1] != 0.f) {
temp = *alpha * a[j + l * a_dim1];
i__3 = *n;
for (i__ = j; i__ <= i__3; ++i__) {
c__[i__ + j * c_dim1] += temp * a[i__ + l * a_dim1];
}
}
}
}
}
} else {

if (upper) {
i__1 = *n;
for (j = 1; j <= i__1; ++j) {
i__2 = j;
for (i__ = 1; i__ <= i__2; ++i__) {
temp = 0.f;
i__3 = *k;
for (l = 1; l <= i__3; ++l) {
temp += a[l + i__ * a_dim1] * a[l + j * a_dim1];
}
if (*beta == 0.f) {
c__[i__ + j * c_dim1] = *alpha * temp;
} else {
c__[i__ + j * c_dim1] = *alpha * temp + *beta * c__[i__ + j * c_dim1];
}
}
}
} else {
i__1 = *n;
for (j = 1; j <= i__1; ++j) {
i__2 = *n;
for (i__ = j; i__ <= i__2; ++i__) {
temp = 0.f;
i__3 = *k;
for (l = 1; l <= i__3; ++l) {
temp += a[l + i__ * a_dim1] * a[l + j * a_dim1];
}
if (*beta == 0.f) {
c__[i__ + j * c_dim1] = *alpha * temp;
} else {
c__[i__ + j * c_dim1] = *alpha * temp + *beta * c__[i__ + j * c_dim1];
}
}
}
}
}
return 0;
} 









int strsm_(char *side, char *uplo, char *transa, char *diag, integer *m, integer *n, real *alpha, real *a, integer *lda, real *b, integer *ldb)
{

integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2, i__3;

static integer i__, j, k, info;
static real temp;
static logical lside;
static integer nrowa;
static logical upper;
static logical nounit;


a_dim1 = *lda;
a_offset = 1 + a_dim1;
a -= a_offset;
b_dim1 = *ldb;
b_offset = 1 + b_dim1;
b -= b_offset;


lside = lsame_(side, (char *) "L");
if (lside) {
nrowa = *m;
} else {
nrowa = *n;
}
nounit = lsame_(diag, (char *) "N");
upper = lsame_(uplo, (char *) "U");
info = 0;
if (! lside && ! lsame_(side, (char *) "R")) {
info = 1;
} else if (! upper && ! lsame_(uplo, (char *) "L")) {
info = 2;
} else if (! lsame_(transa, (char *) "N") && ! lsame_(transa, (char *) "T") && ! lsame_(transa, (char *) "C")) {
info = 3;
} else if (! lsame_(diag, (char *) "U") && ! lsame_(diag, (char *) "N")) {
info = 4;
} else if (*m < 0) {
info = 5;
} else if (*n < 0) {
info = 6;
} else if (*lda < MAX(1,nrowa)) {
info = 9;
} else if (*ldb < MAX(1,*m)) {
info = 11;
}
if (info != 0) {
xerbla_((char *) "STRSM ", &info);
return 0;
}

if (*n == 0) {
return 0;
}

if (*alpha == 0.f) {
i__1 = *n;
for (j = 1; j <= i__1; ++j) {
i__2 = *m;
for (i__ = 1; i__ <= i__2; ++i__) {
b[i__ + j * b_dim1] = 0.f;
}
}
return 0;
}

if (lside) {
if (lsame_(transa, (char *) "N")) {

if (upper) {
i__1 = *n;
for (j = 1; j <= i__1; ++j) {
if (*alpha != 1.f) {
i__2 = *m;
for (i__ = 1; i__ <= i__2; ++i__) {
b[i__ + j * b_dim1] = *alpha * b[i__ + j * b_dim1];
}
}
for (k = *m; k >= 1; --k) {
if (b[k + j * b_dim1] != 0.f) {
if (nounit) {
b[k + j * b_dim1] /= a[k + k * a_dim1];
}
i__2 = k - 1;
for (i__ = 1; i__ <= i__2; ++i__) {
b[i__ + j * b_dim1] -= b[k + j * b_dim1] * a[i__ + k * a_dim1];
}
}
}
}
} else {
i__1 = *n;
for (j = 1; j <= i__1; ++j) {
if (*alpha != 1.f) {
i__2 = *m;
for (i__ = 1; i__ <= i__2; ++i__) {
b[i__ + j * b_dim1] = *alpha * b[i__ + j * b_dim1];
}
}
i__2 = *m;
for (k = 1; k <= i__2; ++k) {
if (b[k + j * b_dim1] != 0.f) {
if (nounit) {
b[k + j * b_dim1] /= a[k + k * a_dim1];
}
i__3 = *m;
for (i__ = k + 1; i__ <= i__3; ++i__) {
b[i__ + j * b_dim1] -= b[k + j * b_dim1] * a[i__ + k * a_dim1];
}
}
}
}
}
} else {

if (upper) {
i__1 = *n;
for (j = 1; j <= i__1; ++j) {
i__2 = *m;
for (i__ = 1; i__ <= i__2; ++i__) {
temp = *alpha * b[i__ + j * b_dim1];
i__3 = i__ - 1;
for (k = 1; k <= i__3; ++k) {
temp -= a[k + i__ * a_dim1] * b[k + j * b_dim1];
}
if (nounit) {
temp /= a[i__ + i__ * a_dim1];
}
b[i__ + j * b_dim1] = temp;
}
}
} else {
i__1 = *n;
for (j = 1; j <= i__1; ++j) {
for (i__ = *m; i__ >= 1; --i__) {
temp = *alpha * b[i__ + j * b_dim1];
i__2 = *m;
for (k = i__ + 1; k <= i__2; ++k) {
temp -= a[k + i__ * a_dim1] * b[k + j * b_dim1];
}
if (nounit) {
temp /= a[i__ + i__ * a_dim1];
}
b[i__ + j * b_dim1] = temp;
}
}
}
}
} else {
if (lsame_(transa, (char *) "N")) {

if (upper) {
i__1 = *n;
for (j = 1; j <= i__1; ++j) {
if (*alpha != 1.f) {
i__2 = *m;
for (i__ = 1; i__ <= i__2; ++i__) {
b[i__ + j * b_dim1] = *alpha * b[i__ + j * b_dim1];
}
}
i__2 = j - 1;
for (k = 1; k <= i__2; ++k) {
if (a[k + j * a_dim1] != 0.f) {
i__3 = *m;
for (i__ = 1; i__ <= i__3; ++i__) {
b[i__ + j * b_dim1] -= a[k + j * a_dim1] * b[i__ + k * b_dim1];
}
}
}
if (nounit) {
temp = 1.f / a[j + j * a_dim1];
i__2 = *m;
for (i__ = 1; i__ <= i__2; ++i__) {
b[i__ + j * b_dim1] = temp * b[i__ + j * b_dim1];
}
}
}
} else {
for (j = *n; j >= 1; --j) {
if (*alpha != 1.f) {
i__1 = *m;
for (i__ = 1; i__ <= i__1; ++i__) {
b[i__ + j * b_dim1] = *alpha * b[i__ + j * b_dim1];
}
}
i__1 = *n;
for (k = j + 1; k <= i__1; ++k) {
if (a[k + j * a_dim1] != 0.f) {
i__2 = *m;
for (i__ = 1; i__ <= i__2; ++i__) {
b[i__ + j * b_dim1] -= a[k + j * a_dim1] * b[i__ + k * b_dim1];
}
}
}
if (nounit) {
temp = 1.f / a[j + j * a_dim1];
i__1 = *m;
for (i__ = 1; i__ <= i__1; ++i__) {
b[i__ + j * b_dim1] = temp * b[i__ + j * b_dim1];
}
}
}
}
} else {

if (upper) {
for (k = *n; k >= 1; --k) {
if (nounit) {
temp = 1.f / a[k + k * a_dim1];
i__1 = *m;
for (i__ = 1; i__ <= i__1; ++i__) {
b[i__ + k * b_dim1] = temp * b[i__ + k * b_dim1];
}
}
i__1 = k - 1;
for (j = 1; j <= i__1; ++j) {
if (a[j + k * a_dim1] != 0.f) {
temp = a[j + k * a_dim1];
i__2 = *m;
for (i__ = 1; i__ <= i__2; ++i__) {
b[i__ + j * b_dim1] -= temp * b[i__ + k *b_dim1];
}
}
}
if (*alpha != 1.f) {
i__1 = *m;
for (i__ = 1; i__ <= i__1; ++i__) {
b[i__ + k * b_dim1] = *alpha * b[i__ + k * b_dim1];
}
}
}
} else {
i__1 = *n;
for (k = 1; k <= i__1; ++k) {
if (nounit) {
temp = 1.f / a[k + k * a_dim1];
i__2 = *m;
for (i__ = 1; i__ <= i__2; ++i__) {
b[i__ + k * b_dim1] = temp * b[i__ + k * b_dim1];
}
}
i__2 = *n;
for (j = k + 1; j <= i__2; ++j) {
if (a[j + k * a_dim1] != 0.f) {
temp = a[j + k * a_dim1];
i__3 = *m;
for (i__ = 1; i__ <= i__3; ++i__) {
b[i__ + j * b_dim1] -= temp * b[i__ + k *b_dim1];
}
}
}
if (*alpha != 1.f) {
i__2 = *m;
for (i__ = 1; i__ <= i__2; ++i__) {
b[i__ + k * b_dim1] = *alpha * b[i__ + k * b_dim1];
}
}
}
}
}
}

return 0;
} 









int sgemm_(char *transa, char *transb, integer *m, integer *n, integer *k, real *alpha, real *a,
integer *lda, real *b, integer *ldb, real *beta, real *c__, integer *ldc)
{

integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2, i__3;

static integer i__, j, l, info;
static logical nota, notb;
static real temp;
static integer nrowa, nrowb;


a_dim1 = *lda;
a_offset = 1 + a_dim1;
a -= a_offset;
b_dim1 = *ldb;
b_offset = 1 + b_dim1;
b -= b_offset;
c_dim1 = *ldc;
c_offset = 1 + c_dim1;
c__ -= c_offset;


nota = lsame_(transa, (char *) "N");
notb = lsame_(transb, (char *) "N");
if (nota) {
nrowa = *m;
} else {
nrowa = *k;
}
if (notb) {
nrowb = *k;
} else {
nrowb = *n;
}

info = 0;
if (! nota && ! lsame_(transa, (char *) "C") && ! lsame_(transa, (char *) "T")) {
info = 1;
} else if (! notb && ! lsame_(transb, (char *) "C") && ! lsame_(transb, (char *) "T")) {
info = 2;
} else if (*m < 0) {
info = 3;
} else if (*n < 0) {
info = 4;
} else if (*k < 0) {
info = 5;
} else if (*lda < MAX(1,nrowa)) {
info = 8;
} else if (*ldb < MAX(1,nrowb)) {
info = 10;
} else if (*ldc < MAX(1,*m)) {
info = 13;
}
if (info != 0) {
xerbla_((char *) "SGEMM ", &info);
return 0;
}

if (*m == 0 || *n == 0 || ((*alpha == 0.f || *k == 0) && *beta == 1.f)) {
return 0;
}

if (*alpha == 0.f) {
if (*beta == 0.f) {
i__1 = *n;
for (j = 1; j <= i__1; ++j) {
i__2 = *m;
for (i__ = 1; i__ <= i__2; ++i__) {
c__[i__ + j * c_dim1] = 0.f;
}
}
} else {
i__1 = *n;
for (j = 1; j <= i__1; ++j) {
i__2 = *m;
for (i__ = 1; i__ <= i__2; ++i__) {
c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
}
}
}
return 0;
}

if (notb) {
if (nota) {

i__1 = *n;
for (j = 1; j <= i__1; ++j) {
if (*beta == 0.f) {
i__2 = *m;
for (i__ = 1; i__ <= i__2; ++i__) {
c__[i__ + j * c_dim1] = 0.f;
}
} else if (*beta != 1.f) {
i__2 = *m;
for (i__ = 1; i__ <= i__2; ++i__) {
c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
}
}
i__2 = *k;
for (l = 1; l <= i__2; ++l) {
if (b[l + j * b_dim1] != 0.f) {
temp = *alpha * b[l + j * b_dim1];
i__3 = *m;
for (i__ = 1; i__ <= i__3; ++i__) {
c__[i__ + j * c_dim1] += temp * a[i__ + l * a_dim1];
}
}
}
}
} else {

i__1 = *n;
for (j = 1; j <= i__1; ++j) {
i__2 = *m;
for (i__ = 1; i__ <= i__2; ++i__) {
temp = 0.f;
i__3 = *k;
for (l = 1; l <= i__3; ++l) {
temp += a[l + i__ * a_dim1] * b[l + j * b_dim1];
}
if (*beta == 0.f) {
c__[i__ + j * c_dim1] = *alpha * temp;
} else {
c__[i__ + j * c_dim1] = *alpha * temp + *beta * c__[i__ + j * c_dim1];
}
}
}
}
} else {
if (nota) {

i__1 = *n;
for (j = 1; j <= i__1; ++j) {
if (*beta == 0.f) {
i__2 = *m;
for (i__ = 1; i__ <= i__2; ++i__) {
c__[i__ + j * c_dim1] = 0.f;
}
} else if (*beta != 1.f) {
i__2 = *m;
for (i__ = 1; i__ <= i__2; ++i__) {
c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
}
}
i__2 = *k;
for (l = 1; l <= i__2; ++l) {
if (b[j + l * b_dim1] != 0.f) {
temp = *alpha * b[j + l * b_dim1];
i__3 = *m;
for (i__ = 1; i__ <= i__3; ++i__) {
c__[i__ + j * c_dim1] += temp * a[i__ + l * a_dim1];
}
}
}
}
} else {

i__1 = *n;
for (j = 1; j <= i__1; ++j) {
i__2 = *m;
for (i__ = 1; i__ <= i__2; ++i__) {
temp = 0.f;
i__3 = *k;
for (l = 1; l <= i__3; ++l) {
temp += a[l + i__ * a_dim1] * b[j + l * b_dim1];
}
if (*beta == 0.f) {
c__[i__ + j * c_dim1] = *alpha * temp;
} else {
c__[i__ + j * c_dim1] = *alpha * temp + *beta * c__[i__ + j * c_dim1];
}
}
}
}
}

return 0;
} 







int sgemv_(char *trans, integer *m, integer *n, real *alpha, real *a, integer *lda, real *x, integer *incx, real *beta, real *y, integer *incy)
{

integer a_dim1, a_offset, i__1, i__2;

static integer i__, j, ix, iy, jx, jy, kx, ky, info;
static real temp;
static integer lenx, leny;


a_dim1 = *lda;
a_offset = 1 + a_dim1;
a -= a_offset;
--x;
--y;


info = 0;
if (! lsame_(trans, (char *) "N") && ! lsame_(trans, (char *) "T") && ! lsame_(trans, (char *) "C")) {
info = 1;
} else if (*m < 0) {
info = 2;
} else if (*n < 0) {
info = 3;
} else if (*lda < MAX(1,*m)) {
info = 6;
} else if (*incx == 0) {
info = 8;
} else if (*incy == 0) {
info = 11;
}
if (info != 0) {
xerbla_((char *) "SGEMV ", &info);
return 0;
}

if (*m == 0 || *n == 0 || (*alpha == 0.f && *beta == 1.f)) {
return 0;
}

if (lsame_(trans, (char *) "N")) {
lenx = *n;
leny = *m;
} else {
lenx = *m;
leny = *n;
}
if (*incx > 0) {
kx = 1;
} else {
kx = 1 - (lenx - 1) * *incx;
}
if (*incy > 0) {
ky = 1;
} else {
ky = 1 - (leny - 1) * *incy;
}

if (*beta != 1.f) {
if (*incy == 1) {
if (*beta == 0.f) {
i__1 = leny;
for (i__ = 1; i__ <= i__1; ++i__) {
y[i__] = 0.f;
}
} else {
i__1 = leny;
for (i__ = 1; i__ <= i__1; ++i__) {
y[i__] = *beta * y[i__];
}
}
} else {
iy = ky;
if (*beta == 0.f) {
i__1 = leny;
for (i__ = 1; i__ <= i__1; ++i__) {
y[iy] = 0.f;
iy += *incy;
}
} else {
i__1 = leny;
for (i__ = 1; i__ <= i__1; ++i__) {
y[iy] = *beta * y[iy];
iy += *incy;
}
}
}
}
if (*alpha == 0.f) {
return 0;
}
if (lsame_(trans, (char *) "N")) {

jx = kx;
if (*incy == 1) {
i__1 = *n;
for (j = 1; j <= i__1; ++j) {
if (x[jx] != 0.f) {
temp = *alpha * x[jx];
i__2 = *m;
for (i__ = 1; i__ <= i__2; ++i__) {
y[i__] += temp * a[i__ + j * a_dim1];
}
}
jx += *incx;
}
} else {
i__1 = *n;
for (j = 1; j <= i__1; ++j) {
if (x[jx] != 0.f) {
temp = *alpha * x[jx];
iy = ky;
i__2 = *m;
for (i__ = 1; i__ <= i__2; ++i__) {
y[iy] += temp * a[i__ + j * a_dim1];
iy += *incy;
}
}
jx += *incx;
}
}
} else {

jy = ky;
if (*incx == 1) {
i__1 = *n;
for (j = 1; j <= i__1; ++j) {
temp = 0.f;
i__2 = *m;
for (i__ = 1; i__ <= i__2; ++i__) {
temp += a[i__ + j * a_dim1] * x[i__];
}
y[jy] += *alpha * temp;
jy += *incy;
}
} else {
i__1 = *n;
for (j = 1; j <= i__1; ++j) {
temp = 0.f;
ix = kx;
i__2 = *m;
for (i__ = 1; i__ <= i__2; ++i__) {
temp += a[i__ + j * a_dim1] * x[ix];
ix += *incx;
}
y[jy] += *alpha * temp;
jy += *incy;
}
}
}

return 0;
} 






int sscal_(integer *n, real *sa, real *sx, integer *incx)
{

integer i__1, i__2;

static integer i__, m, mp1, nincx;


--sx;

if (*n <= 0 || *incx <= 0) {
return 0;
}
if (*incx == 1) {
goto L20;
}

nincx = *n * *incx;
i__1 = nincx;
i__2 = *incx;
for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
sx[i__] = *sa * sx[i__];
}
return 0;

L20:
m = *n % 5;
if (m == 0) {
goto L40;
}
i__2 = m;
for (i__ = 1; i__ <= i__2; ++i__) {
sx[i__] = *sa * sx[i__];
}
if (*n < 5) {
return 0;
}
L40:
mp1 = m + 1;
i__2 = *n;
for (i__ = mp1; i__ <= i__2; i__ += 5) {
sx[i__] = *sa * sx[i__];
sx[i__ + 1] = *sa * sx[i__ + 1];
sx[i__ + 2] = *sa * sx[i__ + 2];
sx[i__ + 3] = *sa * sx[i__ + 3];
sx[i__ + 4] = *sa * sx[i__ + 4];
}
return 0;
} 






doublereal sdot_(integer *n, real *sx, integer *incx, real *sy, integer *incy)
{

integer i__1;
real ret_val;

static integer i__, m, ix, iy, mp1;
static real stemp;


--sy;
--sx;


stemp = 0.f;
ret_val = 0.f;
if (*n <= 0) {
return ret_val;
}
if (*incx == 1 && *incy == 1) {
goto L20;
}

ix = 1;
iy = 1;
if (*incx < 0) {
ix = (-(*n) + 1) * *incx + 1;
}
if (*incy < 0) {
iy = (-(*n) + 1) * *incy + 1;
}
i__1 = *n;
for (i__ = 1; i__ <= i__1; ++i__) {
stemp += sx[ix] * sy[iy];
ix += *incx;
iy += *incy;
}
ret_val = stemp;
return ret_val;

L20:
m = *n % 5;
if (m == 0) {
goto L40;
}
i__1 = m;
for (i__ = 1; i__ <= i__1; ++i__) {
stemp += sx[i__] * sy[i__];
}
if (*n < 5) {
goto L60;
}
L40:
mp1 = m + 1;
i__1 = *n;
for (i__ = mp1; i__ <= i__1; i__ += 5) {
stemp = stemp + sx[i__] * sy[i__] + sx[i__ + 1] * sy[i__ + 1] + sx[i__ + 2] * sy[i__ + 2] + sx[i__ + 3] * sy[i__ + 3] + sx[i__ + 4] * sy[i__ + 4];
}
L60:
ret_val = stemp;
return ret_val;
} 






int spotf2_(char *uplo, integer *n, real *a, integer *lda, integer *info)
{

static integer c__1 = 1;
static real c_b10 = -1.f;
static real c_b12 = 1.f;


integer a_dim1, a_offset, i__1, i__2, i__3;
real r__1;


static integer j;
static real ajj;
static logical upper;


a_dim1 = *lda;
a_offset = 1 + a_dim1;
a -= a_offset;


*info = 0;
upper = lsame_(uplo, (char *) "U");
if (! upper && ! lsame_(uplo, (char *) "L")) {
*info = -1;
} else if (*n < 0) {
*info = -2;
} else if (*lda < MAX(1,*n)) {
*info = -4;
}
if (*info != 0) {
i__1 = -(*info);
xerbla_((char *) "SPOTF2", &i__1);
return 0;
}


if (*n == 0) {
return 0;
}

if (upper) {


i__1 = *n;
for (j = 1; j <= i__1; ++j) {


i__2 = j - 1;
ajj = a[j + j * a_dim1] - sdot_(&i__2, &a[j * a_dim1 + 1], &c__1, &a[j * a_dim1 + 1], &c__1);
if (ajj <= 0.f) {
a[j + j * a_dim1] = ajj;
goto L30;
}
ajj = sqrt(ajj);
a[j + j * a_dim1] = ajj;


if (j < *n) {
i__2 = j - 1;
i__3 = *n - j;
sgemv_((char *) "Transpose", &i__2, &i__3, &c_b10, &a[(j + 1) * a_dim1 + 1], lda, &a[j * a_dim1 + 1], &c__1, &c_b12, &a[j + (j + 1) * a_dim1], lda);
i__2 = *n - j;
r__1 = 1.f / ajj;
sscal_(&i__2, &r__1, &a[j + (j + 1) * a_dim1], lda);
}
}
} else {


i__1 = *n;
for (j = 1; j <= i__1; ++j) {


i__2 = j - 1;
ajj = a[j + j * a_dim1] - sdot_(&i__2, &a[j + a_dim1], lda, &a[j + a_dim1], lda);
if (ajj <= 0.f) {
a[j + j * a_dim1] = ajj;
goto L30;
}
ajj = sqrt(ajj);
a[j + j * a_dim1] = ajj;


if (j < *n) {
i__2 = *n - j;
i__3 = j - 1;
sgemv_((char *) "No transpose", &i__2, &i__3, &c_b10, &a[j + 1 + a_dim1], lda, &a[j + a_dim1], lda, &c_b12, &a[j + 1 + j * a_dim1], &c__1);
i__2 = *n - j;
r__1 = 1.f / ajj;
sscal_(&i__2, &r__1, &a[j + 1 + j * a_dim1], &c__1);
}
}
}
goto L40;

L30:
*info = j;

L40:
return 0;
} 

