
#ifndef CCGL_UTILS_MATH_H
#define CCGL_UTILS_MATH_H

#include <cmath>

#include "basic.h"
#include "utils_array.h"
#ifndef M_E
#define M_E        2.7182818284590452354     
#endif
#ifndef M_LOG2E
#define M_LOG2E    1.4426950408889634074     
#endif
#ifndef M_LOG10E
#define M_LOG10E   0.43429448190325182765    
#endif
#ifndef M_LN2 
#define M_LN2      0.69314718055994530942    
#endif
#ifndef M_LN10
#define M_LN10     2.30258509299404568402    
#endif
#ifndef M_PI
#define M_PI       3.14159265358979323846    
#endif
#ifndef M_PI_2
#define M_PI_2     1.57079632679489661923    
#endif
#ifndef M_PI_4
#define M_PI_4     0.78539816339744830962    
#endif
#ifndef M_1_PI
#define M_1_PI     0.31830988618379067154    
#endif
#ifndef M_2_PI
#define M_2_PI     0.63661977236758134308    
#endif
#ifndef M_2_SQRTPI
#define M_2_SQRTPI 1.12837916709551257390    
#endif
#ifndef M_SQRT2
#define M_SQRT2    1.41421356237309504880    
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2  0.70710678118654752440    
#endif
#ifndef M_2POW23
#define M_2POW23   8388608.0f                
#endif

#ifndef HUGE
#define HUGE ((float)3.40282347e+38)      
#endif
#ifndef MAXFLOAT
#define MAXFLOAT ((float)3.40282347e+38)  
#endif
#ifndef MINFLOAT
#define MINFLOAT ((float)1.175494351e-38)  
#endif


#ifndef M_SQRT3
#define M_SQRT3 1.732051f                  
#endif

#define M_TC     0.63212055882855767840f    
#define M_PI2    6.283185f                  
#define M_GOLDEN 1.618034f                  
#define M_1_PI2  0.15915494309189534561f    



#define M_DIV3  0.3333333333333333333f  
#define M_DIV4  0.25f                   
#define M_DIV5  0.2f                    
#define M_DIV6  0.1666666666666666666f  
#define M_DIV7  0.142857143f            
#define M_DIV8  0.125f                  
#define M_DIV9  0.1111111111111111111f  
#define M_DIV10 0.1f                    
#define M_DIV11 0.090909091f            
#define M_DIV12 0.0833333333333333333f  
#define M_DIV13 0.076923077f            
#define M_DIV14 0.071428571f            
#define M_DIV15 0.0666666666666666666f  
#define M_DIV16 0.0625f                 
#define M_DIV_LN10 0.43429448190325182765f 


#define PH_C    ((float)299792458)             
#define PH_M0   ((float)1.2566370614359172950) 
#define PH_H    ((float)6.62606957e-34)        
#define PH_HBAR ((float)1.05457172e-34)        
#define PH_K    ((float)1.3806488e-23)         
#define PH_ME   ((float)9.10938291e-31)        
#define PH_MP   ((float)1.672614e-27)          
#define PH_MN   ((float)1.674920e-27)          
#define PH_EC   ((float)1.6021917e-19)         
#define PH_F    ((float)9.648670e4)            
#define PH_G    ((float)6.6732e-11)            
#define PH_AVO  ((float)6.022169e23)           


namespace ccgl {

namespace utils_math {

#ifndef Max
#define Max(a, b) ((a) >= (b) ? (a) : (b))
#endif

#ifndef Min
#define Min(a, b) ((a) >= (b) ? (b) : (a))
#endif

#ifndef Abs
#define Abs(x) ((x) >= 0 ? (x) : -(x))
#endif


template <typename T1, typename T2>
bool FloatEqual(T1 v1, T2 v2) {
return Abs(CVT_DBL(v1) - CVT_DBL(v2)) < 1.e-32;
}


float Expo(float xx, float upper = 20.f, float lower = -20.f);


float Power(float a, float n);


template <typename T>
T MaxInArray(const T* a, int n);


template <typename T>
T MinInArray(const T* a, int n);


template <typename T>
T Sum(int row, const T* data);


template <typename T>
T Sum(int row, int*& idx, const T* data);


template <typename T>
void BasicStatistics(const T* values, int num, double** derivedvalues,
T exclude = static_cast<T>(NODATA_VALUE));


template <typename T>
void BasicStatistics(const T*const * values, int num, int lyrs,
double*** derivedvalues, T exclude = static_cast<T>(NODATA_VALUE));


float ApprSqrt(float z);
double ApprSqrt(double z);

template<typename T>
T CalSqrt(T val) {
#if defined(USE_APPR_PAL_MATH)
return ApprSqrt(val);
#else
return sqrt(val);
#endif
}


template <typename T>
static inline T __p_exp_ln2(const T x) {
const T a1 = static_cast<T>(-0.9998684);
const T a2 = static_cast<T>(0.4982926);
const T a3 = static_cast<T>(-0.1595332);
const T a4 = static_cast<T>(0.0293641);
T exp_x = static_cast<T>(1.0) +
a1 * x +
a2 * x * x +
a3 * x * x * x +
a4 * x * x * x * x;
return static_cast<T>(1.0) / exp_x;
}


template <typename T>
static inline T __p_exp_pos(const T x) {
long int k, twok;
static const T ln2 = static_cast<T>(M_LN2);
T x_;
k = x / ln2;
twok = 1ULL << k;
x_ = x - static_cast<T>(k) * ln2;
return static_cast<T>(twok) * __p_exp_ln2(x_);
}

template <typename T>
static inline T ApprExp(const T x) {
if (x >= static_cast<T>(0.0))
return __p_exp_pos(x);
else
return static_cast<T>(1.0) / __p_exp_pos(-x);
}

template<typename T>
T CalExp(T val) {
#if defined(USE_APPR_PAL_MATH)
return ApprExp(val);
#else
return exp(val);
#endif
}


float ApprLn(float z);
double ApprLn(double z);

template<typename T>
T CalLn(T val) {
#if defined(USE_APPR_PAL_MATH)
return ApprLn(val);
#else
return log(val);
#endif
}


float pow_lookup(const float exp, const float log_base);


float inline ApprPow(float a, float b) {
return pow_lookup(b, ApprLn(a));
};

template<typename T1, typename T2>
double CalPow(T1 a, T2 b) {
#if defined(USE_APPR_PAL_MATH)
return CVT_DBL(ApprPow(CVT_FLT(a), CVT_FLT(b)));
#else
return pow(CVT_DBL(a),CVT_DBL(b));
#endif
}


template <typename T>
T MaxInArray(const T* a, const int n) {
T m = a[0];
for (int i = 1; i < n; i++) {
if (a[i] > m) {
m = a[i];
}
}
return m;
}

template <typename T>
T MinInArray(const T* a, const int n) {
T m = a[0];
for (int i = 1; i < n; i++) {
if (a[i] < m) {
m = a[i];
}
}
return m;
}

template <typename T>
T Sum(const int row, const T* data) {
T tmp = 0;
#pragma omp parallel for reduction(+:tmp)
for (int i = 0; i < row; i++) {
tmp += data[i];
}
return tmp;
}

template <typename T>
T Sum(const int row, int*& idx, const T* data) {
T tmp = 0;
#pragma omp parallel for reduction(+:tmp)
for (int i = 0; i < row; i++) {
int j = idx[i];
tmp += data[j];
}
return tmp;
}

template <typename T>
void BasicStatistics(const T* values, const int num, double** derivedvalues,
T exclude ) {
double* tmpstats = new double[6];
double maxv = MISSINGFLOAT;
double minv = MAXIMUMFLOAT;
int validnum = 0;
double sumv = 0.;
double std = 0.;
for (int i = 0; i < num; i++) {
if (FloatEqual(values[i], exclude)) continue;
if (maxv < values[i]) maxv = values[i];
if (minv > values[i]) minv = values[i];
validnum += 1;
sumv += values[i];
}
tmpstats[0] = CVT_DBL(validnum);
double mean = sumv / tmpstats[0];
#pragma omp parallel for reduction(+:std)
for (int i = 0; i < num; i++) {
if (!FloatEqual(values[i], exclude)) {
std += (values[i] - mean) * (values[i] - mean);
}
}
std = sqrt(std / tmpstats[0]);
tmpstats[1] = mean;
tmpstats[2] = maxv;
tmpstats[3] = minv;
tmpstats[4] = std;
tmpstats[5] = maxv - minv;
*derivedvalues = tmpstats;
}

template <typename T>
void BasicStatistics(const T*const * values, const int num, const int lyrs,
double*** derivedvalues, T exclude ) {
double** tmpstats = new double *[6];
for (int i = 0; i < 6; i++) {
tmpstats[i] = new double[lyrs];
}
for (int j = 0; j < lyrs; j++) {
tmpstats[0][j] = 0.;                    
tmpstats[1][j] = 0.;                    
tmpstats[2][j] = CVT_DBL(MISSINGFLOAT); 
tmpstats[3][j] = CVT_DBL(MAXIMUMFLOAT); 
tmpstats[4][j] = 0.;                    
tmpstats[5][j] = 0.;                    
}
double* sumv = nullptr;
utils_array::Initialize1DArray(lyrs, sumv, 0.);
for (int i = 0; i < num; i++) {
for (int j = 0; j < lyrs; j++) {
if (FloatEqual(values[i][j], exclude)) continue;
if (tmpstats[2][j] < values[i][j]) tmpstats[2][j] = values[i][j];
if (tmpstats[3][j] > values[i][j]) tmpstats[3][j] = values[i][j];
tmpstats[0][j] += 1;
sumv[j] += values[i][j];
}
}

for (int j = 0; j < lyrs; j++) {
tmpstats[5][j] = tmpstats[2][j] - tmpstats[3][j];
tmpstats[1][j] = sumv[j] / tmpstats[0][j];
}
for (int j = 0; j < lyrs; j++) {
double tmpstd = 0;
#pragma omp parallel for reduction(+:tmpstd)
for (int i = 0; i < num; i++) {
if (!FloatEqual(values[i][j], exclude)) {
tmpstd += (values[i][j] - tmpstats[1][j]) * (values[i][j] - tmpstats[1][j]);
}
}
tmpstats[4][j] = tmpstd;
}
for (int j = 0; j < lyrs; j++) {
tmpstats[4][j] = sqrt(tmpstats[4][j] / tmpstats[0][j]);
}
utils_array::Release1DArray(sumv);
*derivedvalues = tmpstats;
}

} 
} 

#endif 
