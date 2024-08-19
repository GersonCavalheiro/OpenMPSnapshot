#include "MandelbrotOpenMPSIMD64.h"
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <iostream>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif


void MandelbrotOpenMPSIMD64::calculateFractal(precision_t* cRealArray, precision_t* cImaginaryArray, unsigned short int maxIteration, unsigned int vectorLength, unsigned short int* dest) {
#ifdef __ARM_NEON
if(vectorLength == 0){
throw std::invalid_argument("vectorLength may not be less than 1.");
}
#pragma omp parallel for default(none) num_threads(4) shared(cRealArray, cImaginaryArray, maxIteration, vectorLength, dest) schedule(nonmonotonic:dynamic)
for(unsigned int j = 0; j < (vectorLength/2); j++){
unsigned int offset = j*2;
float64x2_t cReal = vdupq_n_f64(0);
cReal = vsetq_lane_f64((float64_t) cRealArray[offset+0], cReal, 0);
cReal = vsetq_lane_f64((float64_t) cRealArray[offset+1], cReal, 1);
float64x2_t cImaginary = vdupq_n_f64(0);
cImaginary = vsetq_lane_f64((float64_t) cImaginaryArray[offset+0], cImaginary, 0);
cImaginary = vsetq_lane_f64((float64_t) cImaginaryArray[offset+1], cImaginary, 1);
float64x2_t zReal = vdupq_n_f64(0);
float64x2_t zImaginary = vdupq_n_f64(0);
float64x2_t two = vdupq_n_f64(2);
float64x2_t four = vdupq_n_f64(4);
int64x2_t n = vdupq_n_s64(0);

int i = 0;
float64x2_t absSquare = vmlaq_f64(vmulq_f64(zReal, zReal), zImaginary, zImaginary);
int64x2_t absLesserThanTwo = vdupq_n_s64(1);
while(i < maxIteration && vaddvq_s64(absLesserThanTwo) != 0){
float64x2_t nextZReal = vaddq_f64(vmlsq_f64(vmulq_f64(zReal, zReal), zImaginary, zImaginary), cReal);
float64x2_t nextZImaginary = vmlaq_f64(cImaginary, two, vmulq_f64(zReal, zImaginary));
zReal = nextZReal;
zImaginary = nextZImaginary;
n = vsubq_s64(n, absLesserThanTwo);
i++;
absSquare = vmlaq_f64(vmulq_f64(zReal, zReal), zImaginary, zImaginary);
absLesserThanTwo = vreinterpretq_s64_u64(vcltq_f64(absSquare, four));
}
dest[offset+0] = (unsigned short int) vgetq_lane_s64(n, 0);
dest[offset+1] = (unsigned short int) vgetq_lane_s64(n, 1);
}
#else
#pragma omp parallel for default(none) num_threads(4) shared(vectorLength, dest) schedule(static)
for(unsigned int j = 0; j < vectorLength; j++){
dest[j] = 0;
}
#endif
}
