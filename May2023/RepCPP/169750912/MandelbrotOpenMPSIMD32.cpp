#include "MandelbrotOpenMPSIMD32.h"
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <iostream>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif


void MandelbrotOpenMPSIMD32::calculateFractal(precision_t* cRealArray, precision_t* cImaginaryArray, unsigned short int maxIteration, unsigned int vectorLength, unsigned short int* dest) {
#ifdef __ARM_NEON
if(vectorLength == 0){
throw std::invalid_argument("vectorLength may not be less than 1.");
}
#pragma omp parallel for default(none) num_threads(4) shared(cRealArray, cImaginaryArray, maxIteration, vectorLength, dest) schedule(nonmonotonic:dynamic)
for(unsigned int j = 0; j < (vectorLength/4); j++){
unsigned int offset = j * 4;
float32x4_t cReal = vdupq_n_f32(0);
cReal = vsetq_lane_f32((float32_t) cRealArray[offset+0], cReal, 0);
cReal = vsetq_lane_f32((float32_t) cRealArray[offset+1], cReal, 1);
cReal = vsetq_lane_f32((float32_t) cRealArray[offset+2], cReal, 2);
cReal = vsetq_lane_f32((float32_t) cRealArray[offset+3], cReal, 3);
float32x4_t cImaginary = vdupq_n_f32(0);
cImaginary = vsetq_lane_f32((float32_t) cImaginaryArray[offset+0], cImaginary, 0);
cImaginary = vsetq_lane_f32((float32_t) cImaginaryArray[offset+1], cImaginary, 1);
cImaginary = vsetq_lane_f32((float32_t) cImaginaryArray[offset+2], cImaginary, 2);
cImaginary = vsetq_lane_f32((float32_t) cImaginaryArray[offset+3], cImaginary, 3);
float32x4_t zReal = vdupq_n_f32(0);
float32x4_t zImaginary = vdupq_n_f32(0);
float32x4_t two = vdupq_n_f32(2);
float32x4_t four = vdupq_n_f32(4);
int32x4_t n = vdupq_n_s32(0);
int i = 0;
float32x4_t absSquare = vmlaq_f32(vmulq_f32(zReal, zReal), zImaginary, zImaginary);
int32x4_t absLesserThanTwo = vreinterpretq_s32_u32(vcltq_f32(absSquare, four));
while(i < maxIteration && vaddvq_s32(absLesserThanTwo) != 0){
float32x4_t nextZReal = vaddq_f32(vmlsq_f32(vmulq_f32(zReal, zReal), zImaginary, zImaginary), cReal);
float32x4_t nextZImaginary = vmlaq_f32(cImaginary, two, vmulq_f32(zReal, zImaginary));
zReal = nextZReal;
zImaginary = nextZImaginary;
n = vsubq_s32(n, absLesserThanTwo);
i++;
absSquare = vmlaq_f32(vmulq_f32(zReal, zReal), zImaginary, zImaginary);
absLesserThanTwo = vreinterpretq_s32_u32(vcltq_f32(absSquare, four));
}
dest[offset+0] = (unsigned short int) vgetq_lane_s32(n, 0);
dest[offset+1] = (unsigned short int) vgetq_lane_s32(n, 1);
dest[offset+2] = (unsigned short int) vgetq_lane_s32(n, 2);
dest[offset+3] = (unsigned short int) vgetq_lane_s32(n, 3);

}
#else
#pragma omp parallel for default(none) num_threads(4) shared(vectorLength, dest) schedule(static)
for(unsigned int j = 0; j < vectorLength; j++){
dest[j] = 0;
}
#endif
}