#include <stdint.h>
#include <stdio.h>

#include "openmpStructs.h"
#include "openmpUtils.h"
#include "../utils/utils.h"
#include "../utils/structs.h"
#include "../utils/extramath.h"



void initLevelInfo(CurrentLevelInfo *cli, uint32_t *pyrDimensions, Pyramid gaussPyramid){
cli -> lev = -1; 
cli -> nextLevelDimension = 0;
cli -> oldY = 0xfffffff;
updateLevelInfo(cli, pyrDimensions, gaussPyramid);
}


void updateLevelInfo(CurrentLevelInfo *cli, uint32_t *pyrDimensions, Pyramid gaussPyramid){
cli -> lev++; 
cli -> currentNLevels = cli->lev + 1;
cli -> subregionDimension = 3 * ((1 << (cli->lev + 2)) - 1) / 2;
cli -> currentGaussLevel = gaussPyramid[cli->lev];
cli -> width = cli->currentGaussLevel->width;
cli -> prevLevelDimension = cli->nextLevelDimension; 
cli -> nextLevelDimension += pyrDimensions[cli->lev]; 
}


void imgcpy3_parallel(Image3 *dest, Image3 *source, const uint8_t nThreads){
dest->width = source->width;
dest->height = source->height;
uint32_t dim = dest->width * dest->height;

#pragma omp parallel for num_threads(nThreads) schedule(static, 8)
for(int32_t i = 0; i < dim; i++){
dest->pixels[i] = source->pixels[i];
}
}


void clampImage3_parallel(Image3 *img, const uint8_t nThreads){
int32_t dim = img->width * img->height;
Pixel3 *px = img->pixels;
#pragma omp parallel for num_threads(nThreads) schedule(static, 8)
for(int32_t i = 0; i < dim; i++){
px[i].x = clamp(px[i].x, 0, 1);
px[i].y = clamp(px[i].y, 0, 1);
px[i].z = clamp(px[i].z, 0, 1);
}
}