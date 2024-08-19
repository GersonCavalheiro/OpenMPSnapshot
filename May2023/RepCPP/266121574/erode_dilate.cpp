
#include "erode_dilate.h"
#include <omp.h>
#include "image.h"
template<typename type>
void Erode(ImageClass<type> & img_in, ImageClass<type> & img_out, int kernel_size) {

const int width  = img_in.width;
const int height = img_in.height;

type * in  = img_in.pixel;
type * out = img_out.pixel;
#pragma omp parallel for
for (int i = kernel_size/2; i < height- kernel_size/2; i++)
#pragma omp simd
for (int j = kernel_size/2; j < width- kernel_size/2; j++)
{
for (int k = -kernel_size/2; k<= kernel_size/2; k++)
{
int val = (in[i*width +j + k] == 0 ? 0:1);
out[i*width + j] = val;
if(val == 0)
{
break;
}
}
}

}
template void Erode<float>(ImageClass<float> & img_in, ImageClass<float> & img_out, int kernel_size);

template<typename type>
void Dilate(ImageClass<type> & img_in, ImageClass<type> & img_out, int kernel_size)
{

const int width  = img_in.width;
const int height = img_in.height;

type * in  = img_in.pixel;
type * out = img_out.pixel;
#pragma omp parallel for
for (int i = kernel_size/2; i < height- kernel_size/2; i++)
#pragma omp simd
for (int j = kernel_size/2; j < width- kernel_size/2; j++)
{
for (int k = -kernel_size/2; k<= kernel_size/2; k++)
{
int val = (in[i*width +j + k] == 1 ? 1:0);
out[i*width + j] = val;
if(val == 1)
{
break;
}
}
}

}
template void Dilate<float>(ImageClass<float> & img_in, ImageClass<float> & img_out, int kernel_size);
