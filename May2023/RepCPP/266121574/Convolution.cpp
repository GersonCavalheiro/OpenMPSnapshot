#include <omp.h>
#include "image.h"
template<typename type>
void Convolve(ImageClass<type> & img_in, ImageClass<type> & img_out) {

const int width  = img_in.width;
const int height = img_in.height;

type * in  = img_in.pixel;
type * out = img_out.pixel;
#pragma omp parallel for
for (int i = 1; i < height-1; i++)
#pragma omp simd
for (int j = 1; j < width-1; j++) {
type val = -in[(i-1)*width + j-1] -   in[(i-1)*width + j] - in[(i-1)*width + j+1]
-in[(i  )*width + j-1] + 8*in[(i  )*width + j] - in[(i  )*width + j+1]
-in[(i+1)*width + j-1] -   in[(i+1)*width + j] - in[(i+1)*width + j+1];

val = (val < 0   ? 0   : val);
val = (val > 255 ? 255 : val);

out[i*width + j] = val;
}

}

int main(int argc, char ** argv)
{
ImageClass<float> input_image(argv[1]);  
ImageClass<float> result(input_image.width, input_image.height);
printf("\nImage size: %d x %d\n\n", input_image.width, input_image.height);

const double t0 = omp_get_wtime();
Convolve(input_image, result); 
const double t1 = omp_get_wtime();
printf("\nPRINTING STATISTICS\n\n");
const double ts   = t1-t0; 
const double gbps = double(sizeof(float )*input_image.width*input_image.height*2)*1e-9/ts; 
const double fpps = double(input_image.width*input_image.height*2*9)*1e-9/ts; 
printf("%f, %5f, %10f", ts, gbps, fpps);
}