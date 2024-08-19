#include "stencil.h"

template<typename P>
void ApplyStencil(ImageClass<P> & img_in, ImageClass<P> & img_out, int const myRank, int const nRanks) {

const int width  = img_in.width;
const int height = img_in.height;

const double rowsPerProcess = double(height-2)/double(nRanks);
const int myFirstRow = 1 + int(rowsPerProcess*myRank);
const int myLastRow  = 1 + int(rowsPerProcess*(myRank+1));

P * in  = img_in.pixel;
P * out = img_out.pixel;

#pragma omp parallel for
for (int i = myFirstRow; i < myLastRow; i++)
#pragma omp simd
#pragma vector nontemporal
for (int j = 1; j < width-1; j++) {
int val = -in[(i-1)*width + j-1] -   in[(i-1)*width + j] - in[(i-1)*width + j+1] 
-in[(i  )*width + j-1] + 8*in[(i  )*width + j] - in[(i  )*width + j+1] 
-in[(i+1)*width + j-1] -   in[(i+1)*width + j] - in[(i+1)*width + j+1];

val = (val < 0   ? 0   : val);
val = (val > 255 ? 255 : val);

out[i*width + j] = val;
}

}

template void ApplyStencil<float>(ImageClass<float> & img_in, ImageClass<float> & img_out, int const myRank, int const nRanks);
template void ApplyStencil<png_byte>(ImageClass<png_byte> & img_in, ImageClass<png_byte> & img_out, int const myRank, int const nRanks);
