#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "omp.h"
#define LENGTH_KERNEL	5
#define LENGTH_FEATURE0	32
#define LENGTH_FEATURE1	28
#define LENGTH_FEATURE2	14
#define LENGTH_FEATURE3	10
#define	LENGTH_FEATURE4	5
#define LENGTH_FEATURE5	1
#define INPUT			1
#define LAYER1			6
#define LAYER2			6
#define LAYER3			16
#define LAYER4			16
#define LAYER5			120
#define LAYER6         256
#define OUTPUT          10
#define PADDING			2
#define FILE_TEST_IMAGE		"t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL		"t10k-labels-idx1-ubyte"
#define COUNT_TEST		10000
#define NUM_THREAD     4
typedef unsigned char uint8;
typedef uint8 image[28][28];
typedef struct Net
{
_Float64 weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];			
_Float64 weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];			
_Float64 weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];			
_Float64 weight5_6[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][LAYER6];	
_Float64 weight6_7[LAYER6][OUTPUT];	
_Float64 bias0_1[LAYER1];
_Float64 bias2_3[LAYER3];
_Float64 bias4_5[LAYER5];
_Float64 bias5_6[LAYER6];
_Float64 bias6_7[OUTPUT];
}Net;
typedef struct Feature
{
_Float64 input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];
_Float64 layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
_Float64 layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];
_Float64 layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
_Float64 layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];
_Float64 layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
_Float64 layer6[LAYER6];
_Float64 output[OUTPUT];
}Feature;
void forward(Net *net, Feature *features)
{
_Float64 temp = 0;
for (int o0 = 0; o0 < 28; o0++)
{
for (int o1 = 0; o1 < 28; o1++)
{	
temp = features->layer1[0][o0][o1];
temp += features->input[0][o0 + 0][o1 + 0] * net->weight0_1[0][0][0][0] 
+ features->input[0][o0 + 0][o1 + 1] * net->weight0_1[0][0][0][1]
+ features->input[0][o0 + 0][o1 + 2] * net->weight0_1[0][0][0][2]
+ features->input[0][o0 + 0][o1 + 3] * net->weight0_1[0][0][0][3]
+ features->input[0][o0 + 0][o1 + 4] * net->weight0_1[0][0][0][4]
+ features->input[0][o0 + 1][o1 + 0] * net->weight0_1[0][0][1][0]
+ features->input[0][o0 + 1][o1 + 1] * net->weight0_1[0][0][1][1]
+ features->input[0][o0 + 1][o1 + 2] * net->weight0_1[0][0][1][2]
+ features->input[0][o0 + 1][o1 + 3] * net->weight0_1[0][0][1][3]
+ features->input[0][o0 + 1][o1 + 4] * net->weight0_1[0][0][1][4]
+ features->input[0][o0 + 2][o1 + 0] * net->weight0_1[0][0][2][0]
+ features->input[0][o0 + 2][o1 + 1] * net->weight0_1[0][0][2][1]
+ features->input[0][o0 + 2][o1 + 2] * net->weight0_1[0][0][2][2]
+ features->input[0][o0 + 2][o1 + 3] * net->weight0_1[0][0][2][3]
+ features->input[0][o0 + 2][o1 + 4] * net->weight0_1[0][0][2][4]
+ features->input[0][o0 + 3][o1 + 0] * net->weight0_1[0][0][3][0]
+ features->input[0][o0 + 3][o1 + 1] * net->weight0_1[0][0][3][1]
+ features->input[0][o0 + 3][o1 + 2] * net->weight0_1[0][0][3][2]
+ features->input[0][o0 + 3][o1 + 3] * net->weight0_1[0][0][3][3]
+ features->input[0][o0 + 3][o1 + 4] * net->weight0_1[0][0][3][4]
+ features->input[0][o0 + 4][o1 + 0] * net->weight0_1[0][0][4][0]
+ features->input[0][o0 + 4][o1 + 1] * net->weight0_1[0][0][4][1]
+ features->input[0][o0 + 4][o1 + 2] * net->weight0_1[0][0][4][2]
+ features->input[0][o0 + 4][o1 + 3] * net->weight0_1[0][0][4][3]
+ features->input[0][o0 + 4][o1 + 4] * net->weight0_1[0][0][4][4];
features->layer1[0][o0][o1] = temp;
temp = features->layer1[1][o0][o1];
temp += features->input[0][o0 + 0][o1 + 0] * net->weight0_1[0][1][0][0]
+ features->input[0][o0 + 0][o1 + 1] * net->weight0_1[0][1][0][1]
+ features->input[0][o0 + 0][o1 + 2] * net->weight0_1[0][1][0][2]
+ features->input[0][o0 + 0][o1 + 3] * net->weight0_1[0][1][0][3]
+ features->input[0][o0 + 0][o1 + 4] * net->weight0_1[0][1][0][4]
+ features->input[0][o0 + 1][o1 + 0] * net->weight0_1[0][1][1][0]
+ features->input[0][o0 + 1][o1 + 1] * net->weight0_1[0][1][1][1]
+ features->input[0][o0 + 1][o1 + 2] * net->weight0_1[0][1][1][2]
+ features->input[0][o0 + 1][o1 + 3] * net->weight0_1[0][1][1][3]
+ features->input[0][o0 + 1][o1 + 4] * net->weight0_1[0][1][1][4]
+ features->input[0][o0 + 2][o1 + 0] * net->weight0_1[0][1][2][0]
+ features->input[0][o0 + 2][o1 + 1] * net->weight0_1[0][1][2][1]
+ features->input[0][o0 + 2][o1 + 2] * net->weight0_1[0][1][2][2]
+ features->input[0][o0 + 2][o1 + 3] * net->weight0_1[0][1][2][3]
+ features->input[0][o0 + 2][o1 + 4] * net->weight0_1[0][1][2][4]
+ features->input[0][o0 + 3][o1 + 0] * net->weight0_1[0][1][3][0]
+ features->input[0][o0 + 3][o1 + 1] * net->weight0_1[0][1][3][1]
+ features->input[0][o0 + 3][o1 + 2] * net->weight0_1[0][1][3][2]
+ features->input[0][o0 + 3][o1 + 3] * net->weight0_1[0][1][3][3]
+ features->input[0][o0 + 3][o1 + 4] * net->weight0_1[0][1][3][4]
+ features->input[0][o0 + 4][o1 + 0] * net->weight0_1[0][1][4][0]
+ features->input[0][o0 + 4][o1 + 1] * net->weight0_1[0][1][4][1]
+ features->input[0][o0 + 4][o1 + 2] * net->weight0_1[0][1][4][2]
+ features->input[0][o0 + 4][o1 + 3] * net->weight0_1[0][1][4][3]
+ features->input[0][o0 + 4][o1 + 4] * net->weight0_1[0][1][4][4];
features->layer1[1][o0][o1] = temp;
temp = features->layer1[2][o0][o1];
temp += features->input[0][o0 + 0][o1 + 0] * net->weight0_1[0][2][0][0]
+ features->input[0][o0 + 0][o1 + 1] * net->weight0_1[0][2][0][1]
+ features->input[0][o0 + 0][o1 + 2] * net->weight0_1[0][2][0][2]
+ features->input[0][o0 + 0][o1 + 3] * net->weight0_1[0][2][0][3]
+ features->input[0][o0 + 0][o1 + 4] * net->weight0_1[0][2][0][4]
+ features->input[0][o0 + 1][o1 + 0] * net->weight0_1[0][2][1][0]
+ features->input[0][o0 + 1][o1 + 1] * net->weight0_1[0][2][1][1]
+ features->input[0][o0 + 1][o1 + 2] * net->weight0_1[0][2][1][2]
+ features->input[0][o0 + 1][o1 + 3] * net->weight0_1[0][2][1][3]
+ features->input[0][o0 + 1][o1 + 4] * net->weight0_1[0][2][1][4]
+ features->input[0][o0 + 2][o1 + 0] * net->weight0_1[0][2][2][0]
+ features->input[0][o0 + 2][o1 + 1] * net->weight0_1[0][2][2][1]
+ features->input[0][o0 + 2][o1 + 2] * net->weight0_1[0][2][2][2]
+ features->input[0][o0 + 2][o1 + 3] * net->weight0_1[0][2][2][3]
+ features->input[0][o0 + 2][o1 + 4] * net->weight0_1[0][2][2][4]
+ features->input[0][o0 + 3][o1 + 0] * net->weight0_1[0][2][3][0]
+ features->input[0][o0 + 3][o1 + 1] * net->weight0_1[0][2][3][1]
+ features->input[0][o0 + 3][o1 + 2] * net->weight0_1[0][2][3][2]
+ features->input[0][o0 + 3][o1 + 3] * net->weight0_1[0][2][3][3]
+ features->input[0][o0 + 3][o1 + 4] * net->weight0_1[0][2][3][4]
+ features->input[0][o0 + 4][o1 + 0] * net->weight0_1[0][2][4][0]
+ features->input[0][o0 + 4][o1 + 1] * net->weight0_1[0][2][4][1]
+ features->input[0][o0 + 4][o1 + 2] * net->weight0_1[0][2][4][2]
+ features->input[0][o0 + 4][o1 + 3] * net->weight0_1[0][2][4][3]
+ features->input[0][o0 + 4][o1 + 4] * net->weight0_1[0][2][4][4];
features->layer1[2][o0][o1] = temp;
temp = features->layer1[3][o0][o1];
temp += features->input[0][o0 + 0][o1 + 0] * net->weight0_1[0][3][0][0]
+ features->input[0][o0 + 0][o1 + 1] * net->weight0_1[0][3][0][1]
+ features->input[0][o0 + 0][o1 + 2] * net->weight0_1[0][3][0][2]
+ features->input[0][o0 + 0][o1 + 3] * net->weight0_1[0][3][0][3]
+ features->input[0][o0 + 0][o1 + 4] * net->weight0_1[0][3][0][4]
+ features->input[0][o0 + 1][o1 + 0] * net->weight0_1[0][3][1][0]
+ features->input[0][o0 + 1][o1 + 1] * net->weight0_1[0][3][1][1]
+ features->input[0][o0 + 1][o1 + 2] * net->weight0_1[0][3][1][2]
+ features->input[0][o0 + 1][o1 + 3] * net->weight0_1[0][3][1][3]
+ features->input[0][o0 + 1][o1 + 4] * net->weight0_1[0][3][1][4]
+ features->input[0][o0 + 2][o1 + 0] * net->weight0_1[0][3][2][0]
+ features->input[0][o0 + 2][o1 + 1] * net->weight0_1[0][3][2][1]
+ features->input[0][o0 + 2][o1 + 2] * net->weight0_1[0][3][2][2]
+ features->input[0][o0 + 2][o1 + 3] * net->weight0_1[0][3][2][3]
+ features->input[0][o0 + 2][o1 + 4] * net->weight0_1[0][3][2][4]
+ features->input[0][o0 + 3][o1 + 0] * net->weight0_1[0][3][3][0]
+ features->input[0][o0 + 3][o1 + 1] * net->weight0_1[0][3][3][1]
+ features->input[0][o0 + 3][o1 + 2] * net->weight0_1[0][3][3][2]
+ features->input[0][o0 + 3][o1 + 3] * net->weight0_1[0][3][3][3]
+ features->input[0][o0 + 3][o1 + 4] * net->weight0_1[0][3][3][4]
+ features->input[0][o0 + 4][o1 + 0] * net->weight0_1[0][3][4][0]
+ features->input[0][o0 + 4][o1 + 1] * net->weight0_1[0][3][4][1]
+ features->input[0][o0 + 4][o1 + 2] * net->weight0_1[0][3][4][2]
+ features->input[0][o0 + 4][o1 + 3] * net->weight0_1[0][3][4][3]
+ features->input[0][o0 + 4][o1 + 4] * net->weight0_1[0][3][4][4];
features->layer1[3][o0][o1] = temp;
temp = features->layer1[4][o0][o1];
temp += features->input[0][o0 + 0][o1 + 0] * net->weight0_1[0][4][0][0]
+ features->input[0][o0 + 0][o1 + 1] * net->weight0_1[0][4][0][1]
+ features->input[0][o0 + 0][o1 + 2] * net->weight0_1[0][4][0][2]
+ features->input[0][o0 + 0][o1 + 3] * net->weight0_1[0][4][0][3]
+ features->input[0][o0 + 0][o1 + 4] * net->weight0_1[0][4][0][4]
+ features->input[0][o0 + 1][o1 + 0] * net->weight0_1[0][4][1][0]
+ features->input[0][o0 + 1][o1 + 1] * net->weight0_1[0][4][1][1]
+ features->input[0][o0 + 1][o1 + 2] * net->weight0_1[0][4][1][2]
+ features->input[0][o0 + 1][o1 + 3] * net->weight0_1[0][4][1][3]
+ features->input[0][o0 + 1][o1 + 4] * net->weight0_1[0][4][1][4]
+ features->input[0][o0 + 2][o1 + 0] * net->weight0_1[0][4][2][0]
+ features->input[0][o0 + 2][o1 + 1] * net->weight0_1[0][4][2][1]
+ features->input[0][o0 + 2][o1 + 2] * net->weight0_1[0][4][2][2]
+ features->input[0][o0 + 2][o1 + 3] * net->weight0_1[0][4][2][3]
+ features->input[0][o0 + 2][o1 + 4] * net->weight0_1[0][4][2][4]
+ features->input[0][o0 + 3][o1 + 0] * net->weight0_1[0][4][3][0]
+ features->input[0][o0 + 3][o1 + 1] * net->weight0_1[0][4][3][1]
+ features->input[0][o0 + 3][o1 + 2] * net->weight0_1[0][4][3][2]
+ features->input[0][o0 + 3][o1 + 3] * net->weight0_1[0][4][3][3]
+ features->input[0][o0 + 3][o1 + 4] * net->weight0_1[0][4][3][4]
+ features->input[0][o0 + 4][o1 + 0] * net->weight0_1[0][4][4][0]
+ features->input[0][o0 + 4][o1 + 1] * net->weight0_1[0][4][4][1]
+ features->input[0][o0 + 4][o1 + 2] * net->weight0_1[0][4][4][2]
+ features->input[0][o0 + 4][o1 + 3] * net->weight0_1[0][4][4][3]
+ features->input[0][o0 + 4][o1 + 4] * net->weight0_1[0][4][4][4];
features->layer1[4][o0][o1] = temp;
temp = features->layer1[5][o0][o1];
temp += features->input[0][o0 + 0][o1 + 0] * net->weight0_1[0][5][0][0]
+ features->input[0][o0 + 0][o1 + 1] * net->weight0_1[0][5][0][1]
+ features->input[0][o0 + 0][o1 + 2] * net->weight0_1[0][5][0][2]
+ features->input[0][o0 + 0][o1 + 3] * net->weight0_1[0][5][0][3]
+ features->input[0][o0 + 0][o1 + 4] * net->weight0_1[0][5][0][4]
+ features->input[0][o0 + 1][o1 + 0] * net->weight0_1[0][5][1][0]
+ features->input[0][o0 + 1][o1 + 1] * net->weight0_1[0][5][1][1]
+ features->input[0][o0 + 1][o1 + 2] * net->weight0_1[0][5][1][2]
+ features->input[0][o0 + 1][o1 + 3] * net->weight0_1[0][5][1][3]
+ features->input[0][o0 + 1][o1 + 4] * net->weight0_1[0][5][1][4]
+ features->input[0][o0 + 2][o1 + 0] * net->weight0_1[0][5][2][0]
+ features->input[0][o0 + 2][o1 + 1] * net->weight0_1[0][5][2][1]
+ features->input[0][o0 + 2][o1 + 2] * net->weight0_1[0][5][2][2]
+ features->input[0][o0 + 2][o1 + 3] * net->weight0_1[0][5][2][3]
+ features->input[0][o0 + 2][o1 + 4] * net->weight0_1[0][5][2][4]
+ features->input[0][o0 + 3][o1 + 0] * net->weight0_1[0][5][3][0]
+ features->input[0][o0 + 3][o1 + 1] * net->weight0_1[0][5][3][1]
+ features->input[0][o0 + 3][o1 + 2] * net->weight0_1[0][5][3][2]
+ features->input[0][o0 + 3][o1 + 3] * net->weight0_1[0][5][3][3]
+ features->input[0][o0 + 3][o1 + 4] * net->weight0_1[0][5][3][4]
+ features->input[0][o0 + 4][o1 + 0] * net->weight0_1[0][5][4][0]
+ features->input[0][o0 + 4][o1 + 1] * net->weight0_1[0][5][4][1]
+ features->input[0][o0 + 4][o1 + 2] * net->weight0_1[0][5][4][2]
+ features->input[0][o0 + 4][o1 + 3] * net->weight0_1[0][5][4][3]
+ features->input[0][o0 + 4][o1 + 4] * net->weight0_1[0][5][4][4];
features->layer1[5][o0][o1] = temp;
}
}
for ( int o0 = 0; o0 < 28; o0+=2)
for (int  o1 = 0; o1 < 28; o1+=2)
{
double  bias = net->bias0_1[0];
double max = -bias;
max = (features->layer1[0][o0][o1 ] > max) ? features->layer1[0][o0][o1] : max;
max = (features->layer1[0][o0][o1 +1] > max) ? features->layer1[0][o0][o1 + 1] : max;
max = (features->layer1[0][o0+ 1][o1 ] > max) ? features->layer1[0][o0+ 1][o1] : max;
max = (features->layer1[0][o0+ 1][o1 +1] > max) ? features->layer1[0][o0+ 1][o1 + 1] : max;
features->layer2[0][o0>>1][o1>>1] = max+bias;
bias = net->bias0_1[1];
max = -bias;
max = (features->layer1[1][o0][o1 ] > max) ? features->layer1[1][o0][o1] : max;
max = (features->layer1[1][o0][o1 +1] > max) ? features->layer1[1][o0][o1 + 1] : max;
max = (features->layer1[1][o0+ 1][o1 ] > max) ? features->layer1[1][o0+ 1][o1] : max;
max = (features->layer1[1][o0+ 1][o1 +1] > max) ? features->layer1[1][o0+ 1][o1 + 1] : max;
features->layer2[1][o0>>1][o1>>1] = max+bias;
bias = net->bias0_1[2];
max = -bias;
max = (features->layer1[2][o0][o1 ] > max) ? features->layer1[2][o0][o1] : max;
max = (features->layer1[2][o0][o1 +1] > max) ? features->layer1[2][o0][o1 + 1] : max;
max = (features->layer1[2][o0+ 1][o1 ] > max) ? features->layer1[2][o0+ 1][o1] : max;
max = (features->layer1[2][o0+ 1][o1 +1] > max) ? features->layer1[2][o0+ 1][o1 + 1] : max;
features->layer2[2][o0>>1][o1>>1] = max+bias;
bias = net->bias0_1[3];
max = -bias;
max = (features->layer1[3][o0][o1 ] > max) ? features->layer1[3][o0][o1] : max;
max = (features->layer1[3][o0][o1 +1] > max) ? features->layer1[3][o0][o1 + 1] : max;
max = (features->layer1[3][o0+ 1][o1 ] > max) ? features->layer1[3][o0+ 1][o1] : max;
max = (features->layer1[3][o0+ 1][o1 +1] > max) ? features->layer1[3][o0+ 1][o1 + 1] : max;
features->layer2[3][o0>>1][o1>>1] = max+bias;
bias = net->bias0_1[4];
max = -bias;
max = (features->layer1[4][o0][o1 ] > max) ? features->layer1[4][o0][o1] : max;
max = (features->layer1[4][o0][o1 +1] > max) ? features->layer1[4][o0][o1 + 1] : max;
max = (features->layer1[4][o0+ 1][o1 ] > max) ? features->layer1[4][o0+ 1][o1] : max;
max = (features->layer1[4][o0+ 1][o1 +1] > max) ? features->layer1[4][o0+ 1][o1 + 1] : max;
features->layer2[4][o0>>1][o1>>1] = max+bias;
bias = net->bias0_1[5];
max = -bias;
max = (features->layer1[5][o0][o1 ] > max) ? features->layer1[5][o0][o1] : max;
max = (features->layer1[5][o0][o1 +1] > max) ? features->layer1[5][o0][o1 + 1] : max;
max = (features->layer1[5][o0+ 1][o1 ] > max) ? features->layer1[5][o0+ 1][o1] : max;
max = (features->layer1[5][o0+ 1][o1 +1] > max) ? features->layer1[5][o0+ 1][o1 + 1] : max;
features->layer2[5][o0>>1][o1>>1] = max+bias;
}
_Float64 tempBias = 0.0;
for (int y = 0; y < 16; y++)
{
tempBias = net->bias2_3[y];
for (int o0 = 0; o0 < 10; o0++)
for (int o1 = 0; o1 < 10; o1++)
{
temp = features->layer3[y][o0][o1];
temp += features->layer2[0][o0 + 0][o1 + 0] * net->weight2_3[0][y][0][0]
+ features->layer2[0][o0 + 0][o1 + 1] * net->weight2_3[0][y][0][1]
+ features->layer2[0][o0 + 0][o1 + 2] * net->weight2_3[0][y][0][2]
+ features->layer2[0][o0 + 0][o1 + 3] * net->weight2_3[0][y][0][3]
+ features->layer2[0][o0 + 0][o1 + 4] * net->weight2_3[0][y][0][4]
+ features->layer2[0][o0 + 1][o1 + 0] * net->weight2_3[0][y][1][0]
+ features->layer2[0][o0 + 1][o1 + 1] * net->weight2_3[0][y][1][1]
+ features->layer2[0][o0 + 1][o1 + 2] * net->weight2_3[0][y][1][2]
+ features->layer2[0][o0 + 1][o1 + 3] * net->weight2_3[0][y][1][3]
+ features->layer2[0][o0 + 1][o1 + 4] * net->weight2_3[0][y][1][4]
+ features->layer2[0][o0 + 2][o1 + 0] * net->weight2_3[0][y][2][0]
+ features->layer2[0][o0 + 2][o1 + 1] * net->weight2_3[0][y][2][1]
+ features->layer2[0][o0 + 2][o1 + 2] * net->weight2_3[0][y][2][2]
+ features->layer2[0][o0 + 2][o1 + 3] * net->weight2_3[0][y][2][3]
+ features->layer2[0][o0 + 2][o1 + 4] * net->weight2_3[0][y][2][4]
+ features->layer2[0][o0 + 3][o1 + 0] * net->weight2_3[0][y][3][0]
+ features->layer2[0][o0 + 3][o1 + 1] * net->weight2_3[0][y][3][1]
+ features->layer2[0][o0 + 3][o1 + 2] * net->weight2_3[0][y][3][2]
+ features->layer2[0][o0 + 3][o1 + 3] * net->weight2_3[0][y][3][3]
+ features->layer2[0][o0 + 3][o1 + 4] * net->weight2_3[0][y][3][4]
+ features->layer2[0][o0 + 4][o1 + 0] * net->weight2_3[0][y][4][0]
+ features->layer2[0][o0 + 4][o1 + 1] * net->weight2_3[0][y][4][1]
+ features->layer2[0][o0 + 4][o1 + 2] * net->weight2_3[0][y][4][2]
+ features->layer2[0][o0 + 4][o1 + 3] * net->weight2_3[0][y][4][3]
+ features->layer2[0][o0 + 4][o1 + 4] * net->weight2_3[0][y][4][4];
temp += features->layer2[1][o0 + 0][o1 + 0] * net->weight2_3[1][y][0][0]
+ features->layer2[1][o0 + 0][o1 + 1] * net->weight2_3[1][y][0][1]
+ features->layer2[1][o0 + 0][o1 + 2] * net->weight2_3[1][y][0][2]
+ features->layer2[1][o0 + 0][o1 + 3] * net->weight2_3[1][y][0][3]
+ features->layer2[1][o0 + 0][o1 + 4] * net->weight2_3[1][y][0][4]
+ features->layer2[1][o0 + 1][o1 + 0] * net->weight2_3[1][y][1][0]
+ features->layer2[1][o0 + 1][o1 + 1] * net->weight2_3[1][y][1][1]
+ features->layer2[1][o0 + 1][o1 + 2] * net->weight2_3[1][y][1][2]
+ features->layer2[1][o0 + 1][o1 + 3] * net->weight2_3[1][y][1][3]
+ features->layer2[1][o0 + 1][o1 + 4] * net->weight2_3[1][y][1][4]
+ features->layer2[1][o0 + 2][o1 + 0] * net->weight2_3[1][y][2][0]
+ features->layer2[1][o0 + 2][o1 + 1] * net->weight2_3[1][y][2][1]
+ features->layer2[1][o0 + 2][o1 + 2] * net->weight2_3[1][y][2][2]
+ features->layer2[1][o0 + 2][o1 + 3] * net->weight2_3[1][y][2][3]
+ features->layer2[1][o0 + 2][o1 + 4] * net->weight2_3[1][y][2][4]
+ features->layer2[1][o0 + 3][o1 + 0] * net->weight2_3[1][y][3][0]
+ features->layer2[1][o0 + 3][o1 + 1] * net->weight2_3[1][y][3][1]
+ features->layer2[1][o0 + 3][o1 + 2] * net->weight2_3[1][y][3][2]
+ features->layer2[1][o0 + 3][o1 + 3] * net->weight2_3[1][y][3][3]
+ features->layer2[1][o0 + 3][o1 + 4] * net->weight2_3[1][y][3][4]
+ features->layer2[1][o0 + 4][o1 + 0] * net->weight2_3[1][y][4][0]
+ features->layer2[1][o0 + 4][o1 + 1] * net->weight2_3[1][y][4][1]
+ features->layer2[1][o0 + 4][o1 + 2] * net->weight2_3[1][y][4][2]
+ features->layer2[1][o0 + 4][o1 + 3] * net->weight2_3[1][y][4][3]
+ features->layer2[1][o0 + 4][o1 + 4] * net->weight2_3[1][y][4][4];
temp += features->layer2[2][o0 + 0][o1 + 0] * net->weight2_3[2][y][0][0]
+ features->layer2[2][o0 + 0][o1 + 1] * net->weight2_3[2][y][0][1]
+ features->layer2[2][o0 + 0][o1 + 2] * net->weight2_3[2][y][0][2]
+ features->layer2[2][o0 + 0][o1 + 3] * net->weight2_3[2][y][0][3]
+ features->layer2[2][o0 + 0][o1 + 4] * net->weight2_3[2][y][0][4]
+ features->layer2[2][o0 + 1][o1 + 0] * net->weight2_3[2][y][1][0]
+ features->layer2[2][o0 + 1][o1 + 1] * net->weight2_3[2][y][1][1]
+ features->layer2[2][o0 + 1][o1 + 2] * net->weight2_3[2][y][1][2]
+ features->layer2[2][o0 + 1][o1 + 3] * net->weight2_3[2][y][1][3]
+ features->layer2[2][o0 + 1][o1 + 4] * net->weight2_3[2][y][1][4]
+ features->layer2[2][o0 + 2][o1 + 0] * net->weight2_3[2][y][2][0]
+ features->layer2[2][o0 + 2][o1 + 1] * net->weight2_3[2][y][2][1]
+ features->layer2[2][o0 + 2][o1 + 2] * net->weight2_3[2][y][2][2]
+ features->layer2[2][o0 + 2][o1 + 3] * net->weight2_3[2][y][2][3]
+ features->layer2[2][o0 + 2][o1 + 4] * net->weight2_3[2][y][2][4]
+ features->layer2[2][o0 + 3][o1 + 0] * net->weight2_3[2][y][3][0]
+ features->layer2[2][o0 + 3][o1 + 1] * net->weight2_3[2][y][3][1]
+ features->layer2[2][o0 + 3][o1 + 2] * net->weight2_3[2][y][3][2]
+ features->layer2[2][o0 + 3][o1 + 3] * net->weight2_3[2][y][3][3]
+ features->layer2[2][o0 + 3][o1 + 4] * net->weight2_3[2][y][3][4]
+ features->layer2[2][o0 + 4][o1 + 0] * net->weight2_3[2][y][4][0]
+ features->layer2[2][o0 + 4][o1 + 1] * net->weight2_3[2][y][4][1]
+ features->layer2[2][o0 + 4][o1 + 2] * net->weight2_3[2][y][4][2]
+ features->layer2[2][o0 + 4][o1 + 3] * net->weight2_3[2][y][4][3]
+ features->layer2[2][o0 + 4][o1 + 4] * net->weight2_3[2][y][4][4];
temp += features->layer2[3][o0 + 0][o1 + 0] * net->weight2_3[3][y][0][0]
+ features->layer2[3][o0 + 0][o1 + 1] * net->weight2_3[3][y][0][1]
+ features->layer2[3][o0 + 0][o1 + 2] * net->weight2_3[3][y][0][2]
+ features->layer2[3][o0 + 0][o1 + 3] * net->weight2_3[3][y][0][3]
+ features->layer2[3][o0 + 0][o1 + 4] * net->weight2_3[3][y][0][4]
+ features->layer2[3][o0 + 1][o1 + 0] * net->weight2_3[3][y][1][0]
+ features->layer2[3][o0 + 1][o1 + 1] * net->weight2_3[3][y][1][1]
+ features->layer2[3][o0 + 1][o1 + 2] * net->weight2_3[3][y][1][2]
+ features->layer2[3][o0 + 1][o1 + 3] * net->weight2_3[3][y][1][3]
+ features->layer2[3][o0 + 1][o1 + 4] * net->weight2_3[3][y][1][4]
+ features->layer2[3][o0 + 2][o1 + 0] * net->weight2_3[3][y][2][0]
+ features->layer2[3][o0 + 2][o1 + 1] * net->weight2_3[3][y][2][1]
+ features->layer2[3][o0 + 2][o1 + 2] * net->weight2_3[3][y][2][2]
+ features->layer2[3][o0 + 2][o1 + 3] * net->weight2_3[3][y][2][3]
+ features->layer2[3][o0 + 2][o1 + 4] * net->weight2_3[3][y][2][4]
+ features->layer2[3][o0 + 3][o1 + 0] * net->weight2_3[3][y][3][0]
+ features->layer2[3][o0 + 3][o1 + 1] * net->weight2_3[3][y][3][1]
+ features->layer2[3][o0 + 3][o1 + 2] * net->weight2_3[3][y][3][2]
+ features->layer2[3][o0 + 3][o1 + 3] * net->weight2_3[3][y][3][3]
+ features->layer2[3][o0 + 3][o1 + 4] * net->weight2_3[3][y][3][4]
+ features->layer2[3][o0 + 4][o1 + 0] * net->weight2_3[3][y][4][0]
+ features->layer2[3][o0 + 4][o1 + 1] * net->weight2_3[3][y][4][1]
+ features->layer2[3][o0 + 4][o1 + 2] * net->weight2_3[3][y][4][2]
+ features->layer2[3][o0 + 4][o1 + 3] * net->weight2_3[3][y][4][3]
+ features->layer2[3][o0 + 4][o1 + 4] * net->weight2_3[3][y][4][4];
temp += features->layer2[4][o0 + 0][o1 + 0] * net->weight2_3[4][y][0][0]
+ features->layer2[4][o0 + 0][o1 + 1] * net->weight2_3[4][y][0][1]
+ features->layer2[4][o0 + 0][o1 + 2] * net->weight2_3[4][y][0][2]
+ features->layer2[4][o0 + 0][o1 + 3] * net->weight2_3[4][y][0][3]
+ features->layer2[4][o0 + 0][o1 + 4] * net->weight2_3[4][y][0][4]
+ features->layer2[4][o0 + 1][o1 + 0] * net->weight2_3[4][y][1][0]
+ features->layer2[4][o0 + 1][o1 + 1] * net->weight2_3[4][y][1][1]
+ features->layer2[4][o0 + 1][o1 + 2] * net->weight2_3[4][y][1][2]
+ features->layer2[4][o0 + 1][o1 + 3] * net->weight2_3[4][y][1][3]
+ features->layer2[4][o0 + 1][o1 + 4] * net->weight2_3[4][y][1][4]
+ features->layer2[4][o0 + 2][o1 + 0] * net->weight2_3[4][y][2][0]
+ features->layer2[4][o0 + 2][o1 + 1] * net->weight2_3[4][y][2][1]
+ features->layer2[4][o0 + 2][o1 + 2] * net->weight2_3[4][y][2][2]
+ features->layer2[4][o0 + 2][o1 + 3] * net->weight2_3[4][y][2][3]
+ features->layer2[4][o0 + 2][o1 + 4] * net->weight2_3[4][y][2][4]
+ features->layer2[4][o0 + 3][o1 + 0] * net->weight2_3[4][y][3][0]
+ features->layer2[4][o0 + 3][o1 + 1] * net->weight2_3[4][y][3][1]
+ features->layer2[4][o0 + 3][o1 + 2] * net->weight2_3[4][y][3][2]
+ features->layer2[4][o0 + 3][o1 + 3] * net->weight2_3[4][y][3][3]
+ features->layer2[4][o0 + 3][o1 + 4] * net->weight2_3[4][y][3][4]
+ features->layer2[4][o0 + 4][o1 + 0] * net->weight2_3[4][y][4][0]
+ features->layer2[4][o0 + 4][o1 + 1] * net->weight2_3[4][y][4][1]
+ features->layer2[4][o0 + 4][o1 + 2] * net->weight2_3[4][y][4][2]
+ features->layer2[4][o0 + 4][o1 + 3] * net->weight2_3[4][y][4][3]
+ features->layer2[4][o0 + 4][o1 + 4] * net->weight2_3[4][y][4][4];
temp += features->layer2[5][o0 + 0][o1 + 0] * net->weight2_3[5][y][0][0]
+ features->layer2[5][o0 + 0][o1 + 1] * net->weight2_3[5][y][0][1]
+ features->layer2[5][o0 + 0][o1 + 2] * net->weight2_3[5][y][0][2]
+ features->layer2[5][o0 + 0][o1 + 3] * net->weight2_3[5][y][0][3]
+ features->layer2[5][o0 + 0][o1 + 4] * net->weight2_3[5][y][0][4]
+ features->layer2[5][o0 + 1][o1 + 0] * net->weight2_3[5][y][1][0]
+ features->layer2[5][o0 + 1][o1 + 1] * net->weight2_3[5][y][1][1]
+ features->layer2[5][o0 + 1][o1 + 2] * net->weight2_3[5][y][1][2]
+ features->layer2[5][o0 + 1][o1 + 3] * net->weight2_3[5][y][1][3]
+ features->layer2[5][o0 + 1][o1 + 4] * net->weight2_3[5][y][1][4]
+ features->layer2[5][o0 + 2][o1 + 0] * net->weight2_3[5][y][2][0]
+ features->layer2[5][o0 + 2][o1 + 1] * net->weight2_3[5][y][2][1]
+ features->layer2[5][o0 + 2][o1 + 2] * net->weight2_3[5][y][2][2]
+ features->layer2[5][o0 + 2][o1 + 3] * net->weight2_3[5][y][2][3]
+ features->layer2[5][o0 + 2][o1 + 4] * net->weight2_3[5][y][2][4]
+ features->layer2[5][o0 + 3][o1 + 0] * net->weight2_3[5][y][3][0]
+ features->layer2[5][o0 + 3][o1 + 1] * net->weight2_3[5][y][3][1]
+ features->layer2[5][o0 + 3][o1 + 2] * net->weight2_3[5][y][3][2]
+ features->layer2[5][o0 + 3][o1 + 3] * net->weight2_3[5][y][3][3]
+ features->layer2[5][o0 + 3][o1 + 4] * net->weight2_3[5][y][3][4]	
+ features->layer2[5][o0 + 4][o1 + 0] * net->weight2_3[5][y][4][0]
+ features->layer2[5][o0 + 4][o1 + 1] * net->weight2_3[5][y][4][1]
+ features->layer2[5][o0 + 4][o1 + 2] * net->weight2_3[5][y][4][2]
+ features->layer2[5][o0 + 4][o1 + 3] * net->weight2_3[5][y][4][3]
+ features->layer2[5][o0 + 4][o1 + 4] * net->weight2_3[5][y][4][4];
features->layer3[y][o0][o1] = temp;
temp += tempBias;
if (temp < 0)
temp = 0;
features->layer3[y][o0][o1] = temp;
}
}
for (uint8 i = 0; i < 16; i++)
{	
int x0 = 0, x1 = 0, ismax; _Float64 tempD;
x1 = features->layer3[i][0][1] > features->layer3[i][0][0];
x0 = features->layer3[i][1][0] > features->layer3[i][0][x1];
tempD = features->layer3[i][1][1];
x1 += x0 * (- x1);	
ismax = tempD > features->layer3[i][x0][x1];
x0 += ismax * (1 - x0);
x1 += ismax * (1 - x1);
features->layer4[i][0][0] = features->layer3[i][x0][x1];
x0 = 0; x1 = 0;
x1 = features->layer3[i][0][3] > features->layer3[i][0][2];
x0 = features->layer3[i][1][2] > features->layer3[i][0][2 + x1];
tempD = features->layer3[i][1][3];
x1 += x0 * (- x1);	
ismax = tempD > features->layer3[i][x0][2 + x1];
x0 += ismax * (1 - x0);
x1 += ismax * (1 - x1);
features->layer4[i][0][1] = features->layer3[i][x0][2 + x1];
x0 = 0; x1 = 0;
x1 = features->layer3[i][0][5] > features->layer3[i][0][4];
x0 = features->layer3[i][1][4] > features->layer3[i][0][4 + x1];
tempD = features->layer3[i][0 * 2 + 1][5];
x1 += x0 * (- x1);	
ismax = tempD > features->layer3[i][0 * 2 + x0][2 * 2 + x1];
x0 += ismax * (1 - x0);
x1 += ismax * (1 - x1);
features->layer4[i][0][2] = features->layer3[i][0 * 2 + x0][2 * 2 + x1];
x0 = 0; x1 = 0;
x1 = features->layer3[i][0 * 2][3 * 2 + 1] > features->layer3[i][0 * 2][3 * 2];
x0 = features->layer3[i][0 * 2 + 1][3 * 2] > features->layer3[i][0 * 2][3 * 2 + x1];
tempD = features->layer3[i][0 * 2 + 1][3 * 2 + 1];
x1 += x0 * (- x1);	
ismax = tempD > features->layer3[i][0 * 2 + x0][3 * 2 + x1];
x0 += ismax * (1 - x0);
x1 += ismax * (1 - x1);
features->layer4[i][0][3] = features->layer3[i][0 * 2 + x0][3 * 2 + x1];
x0 = 0; x1 = 0;
x1 = features->layer3[i][0 * 2][4 * 2 + 1] > features->layer3[i][0 * 2][4 * 2];
x0 = features->layer3[i][0 * 2 + 1][4 * 2] > features->layer3[i][0 * 2][4 * 2 + x1];
tempD = features->layer3[i][0 * 2 + 1][4 * 2 + 1];
x1 += x0 * (- x1);	
ismax = tempD > features->layer3[i][0 * 2 + x0][4 * 2 + x1];
x0 += ismax * (1 - x0);
x1 += ismax * (1 - x1);
features->layer4[i][0][4] = features->layer3[i][0 * 2 + x0][4 * 2 + x1];
x0 = 0; x1 = 0;
x1 = features->layer3[i][1 * 2][0 * 2 + 1] > features->layer3[i][1 * 2][0 * 2];
x0 = features->layer3[i][1 * 2 + 1][0 * 2] > features->layer3[i][1 * 2][0 * 2 + x1];
tempD = features->layer3[i][1 * 2 + 1][0 * 2 + 1];
x1 += x0 * (- x1);	
ismax = tempD > features->layer3[i][1 * 2 + x0][0 * 2 + x1];
x0 += ismax * (1 - x0);
x1 += ismax * (1 - x1);
features->layer4[i][1][0] = features->layer3[i][1 * 2 + x0][0 * 2 + x1];
x0 = 0; x1 = 0;
x1 = features->layer3[i][1 * 2][1 * 2 + 1] > features->layer3[i][1 * 2][1 * 2];
x0 = features->layer3[i][1 * 2 + 1][1 * 2] > features->layer3[i][1 * 2][1 * 2 + x1];
tempD = features->layer3[i][1 * 2 + 1][1 * 2 + 1];
x1 += x0 * (- x1);	
ismax = tempD > features->layer3[i][1 * 2 + x0][1 * 2 + x1];
x0 += ismax * (1 - x0);
x1 += ismax * (1 - x1);
features->layer4[i][1][1] = features->layer3[i][1 * 2 + x0][1 * 2 + x1];
x0 = 0; x1 = 0;
x1 = features->layer3[i][1 * 2][2 * 2 + 1] > features->layer3[i][1 * 2][2 * 2];
x0 = features->layer3[i][1 * 2 + 1][2 * 2] > features->layer3[i][1 * 2][2 * 2 + x1];
tempD = features->layer3[i][1 * 2 + 1][2 * 2 + 1];
x1 += x0 * (- x1);	
ismax = tempD > features->layer3[i][1 * 2 + x0][2 * 2 + x1];
x0 += ismax * (1 - x0);
x1 += ismax * (1 - x1);
features->layer4[i][1][2] = features->layer3[i][1 * 2 + x0][2 * 2 + x1];
x0 = 0; x1 = 0;
x1 = features->layer3[i][1 * 2][3 * 2 + 1] > features->layer3[i][1 * 2][3 * 2];
x0 = features->layer3[i][1 * 2 + 1][3 * 2] > features->layer3[i][1 * 2][3 * 2 + x1];
tempD = features->layer3[i][1 * 2 + 1][3 * 2 + 1];
x1 += x0 * (- x1);	
ismax = tempD > features->layer3[i][1 * 2 + x0][3 * 2 + x1];
x0 += ismax * (1 - x0);
x1 += ismax * (1 - x1);
features->layer4[i][1][3] = features->layer3[i][1 * 2 + x0][3 * 2 + x1];
x0 = 0; x1 = 0;
x1 = features->layer3[i][1 * 2][4 * 2 + 1] > features->layer3[i][1 * 2][4 * 2];
x0 = features->layer3[i][1 * 2 + 1][4 * 2] > features->layer3[i][1 * 2][4 * 2 + x1];
tempD = features->layer3[i][1 * 2 + 1][4 * 2 + 1];
x1 += x0 * (- x1);	
ismax = tempD > features->layer3[i][1 * 2 + x0][4 * 2 + x1];
x0 += ismax * (1 - x0);
x1 += ismax * (1 - x1);
features->layer4[i][1][4] = features->layer3[i][1 * 2 + x0][4 * 2 + x1];
x0 = 0; x1 = 0;
x1 = features->layer3[i][2 * 2][0 * 2 + 1] > features->layer3[i][2 * 2][0 * 2];
x0 = features->layer3[i][2 * 2 + 1][0 * 2] > features->layer3[i][2 * 2][0 * 2 + x1];
tempD = features->layer3[i][2 * 2 + 1][0 * 2 + 1];
x1 += x0 * (- x1);	
ismax = tempD > features->layer3[i][2 * 2 + x0][0 * 2 + x1];
x0 += ismax * (1 - x0);
x1 += ismax * (1 - x1);
features->layer4[i][2][0] = features->layer3[i][2 * 2 + x0][0 * 2 + x1];
x0 = 0; x1 = 0;
x1 = features->layer3[i][2 * 2][1 * 2 + 1] > features->layer3[i][2 * 2][1 * 2];
x0 = features->layer3[i][2 * 2 + 1][1 * 2] > features->layer3[i][2 * 2][1 * 2 + x1];
tempD = features->layer3[i][2 * 2 + 1][1 * 2 + 1];
x1 += x0 * (- x1);	
ismax = tempD > features->layer3[i][2 * 2 + x0][1 * 2 + x1];
x0 += ismax * (1 - x0);
x1 += ismax * (1 - x1);
features->layer4[i][2][1] = features->layer3[i][2 * 2 + x0][1 * 2 + x1];
x0 = 0; x1 = 0;
x1 = features->layer3[i][2 * 2][2 * 2 + 1] > features->layer3[i][2 * 2][2 * 2];
x0 = features->layer3[i][2 * 2 + 1][2 * 2] > features->layer3[i][2 * 2][2 * 2 + x1];
tempD = features->layer3[i][2 * 2 + 1][2 * 2 + 1];
x1 += x0 * (- x1);	
ismax = tempD > features->layer3[i][2 * 2 + x0][2 * 2 + x1];
x0 += ismax * (1 - x0);
x1 += ismax * (1 - x1);
features->layer4[i][2][2] = features->layer3[i][2 * 2 + x0][2 * 2 + x1];
x0 = 0; x1 = 0;
x1 = features->layer3[i][2 * 2][3 * 2 + 1] > features->layer3[i][2 * 2][3 * 2];
x0 = features->layer3[i][2 * 2 + 1][3 * 2] > features->layer3[i][2 * 2][3 * 2 + x1];
tempD = features->layer3[i][2 * 2 + 1][3 * 2 + 1];
x1 += x0 * (- x1);	
ismax = tempD > features->layer3[i][2 * 2 + x0][3 * 2 + x1];
x0 += ismax * (1 - x0);
x1 += ismax * (1 - x1);
features->layer4[i][2][3] = features->layer3[i][2 * 2 + x0][3 * 2 + x1];
x0 = 0; x1 = 0;
x1 = features->layer3[i][2 * 2][4 * 2 + 1] > features->layer3[i][2 * 2][4 * 2];
x0 = features->layer3[i][2 * 2 + 1][4 * 2] > features->layer3[i][2 * 2][4 * 2 + x1];
tempD = features->layer3[i][2 * 2 + 1][4 * 2 + 1];
x1 += x0 * (- x1);	
ismax = tempD > features->layer3[i][2 * 2 + x0][4 * 2 + x1];
x0 += ismax * (1 - x0);
x1 += ismax * (1 - x1);
features->layer4[i][2][4] = features->layer3[i][2 * 2 + x0][4 * 2 + x1];
x0 = 0; x1 = 0;
x1 = features->layer3[i][3 * 2][0 * 2 + 1] > features->layer3[i][3 * 2][0 * 2];
x0 = features->layer3[i][3 * 2 + 1][0 * 2] > features->layer3[i][3 * 2][0 * 2 + x1];
tempD = features->layer3[i][3 * 2 + 1][0 * 2 + 1];
x1 += x0 * (- x1);	
ismax = tempD > features->layer3[i][3 * 2 + x0][0 * 2 + x1];
x0 += ismax * (1 - x0);
x1 += ismax * (1 - x1);
features->layer4[i][3][0] = features->layer3[i][3 * 2 + x0][0 * 2 + x1];
x0 = 0; x1 = 0;
x1 = features->layer3[i][3 * 2][1 * 2 + 1] > features->layer3[i][3 * 2][1 * 2];
x0 = features->layer3[i][3 * 2 + 1][1 * 2] > features->layer3[i][3 * 2][1 * 2 + x1];
tempD = features->layer3[i][3 * 2 + 1][1 * 2 + 1];
x1 += x0 * (- x1);	
ismax = tempD > features->layer3[i][3 * 2 + x0][1 * 2 + x1];
x0 += ismax * (1 - x0);
x1 += ismax * (1 - x1);
features->layer4[i][3][1] = features->layer3[i][3 * 2 + x0][1 * 2 + x1];
x0 = 0; x1 = 0;
x1 = features->layer3[i][3 * 2][2 * 2 + 1] > features->layer3[i][3 * 2][2 * 2];
x0 = features->layer3[i][3 * 2 + 1][2 * 2] > features->layer3[i][3 * 2][2 * 2 + x1];
tempD = features->layer3[i][3 * 2 + 1][2 * 2 + 1];
x1 += x0 * (- x1);	
ismax = tempD > features->layer3[i][3 * 2 + x0][2 * 2 + x1];
x0 += ismax * (1 - x0);
x1 += ismax * (1 - x1);
features->layer4[i][3][2] = features->layer3[i][3 * 2 + x0][2 * 2 + x1];
x0 = 0; x1 = 0;
x1 = features->layer3[i][3 * 2][3 * 2 + 1] > features->layer3[i][3 * 2][3 * 2];
x0 = features->layer3[i][3 * 2 + 1][3 * 2] > features->layer3[i][3 * 2][3 * 2 + x1];
tempD = features->layer3[i][3 * 2 + 1][3 * 2 + 1];
x1 += x0 * (- x1);	
ismax = tempD > features->layer3[i][3 * 2 + x0][3 * 2 + x1];
x0 += ismax * (1 - x0);
x1 += ismax * (1 - x1);
features->layer4[i][3][3] = features->layer3[i][3 * 2 + x0][3 * 2 + x1];
x0 = 0; x1 = 0;
x1 = features->layer3[i][3 * 2][4 * 2 + 1] > features->layer3[i][3 * 2][4 * 2];
x0 = features->layer3[i][3 * 2 + 1][4 * 2] > features->layer3[i][3 * 2][4 * 2 + x1];
tempD = features->layer3[i][3 * 2 + 1][4 * 2 + 1];
x1 += x0 * (- x1);	
ismax = tempD > features->layer3[i][3 * 2 + x0][4 * 2 + x1];
x0 += ismax * (1 - x0);
x1 += ismax * (1 - x1);
features->layer4[i][3][4] = features->layer3[i][3 * 2 + x0][4 * 2 + x1];
x0 = 0; x1 = 0;
x1 = features->layer3[i][4 * 2][0 * 2 + 1] > features->layer3[i][4 * 2][0 * 2];
x0 = features->layer3[i][4 * 2 + 1][0 * 2] > features->layer3[i][4 * 2][0 * 2 + x1];
tempD = features->layer3[i][4 * 2 + 1][0 * 2 + 1];
x1 += x0 * (- x1);	
ismax = tempD > features->layer3[i][4 * 2 + x0][0 * 2 + x1];
x0 += ismax * (1 - x0);
x1 += ismax * (1 - x1);
features->layer4[i][4][0] = features->layer3[i][4 * 2 + x0][0 * 2 + x1];
x0 = 0; x1 = 0;
x1 = features->layer3[i][4 * 2][1 * 2 + 1] > features->layer3[i][4 * 2][1 * 2];
x0 = features->layer3[i][4 * 2 + 1][1 * 2] > features->layer3[i][4 * 2][1 * 2 + x1];
tempD = features->layer3[i][4 * 2 + 1][1 * 2 + 1];
x1 += x0 * (- x1);	
ismax = tempD > features->layer3[i][4 * 2 + x0][1 * 2 + x1];
x0 += ismax * (1 - x0);
x1 += ismax * (1 - x1);
features->layer4[i][4][1] = features->layer3[i][4 * 2 + x0][1 * 2 + x1];
x0 = 0; x1 = 0;
x1 = features->layer3[i][4 * 2][2 * 2 + 1] > features->layer3[i][4 * 2][2 * 2];
x0 = features->layer3[i][4 * 2 + 1][2 * 2] > features->layer3[i][4 * 2][2 * 2 + x1];
tempD = features->layer3[i][4 * 2 + 1][2 * 2 + 1];
x1 += x0 * (- x1);	
ismax = tempD > features->layer3[i][4 * 2 + x0][2 * 2 + x1];
x0 += ismax * (1 - x0);
x1 += ismax * (1 - x1);
features->layer4[i][4][2] = features->layer3[i][4 * 2 + x0][2 * 2 + x1];
x0 = 0; x1 = 0;
x1 = features->layer3[i][4 * 2][3 * 2 + 1] > features->layer3[i][4 * 2][3 * 2];
x0 = features->layer3[i][4 * 2 + 1][3 * 2] > features->layer3[i][4 * 2][3 * 2 + x1];
tempD = features->layer3[i][4 * 2 + 1][3 * 2 + 1];
x1 += x0 * (- x1);	
ismax = tempD > features->layer3[i][4 * 2 + x0][3 * 2 + x1];
x0 += ismax * (1 - x0);
x1 += ismax * (1 - x1);
features->layer4[i][4][3] = features->layer3[i][4 * 2 + x0][3 * 2 + x1];
x0 = 0; x1 = 0;
x1 = features->layer3[i][4 * 2][4 * 2 + 1] > features->layer3[i][4 * 2][4 * 2];
x0 = features->layer3[i][4 * 2 + 1][4 * 2] > features->layer3[i][4 * 2][4 * 2 + x1];
tempD = features->layer3[i][4 * 2 + 1][4 * 2 + 1];
x1 += x0 * (- x1);	
ismax = tempD > features->layer3[i][4 * 2 + x0][4 * 2 + x1];
x0 += ismax * (1 - x0);
x1 += ismax * (1 - x1);
features->layer4[i][4][4] = features->layer3[i][4 * 2 + x0][4 * 2 + x1];
}
for (uint8 x = 0; x < 15; x++)
for (uint8 y = 0; y < 30; y++)
{
temp = features->layer5[4*y][0][0];
temp += features->layer4[x][0][0] * net->weight4_5[x][4*y][0][0]
+ features->layer4[x][0][1] * net->weight4_5[x][4*y][0][1]
+ features->layer4[x][0][2] * net->weight4_5[x][4*y][0][2]
+ features->layer4[x][0][3] * net->weight4_5[x][4*y][0][3]
+ features->layer4[x][0][4] * net->weight4_5[x][4*y][0][4]                                                             
+ features->layer4[x][1][0] * net->weight4_5[x][4*y][1][0]
+ features->layer4[x][1][1] * net->weight4_5[x][4*y][1][1]
+ features->layer4[x][1][2] * net->weight4_5[x][4*y][1][2]
+ features->layer4[x][1][3] * net->weight4_5[x][4*y][1][3]
+ features->layer4[x][1][4] * net->weight4_5[x][4*y][1][4]                                        
+ features->layer4[x][2][0] * net->weight4_5[x][4*y][2][0]
+ features->layer4[x][2][1] * net->weight4_5[x][4*y][2][1]
+ features->layer4[x][2][2] * net->weight4_5[x][4*y][2][2]
+ features->layer4[x][2][3] * net->weight4_5[x][4*y][2][3]
+ features->layer4[x][2][4] * net->weight4_5[x][4*y][2][4]                                      
+ features->layer4[x][3][0] * net->weight4_5[x][4*y][3][0]
+ features->layer4[x][3][1] * net->weight4_5[x][4*y][3][1]
+ features->layer4[x][3][2] * net->weight4_5[x][4*y][3][2]
+ features->layer4[x][3][3] * net->weight4_5[x][4*y][3][3]
+ features->layer4[x][3][4] * net->weight4_5[x][4*y][3][4]                                    
+ features->layer4[x][4][0] * net->weight4_5[x][4*y][4][0]
+ features->layer4[x][4][1] * net->weight4_5[x][4*y][4][1]
+ features->layer4[x][4][2] * net->weight4_5[x][4*y][4][2]
+ features->layer4[x][4][3] * net->weight4_5[x][4*y][4][3]
+ features->layer4[x][4][4] * net->weight4_5[x][4*y][4][4];
features->layer5[4*y][0][0] = temp;
temp = features->layer5[4*y+1][0][0];
temp += features->layer4[x][0][0] * net->weight4_5[x][4*y+1][0][0]
+ features->layer4[x][0][1] * net->weight4_5[x][4*y+1][0][1]
+ features->layer4[x][0][2] * net->weight4_5[x][4*y+1][0][2]
+ features->layer4[x][0][3] * net->weight4_5[x][4*y+1][0][3]
+ features->layer4[x][0][4] * net->weight4_5[x][4*y+1][0][4]                                                             
+ features->layer4[x][1][0] * net->weight4_5[x][4*y+1][1][0]
+ features->layer4[x][1][1] * net->weight4_5[x][4*y+1][1][1]
+ features->layer4[x][1][2] * net->weight4_5[x][4*y+1][1][2]
+ features->layer4[x][1][3] * net->weight4_5[x][4*y+1][1][3]
+ features->layer4[x][1][4] * net->weight4_5[x][4*y+1][1][4]                                        
+ features->layer4[x][2][0] * net->weight4_5[x][4*y+1][2][0]
+ features->layer4[x][2][1] * net->weight4_5[x][4*y+1][2][1]
+ features->layer4[x][2][2] * net->weight4_5[x][4*y+1][2][2]
+ features->layer4[x][2][3] * net->weight4_5[x][4*y+1][2][3]
+ features->layer4[x][2][4] * net->weight4_5[x][4*y+1][2][4]                                      
+ features->layer4[x][3][0] * net->weight4_5[x][4*y+1][3][0]
+ features->layer4[x][3][1] * net->weight4_5[x][4*y+1][3][1]
+ features->layer4[x][3][2] * net->weight4_5[x][4*y+1][3][2]
+ features->layer4[x][3][3] * net->weight4_5[x][4*y+1][3][3]
+ features->layer4[x][3][4] * net->weight4_5[x][4*y+1][3][4]                                    
+ features->layer4[x][4][0] * net->weight4_5[x][4*y+1][4][0]
+ features->layer4[x][4][1] * net->weight4_5[x][4*y+1][4][1]
+ features->layer4[x][4][2] * net->weight4_5[x][4*y+1][4][2]
+ features->layer4[x][4][3] * net->weight4_5[x][4*y+1][4][3]
+ features->layer4[x][4][4] * net->weight4_5[x][4*y+1][4][4];
features->layer5[4*y+1][0][0] = temp;
temp = features->layer5[4*y+2][0][0];
temp += features->layer4[x][0][0] * net->weight4_5[x][4*y+2][0][0]
+ features->layer4[x][0][1] * net->weight4_5[x][4*y+2][0][1]
+ features->layer4[x][0][2] * net->weight4_5[x][4*y+2][0][2]
+ features->layer4[x][0][3] * net->weight4_5[x][4*y+2][0][3]
+ features->layer4[x][0][4] * net->weight4_5[x][4*y+2][0][4]                                                             
+ features->layer4[x][1][0] * net->weight4_5[x][4*y+2][1][0]
+ features->layer4[x][1][1] * net->weight4_5[x][4*y+2][1][1]
+ features->layer4[x][1][2] * net->weight4_5[x][4*y+2][1][2]
+ features->layer4[x][1][3] * net->weight4_5[x][4*y+2][1][3]
+ features->layer4[x][1][4] * net->weight4_5[x][4*y+2][1][4]                                        
+ features->layer4[x][2][0] * net->weight4_5[x][4*y+2][2][0]
+ features->layer4[x][2][1] * net->weight4_5[x][4*y+2][2][1]
+ features->layer4[x][2][2] * net->weight4_5[x][4*y+2][2][2]
+ features->layer4[x][2][3] * net->weight4_5[x][4*y+2][2][3]
+ features->layer4[x][2][4] * net->weight4_5[x][4*y+2][2][4]                                      
+ features->layer4[x][3][0] * net->weight4_5[x][4*y+2][3][0]
+ features->layer4[x][3][1] * net->weight4_5[x][4*y+2][3][1]
+ features->layer4[x][3][2] * net->weight4_5[x][4*y+2][3][2]
+ features->layer4[x][3][3] * net->weight4_5[x][4*y+2][3][3]
+ features->layer4[x][3][4] * net->weight4_5[x][4*y+2][3][4]                                    
+ features->layer4[x][4][0] * net->weight4_5[x][4*y+2][4][0]
+ features->layer4[x][4][1] * net->weight4_5[x][4*y+2][4][1]
+ features->layer4[x][4][2] * net->weight4_5[x][4*y+2][4][2]
+ features->layer4[x][4][3] * net->weight4_5[x][4*y+2][4][3]
+ features->layer4[x][4][4] * net->weight4_5[x][4*y+2][4][4];
features->layer5[4*y+2][0][0] = temp;
temp = features->layer5[4*y+3][0][0];
temp += features->layer4[x][0][0] * net->weight4_5[x][4*y+3][0][0]
+ features->layer4[x][0][1] * net->weight4_5[x][4*y+3][0][1]
+ features->layer4[x][0][2] * net->weight4_5[x][4*y+3][0][2]
+ features->layer4[x][0][3] * net->weight4_5[x][4*y+3][0][3]
+ features->layer4[x][0][4] * net->weight4_5[x][4*y+3][0][4]                                                             
+ features->layer4[x][1][0] * net->weight4_5[x][4*y+3][1][0]
+ features->layer4[x][1][1] * net->weight4_5[x][4*y+3][1][1]
+ features->layer4[x][1][2] * net->weight4_5[x][4*y+3][1][2]
+ features->layer4[x][1][3] * net->weight4_5[x][4*y+3][1][3]
+ features->layer4[x][1][4] * net->weight4_5[x][4*y+3][1][4]                                        
+ features->layer4[x][2][0] * net->weight4_5[x][4*y+3][2][0]
+ features->layer4[x][2][1] * net->weight4_5[x][4*y+3][2][1]
+ features->layer4[x][2][2] * net->weight4_5[x][4*y+3][2][2]
+ features->layer4[x][2][3] * net->weight4_5[x][4*y+3][2][3]
+ features->layer4[x][2][4] * net->weight4_5[x][4*y+3][2][4]                                      
+ features->layer4[x][3][0] * net->weight4_5[x][4*y+3][3][0]
+ features->layer4[x][3][1] * net->weight4_5[x][4*y+3][3][1]
+ features->layer4[x][3][2] * net->weight4_5[x][4*y+3][3][2]
+ features->layer4[x][3][3] * net->weight4_5[x][4*y+3][3][3]
+ features->layer4[x][3][4] * net->weight4_5[x][4*y+3][3][4]                                    
+ features->layer4[x][4][0] * net->weight4_5[x][4*y+3][4][0]
+ features->layer4[x][4][1] * net->weight4_5[x][4*y+3][4][1]
+ features->layer4[x][4][2] * net->weight4_5[x][4*y+3][4][2]
+ features->layer4[x][4][3] * net->weight4_5[x][4*y+3][4][3]
+ features->layer4[x][4][4] * net->weight4_5[x][4*y+3][4][4];
features->layer5[4*y+3][0][0] = temp;
}
for (uint8 y = 0; y < 120; y++)
{
temp = features->layer5[y][0][0];
temp += features->layer4[15][0][0] * net->weight4_5[15][y][0][0]
+ features->layer4[15][0][1] * net->weight4_5[15][y][0][1]
+ features->layer4[15][0][2] * net->weight4_5[15][y][0][2]
+ features->layer4[15][0][3] * net->weight4_5[15][y][0][3]
+ features->layer4[15][0][4] * net->weight4_5[15][y][0][4]
+ features->layer4[15][1][0] * net->weight4_5[15][y][1][0]
+ features->layer4[15][1][1] * net->weight4_5[15][y][1][1]
+ features->layer4[15][1][2] * net->weight4_5[15][y][1][2]
+ features->layer4[15][1][3] * net->weight4_5[15][y][1][3]
+ features->layer4[15][1][4] * net->weight4_5[15][y][1][4]
+ features->layer4[15][2][0] * net->weight4_5[15][y][2][0]
+ features->layer4[15][2][1] * net->weight4_5[15][y][2][1]
+ features->layer4[15][2][2] * net->weight4_5[15][y][2][2]
+ features->layer4[15][2][3] * net->weight4_5[15][y][2][3]
+ features->layer4[15][2][4] * net->weight4_5[15][y][2][4]
+ features->layer4[15][3][0] * net->weight4_5[15][y][3][0]
+ features->layer4[15][3][1] * net->weight4_5[15][y][3][1]
+ features->layer4[15][3][2] * net->weight4_5[15][y][3][2]
+ features->layer4[15][3][3] * net->weight4_5[15][y][3][3]
+ features->layer4[15][3][4] * net->weight4_5[15][y][3][4]
+ features->layer4[15][4][0] * net->weight4_5[15][y][4][0]
+ features->layer4[15][4][1] * net->weight4_5[15][y][4][1]
+ features->layer4[15][4][2] * net->weight4_5[15][y][4][2]
+ features->layer4[15][4][3] * net->weight4_5[15][y][4][3]
+ features->layer4[15][4][4] * net->weight4_5[15][y][4][4];
features->layer5[y][0][0] = temp;
temp += net->bias4_5[y];
if (temp < 0)
temp = 0;
features->layer5[y][0][0] = temp;
}
for (int x = 118; x > -1; x--)
{
temp = features->layer5[x][0][0];
for (int y = 255; y > -1; y--)
{
features->layer6[y] += temp * net->weight5_6[x][y];
}
}
temp = features->layer5[119][0][0];
for (int y = 255; y > -1; y--)
{
features->layer6[y] += temp * net->weight5_6[119][y] + net->bias5_6[y];
if (features->layer6[y] < 0)
features->layer6[y] = 0;
}
for (int x = 0; x < 128; x++)
{
temp = features->layer6[2*x];
features->output[0] += temp * net->weight6_7[2*x][0];
features->output[1] += temp * net->weight6_7[2*x][1];
features->output[2] += temp * net->weight6_7[2*x][2];
features->output[3] += temp * net->weight6_7[2*x][3];
features->output[4] += temp * net->weight6_7[2*x][4];
features->output[5] += temp * net->weight6_7[2*x][5];
features->output[6] += temp * net->weight6_7[2*x][6];
features->output[7] += temp * net->weight6_7[2*x][7];
features->output[8] += temp * net->weight6_7[2*x][8];
features->output[9] += temp * net->weight6_7[2*x][9];
temp = features->layer6[2*x+1];
features->output[0] += temp * net->weight6_7[2*x+1][0];
features->output[1] += temp * net->weight6_7[2*x+1][1];
features->output[2] += temp * net->weight6_7[2*x+1][2];
features->output[3] += temp * net->weight6_7[2*x+1][3];
features->output[4] += temp * net->weight6_7[2*x+1][4];
features->output[5] += temp * net->weight6_7[2*x+1][5];
features->output[6] += temp * net->weight6_7[2*x+1][6];
features->output[7] += temp * net->weight6_7[2*x+1][7];
features->output[8] += temp * net->weight6_7[2*x+1][8];
features->output[9] += temp * net->weight6_7[2*x+1][9];
}
features->output[0] = features->output[0] + net->bias6_7[0];
features->output[1] = features->output[1] + net->bias6_7[1];
features->output[2] = features->output[2] + net->bias6_7[2];
features->output[3] = features->output[3] + net->bias6_7[3];
features->output[4] = features->output[4] + net->bias6_7[4];
features->output[5] = features->output[5] + net->bias6_7[5];
features->output[6] = features->output[6] + net->bias6_7[6];
features->output[7] = features->output[7] + net->bias6_7[7];
features->output[8] = features->output[8] + net->bias6_7[8];
features->output[9] = features->output[9] + net->bias6_7[9];
}
int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[])
{
FILE *fp_image = fopen(data_file, "rb");
FILE *fp_label = fopen(label_file, "rb");
if (!fp_image||!fp_label) return 1;
fseek(fp_image, 16, SEEK_SET);
fseek(fp_label, 8, SEEK_SET);
fread(data, sizeof(*data)*count, 1, fp_image);
fread(label,count, 1, fp_label);
fclose(fp_image);
fclose(fp_label);
return 0;
}
int main()
{
int i, l, p;
int truePredict = 0;
struct timeval t1, t2;
image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));
if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
{
printf("ERROR!!!\nDataset File Not Find!\n");
free(test_data);
free(test_label);
return 1;
}
Net *net = (Net *)malloc(sizeof(Net));
FILE *fp = fopen("trained.model", "rb");
if (!fp){
printf("ERROR!!!\n");
return 1;
}
fread(net, sizeof(Net), 1, fp);
fclose(fp);
printf("Please wait.....\n");
gettimeofday(&t1, NULL);
#pragma omp parallel for num_threads(NUM_THREAD) private(l) reduction(+:truePredict) schedule(static, COUNT_TEST/NUM_THREAD)
for (int i = 0; i < COUNT_TEST; i++)
{
l = test_label[i];
Feature features = { 0 };
#pragma omp simd
for (int j = 0; j < 28; j++)
{
features.input[0][j + PADDING][2] = ((_Float64)(test_data[i][j][0]) - 128) / 256;
features.input[0][j + PADDING][3] = ((_Float64)(test_data[i][j][1]) - 128) / 256;
features.input[0][j + PADDING][4] = ((_Float64)(test_data[i][j][2]) - 128) / 256;
features.input[0][j + PADDING][5] = ((_Float64)(test_data[i][j][3]) - 128) / 256;
features.input[0][j + PADDING][6] = ((_Float64)(test_data[i][j][4]) - 128) / 256;
features.input[0][j + PADDING][7] = ((_Float64)(test_data[i][j][5]) - 128) / 256;
features.input[0][j + PADDING][8] = ((_Float64)(test_data[i][j][6]) - 128) / 256;
features.input[0][j + PADDING][9] = ((_Float64)(test_data[i][j][7]) - 128) / 256;
features.input[0][j + PADDING][10] = ((_Float64)(test_data[i][j][8]) - 128) / 256;
features.input[0][j + PADDING][11] = ((_Float64)(test_data[i][j][9]) - 128) / 256;
features.input[0][j + PADDING][12] = ((_Float64)(test_data[i][j][10]) - 128) / 256;
features.input[0][j + PADDING][13] = ((_Float64)(test_data[i][j][11]) - 128) / 256;
features.input[0][j + PADDING][14] = ((_Float64)(test_data[i][j][12]) - 128) / 256;
features.input[0][j + PADDING][15] = ((_Float64)(test_data[i][j][13]) - 128) / 256;
features.input[0][j + PADDING][16] = ((_Float64)(test_data[i][j][14]) - 128) / 256;
features.input[0][j + PADDING][17] = ((_Float64)(test_data[i][j][15]) - 128) / 256;
features.input[0][j + PADDING][18] = ((_Float64)(test_data[i][j][16]) - 128) / 256;
features.input[0][j + PADDING][19] = ((_Float64)(test_data[i][j][17]) - 128) / 256;
features.input[0][j + PADDING][20] = ((_Float64)(test_data[i][j][18]) - 128) / 256;
features.input[0][j + PADDING][21] = ((_Float64)(test_data[i][j][19]) - 128) / 256;
features.input[0][j + PADDING][22] = ((_Float64)(test_data[i][j][20]) - 128) / 256;
features.input[0][j + PADDING][23] = ((_Float64)(test_data[i][j][21]) - 128) / 256;
features.input[0][j + PADDING][24] = ((_Float64)(test_data[i][j][22]) - 128) / 256;
features.input[0][j + PADDING][25] = ((_Float64)(test_data[i][j][23]) - 128) / 256;
features.input[0][j + PADDING][26] = ((_Float64)(test_data[i][j][24]) - 128) / 256;
features.input[0][j + PADDING][27] = ((_Float64)(test_data[i][j][25]) - 128) / 256;
features.input[0][j + PADDING][28] = ((_Float64)(test_data[i][j][26]) - 128) / 256;
features.input[0][j + PADDING][29] = ((_Float64)(test_data[i][j][27]) - 128) / 256;
}
forward(net, &features);
_Float64 *output = (_Float64 *)features.output;
int result = 0;
_Float64 maxvalue = *output;
if (output[1] > maxvalue)
{
maxvalue = output[1];
result = 1;
}
if (output[2] > maxvalue)
{
maxvalue = output[2];
result = 2;
}
if (output[3] > maxvalue)
{
maxvalue = output[3];
result = 3;
}
if (output[4] > maxvalue)
{
maxvalue = output[4];
result = 4;
}
if (output[5] > maxvalue)
{
maxvalue = output[5];
result = 5;
}
if (output[6] > maxvalue)
{
maxvalue = output[6];
result = 6;
}
if (output[7] > maxvalue)
{
maxvalue = output[7];
result = 7;
}
if (output[8] > maxvalue)
{
maxvalue = output[8];
result = 8;
}
if (output[9] > maxvalue)
{
maxvalue = output[9];
result = 9;
}
p = result;
if (l == p)
truePredict++;
}
gettimeofday(&t2, NULL);
printf("%d / %d\n", truePredict, COUNT_TEST);
printf("%ld seconds and %ld microseconds\n",(long)(t2.tv_sec - t1.tv_sec), (long)(t2.tv_usec - t1.tv_usec));
free(net);
free(test_data);
free(test_label);
return 0;
}
