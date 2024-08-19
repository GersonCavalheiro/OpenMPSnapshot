




#ifndef _ACF_KERNEL_H_
#define _ACF_KERNEL_H_

#define LOG2_GRID_SIZE 14

#include "model_io.cpp"
#include "histogram_kernel.cpp"

void ACFKernelSymm(cartesian g_idata1, unsigned int*__restrict g_odata,
nd_item<3> item, double3 *__restrict sdata,
double *__restrict binbounds)
{
int tx = (item.get_group(2) << 7) + item.get_local_id(2);
int by = (item.get_group(1) << 7);
if (item.get_group(2) < item.get_group(1)) { 
by <<= (LOG2_GRID_SIZE - 2);
by += tx;
#pragma unroll
for(int i=0; i<128; i+=4) {
g_odata[by+(i<<(LOG2_GRID_SIZE - 2))] = 2088533116; 
}
} else if (item.get_group(2) >
item.get_group(1)) { 
double temp;
unsigned int temp2;
double3 vec1, vec2;

vec1.x() = g_idata1.x[tx];
vec1.y() = g_idata1.y[tx];
vec1.z() = g_idata1.z[tx];
sdata[item.get_local_id(2)].x() = g_idata1.x[by + item.get_local_id(2)];
sdata[item.get_local_id(2)].y() = g_idata1.y[by + item.get_local_id(2)];
sdata[item.get_local_id(2)].z() = g_idata1.z[by + item.get_local_id(2)];

item.barrier(access::fence_space::local_space);

by <<= (LOG2_GRID_SIZE - 2);
by += tx;

#pragma unroll
for(int i=0; i<128; i+=4) {
temp2 = 0;
#pragma unroll
for(int j=0; j<4; j++) {
vec2 = sdata[i+j];
temp = vec1.x() * vec2.x() + vec1.y() * vec2.y() + vec1.z() * vec2.z();
if(temp < binbounds[30]) temp2 += (124<<(j<<3));
else if(temp < binbounds[29]) temp2 += (120<<(j<<3));
else if(temp < binbounds[28]) temp2 += (116<<(j<<3));
else if(temp < binbounds[27]) temp2 += (112<<(j<<3));
else if(temp < binbounds[26]) temp2 += (108<<(j<<3));
else if(temp < binbounds[25]) temp2 += (104<<(j<<3));
else if(temp < binbounds[24]) temp2 += (100<<(j<<3));
else if(temp < binbounds[23]) temp2 += (96<<(j<<3));
else if(temp < binbounds[22]) temp2 += (92<<(j<<3));
else if(temp < binbounds[21]) temp2 += (88<<(j<<3));
else if(temp < binbounds[20]) temp2 += (84<<(j<<3));
else if(temp < binbounds[19]) temp2 += (80<<(j<<3));
else if(temp < binbounds[18]) temp2 += (76<<(j<<3));
else if(temp < binbounds[17]) temp2 += (72<<(j<<3));
else if(temp < binbounds[16]) temp2 += (68<<(j<<3));
else if(temp < binbounds[15]) temp2 += (64<<(j<<3));
else if(temp < binbounds[14]) temp2 += (60<<(j<<3));
else if(temp < binbounds[13]) temp2 += (56<<(j<<3));
else if(temp < binbounds[12]) temp2 += (52<<(j<<3));
else if(temp < binbounds[11]) temp2 += (48<<(j<<3));
else if(temp < binbounds[10]) temp2 += (44<<(j<<3));
else if(temp < binbounds[9]) temp2 += (40<<(j<<3));
else if(temp < binbounds[8]) temp2 += (36<<(j<<3));
else if(temp < binbounds[7]) temp2 += (32<<(j<<3));
else if(temp < binbounds[6]) temp2 += (28<<(j<<3));
else if(temp < binbounds[5]) temp2 += (24<<(j<<3));
else if(temp < binbounds[4]) temp2 += (20<<(j<<3));
else if(temp < binbounds[3]) temp2 += (16<<(j<<3));
else if(temp < binbounds[2]) temp2 += (12<<(j<<3));
else if(temp < binbounds[1]) temp2 += (8<<(j<<3));
else if(temp < binbounds[0]) temp2 += (4<<(j<<3));
else temp2 += (0<<(j<<3));
}
g_odata[by+(i<<(LOG2_GRID_SIZE - 2))] = temp2;
}
}
else {  
double temp;
unsigned int temp2;
double3 vec1, vec2;

vec1.x() = g_idata1.x[tx];
vec1.y() = g_idata1.y[tx];
vec1.z() = g_idata1.z[tx];
sdata[item.get_local_id(2)].x() = g_idata1.x[by + item.get_local_id(2)];
sdata[item.get_local_id(2)].y() = g_idata1.y[by + item.get_local_id(2)];
sdata[item.get_local_id(2)].z() = g_idata1.z[by + item.get_local_id(2)];

item.barrier(access::fence_space::local_space);

by <<= (LOG2_GRID_SIZE - 2);
by += tx;

#pragma unroll
for(int i=0; i<128; i+=4) {
temp2 = 0;
#pragma unroll
for(int j=0; j<4; j++) {
if (item.get_local_id(2) <= i + j) temp2 += (124 << (j << 3));
else { 
vec2 = sdata[i+j];
temp =
vec1.x() * vec2.x() + vec1.y() * vec2.y() + vec1.z() * vec2.z();
if(temp < binbounds[30]) temp2 += (124<<(j<<3));
else if(temp < binbounds[29]) temp2 += (120<<(j<<3));
else if(temp < binbounds[28]) temp2 += (116<<(j<<3));
else if(temp < binbounds[27]) temp2 += (112<<(j<<3));
else if(temp < binbounds[26]) temp2 += (108<<(j<<3));
else if(temp < binbounds[25]) temp2 += (104<<(j<<3));
else if(temp < binbounds[24]) temp2 += (100<<(j<<3));
else if(temp < binbounds[23]) temp2 += (96<<(j<<3));
else if(temp < binbounds[22]) temp2 += (92<<(j<<3));
else if(temp < binbounds[21]) temp2 += (88<<(j<<3));
else if(temp < binbounds[20]) temp2 += (84<<(j<<3));
else if(temp < binbounds[19]) temp2 += (80<<(j<<3));
else if(temp < binbounds[18]) temp2 += (76<<(j<<3));
else if(temp < binbounds[17]) temp2 += (72<<(j<<3));
else if(temp < binbounds[16]) temp2 += (68<<(j<<3));
else if(temp < binbounds[15]) temp2 += (64<<(j<<3));
else if(temp < binbounds[14]) temp2 += (60<<(j<<3));
else if(temp < binbounds[13]) temp2 += (56<<(j<<3));
else if(temp < binbounds[12]) temp2 += (52<<(j<<3));
else if(temp < binbounds[11]) temp2 += (48<<(j<<3));
else if(temp < binbounds[10]) temp2 += (44<<(j<<3));
else if(temp < binbounds[9]) temp2 += (40<<(j<<3));
else if(temp < binbounds[8]) temp2 += (36<<(j<<3));
else if(temp < binbounds[7]) temp2 += (32<<(j<<3));
else if(temp < binbounds[6]) temp2 += (28<<(j<<3));
else if(temp < binbounds[5]) temp2 += (24<<(j<<3));
else if(temp < binbounds[4]) temp2 += (20<<(j<<3));
else if(temp < binbounds[3]) temp2 += (16<<(j<<3));
else if(temp < binbounds[2]) temp2 += (12<<(j<<3));
else if(temp < binbounds[1]) temp2 += (8<<(j<<3));
else if(temp < binbounds[0]) temp2 += (4<<(j<<3));
else temp2 += (0<<(j<<3));
}
}
g_odata[by+(i<<(LOG2_GRID_SIZE - 2))] = temp2;
}
}
}


void ACFKernel(cartesian g_idata1, cartesian g_idata2, unsigned int*__restrict g_odata,
nd_item<3> item, double3 *__restrict sdata, double *__restrict binbounds) 
{
double temp;
unsigned int temp2;
double3 vec1, vec2;
int tx = (item.get_group(2) << 7) + item.get_local_id(2);
int by = (item.get_group(1) << 7);

vec1.x() = g_idata2.x[tx];
vec1.y() = g_idata2.y[tx];
vec1.z() = g_idata2.z[tx];
sdata[item.get_local_id(2)].x() = g_idata1.x[by + item.get_local_id(2)];
sdata[item.get_local_id(2)].y() = g_idata1.y[by + item.get_local_id(2)];
sdata[item.get_local_id(2)].z() = g_idata1.z[by + item.get_local_id(2)];

item.barrier(access::fence_space::local_space);

by <<= (LOG2_GRID_SIZE - 2);
by += tx;

#pragma unroll
for(int i=0; i<128; i+=4) {   
temp2 = 0;
#pragma unroll
for(int j=0; j<4; j++) {    
vec2 = sdata[i+j];
temp = vec1.x() * vec2.x() + vec1.y() * vec2.y() + vec1.z() * vec2.z();
if(temp < binbounds[30]) temp2 += (124<<(j<<3));
else if(temp < binbounds[29]) temp2 += (120<<(j<<3));
else if(temp < binbounds[28]) temp2 += (116<<(j<<3));
else if(temp < binbounds[27]) temp2 += (112<<(j<<3));
else if(temp < binbounds[26]) temp2 += (108<<(j<<3));
else if(temp < binbounds[25]) temp2 += (104<<(j<<3));
else if(temp < binbounds[24]) temp2 += (100<<(j<<3));
else if(temp < binbounds[23]) temp2 += (96<<(j<<3));
else if(temp < binbounds[22]) temp2 += (92<<(j<<3));
else if(temp < binbounds[21]) temp2 += (88<<(j<<3));
else if(temp < binbounds[20]) temp2 += (84<<(j<<3));
else if(temp < binbounds[19]) temp2 += (80<<(j<<3));
else if(temp < binbounds[18]) temp2 += (76<<(j<<3));
else if(temp < binbounds[17]) temp2 += (72<<(j<<3));
else if(temp < binbounds[16]) temp2 += (68<<(j<<3));
else if(temp < binbounds[15]) temp2 += (64<<(j<<3));
else if(temp < binbounds[14]) temp2 += (60<<(j<<3));
else if(temp < binbounds[13]) temp2 += (56<<(j<<3));
else if(temp < binbounds[12]) temp2 += (52<<(j<<3));
else if(temp < binbounds[11]) temp2 += (48<<(j<<3));
else if(temp < binbounds[10]) temp2 += (44<<(j<<3));
else if(temp < binbounds[9]) temp2 += (40<<(j<<3));
else if(temp < binbounds[8]) temp2 += (36<<(j<<3));
else if(temp < binbounds[7]) temp2 += (32<<(j<<3));
else if(temp < binbounds[6]) temp2 += (28<<(j<<3));
else if(temp < binbounds[5]) temp2 += (24<<(j<<3));
else if(temp < binbounds[4]) temp2 += (20<<(j<<3));
else if(temp < binbounds[3]) temp2 += (16<<(j<<3));
else if(temp < binbounds[2]) temp2 += (12<<(j<<3));
else if(temp < binbounds[1]) temp2 += (8<<(j<<3));
else if(temp < binbounds[0]) temp2 += (4<<(j<<3));
else temp2 += (0<<(j<<3));
}
g_odata[by+(i<<(LOG2_GRID_SIZE - 2))] = temp2;
}
}

#endif
