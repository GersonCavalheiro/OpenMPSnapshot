#ifndef RESOUTION
#define RESOLUTION 1
#endif
#ifndef INV_RESOLUTION
#define INV_RESOLUTION 1
#endif
#ifndef RADIUS
#define RADIUS 1
#endif
#ifndef RADIUS_FINAL
#define RADIUS_FINAL 1.001f
#endif
#if defined (DOUBLE_FP)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

typedef struct {
#if defined (DOUBLE_FP)
double data[3][3];
#else
float data[3][3];
#endif
} Mat33;

#if defined (DOUBLE_FP)
typedef double Vec3[3];
#else
typedef float Vec3[3];
#endif

typedef struct {
Mat33 invCovariance;
Vec3 mean;
int first;
} Voxel;

typedef struct {
Mat33 invCovariance;
Vec3 mean;
int point;
} PointVoxel;

typedef struct {
float data[4];
} PointXYZI;

typedef struct {
float x;
float y;
float z;
int iNext;
} PointQueue;


__kernel
void __attribute__ ((reqd_work_group_size(NUMWORKITEMS_PER_WORKGROUP,1,1)))
findMinMax(
__global const PointXYZI* restrict input,
int input_size,
__global PointXYZI* restrict gmins,
__global PointXYZI* restrict gmaxs
) {
int gid   = get_global_id(0);
int lid   = get_local_id(0);
int wgid  = get_group_id(0);
int lsize = get_local_size(0);

__local PointXYZI lmins [NUMWORKITEMS_PER_WORKGROUP];
__local PointXYZI lmaxs [NUMWORKITEMS_PER_WORKGROUP];

float mymin [3];
float mymax [3];

for (char n = 0; n < 3; n++) {
lmins [lid].data[n] = INFINITY;
lmaxs [lid].data[n] = - INFINITY;
mymin[n] = INFINITY;
mymax[n] = -INFINITY;
}

int num_wg = get_num_groups(0); 

int num_elems_per_wg =  lsize;

int offset_wg = num_elems_per_wg * wgid; 
int offset_wi = offset_wg + lid;         

int upper_bound = num_elems_per_wg * (wgid + 1);

PointXYZI temp;
if (offset_wi < input_size) {
for (int i = offset_wi; i < upper_bound; i += lsize) {
temp = input [i];
for (char n = 0; n < 3; n++) {
if (temp.data[n] < mymin[n]) {
mymin[n] = temp.data[n];
}

if (temp.data[n] > mymax[n]) {
mymax[n] = temp.data[n];
}
}
}

for (char n = 0; n < 3; n++) {
lmins[lid].data[n] = mymin[n];
lmaxs[lid].data[n] = mymax[n];
}
}
barrier(CLK_LOCAL_MEM_FENCE);
lsize = lsize >> 1; 
while (lsize > 0) {
if (lid < lsize) {
for (char n = 0; n < 3; n++) {
if (lmins[lid].data[n] > lmins[lid+lsize].data[n]) {
lmins[lid].data[n] = lmins[lid+lsize].data[n];
}
if (lmaxs[lid].data[n] < lmaxs[lid+lsize].data[n]) {
lmaxs[lid].data[n] = lmaxs[lid+lsize].data[n];
}
}
}
lsize = lsize >> 1; 
barrier(CLK_LOCAL_MEM_FENCE);
}
if (lid == 0) {
gmins [wgid] = lmins[0];
gmaxs [wgid] = lmaxs[0];
}
}
