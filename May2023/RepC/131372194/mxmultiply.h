#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <arm_neon.h>
#define NUMTHREADS                             4
#define VECREG_LEN                             4 
#define VECLOD(ptr)                            vld1q_f32(ptr) 
#define VMULSC(vecreg, scalar)                 vmulq_n_f32(vecreg, scalar)
#define VMULAC(vecreg1, vecreg2, vecreg3)      vmlaq_f32(vecreg1, vecreg2, vecreg3)
#define VEXTLN(vecreg, lane)                   vgetq_lane_f32(vecreg, lane)
void mxinitf(size_t len, float *mx, size_t mod)
{
size_t i;
for(i = 0; i < (len * len); i++)
*(mx + i) = (float)((i * i) % mod);
}
void rmxinitf(size_t len, float *mx, float ubound)
{
size_t i;
srand(time(0));
for(i = 0; i < (len * len); i++)
*(mx + i) = ((float)rand() / (float)RAND_MAX) * ubound;
}
void printmxf(size_t ncols, size_t nrows, float *mx, char omode) 
{
size_t i, j;
if(omode == 'c')
{
for(i = 0; i < ncols; i++)
{    
for(j = 0; j < nrows; j++) 
printf("%-10.3f", *(mx + (nrows * i) + j));
printf("\n");
}
}
else if(omode == 'r')
{
for(i = 0; i < nrows; i++)
{    
for(j = 0; j < ncols; j++) 
printf("%-10.3f", *(mx + (nrows * j) + i));
printf("\n");
}
}
}
void fprintmxf(size_t ncols, size_t nrows, float *mx, char omode, char *str) 
{
size_t i, j;
FILE *file = fopen(str, "w");
if(omode == 'c')
{
for(i = 0; i < ncols; i++)
{    
for(j = 0; j < nrows; j++) 
fprintf(file, "%-10.3f", *(mx + (nrows * i) + j));
fprintf(file, "\n");
}
}
else if(omode == 'r')
{
for(i = 0; i < nrows; i++)
{    
for(j = 0; j < ncols; j++) 
fprintf(file, "%-10.3f", *(mx + (nrows * j) + i));
fprintf(file, "\n");
}
}
fclose(file);
}
void mxtransposef(size_t len, float *mx)
{
size_t i, j;
float tmp;
for(i = 0; i < len; i++)
{
for(j = (i + 1); j < len; j++)
{
tmp = *(mx + (len * j) + i);
*(mx + (len * j) + i) = *(mx + (len * i) + j);
*(mx + (len * i) + j) = tmp;
}
}
}
size_t gsif(size_t len, size_t task_id, size_t num_tasks)
{
return ((task_id * len) / num_tasks);
}
size_t geif(size_t len, size_t task_id, size_t num_tasks)
{
return (((task_id + 1) * len) / num_tasks);
}
void mxmultiplyf(size_t len, float *mxa, float *mxb, float *mxc) 
{
size_t i, j, k;
for(i = 0; i < len; i++)
{
for(j = 0; j < len; j++)
{
for(k = 0; k < len; k++)
(*(mxc + (len * i) + k)) += (*(mxa + (len * j) + k)) * (*(mxb + (len * i) + j));
}  
}
}
void mmxmultiplyf(size_t len, float *mxa, float *mxb, float *mxc) 
{
size_t i, j, k;
#pragma omp parallel num_threads(NUMTHREADS)
{
#pragma omp for private(i, j, k)
for(i = 0; i < len; i++)
{
for(j = 0; j < len; j++)
{
for(k = 0; k < len; k++)
(*(mxc + (len * i) + k)) += (*(mxa + (len * j) + k)) * (*(mxb + (len * i) + j));
}  
}
}
}
void pmxmultiplyf(size_t len, float *mxa, float *mxb, float *mxc, size_t task_id, size_t num_tasks)
{
size_t i, j, k, si, ei;
si = (task_id * len) / num_tasks; 
ei = ((task_id + 1) * len) / num_tasks;
for(i = 0; i < len; i++)
{
for(j = si; j < ei; j++)
{
for(k = 0; k < len; k++)
(*(mxc + (len * i) + k)) += (*(mxa + (len * j) + k)) * (*(mxb + (len * i) + j));
}  
}
}
float *pmxmultiplyfs(size_t ncolsmxa, size_t nrowsmxa, float *mxa, size_t ncolsmxb, size_t nrowsmxb, float *mxb) 
{
size_t i, j, k;
float *rmx = calloc(nrowsmxa * ncolsmxb, sizeof(float));
for(i = 0; i < ncolsmxa; i++)
{
for(j = 0; j < ncolsmxb; j++)
{
for(k = 0; k < nrowsmxa; k++)
(*(rmx + (nrowsmxa * j) + k)) += (*(mxa + (nrowsmxa * i) + k)) * (*(mxb + (ncolsmxb * i) + j));
} 
}
return rmx;
}
void vmxmultiplyf(size_t len, float *mxa, float *mxb, float *mxc)
{
float32x4_t a, b, c;
size_t i, j, k, reps = len / VECREG_LEN;
float *mxap, *mxbp, *mxcp, tmp;
for(i = 0; i < len; i++)
{
for(j = (i + 1); j < len; j++)
{
tmp = *(mxa + (len * j) + i);
*(mxa + (len * j) + i) = *(mxa + (len * i) + j);
*(mxa + (len * i) + j) = tmp;
}
}
for(i = 0; i < len; i++)
{
mxap = mxa;
for(j = 0; j < len; j++)
{
c = VMULSC(c, 0.0); 
mxbp = mxb + (len * i);
for(k = 0; k < reps; k++)
{
a = VECLOD(mxap); 
b = VECLOD(mxbp);
c = VMULAC(c, a, b); 
mxap += VECREG_LEN;
mxbp += VECREG_LEN;
}
mxcp = mxc + (len * i) + j;
tmp = VEXTLN(c, 0) + VEXTLN(c, 1) + VEXTLN(c, 2) + VEXTLN(c, 3);
*(mxcp) = tmp; 
} 
}
}
float *mpvmxmultiplyf(size_t len, float *mxa, float *mxb, int task_id, int num_tasks, size_t *ncols)
{
size_t si, ei;
float *buf;
si = (task_id * len) / num_tasks;
ei = ((task_id + 1) * len) / num_tasks;
*ncols = ei - si;
buf = calloc(len * (*ncols), sizeof(float));
#pragma omp parallel firstprivate(len, mxa, mxb, buf, ei, si) num_threads(NUMTHREADS)
{ 
float32x4_t a, b, c;
size_t i, j, k, reps = len / VECREG_LEN;
float *mxap, *mxbp, *bufp, tmp;
#pragma omp for
for(i = 0; i < len; i++)
{
for(j = (i + 1); j < len; j++)
{
tmp = *(mxa + (len * j) + i);
*(mxa + (len * j) + i) = *(mxa + (len * i) + j);
*(mxa + (len * i) + j) = tmp;
}
}
#pragma omp for 
for(i = si; i < ei; i++)
{
mxap = mxa;
for(j = 0; j < len; j++)
{
c = VMULSC(c, 0.0); 
mxbp = mxb + (len * i);
for(k = 0; k < reps; k++)
{
a = VECLOD(mxap); 
b = VECLOD(mxbp);
c = VMULAC(c, a, b); 
mxap += VECREG_LEN;
mxbp += VECREG_LEN;
}
bufp = buf + (len * (i - si)) + j;
tmp = VEXTLN(c, 0) + VEXTLN(c, 1) + VEXTLN(c, 2) + VEXTLN(c, 3);
*(bufp) = tmp; 
} 
}
}
return buf;
}
float *mpvmxmultiplyfs(size_t len, float *mxa, size_t ncols, float *mxb)
{
float *buf = calloc(len * ncols, sizeof(float));
#pragma omp parallel firstprivate(len, mxa, ncols, mxb, buf) num_threads(NUMTHREADS)
{ 
float32x4_t a, b, c;
size_t i, j, k, reps = len / VECREG_LEN;
float *mxap, *mxbp, *bufp, tmp;
#pragma omp for
for(i = 0; i < len; i++)
{
for(j = (i + 1); j < len; j++)
{
tmp = *(mxa + (len * j) + i);
*(mxa + (len * j) + i) = *(mxa + (len * i) + j);
*(mxa + (len * i) + j) = tmp;
}
}
#pragma omp for 
for(i = 0; i < ncols; i++)
{
mxap = mxa;
for(j = 0; j < len; j++)
{
c = VMULSC(c, 0.0); 
mxbp = mxb + (len * i);
for(k = 0; k < reps; k++)
{
a = VECLOD(mxap); 
b = VECLOD(mxbp);
c = VMULAC(c, a, b); 
mxap += VECREG_LEN;
mxbp += VECREG_LEN;
}
bufp = buf + (len * i) + j;
tmp = VEXTLN(c, 0) + VEXTLN(c, 1) + VEXTLN(c, 2) + VEXTLN(c, 3);
*(bufp) = tmp; 
} 
}
}
return buf;
}
