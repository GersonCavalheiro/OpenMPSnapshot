#include <sycl/sycl.hpp>
#include "indices.hpp"



#define C_a 1.387039845322148f 
#define C_b 1.306562964876377f 
#define C_c 1.175875602419359f 
#define C_d 0.785694958387102f 
#define C_e 0.541196100146197f 
#define C_f 0.275899379282943f 



#define C_norm 0.3535533905932737f 

#define BLOCK_SIZE          8


#define KER2_BLOCK_WIDTH          128



#define KER2_BLOCK_HEIGHT         8



#define KER2_SMEMBLOCK_STRIDE     (KER2_BLOCK_WIDTH+1)



void InplaceDCTvector(float *Vect0, int Step)
{
float *Vect1 = Vect0 + Step;
float *Vect2 = Vect1 + Step;
float *Vect3 = Vect2 + Step;
float *Vect4 = Vect3 + Step;
float *Vect5 = Vect4 + Step;
float *Vect6 = Vect5 + Step;
float *Vect7 = Vect6 + Step;

float X07P = (*Vect0) + (*Vect7);
float X16P = (*Vect1) + (*Vect6);
float X25P = (*Vect2) + (*Vect5);
float X34P = (*Vect3) + (*Vect4);

float X07M = (*Vect0) - (*Vect7);
float X61M = (*Vect6) - (*Vect1);
float X25M = (*Vect2) - (*Vect5);
float X43M = (*Vect4) - (*Vect3);

float X07P34PP = X07P + X34P;
float X07P34PM = X07P - X34P;
float X16P25PP = X16P + X25P;
float X16P25PM = X16P - X25P;

(*Vect0) = C_norm * (X07P34PP + X16P25PP);
(*Vect2) = C_norm * (C_b * X07P34PM + C_e * X16P25PM);
(*Vect4) = C_norm * (X07P34PP - X16P25PP);
(*Vect6) = C_norm * (C_e * X07P34PM - C_b * X16P25PM);

(*Vect1) = C_norm * (C_a * X07M - C_c * X61M + C_d * X25M - C_f * X43M);
(*Vect3) = C_norm * (C_c * X07M + C_f * X61M - C_a * X25M + C_d * X43M);
(*Vect5) = C_norm * (C_d * X07M + C_a * X61M + C_f * X25M - C_c * X43M);
(*Vect7) = C_norm * (C_f * X07M + C_d * X61M + C_c * X25M + C_a * X43M);
}



void InplaceIDCTvector(float *Vect0, int Step)
{
float *Vect1 = Vect0 + Step;
float *Vect2 = Vect1 + Step;
float *Vect3 = Vect2 + Step;
float *Vect4 = Vect3 + Step;
float *Vect5 = Vect4 + Step;
float *Vect6 = Vect5 + Step;
float *Vect7 = Vect6 + Step;

float Y04P   = (*Vect0) + (*Vect4);
float Y2b6eP = C_b * (*Vect2) + C_e * (*Vect6);

float Y04P2b6ePP = Y04P + Y2b6eP;
float Y04P2b6ePM = Y04P - Y2b6eP;
float Y7f1aP3c5dPP = C_f * (*Vect7) + C_a * (*Vect1) + C_c * (*Vect3) + C_d * (*Vect5);
float Y7a1fM3d5cMP = C_a * (*Vect7) - C_f * (*Vect1) + C_d * (*Vect3) - C_c * (*Vect5);

float Y04M   = (*Vect0) - (*Vect4);
float Y2e6bM = C_e * (*Vect2) - C_b * (*Vect6);

float Y04M2e6bMP = Y04M + Y2e6bM;
float Y04M2e6bMM = Y04M - Y2e6bM;
float Y1c7dM3f5aPM = C_c * (*Vect1) - C_d * (*Vect7) - C_f * (*Vect3) - C_a * (*Vect5);
float Y1d7cP3a5fMM = C_d * (*Vect1) + C_c * (*Vect7) - C_a * (*Vect3) + C_f * (*Vect5);

(*Vect0) = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
(*Vect7) = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
(*Vect4) = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
(*Vect3) = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);

(*Vect1) = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
(*Vect5) = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
(*Vect2) = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
(*Vect6) = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);
}




void DCT2D8x8(sycl::nd_item<2> &item, 
float *__restrict block, 
float *__restrict dst,
const float *__restrict src,
const uint size)
{
const int lidx = item.get_local_id(1);
const int lidy = item.get_local_id(0);
const int bidx = item.get_group(1);

if (bidx * KER2_BLOCK_HEIGHT * KER2_BLOCK_WIDTH + (lidy+1) * BLOCK_SIZE*BLOCK_SIZE-1 >= size) return;

int offset = lidy * (BLOCK_SIZE*BLOCK_SIZE) + lidx;

src += bidx * KER2_BLOCK_HEIGHT * KER2_BLOCK_WIDTH;
dst += bidx * KER2_BLOCK_HEIGHT * KER2_BLOCK_WIDTH;

src += offset;
dst += offset;

float *bl_ptr = block + offset;

#pragma unroll

for (unsigned int i = 0; i < BLOCK_SIZE; i++)
bl_ptr[i * BLOCK_SIZE] = src[i * BLOCK_SIZE]; 

InplaceDCTvector(bl_ptr - lidx + BLOCK_SIZE * lidx, 1);

InplaceDCTvector(bl_ptr, BLOCK_SIZE);

for (unsigned int i = 0; i < BLOCK_SIZE; i++)
dst[i * BLOCK_SIZE] = bl_ptr[i * BLOCK_SIZE];
}




void IDCT2D8x8(sycl::nd_item<2> &item,
float *__restrict block, 
float *__restrict dst,
const float *__restrict src,
const uint size)
{
const int lidx = item.get_local_id(1);
const int lidy = item.get_local_id(0);
const int bidx = item.get_group(1);

if (bidx * KER2_BLOCK_HEIGHT * KER2_BLOCK_WIDTH + (lidy+1) * BLOCK_SIZE*BLOCK_SIZE-1 >= size) return;

int offset = lidy * (BLOCK_SIZE*BLOCK_SIZE) + lidx;

src += bidx * KER2_BLOCK_HEIGHT * KER2_BLOCK_WIDTH; 
dst += bidx * KER2_BLOCK_HEIGHT * KER2_BLOCK_WIDTH; 

src += offset;
dst += offset;

float *bl_ptr = block + offset;

#pragma unroll

for (unsigned int i = 0; i < BLOCK_SIZE; i++)
bl_ptr[i * BLOCK_SIZE] = src[i * BLOCK_SIZE];

InplaceIDCTvector(bl_ptr - lidx + BLOCK_SIZE * lidx, 1);

InplaceIDCTvector(bl_ptr, BLOCK_SIZE);    

for (unsigned int i = 0; i < BLOCK_SIZE; i++)
dst[i * BLOCK_SIZE] = bl_ptr[i * BLOCK_SIZE];
}

void run_DCT2D8x8(  
sycl::queue &q,
float *transformed_stacks,
float *gathered_stacks,
const uint size,
const sycl::range<2> lws,  
const sycl::range<2> gws)
{
q.submit([&] (sycl::handler &cgh) {
sycl::local_accessor<float, 1>
lmem(sycl::range<1>(KER2_BLOCK_HEIGHT * KER2_SMEMBLOCK_STRIDE), cgh);
cgh.parallel_for<class dct>(
sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
DCT2D8x8(item, lmem.get_pointer(), transformed_stacks, gathered_stacks, size);
});
});
}

void run_IDCT2D8x8(
sycl::queue &q,
float *gathered_stacks,
float *transformed_stacks,
const uint size,
const sycl::range<2> lws,  
const sycl::range<2> gws)
{
q.submit([&] (sycl::handler &cgh) {
sycl::local_accessor<float, 1>
lmem(sycl::range<1>(KER2_BLOCK_HEIGHT * KER2_SMEMBLOCK_STRIDE), cgh);
cgh.parallel_for<class idct>(
sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
IDCT2D8x8 (item, lmem.get_pointer(), gathered_stacks, transformed_stacks, size);
});
});
}
