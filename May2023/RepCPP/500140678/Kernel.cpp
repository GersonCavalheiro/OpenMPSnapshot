#include "CommonIncl.h"
#include "Kernel.h"

void cubic_kernel(rr_float dist, const rr_float2& diff, rr_float& w, rr_float2& dwdr) {
rr_float q = get_kernel_q(dist);

if (q <= 1) {
w = cubic_kernel_q1(q);
dwdr = cubic_kernel_q1_grad(q, diff);
}
else if (q <= 2) {
w = cubic_kernel_q2(q);
dwdr = cubic_kernel_q2_grad(q, dist, diff);
}
else {
w = 0.f;
dwdr = { 0.f };
}
}

void gauss_kernel(rr_float dist, const rr_float2& diff, rr_float& w, rr_float2& dwdr) {
rr_float q = get_kernel_q(dist);
if (q <= 3) {
w = gauss_kernel_q3(q);
dwdr = gauss_kernel_q3_grad(w, diff);
}
else {
w = 0;
dwdr = { 0.f };
}
}

void quintic_kernel(rr_float dist, const rr_float2& diff, rr_float& w, rr_float2& dwdr) {
rr_float q = get_kernel_q(dist);
if (q <= 1) {
w = quintic_kernel_q1(q);
dwdr = quintic_kernel_q1_grad(q, dist, diff);
}
else if (q <= 2) {
w = quintic_kernel_q2(q);
dwdr = quintic_kernel_q2_grad(q, dist, diff);
}
else if (q <= 3) {
w = quintic_kernel_q3(q);
dwdr = quintic_kernel_q3_grad(q, dist, diff);
}
else {
w = 0.f;
dwdr = { 0.f };
}
}

void desbrun_kernel(rr_float dist, const rr_float2& diff, rr_float& w, rr_float2& dwdr) {
rr_float q = get_kernel_q(dist);
if (q <= 2) {
w = desbrun_kernel_q2(q);
dwdr = desbrun_kernel_q2_grad(q, dist, diff);
}
else {
w = 0.f;
dwdr = { 0.f };
}
}

void kernel_self(
rr_float& w_ii, 
rr_float2& dwdr_ii, 
rr_uint skf)
{
kernel(0.f, rr_float2{ 0.f }, w_ii, dwdr_ii, skf);
}

void kernel(
const rr_float2& ri,
const rr_float2& rj,
rr_float& w, 
rr_float2& dwdr, 
rr_uint skf)
{
rr_float2 diff = ri - rj;
rr_float dist = length(diff);
kernel(dist, diff, w, dwdr, skf);
}

void kernel(
const rr_float dist,
const rr_float2& diff, 
rr_float& w, 
rr_float2& dwdr, 
rr_uint skf)
{
switch (skf) {
case 1: cubic_kernel(dist, diff, w, dwdr); break;
case 2: gauss_kernel(dist, diff, w, dwdr); break;
case 3: quintic_kernel(dist, diff, w, dwdr); break;
case 4: desbrun_kernel(dist, diff, w, dwdr); break;
default: cubic_kernel(dist, diff, w, dwdr); break;
}
}

rr_float kernel_w(rr_float dist, rr_uint skf) {
switch (skf) {
case 1: return cubic_kernel_w(dist);
case 2: return gauss_kernel_w(dist);
case 3: return quintic_kernel_w(dist);
case 4: return desbrun_kernel_w(dist);
default: return cubic_kernel_w(dist); 
}
}

rr_float2 kernel_dwdr(rr_float dist, const rr_float2& diff, rr_uint skf) {
switch (skf) {
case 1: return cubic_kernel_dwdr(dist, diff);
case 2: return gauss_kernel_dwdr(dist, diff);
case 3: return quintic_kernel_dwdr(dist, diff);
case 4: return desbrun_kernel_dwdr(dist, diff);
default: return cubic_kernel_dwdr(dist, diff);
}
}

void calculate_kernels(
const rr_uint ntotal,
const heap_darray<rr_float2>& r,
const heap_darray_md<rr_uint>& neighbours, 
heap_darray_md<rr_float>& w, 
heap_darray_md<rr_float2>& dwdr, 
rr_uint skf)
{
#pragma omp parallel for
for (rr_iter j = 0; j < ntotal; j++) { 
rr_uint i;
for (rr_iter n = 0;
i = neighbours(n, j), i != ntotal; 
++n)
{
rr_float2 diff = r(i) - r(j);
rr_float dist = length(diff);

kernel(dist, diff, w(n, j), dwdr(n, j), skf);
}
}
}
void calculate_kernels_w(
const rr_uint ntotal,
const heap_darray<rr_float2>& r,
const heap_darray_md<rr_uint>& neighbours, 
heap_darray_md<rr_float>& w, 
rr_uint skf)
{
#pragma omp parallel for
for (rr_iter j = 0; j < ntotal; j++) { 
rr_uint i;
for (rr_iter n = 0;
i = neighbours(n, j), i != ntotal; 
++n)
{
rr_float2 diff = r(i) - r(j);
rr_float dist = length(diff);

w(n, j) = kernel_w(dist, skf);
}
}
}
void calculate_kernels_dwdr(
const rr_uint ntotal,
const heap_darray<rr_float2>& r,
const heap_darray_md<rr_uint>& neighbours, 
heap_darray_md<rr_float2>& dwdr, 
rr_uint skf)
{
#pragma omp parallel for
for (rr_iter j = 0; j < ntotal; j++) { 
rr_uint i;
for (rr_iter n = 0;
i = neighbours(n, j), i != ntotal; 
++n)
{
rr_float2 diff = r(i) - r(j);
rr_float dist = length(diff);

dwdr(n, j) = kernel_dwdr(dist, diff, skf);
}
}
}