

#include <omp.h>
#include <stdint.h>



void RAND_ADD( uint64_t *restrict ARRAY,
uint64_t *restrict IDX,
uint64_t iters,
uint64_t pes ){

#pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters)
{
#pragma omp parallel
{
uint64_t i = 0, ret;
uint64_t start = (uint64_t) (omp_get_team_num() * iters);

#pragma omp for simd
for( i=start; i<(start+iters); i++ ){
#pragma omp atomic capture
{
ret = ARRAY[IDX[i]];
ARRAY[IDX[i]] += 1;
}
}
}
}
}

void STRIDE1_ADD( uint64_t *restrict ARRAY,
uint64_t *restrict IDX,
uint64_t iters,
uint64_t pes ){

#pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters)
{
#pragma omp parallel
{
uint64_t i = 0, ret;
uint64_t start = (uint64_t) (omp_get_team_num() * iters);

#pragma omp for simd
for( i=start; i<(start+iters); i++ ){
#pragma omp atomic capture
{
ret = ARRAY[i];
ARRAY[i] += 1;
}
}
}
}
}

void STRIDEN_ADD( uint64_t *restrict ARRAY,
uint64_t *restrict IDX,
uint64_t iters,
uint64_t pes,
uint64_t stride ){

#pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters, stride)
{
#pragma omp parallel
{
uint64_t i = 0, ret;
uint64_t start = (uint64_t) (omp_get_team_num() * iters * stride);

#pragma omp for simd
for( i=start; i<(start+(iters*stride)); i+=stride ){
#pragma omp atomic capture
{
ret = ARRAY[i];
ARRAY[i] += 1;
}
}
}
}
}


void PTRCHASE_ADD( uint64_t *restrict ARRAY,
uint64_t *restrict IDX,
uint64_t iters,
uint64_t pes ){

#pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters)
{

uint64_t zero = 0;      

uint64_t i = 0;
uint64_t start = (uint64_t) (omp_get_team_num() * iters);

for( i=0; i<iters; i++ ){
#pragma omp atomic capture
{
start = IDX[start];
IDX[start] += zero;
}       
}
}
}

void SG_ADD( uint64_t *restrict ARRAY,
uint64_t *restrict IDX,
uint64_t iters,
uint64_t pes ){

#pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters)
{
#pragma omp parallel
{      

uint64_t zero = 0;

uint64_t i = 0;
uint64_t src = 0;
uint64_t dest = 0;
uint64_t val = 0;
uint64_t ret;
uint64_t start = (uint64_t) (omp_get_team_num() * iters);

#pragma omp for simd
for( i=start; i<(start+iters); i++ ){
#pragma omp atomic capture
{
src = IDX[i];
IDX[i] += zero;
}

#pragma omp atomic capture
{
dest = IDX[i+1];
IDX[i+1] += zero;
}

#pragma omp atomic capture
{
val = ARRAY[src];
ARRAY[src] += 1;
}

#pragma omp atomic capture
{
ret = ARRAY[dest];
ARRAY[dest] += val;
}
}
}
}
}

void CENTRAL_ADD( uint64_t *restrict ARRAY,
uint64_t *restrict IDX,
uint64_t iters,
uint64_t pes ){

#pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters)
{
#pragma omp parallel
{
uint64_t i = 0, ret;

#pragma omp for simd
for( i=0; i<iters; i++ ){
#pragma omp atomic capture
{
ret = ARRAY[0];
ARRAY[0] += 1;
}
}
}
}
}

void SCATTER_ADD( uint64_t *restrict ARRAY,
uint64_t *restrict IDX,
uint64_t iters,
uint64_t pes ){

#pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters)
{
#pragma omp parallel
{

uint64_t zero = 0;

uint64_t i = 0;
uint64_t dest = 0;
uint64_t val = 0;
uint64_t ret;
uint64_t start = (uint64_t) (omp_get_team_num() * iters);

#pragma omp for simd
for( i=start; i<(start+iters); i++ ){
#pragma omp atomic capture
{
dest = IDX[i+1];
IDX[i+1] += zero;
}

#pragma omp atomic capture
{
val = ARRAY[i];
ARRAY[i] += 1;
}

#pragma omp atomic capture
{
ret = ARRAY[dest];
ARRAY[dest] += val;
}
}
}
}
}

void GATHER_ADD( uint64_t *restrict ARRAY,
uint64_t *restrict IDX,
uint64_t iters,
uint64_t pes ){

#pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters)
{
#pragma omp parallel
{ 

uint64_t zero = 0;

uint64_t i = 0;
uint64_t dest = 0;
uint64_t val = 0;
uint64_t ret;
uint64_t start = (uint64_t) (omp_get_team_num() * iters);

#pragma omp for simd
for( i=start; i<(start+iters); i++ ){
#pragma omp atomic capture
{
dest = IDX[i+1];
IDX[i+1] += zero;
}

#pragma omp atomic capture
{
val = ARRAY[dest];
ARRAY[dest] += 1;
}

#pragma omp atomic capture
{
ret = ARRAY[i];
ARRAY[i] += val;
}
}
}
}
}


