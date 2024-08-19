

#include <omp.h>
#include <stdint.h>




void RAND_ADD( uint64_t *restrict ARRAY,
uint64_t *restrict IDX,
uint64_t iters,
uint64_t pes ){

uint64_t i      = 0;
uint64_t start  = 0;

#pragma omp parallel private(start,i)
{
start = (uint64_t)(omp_get_thread_num()) * iters;
for( i=start; i<(start+iters); i++ ){
__atomic_fetch_add( &ARRAY[IDX[i]], (uint64_t)(0x1), __ATOMIC_RELAXED );
}
}
}

void RAND_CAS( uint64_t *restrict ARRAY,
uint64_t *restrict IDX,
uint64_t iters,
uint64_t pes ){

uint64_t i      = 0;
uint64_t start  = 0;

#pragma omp parallel private(start,i)
{
start = (uint64_t)(omp_get_thread_num()) * iters;
for( i=start; i<(start+iters); i++ ){
__atomic_compare_exchange_n( &ARRAY[IDX[i]], &ARRAY[IDX[i]], ARRAY[IDX[i]],
0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
}
}
}

void STRIDE1_ADD( uint64_t *restrict ARRAY,
uint64_t *restrict IDX,
uint64_t iters,
uint64_t pes ){

uint64_t i      = 0;
uint64_t start  = 0;

#pragma omp parallel private(start,i)
{
start = (uint64_t)(omp_get_thread_num()) * iters;
for( i=start; i<(start+iters); i++ ){
__atomic_fetch_add( &ARRAY[i], (uint64_t)(0xF), __ATOMIC_RELAXED );
}
}
}

void STRIDE1_CAS( uint64_t *restrict ARRAY,
uint64_t *restrict IDX,
uint64_t iters,
uint64_t pes ){

uint64_t i      = 0;
uint64_t start  = 0;

#pragma omp parallel private(start,i)
{
start = (uint64_t)(omp_get_thread_num()) * iters;
for( i=start; i<(start+iters); i++ ){
__atomic_compare_exchange_n( &ARRAY[i], &ARRAY[i], ARRAY[i],
0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
}
}
}

void STRIDEN_ADD( uint64_t *restrict ARRAY,
uint64_t *restrict IDX,
uint64_t iters,
uint64_t pes,
uint64_t stride ){

uint64_t i      = 0;
uint64_t start  = 0;

#pragma omp parallel private(start,i)
{
start = (uint64_t)(omp_get_thread_num()) * iters * stride;
for( i=start; i<(start+iters*stride); i+=stride ){
__atomic_fetch_add( &ARRAY[i], (uint64_t)(0xF), __ATOMIC_RELAXED );
}
}
}

void STRIDEN_CAS( uint64_t *restrict ARRAY,
uint64_t *restrict IDX,
uint64_t iters,
uint64_t pes,
uint64_t stride ){
uint64_t i      = 0;
uint64_t start  = 0;

#pragma omp parallel private(start,i)
{
start = (uint64_t)(omp_get_thread_num()) * iters * stride;
for( i=start; i<(start+iters*stride); i+=stride ){
__atomic_compare_exchange_n( &ARRAY[i], &ARRAY[i], ARRAY[i],
0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
}
}
}

void PTRCHASE_ADD( uint64_t *restrict ARRAY,
uint64_t *restrict IDX,
uint64_t iters,
uint64_t pes ){

uint64_t i      = 0;
uint64_t start  = 0;

#pragma omp parallel private(start,i)
{
start = (uint64_t)(omp_get_thread_num()) * iters;
for( i=0; i<iters; i++ ){
start = __atomic_fetch_add( &IDX[start],
(uint64_t)(0x00ull),
__ATOMIC_RELAXED );
}
}
}

void PTRCHASE_CAS( uint64_t *restrict ARRAY,
uint64_t *restrict IDX,
uint64_t iters,
uint64_t pes ){

uint64_t i      = 0;
uint64_t start  = 0;

#pragma omp parallel private(start,i)
{
start = (uint64_t)(omp_get_thread_num()) * iters;
for( i=0; i<iters; i++ ){
__atomic_compare_exchange_n( &IDX[start], &start, IDX[start],
0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
}
}
}

void SG_ADD( uint64_t *restrict ARRAY,
uint64_t *restrict IDX,
uint64_t iters,
uint64_t pes ){

uint64_t i      = 0;
uint64_t start  = 0;
uint64_t src    = 0;
uint64_t dest   = 0;
uint64_t val    = 0;

#pragma omp parallel private(start,i,src,dest,val)
{
start = (uint64_t)(omp_get_thread_num()) * iters;
for( i=start; i<(start+iters); i++ ){
src  = __atomic_fetch_add( &IDX[i], (uint64_t)(0x00ull), __ATOMIC_RELAXED );
dest = __atomic_fetch_add( &IDX[i+1], (uint64_t)(0x00ull), __ATOMIC_RELAXED );
val = __atomic_fetch_add( &ARRAY[src], (uint64_t)(0x01ull), __ATOMIC_RELAXED );
__atomic_fetch_add( &ARRAY[dest], val, __ATOMIC_RELAXED );
}
}
}

void SG_CAS( uint64_t *restrict ARRAY,
uint64_t *restrict IDX,
uint64_t iters,
uint64_t pes ){

uint64_t i      = 0;
uint64_t start  = 0;
uint64_t src    = 0;
uint64_t dest   = 0;
uint64_t val    = 0;

#pragma omp parallel private(start,i,src,dest,val)
{
start = (uint64_t)(omp_get_thread_num()) * iters;
val   = 0x00ull;
src   = 0x00ull;
dest  = 0x00ull;
for( i=start; i<(start+iters); i++ ){
__atomic_compare_exchange_n( &IDX[i], &src, IDX[i],
0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
__atomic_compare_exchange_n( &IDX[i+1], &dest, IDX[i+1],
0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
__atomic_compare_exchange_n( &ARRAY[src], &val, ARRAY[src],
0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
__atomic_compare_exchange_n( &ARRAY[dest], &ARRAY[dest], val,
0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
}
}
}

void CENTRAL_ADD( uint64_t *restrict ARRAY,
uint64_t *restrict IDX,
uint64_t iters,
uint64_t pes ){
uint64_t i      = 0;

#pragma omp parallel private(i)
{
for( i=0; i<iters; i++ ){
__atomic_fetch_add( &ARRAY[0], (uint64_t)(0x1), __ATOMIC_RELAXED );
}
}
}

void CENTRAL_CAS( uint64_t *restrict ARRAY,
uint64_t *restrict IDX,
uint64_t iters,
uint64_t pes ){
uint64_t i      = 0;

#pragma omp parallel private(i)
{
for( i=0; i<iters; i++ ){
__atomic_compare_exchange_n( &ARRAY[0], &ARRAY[0], ARRAY[0],
0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
}
}
}

void SCATTER_ADD( uint64_t *restrict ARRAY,
uint64_t *restrict IDX,
uint64_t iters,
uint64_t pes ){

uint64_t i      = 0;
uint64_t start  = 0;
uint64_t dest   = 0;
uint64_t val    = 0;

#pragma omp parallel private(start,i,dest,val)
{
start = (uint64_t)(omp_get_thread_num()) * iters;
for( i=start; i<(start+iters); i++ ){
dest = __atomic_fetch_add( &IDX[i+1], (uint64_t)(0x00ull), __ATOMIC_RELAXED );
val = __atomic_fetch_add( &ARRAY[i], (uint64_t)(0x01ull), __ATOMIC_RELAXED );
__atomic_fetch_add( &ARRAY[dest], val, __ATOMIC_RELAXED );
}
}
}

void SCATTER_CAS( uint64_t *restrict ARRAY,
uint64_t *restrict IDX,
uint64_t iters,
uint64_t pes ){

uint64_t i      = 0;
uint64_t start  = 0;
uint64_t dest   = 0;
uint64_t val    = 0;

#pragma omp parallel private(start,i,dest,val)
{
start = (uint64_t)(omp_get_thread_num()) * iters;
dest  = 0x00ull;
val   = 0x00ull;
for( i=start; i<(start+iters); i++ ){
__atomic_compare_exchange_n( &IDX[i+1], &dest, IDX[i+1],
0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
__atomic_compare_exchange_n( &ARRAY[i], &val, ARRAY[i],
0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
__atomic_compare_exchange_n( &ARRAY[dest], &ARRAY[dest], val,
0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
}
}
}

void GATHER_ADD( uint64_t *restrict ARRAY,
uint64_t *restrict IDX,
uint64_t iters,
uint64_t pes ){

uint64_t i      = 0;
uint64_t start  = 0;
uint64_t dest   = 0;
uint64_t val    = 0;

#pragma omp parallel private(start,i,dest,val)
{
start = (uint64_t)(omp_get_thread_num()) * iters;
for( i=start; i<(start+iters); i++ ){
dest = __atomic_fetch_add( &IDX[i+1], (uint64_t)(0x00ull), __ATOMIC_RELAXED );
val = __atomic_fetch_add( &ARRAY[dest], (uint64_t)(0x01ull), __ATOMIC_RELAXED );
__atomic_fetch_add( &ARRAY[i], val, __ATOMIC_RELAXED );
}
}
}

void GATHER_CAS( uint64_t *restrict ARRAY,
uint64_t *restrict IDX,
uint64_t iters,
uint64_t pes ){

uint64_t i      = 0;
uint64_t start  = 0;
uint64_t dest   = 0;
uint64_t val    = 0;

#pragma omp parallel private(start,i,dest,val)
{
start = (uint64_t)(omp_get_thread_num()) * iters;
dest  = 0x00ull;
val   = 0x00ull;
for( i=start; i<(start+iters); i++ ){
__atomic_compare_exchange_n( &IDX[i+1], &dest, IDX[i+1],
0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
__atomic_compare_exchange_n( &ARRAY[dest], &val, ARRAY[dest],
0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
__atomic_compare_exchange_n( &ARRAY[i], &ARRAY[i], val,
0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
}
}
}


