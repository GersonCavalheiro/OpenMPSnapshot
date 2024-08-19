#ifndef __REDUCE_KERNELS_FUNCTIONAL_CORE_H__
#define __REDUCE_KERNELS_FUNCTIONAL_CORE_H__

#include <omp.h>

#include "../../../macros/macros.h"
#include "../../../types/types.h"
#include "../../../meta/meta.h"

namespace __core__ {
namespace __functional__ {
namespace __reduce__ {
namespace __array__ {
namespace __private__ {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
template <typename fn_T,int threadnum,typename RT,typename T,typename IT> __optimize__
RT __reduce_ckernel__(RT& result,CRESTRICT_Q(T*) arr,IT size) {
RT result_mem[threadnum];
if(threadnum>1) {
#pragma omp parallel num_threads(threadnum) shared(result_mem,result)
{
int threadid=-1;
RT result_local;
#pragma omp for
for(IT i=0;i<size;++i) {
if(threadid==-1) {
threadid=omp_get_thread_num();
result_local=arr[i];
}
else
result_local=fn_T::fn(arr[i],result_local);
}
result_mem[threadid]=result_local;
int thread_count=omp_get_num_threads();
#pragma omp barrier
#pragma omp single
{
result=result_mem[0];
for(int i=1;i<thread_count;++i)
result=fn_T::fn(result_mem[i],result);
}
}
}
else {
for(IT i=0;i<size;++i) {
if(i==0)
result=arr[0];
else
result=fn_T::fn(arr[i],result);
}
}
return result;
}
template <typename fn_T,typename IV,int threadnum,typename RT,typename T,typename IT> __optimize__
RT __reduce_ckernel__(RT& result,CRESTRICT_Q(T*) arr,IT size) {
RT result_mem[threadnum];
if(threadnum>1) {
result=IV::value;
#pragma omp parallel num_threads(threadnum) shared(result_mem,result)
{
int threadid=omp_get_thread_num();
RT result_local=IV::value;
#pragma omp for
for(IT i=0;i<size;++i)
result_local=fn_T::fn(arr[i],result_local);
result_mem[threadid]=result_local;
int thread_count=omp_get_num_threads();
#pragma omp barrier
#pragma omp single
{
for(int i=0;i<thread_count;++i)
result=fn_T::fn(result_mem[i],result);
}
}
}
else {
result=IV::value;
for(IT i=0;i<size;++i)
result=fn_T::fn(arr[i],result);
}
return result;
}
template <typename fn_T,int threadnum,typename RT,typename T,typename IT> __optimize__
RT __reduce_ckernel__(RT& result,CRESTRICT_Q(T*) arr,IT size,const RT &ival) {
RT result_mem[threadnum];
if(threadnum>1) {
#pragma omp parallel num_threads(threadnum) shared(result_mem,result)
{
int threadid=omp_get_thread_num();
RT result_local=ival;
#pragma omp for
for(IT i=0;i<size;++i)
result_local=fn_T::fn(arr[i],result_local);
result_mem[threadid]=result_local;
int thread_count=omp_get_num_threads();
#pragma omp barrier
#pragma omp single
{
result=result_mem[0];
for(int i=1;i<thread_count;++i)
result=fn_T::fn(result_mem[i],result);
}
}
}
else {
result=ival;
for(IT i=0;i<size;++i)
result=fn_T::fn(arr[i],result);
}
return result;
}

template <typename reduce_FT,typename apply_FT,int threadnum,typename RT,typename T,typename IT> __optimize__
RT __reduce_apply_ckernel__(RT& result,CRESTRICT_Q(T*) arr,IT size) {
RT result_mem[threadnum];
if(threadnum>1) {
#pragma omp parallel num_threads(threadnum) shared(result_mem,result)
{
int threadid=-1;
RT result_local;
#pragma omp for
for(IT i=0;i<size;++i) {
if(threadid==-1) {
threadid=omp_get_thread_num();
result_local=apply_FT::fn(arr[i]);
}
else
result_local=reduce_FT::fn(apply_FT::fn(arr[i]),result_local);
}
result_mem[threadid]=result_local;
int thread_count=omp_get_num_threads();
#pragma omp barrier
#pragma omp single
{
result=result_mem[0];
for(int i=1;i<thread_count;++i)
result=reduce_FT::fn(result_mem[i],result);
}
}
}
else {
for(IT i=0;i<size;++i) {
if(i==0)
result=apply_FT::fn(arr[0]);
else
result=reduce_FT::fn(apply_FT::fn(arr[i]),result);
}
}
return result;
}
template <typename reduce_FT,typename apply_FT,typename IV,int threadnum,typename RT,typename T,typename IT> __optimize__
RT __reduce_apply_ckernel__(RT& result,CRESTRICT_Q(T*) arr,IT size) {
RT result_mem[threadnum];
if(threadnum>1) {
#pragma omp parallel num_threads(threadnum) shared(result_mem,result)
{
int threadid=omp_get_thread_num();
RT result_local=IV::value;
#pragma omp for
for(IT i=0;i<size;++i)
result_local=reduce_FT::fn(apply_FT::fn(arr[i]),result_local);
result_mem[threadid]=result_local;
int thread_count=omp_get_num_threads();
#pragma omp barrier
#pragma omp single
{
result=result_mem[0];
for(int i=1;i<thread_count;++i)
result=reduce_FT::fn(result_mem[i],result);
}
}
}
else {
result=IV::value;
for(IT i=0;i<size;++i)
result=reduce_FT::fn(apply_FT::fn(arr[i]),result);
}
return result;
}
template <typename reduce_FT,typename apply_FT,int threadnum,typename RT,typename T,typename IT> __optimize__
RT __reduce_apply_ckernel__(RT& result,CRESTRICT_Q(T*) arr,IT size,const RT &ival) {
RT result_mem[threadnum];
if(threadnum>1) {
#pragma omp parallel num_threads(threadnum) shared(result_mem,result)
{
int threadid=omp_get_thread_num();
RT result_local=ival;
#pragma omp for
for(IT i=0;i<size;++i)
result_local=reduce_FT::fn(apply_FT::fn(arr[i]),result_local);
result_mem[threadid]=result_local;
int thread_count=omp_get_num_threads();
#pragma omp barrier
#pragma omp single
{
result=result_mem[0];
for(int i=1;i<thread_count;++i)
result=reduce_FT::fn(result_mem[i],result);
}
}
}
else {
result=ival;
for(IT i=0;i<size;++i)
result=reduce_FT::fn(apply_FT::fn(arr[i]),result);
}
return result;
}

template <typename reduce_FT,typename apply_FT,int threadnum,typename RT,typename V,typename U,typename IT> __optimize__
RT __reduce_apply_ckernel__(RT& result,CRESTRICT_Q(V*) arr1,CRESTRICT_Q(U*) arr2,IT size) {
RT result_mem[threadnum];
if(threadnum>1) {
#pragma omp parallel num_threads(threadnum) shared(result_mem,result)
{
int threadid=-1;
RT result_local;
#pragma omp for
for(IT i=0;i<size;++i) {
if(threadid==-1) {
threadid=omp_get_thread_num();
result_local=apply_FT::fn(arr1[i],arr2[i]);
}
else
result_local=reduce_FT::fn(apply_FT::fn(arr1[i],arr2[i]),result_local);
}
result_mem[threadid]=result_local;
int thread_count=omp_get_num_threads();
#pragma omp barrier
#pragma omp single
{
result=result_mem[0];
for(int i=1;i<thread_count;++i)
result=reduce_FT::fn(result_mem[i],result);
}
}
}
else {
for(IT i=0;i<size;++i) {
if(i==0)
result=apply_FT::fn(arr1[0]);
else
result=reduce_FT::fn(apply_FT::fn(arr1[i],arr2[i]),result);
}
}
return result;
}
template <typename reduce_FT,typename apply_FT,typename IV,int threadnum,typename RT,typename V,typename U,typename IT> __optimize__
RT __reduce_apply_ckernel__(RT& result,CRESTRICT_Q(V*) arr1,CRESTRICT_Q(U*) arr2,IT size) {
RT result_mem[threadnum];
if(threadnum>1) {
#pragma omp parallel num_threads(threadnum) shared(result_mem,result)
{
int threadid=omp_get_thread_num();
RT result_local=IV::value;
#pragma omp for
for(IT i=0;i<size;++i)
result_local=reduce_FT::fn(apply_FT::fn(arr1[i],arr2[i]),result_local);
result_mem[threadid]=result_local;
int thread_count=omp_get_num_threads();
#pragma omp barrier
#pragma omp single
{
result=result_mem[0];
for(int i=1;i<thread_count;++i)
result=reduce_FT::fn(result_mem[i],result);
}
}
}
else {
result=IV::value;
for(IT i=0;i<size;++i)
result=reduce_FT::fn(apply_FT::fn(arr1[i],arr2[i]),result);
}
return result;
}
template <typename reduce_FT,typename apply_FT,int threadnum,typename RT,typename V,typename U,typename IT> __optimize__
RT __reduce_apply_ckernel__(RT& result,CRESTRICT_Q(V*) arr1,CRESTRICT_Q(U*) arr2,IT size,const RT &ival) {
RT result_mem[threadnum];
if(threadnum>1) {
#pragma omp parallel num_threads(threadnum) shared(result_mem,result)
{
int threadid=omp_get_thread_num();
RT result_local=ival;
#pragma omp for
for(IT i=0;i<size;++i)
result_local=reduce_FT::fn(apply_FT::fn(arr1[i],arr2[i]),result_local);
result_mem[threadid]=result_local;
int thread_count=omp_get_num_threads();
#pragma omp barrier
#pragma omp single
{
result=result_mem[0];
for(int i=1;i<thread_count;++i)
result=reduce_FT::fn(result_mem[i],result);
}
}
}
else {
result=ival;
for(IT i=0;i<size;++i)
result=reduce_FT::fn(apply_FT::fn(arr1[i],arr2[i]),result);
}
return result;
}
#pragma GCC diagnostic pop
}
}
}
}
}
#endif
