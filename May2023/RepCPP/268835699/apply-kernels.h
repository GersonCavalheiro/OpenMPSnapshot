#ifndef __ARRAY_APPLY_KERNELS_H__
#define __ARRAY_APPLY_KERNELS_H__

#include <type_traits>

#include <omp.h>

#include "../../../macros/macros.h"
#include "../../../meta/meta.h"
#include "../../../types/types.h"

namespace __core__ {
namespace __functional__ {
namespace __apply__ {
namespace __array__ {
namespace __private__ {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
template <typename fn_T,int threadnum,typename T,typename IT,typename... Args> __optimize__
void __apply_function_ckernel__(T* arr,IT size,Args... args) {
if(threadnum>1) {
#pragma omp parallel for num_threads(threadnum)
for(IT i=0;i<size;++i)
arr[i]=fn_T::fn(arr[i],args...);
}
else {
for(IT i=0;i<size;++i)
arr[i]=fn_T::fn(arr[i],args...);
}
}
template <typename fn_T,int threadnum,typename T,typename U,typename IT,typename... Args> __optimize__
void __apply_function_ckernel__(T* arr_dst,U* arr_src,IT size,Args... args) {
if(threadnum>1) {
#pragma omp parallel for num_threads(threadnum)
for(IT i=0;i<size;++i)
arr_dst[i]=fn_T::fn(arr_src[i],args...);
}
else {
for(IT i=0;i<size;++i)
arr_dst[i]=fn_T::fn(arr_src[i],args...);
}
}
template <typename fn_T,int threadnum,typename T,typename V,typename U,typename IT,typename... Args> __optimize__
void __apply_function_ckernel__(T* arr_dst,V* arr1,U* arr2,IT size,Args... args) {
if(threadnum>1) {
#pragma omp parallel for num_threads(threadnum)
for(IT i=0;i<size;++i)
arr_dst[i]=fn_T::fn(arr1[i],arr2[i],args...);
}
else {
for(IT i=0;i<size;++i)
arr_dst[i]=fn_T::fn(arr1[i],arr2[i],args...);
}
}
template <typename fn_T,int threadnum,typename T,typename V,typename U,typename W,typename IT,typename... Args> __optimize__
void __apply_function_ckernel__(T* arr_dst,V* arr1,U* arr2,W* arr3,IT size,Args... args) {
if(threadnum>1) {
#pragma omp parallel for num_threads(threadnum)
for(IT i=0;i<size;++i)
arr_dst[i]=fn_T::fn(arr1[i],arr2[i],arr3[i],args...);
}
else {
for(IT i=0;i<size;++i)
arr_dst[i]=fn_T::fn(arr1[i],arr2[i],arr3[i],args...);
}
}

template <typename fn_T,int threadnum,typename T,typename IT,typename... Args> __optimize__
void __apply_function_indexed_ckernel__(T* arr,IT size,Args... args) {
if(threadnum>1) {
#pragma omp parallel for num_threads(threadnum)
for(IT i=0;i<size;++i)
arr[i]=fn_T::fn(i,arr[i],args...);
}
else {
for(IT i=0;i<size;++i)
arr[i]=fn_T::fn(i,arr[i],args...);
}
}
template <typename fn_T,int threadnum,typename T,typename U,typename IT,typename... Args> __optimize__
void __apply_function_indexed_ckernel__(T* arr_dst,U* arr_src,IT size,Args... args) {
if(threadnum>1) {
#pragma omp parallel for num_threads(threadnum)
for(IT i=0;i<size;++i)
arr_dst[i]=fn_T::fn(i,arr_src[i],args...);
}
else {
for(IT i=0;i<size;++i)
arr_dst[i]=fn_T::fn(i,arr_src[i],args...);
}
}
template <typename fn_T,int threadnum,typename T,typename V,typename U,typename IT,typename... Args> __optimize__
void __apply_function_indexed_ckernel__(T* arr_dst,V* arr1,U* arr2,IT size,Args... args) {
if(threadnum>1) {
#pragma omp parallel for num_threads(threadnum)
for(IT i=0;i<size;++i)
arr_dst[i]=fn_T::fn(i,arr1[i],arr2[i],args...);
}
else {
for(IT i=0;i<size;++i)
arr_dst[i]=fn_T::fn(i,arr1[i],arr2[i],args...);
}
}
template <typename fn_T,int threadnum,typename T,typename V,typename U,typename W,typename IT,typename... Args> __optimize__
void __apply_function_indexed_ckernel__(T* arr_dst,V* arr1,U* arr2,W* arr3,IT size,Args... args) {
if(threadnum>1) {
#pragma omp parallel for num_threads(threadnum)
for(IT i=0;i<size;++i)
arr_dst[i]=fn_T::fn(i,arr1[i],arr2[i],arr3[i],args...);
}
else {
for(IT i=0;i<size;++i)
arr_dst[i]=fn_T::fn(i,arr1[i],arr2[i],arr3[i],args...);
}
}

template <typename fn_T,int threadnum,typename IT,typename... Args> __optimize__
void __apply_function_meta_ckernel__(IT size,Args... args) {
if(threadnum>1) {
#pragma omp parallel for num_threads(threadnum)
for(IT i=0;i<size;++i)
fn_T::fn(i,size,args...);
}
else {
for(IT i=0;i<size;++i)
fn_T::fn(i,size,args...);
}
}
template <typename fn_T,int threadnum,typename IT,typename... Args> __optimize__
void __apply_function_meta_ckernel__(IT size,IT private_mem_size,Args... args) {
if(threadnum>1) {
#pragma omp parallel num_threads(threadnum)
{
uchar *private_mem=malloc(private_mem_size);
#pragma omp for
for(IT i=0;i<size;++i)
fn_T::fn(i,size,private_mem,args...);
free(private_mem);
}
}
else {
uchar *private_mem=malloc(private_mem_size);
for(IT i=0;i<size;++i)
fn_T::fn(i,size,private_mem,args...);
free(private_mem);
}
}
#pragma GCC diagnostic pop
}
}
}
}
}
#endif
