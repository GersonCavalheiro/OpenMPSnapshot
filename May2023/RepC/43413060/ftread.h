#pragma once
#include <stdio.h>
#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif
int ftread(void *ptr,size_t size,size_t nitems,FILE *stream) __attribute__((warn_unused_result));
int my_ftread(void *ptr,size_t size,size_t nitems,FILE *stream) __attribute__((warn_unused_result));
#ifdef __cplusplus
}
#endif
