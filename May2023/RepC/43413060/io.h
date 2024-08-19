#pragma once
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif
int64_t read_positions(const char *filename, const char *format, const size_t size, const int num_fields, ...) __attribute__((warn_unused_result));
int64_t read_columns_into_array(const char *filename, const char *format, const size_t size, const int num_fields, void **data) __attribute__((warn_unused_result));
#ifdef __cplusplus
}
#endif
