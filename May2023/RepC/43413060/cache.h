#pragma once
#ifdef __cplusplus
extern "C" {
#endif
#define DEFAULT_CACHE_LINE_SIZE  64
#include <stddef.h>
size_t cache_line_size(void);
#ifdef __cplusplus
}
#endif
