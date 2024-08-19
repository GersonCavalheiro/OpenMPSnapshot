#pragma once
#include "stdafx.h"
typedef struct
{
BITARRAY_WORD * data;
uint64_t capacity;
uint64_t actual_capacity;
bool oddonly;
} bitarray;
uint64_t bitarray_get_required_mem(const uint64_t capacity, const bool oddonly);
bitarray * bitarray_create(const uint64_t capacity, const bool oddonly);
void bitarray_set(bitarray * const b, const uint64_t i);
void bitarray_unset(bitarray * const b, const uint64_t i);
void bitarray_setval(bitarray * const b, const uint64_t i, const bool val);
bool bitarray_get(bitarray * const b, const uint64_t i);
uint64_t bitarray_count(bitarray * const arr, const bool value);
void bitarray_delete(bitarray * arr);
BITARRAY_WORD bitarray_serialize_to_file(const bitarray * const arr, FILE * const file);
