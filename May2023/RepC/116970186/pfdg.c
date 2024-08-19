#include "stdafx.h"
#include "pfdg.h"
#include <math.h>
#include "pattern.h"
#include "utils.h"
#include "pfmath.h"
#include "io.h"
uint64_t pfdg_mem_get_base(const uint64_t start, const uint64_t end)
{
const uint64_t pattern_len = (PFDG_PATTERN_LENGTH + 1ULL) * sizeof(BITARRAY_WORD);
const uint64_t capacity = (uint64_t)sqrt((double)end) + 1;
return pattern_len + bitarray_get_required_mem(capacity, true);
}
uint64_t pfdg_mem_get_chunk_size(const uint64_t start, const uint64_t end, const uint64_t chunks)
{
const uint64_t len = end - start;
uint64_t chunk_size = DIVUP(len, chunks);
chunk_size = DIVUP(chunk_size, BITS(BITARRAY_WORD) * 2) * BITS(BITARRAY_WORD) * 2;
return bitarray_get_required_mem(chunk_size, true);
}
uint64_t pfdg_mem_get_chunk_count_by_size(const uint64_t start, const uint64_t end, const uint64_t chunk_size)
{
uint64_t actual_size = (chunk_size / 32) * 32;
uint64_t len = end - start;
len = DIVUP(len, BITS(BITARRAY_WORD) * 2) * 8;
return DIVUP(len, actual_size);
}
uint64_t pfdg_get_file_size(const uint64_t start, const uint64_t end)
{
const uint64_t len = end - start;
return sizeof(pfdg_file_header) + DIVUP(len, BITS(BITARRAY_WORD) * 2) * sizeof(BITARRAY_WORD);
}
bitarray* pfdg_init_bitarray(const uint64_t capacity, const uint64_t offset, const bool use_pattern)
{
bitarray* arr = bitarray_create(capacity, true);
if (!arr) return 0;
const uint64_t len = arr->actual_capacity / BITS(BITARRAY_WORD);
if (use_pattern)
{
uint64_t i = 0;
if (offset < 2)
{
arr->data[0] = pfdg_pattern_prefix;
++i;
}
const uint64_t align = (offset / BITS(BITARRAY_WORD) / 2 - 1 + i) % PFDG_PATTERN_LENGTH;
if (align > 0)
{
const uint64_t words = MIN(len, (PFDG_PATTERN_LENGTH - align));
const uint64_t cp = words * sizeof(BITARRAY_WORD);
memcpy_aligned8(arr->data, cp, pfdg_pattern + align, cp);
i += words;
}
for (; i < len; i += PFDG_PATTERN_LENGTH)
{
const uint64_t cp = (len - i) * sizeof(BITARRAY_WORD);
memcpy_aligned8(arr->data + i, cp, pfdg_pattern, MIN(cp, PFDG_PATTERN_LENGTH * sizeof(BITARRAY_WORD)));
}
}
else
{
for (uint64_t i = 0; i < len; ++i)
arr->data[i] = 0;
}
if (offset < 2) bitarray_set(arr, 1);
const uint64_t extra_bits = arr->actual_capacity - DIVUP(capacity, 2);
arr->data[len - 1] |= (((BITARRAY_WORD)1 << extra_bits) - 1) << (BITS(BITARRAY_WORD) - extra_bits);
return arr;
}
void pfdg_mark(bitarray* const arr, const uint64_t prime, const uint64_t offset)
{
uint64_t i = DIVUP(offset, prime);
if (i < prime) i = prime;
else if (i % 2 == 0) ++i;
uint64_t limit = (uint64_t)ceil(((double)(arr->capacity + offset) / (double) prime - 1) / 2.0);
uint64_t j; 
#pragma omp simd
for (j = i / 2; j < limit; j++)
bitarray_set(arr, (j * 2 + 1) * prime - offset);
}
void pfdg_sieve_seed(bitarray* const arr, const bool skip)
{
for (uint64_t i = skip ? PFDG_PATTERN_SKIP : 3; i < arr->capacity; i += 2)
if (!bitarray_get(arr, i))
pfdg_mark(arr, i, 0);
}
void pfdg_sieve(bitarray* const arr, bitarray* const known, const uint64_t offset, const bool skip)
{
for (uint64_t i = skip ? PFDG_PATTERN_SKIP : 3; i < known->capacity; i += 2)
if (!bitarray_get(known, i))
pfdg_mark(arr, i, offset);
}
bool pfdg_sieve_parallel(const uint64_t start, const uint64_t end, uint64_t chunks, const uint64_t buffers, const char * const file, uint64_t * const prime_count)
{
bitarray* const known = pfdg_init_bitarray((uint64_t)sqrt((double)end) + 1, 0, true);
if (!known) return UINT32_MAX;
pfdg_sieve_seed(known, true);
const uint64_t len = end - start;
*prime_count = 0;
uint64_t chunk_size = DIVUP(len, chunks);
chunk_size = DIVUP(chunk_size, BITS(BITARRAY_WORD) * 2) * BITS(BITARRAY_WORD) * 2;
chunks = DIVUP(len, chunk_size);
FILE* f = NULL;
if (file != NULL)
{
f = fopen(file, "w+b");
if (!f)
{
pfdg_file_header h = PFDG_HEADER_INITIALIZER;
h.data_length = DIVUP(len, BITS(BITARRAY_WORD) * 2) * sizeof(BITARRAY_WORD);
h.number_first = start;
h.number_last = end;
fwrite(&h, sizeof(pfdg_file_header), 1, f);
}
io_init(f, buffers);
}
bool abort = false;
int64_t i;
#pragma omp parallel for ordered schedule(static,1)
for (i = 0; i < (int64_t)chunks; ++i)
{
#pragma omp flush (abort)
if (!abort)
{
const uint64_t offset = start + chunk_size * i;
bitarray* const arr = pfdg_init_bitarray(i == chunks - 1 ? len - chunk_size * i : chunk_size, offset, true);
if (!arr)
{
abort = true;
#pragma omp flush (abort)
}
else
{
pfdg_sieve(arr, known, offset, true);
const uint64_t count = bitarray_count(arr, false);
#pragma omp atomic
*prime_count += count;
if (f != NULL)
{
#pragma omp ordered
{
io_enqueue(arr);
}
}
else
{
bitarray_delete(arr);
}
}
}
}
if (f != NULL) io_end();
if (abort) return false;
if (f != NULL)
{
fseek(f, 40, SEEK_SET);
fwrite(prime_count, sizeof(uint64_t), 1, f);
fclose(f);
}
return true;
}
bool pfdg_generate_pattern(const uint64_t last_prime, const char * const file)
{
bitarray* known = pfdg_init_bitarray(last_prime + 1, 0, true);
if (!known) return false;
pfdg_sieve_seed(known, false);
uint64_t len = 1;
for (uint64_t i = 3; i <= last_prime; i+=2)
if (!bitarray_get(known, i))
len *= i;
bitarray* arr = pfdg_init_bitarray((len + 1) * 2 * BITS(BITARRAY_WORD), 0, false);
if (!arr) return false;
pfdg_sieve(arr, known, 0, false);
bitarray_delete(known);
FILE * f = fopen(file, "w+");
uint64_t next_prime = last_prime + 2;
for (; next_prime < arr->capacity; next_prime += 2)
if (!bitarray_get(arr, next_prime))
break;
fprintf(f, "#pragma once\n\n#include \"stdafx.h\"\n\n
fprintf(f, "
for (uint64_t i = 1; i < arr->actual_capacity / BITS(BITARRAY_WORD); ++i)
fprintf(f, "\t0x%.16llX,\n", arr->data[i]);
fprintf(f, "};\n");
fclose(f);
bitarray_delete(arr);
return true;
}
