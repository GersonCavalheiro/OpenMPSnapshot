#include "stdafx.h"
#include "pfdg.h"
#include "timer.h"
#include "utils.h"
#include "pattern.h"
#include "cpuid.h"
#include "cli.h"
#include <math.h>
void print_timestamp_diff(const PFDG_TIMESTAMP t_start, const PFDG_TIMESTAMP t_end)
{
const PFDG_TIMESTAMP diff = pfdg_timestamp_diff(t_start, t_end);
printf("Time: %f seconds\n", pfdg_timestamp_microseconds(diff) / 1000000.0);
}
void fill_test_omp(const uint64_t test_size, const int threads)
{
const uint64_t pattern_length = 1152;
printf("\n\nPattern fill %llu bytes to %llu MiB, %d threads test:\n", pattern_length * sizeof(BITARRAY_WORD), test_size / 1048576, threads);
omp_set_num_threads(threads);
uint64_t i;
PFDG_TIMESTAMP t_start, t_end;
pfdg_timestamp_get(&t_start);
uint64_t* mem1 = ALIGNED_MALLOC(test_size, 32);
#pragma omp parallel for
for (i = 0; i < test_size / pattern_length / 8; ++i)
{
const uint64_t cp = (test_size - i * pattern_length) * sizeof(BITARRAY_WORD);
memcpy_aligned32_avx(mem1 + i * pattern_length, cp, pfdg_pattern, MIN(cp, pattern_length * sizeof(BITARRAY_WORD)));
}
printf("AVX-256 ");
pfdg_timestamp_get(&t_end);
print_timestamp_diff(t_start, t_end);
ALIGNED_FREE(mem1);
pfdg_timestamp_get(&t_start);
mem1 = ALIGNED_MALLOC(test_size, 16);
#pragma omp parallel for
for (i = 0; i < test_size / pattern_length / 8; ++i)
{
const uint64_t cp = (test_size - i * pattern_length) * sizeof(BITARRAY_WORD);
memcpy_aligned16_sse(mem1 + i * pattern_length, cp, pfdg_pattern, MIN(cp, pattern_length * sizeof(BITARRAY_WORD)));
}
printf("AVX/SSE-128 ");
pfdg_timestamp_get(&t_end);
print_timestamp_diff(t_start, t_end);
ALIGNED_FREE(mem1);
pfdg_timestamp_get(&t_start);
mem1 = malloc(test_size);
#pragma omp parallel for
for (i = 0; i < test_size / pattern_length / 8; ++i)
{
const uint64_t cp = (test_size - i * pattern_length) * sizeof(BITARRAY_WORD);
memcpy_aligned8(mem1 + i * pattern_length, cp, pfdg_pattern, MIN(cp, pattern_length * sizeof(BITARRAY_WORD)));
}
printf("64-bit ");
pfdg_timestamp_get(&t_end);
print_timestamp_diff(t_start, t_end);
free(mem1);
pfdg_timestamp_get(&t_start);
mem1 = malloc(test_size);
#pragma omp parallel for
for (i = 0; i < test_size / pattern_length / 8; ++i)
{
const uint64_t cp = (test_size - i * pattern_length) * sizeof(BITARRAY_WORD);
memcpy(mem1 + i * pattern_length, pfdg_pattern, MIN(cp, pattern_length * sizeof(BITARRAY_WORD)));
}
printf("Built-in ");
pfdg_timestamp_get(&t_end);
print_timestamp_diff(t_start, t_end);
free(mem1);
}
void memcpy_test()
{
const uint64_t test_size = 1048576 * 1024;
const uint64_t check_value = 0xFFEEDDCCBBAA9988;
pfdg_timestamp_init();
PFDG_TIMESTAMP t_start, t_end;
printf("Sequential copy %llu MiB test:\n", test_size / 1048576);
pfdg_timestamp_get(&t_start);
uint64_t* mem1 = ALIGNED_MALLOC(test_size, 32);
uint64_t* mem2 = ALIGNED_MALLOC(test_size, 32);
mem1[test_size / 8 - 1] = check_value;
mem2[test_size / 8 - 1] = 0;
memcpy_aligned32_avx(mem2, test_size, mem1, test_size);
printf("\nAVX-256: ");
if (mem2[test_size / 8 - 1] == check_value) printf("Success\n");
else printf("Failed\n");
pfdg_timestamp_get(&t_end);
print_timestamp_diff(t_start, t_end);
ALIGNED_FREE(mem1);
ALIGNED_FREE(mem2);
pfdg_timestamp_get(&t_start);
mem1 = ALIGNED_MALLOC(test_size, 16);
mem2 = ALIGNED_MALLOC(test_size, 16);
mem1[test_size / 8 - 1] = check_value;
mem2[test_size / 8 - 1] = 0;
memcpy_aligned16_sse(mem2, test_size, mem1, test_size);
printf("\nAVX/SSE-128: ");
if (mem2[test_size / 8 - 1] == check_value) printf("Success\n");
else printf("Failed\n");
pfdg_timestamp_get(&t_end);
print_timestamp_diff(t_start, t_end);
ALIGNED_FREE(mem1);
ALIGNED_FREE(mem2);
pfdg_timestamp_get(&t_start);
mem1 = malloc(test_size);
mem2 = malloc(test_size);
mem1[test_size / 8 - 1] = check_value;
mem2[test_size / 8 - 1] = 0;
memcpy_aligned8(mem2, test_size, mem1, test_size);
printf("\n64-bit: ");
if (mem2[test_size / 8 - 1] == check_value) printf("Success\n");
else printf("Failed\n");
pfdg_timestamp_get(&t_end);
print_timestamp_diff(t_start, t_end);
free(mem1);
free(mem2);
pfdg_timestamp_get(&t_start);
mem1 = malloc(test_size);
mem2 = malloc(test_size);
mem1[test_size / 8 - 1] = check_value;
mem2[test_size / 8 - 1] = 0;
memcpy(mem2, mem1, test_size);
printf("\nBuilt-in: ");
if (mem2[test_size / 8 - 1] == check_value) printf("Success\n");
else printf("Failed\n");
pfdg_timestamp_get(&t_end);
print_timestamp_diff(t_start, t_end);
free(mem1);
free(mem2);
fill_test_omp(test_size * 2, 4);
fill_test_omp(test_size * 2, 3);
fill_test_omp(test_size * 2, 2);
fill_test_omp(test_size * 2, 1);
}
int main(const int argc, char** argv)
{
cpuid_init();
pfdg_args_t* args = pfdg_cli_parse(argc, argv);
if (args == NULL)
{
printf("Out of memory");
return 2;
}
if (args->error != pfdg_success)
{
printf("PrimeFDG\nCopyright (c) 2022 by Oron Feinerman\n");
switch (args->error)
{
case pfdg_error_command:
if (args->message == NULL)
printf("Error: No command specified");
else
printf("Error: Unrecognized command \"%s\"", args->message);
break;
case pfdg_error_number_of_args:
printf("Error: Wrong number of arguments");
break;
case pfdg_error_arg:
if (args->message == NULL)
printf("Error: Argument parsing failed");
else
printf("Error: Unrecognized argument \"%s\"", args->message);
break;
case pfdg_error_arg_value:
if (args->message == NULL)
printf("Error: Argument value parsing failed");
else
printf("Error: Inavlid argument value \"%s\"", args->message);
break;
case pfdg_error_arg_precondition:
if (args->message == NULL)
printf("Error: Argument validation failed");
else
printf("Error: Argument \"%s\" value does not meet precondition", args->message);
break;
case pfdg_error_mem_base:
if (args->message == NULL)
printf("Error: Not enough memory for essential data - Try increasing the value of \"maxmem\"");
else
printf("Error: Not enough memory for essential data (%s more bytes required) - Try increasing the value of \"maxmem\"", args->message);
break;
case pfdg_error_mem_min:
if (args->message == NULL)
printf("Error: Not enough memory for per-thread computation data - Try increasing the value of \"maxmem\" or \"chunks\", or decreasing the value of \"threads\"");
else
printf("Error: Not enough memory for per-thread computation data (%s more bytes required) - Try increasing the value of \"maxmem\" or \"chunks\", or decreasing the value of \"threads\"", args->message);
break;
case pfdg_error_mem_chunk_size:
printf("Error: Not enough memory for per-thread computation data - Try increasing the value of \"maxmem\" or decreasing the value of \"threads\"");
break;
}
printf("\nRun \"primefdg help\" for usage information.\n");
return 1;
}
if (args->command == pfdg_command_help)
{
printf("PrimeFDG\nCopyright (c) 2022 by Oron Feinerman\n\nUsage: primefdg <command> [<argument>=<value>...]\n(the syntax is similar to dd)\n");
printf("\nAvailable commands:\n");
printf("%s: Prints this message. This command does not accept arguments.\n", PFDG_STR_CMD(pfdg_command_help));
printf("%s: Runs a memcpy speed test. This command does not accept arguments.\n", PFDG_STR_CMD(pfdg_command_test));
printf("%s: Finds prime numbers within a given range.\n", PFDG_STR_CMD(pfdg_command_sieve));
printf("%s: Calculates the estimated memory usage of a %s command with the same arguments.\n", PFDG_STR_CMD(pfdg_command_mem), PFDG_STR_CMD(pfdg_command_sieve));
printf("\nAvailable arguments:\n");
printf("%s: The start of the range in which to find prime numbers. Supports exponent specifiers. Values below 3 are set to 3. If unspecified, will be set to 0.\n", PFDG_STR_ARG(pfdg_arg_start));
printf("%s: The end of the range in which to find prime numbers. Supports exponent specifiers. Required.\n", PFDG_STR_ARG(pfdg_arg_end));
printf("%s: The number of threads to use for computation. If unspecified or equal to 0, will be set to the number of CPU threads available.\n", PFDG_STR_ARG(pfdg_arg_threads));
printf("%s: The number of tasks that the computation will be split into. Supports exponent specifiers. This is an advanced option that can be used to tune performance. If unspecified, the value will be calculated as such: %s * 2^MAX(floor(log10(%s)) - 1, 0)\n", PFDG_STR_ARG(pfdg_arg_chunks), PFDG_STR_ARG(pfdg_arg_threads), PFDG_STR_ARG(pfdg_arg_end));
printf("%s: Limits the maximum amount of memory to use for computation. Supports exponent specifiers. If the value of this argument is larger than the amount of memory required by default, and an output file is specified, the extra memory will be used for buffering. Note that the actual memory usage will usually be slightly higher.\n", PFDG_STR_ARG(pfdg_arg_maxmem));
printf("%s: The path to a file to which the computation results will be stored. If unspecified, the results will be discarded immediately.\n", PFDG_STR_ARG(pfdg_arg_outfile));
printf("\nUsage example: primefdg sieve end=1e12 threads=8\n");
printf("\nExponent specifiers (base 10): k,m,g,t,e\nExponent specifiers (base 2): K,M,G,T\n");
return 0;
}
if (args->command == pfdg_command_test)
{
memcpy_test();
return 0;
}
printf("Using %i threads, %llu chunks\n", args->threads, args->chunks);
printf("Maximum memory usage: ");
pfdg_cli_print_bytes(args->maxmem);
printf("\n");
if (args->file_size > 0)
{
printf("Expected file size: ");
pfdg_cli_print_bytes(args->file_size);
printf("\n");
}
if (args->command == pfdg_command_mem)
{
return 0;
}
pfdg_timestamp_init();
PFDG_TIMESTAMP t_start, t_end;
pfdg_timestamp_get(&t_start);
uint64_t res;
const bool r = pfdg_sieve_parallel(args->start, args->end, args->chunks, args->buffer_count, args->outfile, &res);
if (!r)
{
printf("Out of memory!");
return 1;
}
pfdg_timestamp_get(&t_end);
print_timestamp_diff(t_start, t_end);
printf("Number of primes found: %llu\n", res);
return 0;
}
