#pragma once
#include <stdint.h>
#include "stdafx.h"
typedef enum {
pfdg_command_none = 0,
pfdg_command_sieve = 1,
pfdg_command_mem = 2,
pfdg_command_read = 3,
pfdg_command_help = 4,
pfdg_command_test = 5
} pfdg_command_t;
static const char* const pfdg_str_cmd[] =
{
"",
"sieve",
"mem",
"read",
"help",
"test",
0
};
#define PFDG_STR_CMD(x) pfdg_str_cmd[x]
typedef enum {
pfdg_arg_none = 0,
pfdg_arg_start = 1,
pfdg_arg_end = 2,
pfdg_arg_threads = 3,
pfdg_arg_chunks = 4,
pfdg_arg_maxmem = 5,
pfdg_arg_outfile = 6,
pfdg_arg_infile = 7
} pfdg_arg_t;
static const char* const pfdg_str_arg[] =
{
"",
"start",
"end",
"threads",
"chunks",
"maxmem",
"outfile",
"infile",
0
};
#define PFDG_STR_ARG(x) pfdg_str_arg[x]
typedef struct {
pfdg_command_t command;
uint64_t start;
uint64_t end;
int threads;
int buffer_count;
uint64_t chunks;
uint64_t maxmem;
uint64_t file_size;
char* outfile;
char* infile;
pfdg_error_t error;
const char* message;
} pfdg_args_t;
pfdg_args_t* pfdg_cli_parse(const int argc, char** argv);
void pfdg_cli_destroy(pfdg_args_t* args);
void pfdg_cli_print_bytes(uint64_t bytes);
