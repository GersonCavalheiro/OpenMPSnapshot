#pragma once
#include "bitarray.h"
void io_init(FILE * file, const uint64_t queue_length);
void io_enqueue(bitarray * source);
void io_end();
