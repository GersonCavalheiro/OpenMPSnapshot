#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>
#include <stdio.h>
#include <sys/mta_task.h>
#include "make_graph.h"
int main(int argc, char* argv[]) {
int log_numverts;
unsigned int start, time_taken;
size_t i;
int64_t nedges, actual_nedges;
packed_edge* result;
log_numverts = 16; 
if (argc >= 2) log_numverts = atoi(argv[1]);
#pragma mta fence
start = mta_get_clock(0);
make_graph(log_numverts, INT64_C(16) << log_numverts, 1, 2, &nedges, &result);
#pragma mta fence
time_taken = mta_get_clock(start);
actual_nedges = nedges;
fprintf(stderr, "%" PRIu64 " edge%s generated in %fs (%f Medges/s)\n", actual_nedges, (actual_nedges == 1 ? "" : "s"), time_taken * mta_clock_period(), 1. * actual_nedges / time_taken * 1.e-6 / mta_clock_period());
free(result);
return 0;
}
