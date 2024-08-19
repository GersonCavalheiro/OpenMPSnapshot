#include <par-res-kern_general.h>
void bail_out(int error) {
#pragma omp barrier
if (error != 0) exit(EXIT_FAILURE);
}
