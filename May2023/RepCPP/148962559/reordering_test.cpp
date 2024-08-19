



#include <omp.h>

#ifdef MKL
#include <mkl.h>
#endif

#include "Utils.hpp"

#include <algorithm>
#include <utils/log.h>
#include <utils/yche_serialization.h>

#include "reordering/other-reorderings/SpMP/CSR.hpp"

using namespace std;
using namespace SpMP;

static const size_t LLC_CAPACITY = 32 * 1024 * 1024;
static const double *bufToFlushLlc = NULL;

void flushLlc() {
double sum = 0;
#pragma omp parallel for reduction(+:sum)
for (size_t i = 0; i < LLC_CAPACITY / sizeof(bufToFlushLlc[0]); ++i) {
sum += bufToFlushLlc[i];
}
FILE *fp = fopen("/dev/null", "w");
fprintf(fp, "%f", sum);
fclose(fp);
}

typedef enum {
BFS = 0,
RCM_WO_SOURCE_SELECTION,
RCM,
} Option;


void WriteToFile(string &my_path, int *p, int n) {
log_info("%s", my_path.c_str());
FILE *pFile = fopen(my_path.c_str(), "wb");
YcheSerializer serializer;
serializer.write_array(pFile, p, n);

fflush(pFile);
fclose(pFile);
}

int main(int argc, char **argv) {
#ifdef USE_LOG
FILE *log_f;
if (argc >= 3) {
log_f = fopen(argv[2], "a+");
log_set_fp(log_f);
}
#endif

if (argc < 2) {
log_error("Usage: reordering_test matrix_in_matrix_market_format");
return -1;
}


auto *A = new CSR(argv[1], 0, true );
auto nnz = A->getNnz();
double bytes = (double) (sizeof(double) + sizeof(int)) * nnz + sizeof(double) * (A->m + A->n);
log_info("m = %d nnz = %lld %f bytes = %f", A->m, nnz, (double) nnz / A->m, bytes);

log_info("original bandwidth %d", A->getBandwidth());

auto *x = MALLOC(double, A->m);
auto *y = MALLOC(double, A->m);

bufToFlushLlc = (double *) _mm_malloc(LLC_CAPACITY, 64);
flushLlc();

auto *perm = MALLOC(int, A->m);
auto *inversePerm = MALLOC(int, A->m);

for (int o = BFS; o <= RCM; ++o) {
auto option = (Option) o;

switch (option) {
case BFS:
log_info("BFS reordering");
break;
case RCM_WO_SOURCE_SELECTION:
log_info("RCM reordering w/o source selection heuristic");
break;
case RCM:
log_info("RCM reordering");
break;
default:
assert(false);
break;
}

double t = -omp_get_wtime();
string my_path;
switch (option) {
case BFS:
A->getBFSPermutation(perm, inversePerm);
my_path = string(argv[1]) + "/" + "bfs.dict";
break;
case RCM_WO_SOURCE_SELECTION:
A->getRCMPermutation(perm, inversePerm, false);
my_path = string(argv[1]) + "/" + "rcm_wo_src_sel.dict";
break;
case RCM:
A->getRCMPermutation(perm, inversePerm);
my_path = string(argv[1]) + "/" + "rcm_with_src_sel.dict";
break;
}
t += omp_get_wtime();

log_info("Constructing permutation takes %g (%.2f gbps)", t, nnz * 4 / t / 1e9);

isPerm(perm, A->m);
isPerm(inversePerm, A->m);
log_info("Finish Get Permutation...");


WriteToFile(my_path, perm, A->m);
}

FREE(x);
FREE(y);

delete A;
}
