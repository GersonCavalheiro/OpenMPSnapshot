#include "pr.h"
#include "timer.h"
#include <vector>
#include "prop_blocking.h"
#define PR_VARIANT "pb" 

void PRSolver(int m, int nnz, IndexT *in_row_offsets, IndexT *in_column_indices, IndexT *out_row_offsets, IndexT *out_column_indices, int *degrees, ScoreT *scores) {
int num_threads = 1;
#pragma omp parallel
{
num_threads = omp_get_num_threads();
}
printf("Launching PR solver (%d threads) using propogation blocking...\n", num_threads);
const ScoreT base_score = (1.0f - kDamp) / m;
vector<ScoreT> sums(m, 0);
int num_bins = (m-1) / BIN_WIDTH + 1; 

int iter = 0;
double error = 0;
preprocessing(m, nnz, out_row_offsets, out_column_indices);

Timer t;
t.Start();
do {
iter ++;
Timer tt;
tt.Start();
#pragma omp parallel for schedule(dynamic, 1024)
for (int u = 0; u < m; u ++) {
const IndexT row_begin = out_row_offsets[u];
const IndexT row_end = out_row_offsets[u+1];
int degree = degrees[u];
ScoreT c = scores[u] / (ScoreT)degree; 
for (IndexT offset = row_begin; offset < row_end; offset ++) {
IndexT v = out_column_indices[offset];
int dest_bin = v >> BITS; 
value_bins[dest_bin][pos[offset]] = c;
}
}
tt.Stop();
if (iter == 1) printf("\truntime [binning] = %f ms.\n", tt.Millisecs());
tt.Start();


tt.Stop();
if (iter == 1) printf("\truntime [accumulate] = %f ms.\n", tt.Millisecs());
tt.Start();
error = 0;
#pragma omp parallel for reduction(+ : error)
for (int u = 0; u < m; u ++) {
ScoreT new_score = base_score + kDamp * sums[u];
error += fabs(new_score - scores[u]);
scores[u] = new_score;
sums[u] = 0;
}
tt.Stop();
if (iter == 1) printf("\truntime [l1norm] = %f ms.\n", tt.Millisecs());
printf(" %2d    %lf\n", iter, error);
if (error < EPSILON) break;
} while(iter < MAX_ITER);
t.Stop();

printf("\titerations = %d.\n", iter);
printf("\truntime [%s] = %f ms.\n", PR_VARIANT, t.Millisecs());
return;
}

