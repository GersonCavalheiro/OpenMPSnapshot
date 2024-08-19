#include "knn.h"
#include <omp.h>
#include "sedm.h"
#include "qselect.h"
#include "matrix.h"
#include "def.h"
knn_result *knn(
const elem_t *X, size_t n, const elem_t *Y, size_t Y_begin, size_t m, 
size_t d, size_t k, size_t t, knn_result **prev_result) {
knn_result *res = (knn_result *) malloc(sizeof(knn_result));
res->m = n;
res->k = k;
res->n_idx = (size_t *) malloc(n * k * sizeof(size_t));
res->n_dist = (elem_t *) malloc(n * k * sizeof(elem_t));
int threadnum = omp_get_max_threads();
omp_set_num_threads(threadnum);
t = min(t, n);
elem_t *D = (elem_t *) malloc(t * m * sizeof(elem_t));
size_t *ind = (size_t *) malloc(t * m * sizeof(size_t));
for(size_t X_begin = 0 ; X_begin < n ; X_begin += t) {
size_t X_end = min(X_begin + t, n);
size_t slice_size = X_end - X_begin;
const elem_t *X_slice = MATRIX_ROW(X, X_begin, n, d);
sedm(X_slice, slice_size, Y, m, d, D);
#pragma omp parallel for
for(int tid = 0 ; tid < slice_size ; tid++) {
elem_t *Di = MATRIX_ROW(D, tid, t, m);
size_t *ind_i = MATRIX_ROW(ind, tid, t, m);
gen_indices(ind_i, Y_begin, m);
qselect(k, Di, ind_i, m);
for(size_t i = 0 ; i < k - 1 ; i++) {
for(size_t j = 0 ; j < k - i - 1 ; j++) {
if(Di[j] > Di[j + 1]) {
SWAP(Di[j], Di[j + 1]);
SWAP(ind_i[j], ind_i[j + 1]);
}
}
}
if(*prev_result == NULL) {
for(size_t j = 0 ; j < k ; j++) {
MATRIX_ELEM(res->n_dist, X_begin + tid, j, n, k) = Di[j];
MATRIX_ELEM(res->n_idx, X_begin + tid, j, n, k) = ind_i[j];
}
} else {
elem_t *prev_dist = MATRIX_ROW((*prev_result)->n_dist, X_begin + tid, n, k);
size_t *prev_idx = MATRIX_ROW((*prev_result)->n_idx, X_begin + tid, n, k);
size_t p_idx = 0;
size_t d_idx = 0;
for(size_t j = 0 ; j < k ; j++) {
if(prev_dist[p_idx] <= Di[d_idx]) {
MATRIX_ELEM(res->n_dist, X_begin + tid, j, n, k) = prev_dist[p_idx];
MATRIX_ELEM(res->n_idx, X_begin + tid, j, n, k) = prev_idx[p_idx];
p_idx += 1;
} else {
MATRIX_ELEM(res->n_dist, X_begin + tid, j, n, k) = Di[d_idx];
MATRIX_ELEM(res->n_idx, X_begin + tid, j, n, k) = ind_i[d_idx];
d_idx += 1;
}
}
}
}
}
free(D);
free(ind);
if(*prev_result != NULL)
delete_knn(*prev_result);
return res;
}
void delete_knn(knn_result *knn) {
free(knn->n_idx);
free(knn->n_dist);
free(knn);
}
