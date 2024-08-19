#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <omp.h>
#include "utils.h"
#define MAXN 10000005
#define MAX_THREAD 4
#define MIN(X, Y) ((X) < (Y)) ? (X) : (Y)
uint32_t prefix_sum[MAXN];
int main() {
int n;
uint32_t key;
while (scanf("%d %" PRIu32, &n, &key) == 2) {
int num_thread = MIN(n, MAX_THREAD);
int block_size = (n + num_thread - 1) / num_thread;
omp_set_num_threads(num_thread);
#pragma omp parallel for
for (int t = 0; t < num_thread; t ++) {
int left = block_size * t + 1, right = MIN(block_size * (t + 1), n);
uint32_t sum = 0;
for (int i = left; i <= right; i ++) {
sum += encrypt(i, key);
prefix_sum[i] = sum;
}
}
for (int t = 0; t < num_thread; t ++) {
int left = block_size * t + 1, right = MIN(block_size * (t + 1), n);
if (left <= right) prefix_sum[right] += prefix_sum[left-1];
}
#pragma omp parallel for
for (int t = 1; t < num_thread; t ++) {
int left = block_size * t + 1, right = MIN(block_size * (t + 1), n);
int prev_block_sum = prefix_sum[left-1];
for (int i = left; i < right; i ++)
prefix_sum[i] += prev_block_sum;
}
output(prefix_sum, n);
}
return 0;
}