#include <stdio.h>
#include <omp.h>
#define MAXN 10005
#define MAXM 1000005
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
int weights[MAXN], values[MAXN];
int dp[2][MAXM] = {};
int main() {
int N, M;
scanf("%d%d", &N, &M);
for (int i = 0; i < N; i ++)
scanf("%d%d", &weights[i], &values[i]);
#pragma omp parallel
{
for (int i = 0; i < N; i ++) {
#pragma omp for
for (int j = M; j >= weights[i]; j --)
dp[1-i%2][j] = MAX(dp[i%2][j], dp[i%2][j-weights[i]] + values[i]);
#pragma omp for
for (int j = weights[i] - 1; j >= 0; j --)
dp[1-i%2][j] = dp[i%2][j];
}
}
printf("%d\n", dp[N%2][M]);
return 0;
}