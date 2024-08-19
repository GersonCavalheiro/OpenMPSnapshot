#include "solver.hpp"

#include <vector>
#include "omp.h"

int LCSSolver::Solve(size_t N, const string& s1, const string& s2) {
vector<vector<int>> dp(N + 1, vector<int>(N + 1, 0));
for (int i = 1; i <= N; ++i) {
for (int j = 1; j <= N; ++j) {
if (s1[i - 1] == s2[j - 1]) {
dp[i][j] = dp[i - 1][j - 1] + 1;
} else {
dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
}
}
}
return dp[N][N];
}

int LCSSolver::SolveLessMemory(size_t N, const string& s1, const string& s2) {
vector<vector<int>> dp(2, vector<int>(N + 1, 0));
for (int i = 1; i <= N; ++i) {
for (int j = 1; j <= N; ++j) {
if (s1[i - 1] == s2[j - 1]) {
dp[i & 1][j] = dp[(i - 1) & 1][j - 1] + 1;
} else {
dp[i & 1][j] = max(dp[(i - 1) & 1][j], dp[i & 1][j - 1]);
}
}
}
return dp[N & 1][N];
}

int LCSSolver::SolveParallel(size_t N, const string& s1, const string& s2) {
vector<vector<int>> dp(N + N + 1);
for (int i = 0; i <= N + N; ++i) {
dp[i].resize(min(i + 1, static_cast<int>(N) + 1));
}
for (int sum = 2; sum <= N + N; ++sum) {
int start = 1, finish = sum - 1;
if (sum > N) {
start = sum - N;
finish = N;
}
#pragma omp parallel for schedule(static)
for (int i = start; i <= finish; ++i) {
int j = sum - i;
if (s1[i - 1] == s2[j - 1]) {
dp[sum][i] = dp[sum - 2][i - 1] + 1;
} else {
dp[sum][i] = max(dp[sum - 1][i - 1], dp[sum - 1][i]);
}
}
}
return dp[N + N][N];
}

int LCSSolver::SolveParallelLessMemory(size_t N, const string& s1, const string& s2) {
vector<vector<int>> dp(3, vector<int>(N + 1, 0));
for (int sum = 2; sum <= N + N; ++sum) {
int cur_sum = sum % 3;
int prev_sum = (cur_sum + 2) % 3;
int prev_prev_sum = (cur_sum + 1) % 3;
int start = 1, finish = sum - 1;
if (sum > N) {
start = sum - N;
finish = N;
}
#pragma omp parallel for schedule(static)
for (int i = start; i <= finish; ++i) {
int j = sum - i;
if (s1[i - 1] == s2[j - 1]) {
dp[cur_sum][i] = dp[prev_prev_sum][i - 1] + 1;
} else {
dp[cur_sum][i] = max(dp[prev_sum][i - 1], dp[prev_sum][i]);
}
}
}

return dp[(N + N) % 3][N];
}

