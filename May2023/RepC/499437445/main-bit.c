#include <stdio.h>
#include <omp.h>
#define MAXN 20
int barriers[MAXN] = {0}, N, all;
int dfsQueen(int row, int cols, int diags, int antiDiags) {
if (row == N) return 1;
int ans = 0, canPlace = all & ~(cols | diags | antiDiags) & ~barriers[row];
while (canPlace) {
int place = canPlace & -canPlace; 
canPlace ^= place;
ans += dfsQueen(row + 1, cols | place, (diags | place) << 1, (antiDiags | place) >> 1);
}
return ans;
}
int main()
{
int count = 1;
char row[MAXN];
while (scanf("%d", &N) == 1) {
for (int i = 0; i < N; i++) {
scanf("%s", row);
for (int j = 0; j < N; j++)
if (row[j] == '*')
barriers[i] |= (1 << j);
}
all = (1 << N) - 1;
int ans = 0;
#pragma omp parallel for reduction(+:ans)
for (int i = 0; i < N; i ++) {
int cols = (1 << i);
if (cols & barriers[0]) continue;
ans += dfsQueen(1, cols, cols << 1, cols >> 1);
}
printf("Case %d: %d\n", count ++, ans);
}
return 0;
}