#define N  4
unsigned long long int square[N][N];
void sequential_(int x, int y)
{
#pragma analysis_check assert defined(square[x][y], x, y) upper_exposed(square[x][y], square[x-1][y], square[x][y-1])
for (x = 1; x < N; x++) {
for (y = 1; y < N; y++) {
square[x][y] = square[x-1][y] + square[x][y] + square[x][y-1];
}
}
}
void sequential(int a, int b)
{
#pragma analysis_check assert defined(square[a][b]) upper_exposed(square[a][b], square[a-1][b], square[a][b-1], a, b)
sequential_(a, b);
}
void ompss(int h, int j)
{
#pragma analysis_check assert defined(square[h][j]) upper_exposed(square[h][j], square[h-1][j], square[h][j], square[h][j-1], h, j)
sequential(h, j);
}