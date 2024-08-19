const int size = 2000;
float a[size][size];
float b[size][size];
float c[size][size];
int main()
{
for (int i = 0; i < size; ++i) {
for (int j = 0; j < size; ++j) {
a[i][j] = (float)i + j;
b[i][j] = (float)i - j;
c[i][j] = 0.0f;
}
}
#pragma omp parallel for default(none) shared(a,b,c)
for (int i = 0; i < size; ++i) {
for (int j = 0; j < size; ++j) {
for (int k = 0; k < size; ++k) {
c[i][j] += a[i][k] * b[k][j];
}
}
}
return 0;
}
