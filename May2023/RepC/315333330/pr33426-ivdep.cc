void foo(int n, int *a, int *b, int *c, int *d, int *e) {
int i;
#pragma GCC ivdep
for (i = 0; i < n; ++i) {
a[i] = b[i] + c[i];
}
}
