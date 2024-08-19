int ar[100];
void foo(int *a) {
#pragma GCC ivdep
for (auto &i : ar) {
i *= *a;
}
}
