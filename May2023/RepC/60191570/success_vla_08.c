int foo(int n) {
int v[n];
int v2[10];
#pragma omp task inout(v, v2)
{}
}
