typedef struct A {
int n;
} A;
void foo(int N, int *p) {
A var = { N };
#pragma omp task inout([var.n]p)
{}
}
