void foo() { }
int main() {
#pragma omp for collapse(1)
for (int i = 1; i <= 151; i+=31)
foo();
}
