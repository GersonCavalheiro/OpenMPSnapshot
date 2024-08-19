#pragma oss task verified inout(*x)
void foo(int *x) {
}
int main() {
int x;
#pragma oss task verified inout(x)
{}
foo(&x);
#pragma oss taskwait
}
