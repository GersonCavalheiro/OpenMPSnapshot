void bar(int&);
int x=0;
int y;
void foo() {
#pragma analysis_check assert upper_exposed(x) defined(x,y)
#pragma omp task
{
#pragma analysis_check assert defined(x)
#pragma omp task
x = 1;
y = x;
}
#pragma analysis_check assert undefined(x,y)
#pragma omp task
bar(x);
}
