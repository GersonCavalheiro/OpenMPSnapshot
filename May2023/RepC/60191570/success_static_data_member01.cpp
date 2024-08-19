#include<assert.h>
struct Test 
{
static int n;
static int m;
int x; 
#pragma omp task inout(n,m,x)
void foo() 
{
n++;
m++;
x++;
}
};
void inc(int * ptr) 
{
(*ptr)++;
}
int Test::n = 1;
int Test::m = 1;
int main() 
{
Test a,b;
int * ptr_n = &(Test::n);
assert(Test::n == 1 && Test::m == 1);
a.foo();    
#pragma omp taskwait
assert(Test::n == 2 && Test::m == 2);
b.foo();    
#pragma omp taskwait
assert(Test::n == 3 && Test::m == 3);
inc(ptr_n);
assert(Test::n == 4 && Test::m == 3);
}
