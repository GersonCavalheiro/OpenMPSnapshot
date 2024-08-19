#include<assert.h>
struct Test 
{
static int n;
static int m;
int x; 
#pragma omp task inout(n,m,x,s)
void foo(int & s) 
{
n++;
m++;
x++;
s++;
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
a.foo(Test::m);    
#pragma omp taskwait
assert(Test::n == 2 && Test::m == 3);
b.foo(Test::m);    
#pragma omp taskwait
assert(Test::n == 3 && Test::m == 5);
inc(ptr_n);
assert(Test::n == 4 && Test::m == 5);
}
