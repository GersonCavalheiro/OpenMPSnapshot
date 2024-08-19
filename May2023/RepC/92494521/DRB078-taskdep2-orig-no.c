#include <assert.h> 
#include <unistd.h>
int main()
{
int i=0;
#pragma omp parallel
#pragma omp single
{
#pragma omp task depend (out:i)
{
sleep(3);
i = 1;    
}
#pragma omp task depend (out:i)
i = 2;    
}
assert (i==2);
return 0;
} 