#include<stdlib.h>
#include<sched.h>
#include<stdio.h>
#include<omp.h>
void func1()
{
int x = 2;
#pragma omp task shared(x) mergeable
{
x++;
}
#pragma omp taskwait
printf("VALUE : %d\n",x); 
}
void func2()
{
int x = 2;
#pragma omp task mergeable
{
x++;
}
#pragma omp taskwait
printf("VALUE : %d\n",x); 
}
int main()
{
func1();
func2();
}
