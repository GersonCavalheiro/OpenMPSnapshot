#include<iostream>
#include<cassert>
using namespace std;
class A {
public:
static int counter; 
static int pcounter; 
#pragma omp threadprivate(pcounter)
};
int A::counter=0; 
int A::pcounter=0; 
A a; 
void foo()
{
a.counter++; 
a.pcounter++; 
}
int main()
{ 
#pragma omp parallel 
{
foo();
}
assert (A::pcounter == 1);
cout<<A::counter <<" "<< A::pcounter<<endl;
return 0;   
}
