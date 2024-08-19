void bar(int* x)
{
#pragma oss task in([1](char*)x)
*x = ~(*x); 
#pragma oss taskwait
}
