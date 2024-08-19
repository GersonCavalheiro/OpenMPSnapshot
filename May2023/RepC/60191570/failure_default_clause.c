void foo( int a )
{
int i;
#pragma omp parallel for default(auto)
for(i=0; i<1; ++i)
a++;
}