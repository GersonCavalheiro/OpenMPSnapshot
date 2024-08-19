int main()
{
int n = 0;
#pragma omp target map(tofrom: n)
#pragma omp parallel for reduction (+: n)
for (int i = [](){ return 3; }(); i < 10; ++i)
n++;
return n;
}
