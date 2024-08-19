inline const int& max(const int& a, const int& b)
{
return a < b ? b : a;
}
int main()
{
#pragma omp simd
for ( int i = 0; i < 20; ++i )
{
const int j = max(i, 1);
}
return 0;
}
