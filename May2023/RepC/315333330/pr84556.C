void
foo ()
{
auto x = [] ()
{
#pragma omp simd
for (int i = 0; i < 8; ++i)
;
};
}
