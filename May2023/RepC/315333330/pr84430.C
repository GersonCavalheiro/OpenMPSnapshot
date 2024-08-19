void
foo ()
{
auto a = [] {
#pragma omp simd
for (int i = 0; i < 10; ++i)
;
};
}
