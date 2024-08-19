#pragma omp declare simd notinbranch uniform(y)
float
bar (float x, float *y, int)
{
return y[0] + y[1] * x;
}
