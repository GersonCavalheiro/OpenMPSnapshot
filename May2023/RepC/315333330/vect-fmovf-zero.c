#pragma GCC target "+nosve"
#define N 32
void
foo (float *output)
{
int i = 0;
for (i = 0; i < N; i++)
output[i] = 0.0;
}
