float coeffs[10] = {0.9, 1.1, 1.2, 1.3, 1.4,
1.5, 1.6, 1.7, 1.8, 1.9};
void foo (float *out, float d0)
{
int i;
#pragma analysis_check assert induction_var(i:0:15:1)
for (i=0; i < 16; i++)
{
float tmp  = coeffs[0] * d0;
tmp += coeffs[1] * d0;
out[i] = tmp;
}
}
void bar (float *out, float d0)
{
int i, tmp;
#pragma analysis_check assert induction_var(i:0:15:1)
for (i=0; i < 16; i++)
{
tmp = coeffs[0] * d0;
tmp += coeffs[1] * d0;
out[i] = tmp;
}
}
