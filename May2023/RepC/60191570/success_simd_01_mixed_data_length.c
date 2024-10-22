int test(void)
{
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 4)
int i;
unsigned char __attribute__((aligned(64))) a[102];
float __attribute__((aligned(64))) b[102];
#pragma omp simd 
for (i=0; i<101; i++)
{
a[i] = (unsigned char)2;
}
#pragma omp simd 
for (i=0; i<101; i++)
{
b[i] = 10.0f;
}
a[101] = 8;
b[101] = 7.0f;
#pragma omp simd
for (i=0; i<101; i++)
{
b[i] += 6.0f;
a[i] = b[i];
}
for (i=0; i<101; i++)
{
if (a[i] != 16)
{
printf("ERROR: a[%d] == %d\n", i, a[i]);
return 1;
}
if (b[i] != 16)
{
printf("ERROR: b[%d] == %d\n", i, a[i]);
return 1;
}
}
if (a[101] != 8)
{
printf("ERROR: a[%d] == %d\n", i, a[101]);
return 1;
}
if (b[101] != 7.0f)
{
printf("ERROR: b[%d] == %d\n", i, a[101]);
return 1;
}
#else
#warning "This compiler is not supported"
#endif
return 0;
}
int main(int argc, char *argv[])
{
return test();
}
