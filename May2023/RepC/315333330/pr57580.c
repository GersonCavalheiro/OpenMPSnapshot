#define PS \
_Pragma("omp parallel num_threads(2)") \
{ \
_Pragma("omp single") \
{ \
ret = 0; \
} \
}
int
main ()
{
int ret;
_Pragma("omp parallel num_threads(3)")
{
_Pragma("omp single")
{
ret = 0;
}
}
_Pragma("omp parallel num_threads(4)") { _Pragma("omp single") { ret = 0; } }
{ _Pragma("omp parallel num_threads(5)") { _Pragma("omp single") { ret = 0; } } }
PS
PS
return ret;
}
