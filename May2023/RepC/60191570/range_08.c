void foo()
{
#pragma omp parallel
#pragma omp single
{
unsigned int lim = 256;
unsigned int ub = lim / 128;
int i;
#pragma analysis_check assert range(ub:2U:2U:0)
for(i = 0; i < ub; i++) 
#pragma analysis_check assert range(i:0:1U:0)
sleep(1);
}
}