int h;
int main(void)
{
int i,j;
for (i = 0; i < 10; ++i) {
if (i % 2) {
#pragma omp task out(h)
h=i-1;
} else {
#pragma omp task in(h)
h+=1;
}
}
#pragma analysis_check assert live_in(h) live_out(h) dead(i, j)
#pragma omp task inout(h)
h += 4;
return 0;
}