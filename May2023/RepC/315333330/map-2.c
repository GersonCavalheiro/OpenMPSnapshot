void
foo (int *p, int (*q)[10], int r[10], int s[10][10])
{
int a[10], b[10][10];
#pragma omp target map (tofrom: p[-1:2])
;
#pragma omp target map (tofrom: q[-1:2][0:10])
;
#pragma omp target map (tofrom: q[-1:2][-2:10]) 
;
#pragma omp target map (tofrom: r[-1:2])
;
#pragma omp target map (tofrom: s[-1:2][:])
;
#pragma omp target map (tofrom: s[-1:2][-2:10]) 
;
#pragma omp target map (tofrom: a[-1:2])	 
;
#pragma omp target map (tofrom: b[-1:2][0:])	 
;
#pragma omp target map (tofrom: b[1:2][-2:10]) 
;
#pragma omp target map (tofrom: p[2:-3])	 
;
#pragma omp target map (tofrom: q[2:-3][:])	 
;
#pragma omp target map (tofrom: q[2:3][0:-1])	 
;
#pragma omp target map (tofrom: r[2:-5])	 
;
#pragma omp target map (tofrom: s[2:-5][:])	 
;
#pragma omp target map (tofrom: s[2:5][0:-4])	 
;
#pragma omp target map (tofrom: a[2:-5])	 
;
#pragma omp target map (tofrom: b[2:-5][0:10]) 
;
#pragma omp target map (tofrom: b[2:5][0:-4]) 
;
}
