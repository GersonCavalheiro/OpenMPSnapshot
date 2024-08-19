#pragma omp declare target
int a[] = { 1, 2, 3 };
extern int b[];			
extern int c[];			
extern int d[];			
int d[3];
#pragma omp end declare target
int c[3];
int e[] = { 1, 2, 3 };
#pragma omp declare target to (e)
extern int f[];
#pragma omp declare target to (f) 
extern int g[];
#pragma omp declare target to (g) 
int g[3];
extern int h[];
int h[3];
#pragma omp declare target to (h)
int i[] = { 1, 2, 3 };
int j[] = { 1, 2, 3 };
extern int k[];
extern int l[];
extern int m[];
extern int n[];
extern int o[];
extern int p[];
int k[3];
int l[3];
int q;
void
foo (void)
{
#pragma omp target update to (q) to (i)
#pragma omp target map (tofrom: j)
;
#pragma omp target update from (q) from (k)
#pragma omp target map (to: l)
;
#pragma omp target update from (q) from (m)	
#pragma omp target map (from: n)		
;
#pragma omp target update to (q) to (o)	
#pragma omp target map (from: p)		
;
}
int o[3];
int p[3];
