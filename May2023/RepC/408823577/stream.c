# include <stdio.h>
# include <math.h>
# include <float.h>
# include <limits.h>
# include <sys/time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
# define N	40000000
# define NTIMES	50
# define OFFSET	0
# define HLINE "-------------------------------------------------------------\n"
# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif
static double	a[N+OFFSET],
b[N+OFFSET],
c[N+OFFSET];
static double	avgtime[4] = {0}, maxtime[4] = {0},
mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};
static char	*label[4] = {"Copy:      ", "Scale:     ",
"Add:       ", "Triad:     "};
static double	bytes[4] = {
2 * sizeof(double) * N,
2 * sizeof(double) * N,
3 * sizeof(double) * N,
3 * sizeof(double) * N
};
extern double mysecond();
extern void checkSTREAMresults();
#ifdef TUNED
extern void tuned_STREAM_Copy();
extern void tuned_STREAM_Scale(double scalar);
extern void tuned_STREAM_Add();
extern void tuned_STREAM_Triad(double scalar);
#endif
int
main()
{
int			quantum, checktick();
int			BytesPerWord;
register int	j, k;
double		scalar, t, times[4][NTIMES];
printf(HLINE);
BytesPerWord = sizeof(double);
printf("This system uses %d bytes per DOUBLE PRECISION word.\n",
BytesPerWord);
printf(HLINE);
printf("Array size = %d, Offset = %d\n" , N, OFFSET);
printf("Total memory required = %.1f MB.\n",
(3.0 * BytesPerWord) * ( (double) N / 1048576.0));
printf("Each test is run %d times, but only\n", NTIMES);
printf("the *best* time for each is used.\n");
#ifdef _OPENMP
printf(HLINE);
#pragma omp parallel private(k)
{
k = omp_get_num_threads();
printf ("Number of Threads requested = %i\n",k);
}
#endif
for (j=0; j<N; j++) {
a[j] = 1.0;
b[j] = 2.0;
c[j] = 0.0;
}
printf(HLINE);
if  ( (quantum = checktick()) >= 1) 
printf("Your clock granularity/precision appears to be "
"%d microseconds.\n", quantum);
else
printf("Your clock granularity appears to be "
"less than one microsecond.\n");
t = mysecond();
#pragma omp parallel for
for (j = 0; j < N; j++)
a[j] = 2.0E0 * a[j];
t = 1.0E6 * (mysecond() - t);
printf("Each test below will take on the order"
" of %d microseconds.\n", (int) t  );
printf("   (= %d clock ticks)\n", (int) (t/quantum) );
printf("Increase the size of the arrays if this shows that\n");
printf("you are not getting at least 20 clock ticks per test.\n");
printf(HLINE);
printf("WARNING -- The above is only a rough guideline.\n");
printf("For best results, please be sure you know the\n");
printf("precision of your system timer.\n");
printf(HLINE);
scalar = 3.0;
for (k=0; k<NTIMES; k++)
{
times[0][k] = mysecond();
#ifdef TUNED
tuned_STREAM_Copy();
#else
#pragma omp parallel for
for (j=0; j<N; j++)
c[j] = a[j];
#endif
times[0][k] = mysecond() - times[0][k];
times[1][k] = mysecond();
#ifdef TUNED
tuned_STREAM_Scale(scalar);
#else
#pragma omp parallel for
for (j=0; j<N; j++)
b[j] = scalar*c[j];
#endif
times[1][k] = mysecond() - times[1][k];
times[2][k] = mysecond();
#ifdef TUNED
tuned_STREAM_Add();
#else
#pragma omp parallel for
for (j=0; j<N; j++)
c[j] = a[j]+b[j];
#endif
times[2][k] = mysecond() - times[2][k];
times[3][k] = mysecond();
#ifdef TUNED
tuned_STREAM_Triad(scalar);
#else
#pragma omp parallel for
for (j=0; j<N; j++)
a[j] = b[j]+scalar*c[j];
#endif
times[3][k] = mysecond() - times[3][k];
}
for (k=1; k<NTIMES; k++) 
{
for (j=0; j<4; j++)
{
avgtime[j] = avgtime[j] + times[j][k];
mintime[j] = MIN(mintime[j], times[j][k]);
maxtime[j] = MAX(maxtime[j], times[j][k]);
}
}
printf("Function      Rate (MB/s)   Avg time     Min time     Max time\n");
for (j=0; j<4; j++) {
avgtime[j] = avgtime[j]/(double)(NTIMES-1);
printf("%s%11.4f  %11.4f  %11.4f  %11.4f\n", label[j],
1.0E-06 * bytes[j]/mintime[j],
avgtime[j],
mintime[j],
maxtime[j]);
}
printf(HLINE);
checkSTREAMresults();
printf(HLINE);
return 0;
}
# define	M	20
int
checktick()
{
int		i, minDelta, Delta;
double	t1, t2, timesfound[M];
for (i = 0; i < M; i++) {
t1 = mysecond();
while( ((t2=mysecond()) - t1) < 1.0E-6 )
;
timesfound[i] = t1 = t2;
}
minDelta = 1000000;
for (i = 1; i < M; i++) {
Delta = (int)( 1.0E6 * (timesfound[i]-timesfound[i-1]));
minDelta = MIN(minDelta, MAX(Delta,0));
}
return(minDelta);
}
#include <sys/time.h>
double mysecond()
{
struct timeval tp;
struct timezone tzp;
int i;
i = gettimeofday(&tp,&tzp);
return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}
void checkSTREAMresults ()
{
double aj,bj,cj,scalar;
double asum,bsum,csum;
double epsilon;
int	j,k;
aj = 1.0;
bj = 2.0;
cj = 0.0;
aj = 2.0E0 * aj;
scalar = 3.0;
for (k=0; k<NTIMES; k++)
{
cj = aj;
bj = scalar*cj;
cj = aj+bj;
aj = bj+scalar*cj;
}
aj = aj * (double) (N);
bj = bj * (double) (N);
cj = cj * (double) (N);
asum = 0.0;
bsum = 0.0;
csum = 0.0;
for (j=0; j<N; j++) {
asum += a[j];
bsum += b[j];
csum += c[j];
}
#ifdef VERBOSE
printf ("Results Comparison: \n");
printf ("        Expected  : %f %f %f \n",aj,bj,cj);
printf ("        Observed  : %f %f %f \n",asum,bsum,csum);
#endif
#define abs(a) ((a) >= 0 ? (a) : -(a))
epsilon = 1.e-8;
if (abs(aj-asum)/asum > epsilon) {
printf ("Failed Validation on array a[]\n");
printf ("        Expected  : %f \n",aj);
printf ("        Observed  : %f \n",asum);
}
else if (abs(bj-bsum)/bsum > epsilon) {
printf ("Failed Validation on array b[]\n");
printf ("        Expected  : %f \n",bj);
printf ("        Observed  : %f \n",bsum);
}
else if (abs(cj-csum)/csum > epsilon) {
printf ("Failed Validation on array c[]\n");
printf ("        Expected  : %f \n",cj);
printf ("        Observed  : %f \n",csum);
}
else {
printf ("Solution Validates\n");
}
}
void tuned_STREAM_Copy()
{
int j;
#pragma omp parallel for
for (j=0; j<N; j++)
c[j] = a[j];
}
void tuned_STREAM_Scale(double scalar)
{
int j;
#pragma omp parallel for
for (j=0; j<N; j++)
b[j] = scalar*c[j];
}
void tuned_STREAM_Add()
{
int j;
#pragma omp parallel for
for (j=0; j<N; j++)
c[j] = a[j]+b[j];
}
void tuned_STREAM_Triad(double scalar)
{
int j;
#pragma omp parallel for
for (j=0; j<N; j++)
a[j] = b[j]+scalar*c[j];
}
