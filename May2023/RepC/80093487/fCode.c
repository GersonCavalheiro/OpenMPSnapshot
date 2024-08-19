#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>
int main()
{ 
int iThreadCnt = 1, iLoopCntr = 1000*1000*1000;
struct timeval start, end;
float fTimeTaken = 0;
int iCntr = 0;
register __m256 vec1 = _mm256_setr_ps(4.0, 5.0, 13.0, 6.0, 4.0, 5.0, 13.0, 6.0);
register __m256 vec2 = _mm256_setr_ps(9.0, 3.0, 6.0, 7.0, 4.0, 5.0, 13.0, 6.0);
register __m256 vec3 = _mm256_setr_ps(1.0, 1.0, 1.0, 1.0, 4.0, 5.0, 13.0, 6.0);
register __m256 vec4 = _mm256_setr_ps(4.0, 5.0, 13.0, 6.0, 4.0, 5.0, 13.0, 6.0);
register __m256 vec5 = _mm256_setr_ps(9.0, 3.0, 6.0, 7.0, 4.0, 5.0, 13.0, 6.0);
register __m256 vec6 = _mm256_setr_ps(1.0, 1.0, 1.0, 1.0, 4.0, 5.0, 13.0, 6.0);
register __m256 vec7 = _mm256_setr_ps(4.0, 5.0, 13.0, 6.0, 4.0, 5.0, 13.0, 6.0);
register __m256 vec8 = _mm256_setr_ps(9.0, 3.0, 6.0, 7.0, 4.0, 5.0, 13.0, 6.0);
register __m256 vec9 = _mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 13.0, 6.0);
gettimeofday(&start, NULL);
#pragma omp parallel for default(shared)	                
for (iCntr=0; iCntr < iLoopCntr; iCntr++)
{
if(iCntr == 0)
{
iThreadCnt = omp_get_num_threads();
}
__m256 result1 = _mm256_fmadd_ps(vec1, vec2, vec3);
__m256 result2 = _mm256_fmadd_ps(vec4, vec5, vec6);
__m256 result3 = _mm256_fmadd_ps(vec7, vec8, vec9);
__m256 result4 = _mm256_fmadd_ps(result1, result2, result3);		
asm("");
}
gettimeofday(&end, NULL);
fTimeTaken = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)); 
printf("Number of flops = %f per sec.\n", (2 * 4 * (float)iLoopCntr * (float)iThreadCnt * (256./32.) * 1000000) / fTimeTaken);
return 0;
}
