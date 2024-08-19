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
register  __m256i vec1 = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
register  __m256i vec2 = _mm256_set_epi32(9, 3, 6, 7, 9, 3, 6, 7);
register  __m256i vec3 = _mm256_set_epi32(1, 1, 1, 1, 1, 1, 1, 1);
register  __m256i vec4 = _mm256_set_epi32(4, 5, 3, 6, 4, 5, 1, 6);
gettimeofday(&start, NULL);
#pragma omp parallel for default(shared)	                
for (iCntr=0; iCntr < iLoopCntr; iCntr++)
{
if(iCntr == 0)
{
iThreadCnt = omp_get_num_threads();
}
__m256i result1 = _mm256_add_epi32(vec1, vec2);
__m256i result2 = _mm256_add_epi32(vec3, vec4);
__m256i result3 = _mm256_sub_epi32(result1, result2);
__m256i result4 = _mm256_add_epi32(result1, result2);
asm("");
}		
gettimeofday(&end, NULL);
fTimeTaken = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)); 
printf("Number of iops = %f per sec.\n", (4 * (float)iLoopCntr * (float)iThreadCnt * (256./32.) * 1000000) / fTimeTaken);
return 0;
}
