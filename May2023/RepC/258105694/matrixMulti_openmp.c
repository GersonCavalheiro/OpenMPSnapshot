#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define NRA 160               
#define NCA 160               
#define NCB 90               
#define nthreads 4           
int
main (int argc, char *argv[]) 
{
int	tid,                   
chunk,                 
i,j,k;
double	a[NRA][NCA],           
b[NCA][NCB],           
c[NRA][NCB];           
omp_set_num_threads(nthreads);
chunk = NRA/nthreads;                    
srand(time(0));
double start,dif,end;
printf("\n\n");
#pragma omp parallel shared(a,b,c,chunk) private(start,dif,end,tid,i,j,k)
{
start  = omp_get_wtime();
tid = omp_get_thread_num();
if (tid == 0)
{
printf("openMP se inicializa con %d threads.\n",nthreads);
printf("Inicializando arrays...\n");
}
#pragma omp for ordered schedule (static, chunk) 
for (i=0; i<NRA; i++)
for (j=0; j<NCA; j++)
a[i][j]= rand()%3+1;
#pragma omp for ordered schedule (static, chunk)
for (i=0; i<NCA; i++)
for (j=0; j<NCB; j++)
b[i][j]= rand()%3+1;
#pragma omp for ordered schedule (static, chunk)
for (i=0; i<NRA; i++)
for (j=0; j<NCB; j++)
c[i][j]= 0;  
printf("Thread %d empieza a multiplicar...\n",tid);
#pragma omp for ordered schedule (static, chunk)
for (i=0; i<NRA; i++)    
{
printf("Thread=%d hace la fila=%d\n",tid,i);
for(j=0; j<NCB; j++)
for (k=0; k<NCA; k++)
c[i][j] += a[i][k] * b[k][j];
}
#pragma omp reduction(max:dif)
end = omp_get_wtime();
dif = end-start;
if(tid==0){
printf("El proceso paralelizado en openMP demoro %.8fs de tiempo.\n",dif);
}
}   
printf("\n\n");
return 0;
}
