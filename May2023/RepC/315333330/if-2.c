void
foo (int a, int b, int *p, int *q, int task)
{
int i;
#pragma omp parallel if (a) if (b) 
;
#pragma omp parallel if (a) if (parallel: b) 
;
#pragma omp parallel if (parallel: a) if (b) 
;
#pragma omp parallel if (parallel:a) if (parallel:a) 
;
#pragma omp parallel if (task:a)  if (taskloop: b) 
;
#pragma omp parallel if (target update:a) 
;
#pragma omp parallel for simd if (target update: a) 
for (i = 0; i < 16; i++)
;
#pragma omp task if (task)
;
#pragma omp task if (task: task)
;
#pragma omp task if (parallel: a) 
;
#pragma omp taskloop if (task : a) 
for (i = 0; i < 16; i++)
;
#pragma omp target if (taskloop: a) 
;
#pragma omp target teams distribute parallel for simd if (target exit data : a) 
for (i = 0; i < 16; i++)
;
#pragma omp target data if (target: a) map (p[0:2]) 
;
#pragma omp target enter data if (target data: a) map (to: p[0:2]) 
#pragma omp target exit data if (target enter data: a) map (from: p[0:2]) 
#pragma omp target update if (target exit data:a) to (q[0:3]) 
}
