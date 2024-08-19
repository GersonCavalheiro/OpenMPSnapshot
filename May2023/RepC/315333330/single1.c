void
foo (int i)
{
#pragma omp single copyprivate (i)
;
#pragma omp single nowait
;
#pragma omp single copyprivate (i) nowait	
;
#pragma omp single nowait copyprivate (i)	
;
}
