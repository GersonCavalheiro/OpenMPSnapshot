#include "chol_nllmain.h"
#include "tasks_nested_gemm.h"
#include "tasks_nested_syrk.h"
#include "tasks_nested_trsm.h"
#include "tasks_nested_potrf.h"
#include "fptype.h"
int CHOL_NLL(int mt, int sb, int b, int t, fp_t **Ah) 
{
int k;
for ( k = 0; k < mt; k++ ) {
int j;
for ( j=0; j<k; j++ ) {
NTASK_SYRK(sb, b, t, Ah[j * mt + k], Ah[k * mt + k]);
}
NTASK_POTRF(sb, b, t, Ah[k*mt + k]);
int i;
for ( i = k+1; i < mt; i++ ) {
int j;
for ( j=0; j<k; j++ ) {
NTASK_GEMM(sb, b, t, Ah[j * mt + i], Ah[j * mt + k], Ah[k * mt + i]);
}
NTASK_TRSM(sb, b, t, Ah[k * mt + k], Ah[k * mt + i]);
}
#if 0
for (i = k+1; i < mt; i++) {
#pragma omp task in([sb*sb]Ah[k*mt + k]) inout([sb*sb]Ah[k*mt + i]) priority( (mt-i)+10 ) untied
trsm_ntask(sb, b, t, Ah[k * mt + k], Ah[k * mt + i]);
}
#endif
}
return 0;
}
