#include "matrizv3.h"
#include "matriz-operacoesv3.h"
int multiplicarOMP(mymatriz *mat_a, mymatriz *mat_b, mymatriz *mat_c, int tid, int n_threads)
{
int i_max = mat_a->lin;
int j_max = mat_b->col;
int k_max = mat_a->col;
#pragma omp parallel for schedule(guided) shared(mat_a,mat_b,mat_c)
for (int i = tid; i < i_max; i += n_threads){
for (int k = 0; k < k_max; k++){
for (int j = 0; j < j_max; j++){
mat_c->matriz[i][j] += mat_a->matriz[i][k] * mat_b->matriz[k][j];
}
}
}
return 0;
}
int multiplicarOMPblocos(matriz_bloco_t *mat_suba, matriz_bloco_t *mat_subb, matriz_bloco_t *mat_subc)
{
if ( (mat_suba == NULL) || (mat_subb == NULL) || (mat_subc == NULL)) {
printf ("** Erro: Memoria Insuficiente **\n");
return (-1);
}
#pragma omp parallel for schedule(guided) shared(mat_suba,mat_subb,mat_subc)
for (int i = mat_suba->bloco->lin_inicio; i < mat_suba->bloco->lin_fim; i++)
{
for (int j = mat_subb->bloco->col_inicio; j < mat_subb->bloco->col_fim; j++)
{
for (int k = mat_suba->bloco->col_inicio; k < mat_suba->bloco->col_fim; k++)
{
mat_subc->matriz->matriz[i][j] += mat_suba->matriz->matriz[i][k] * mat_subb->matriz->matriz[k][j];
}
}
}
return 0;
}
