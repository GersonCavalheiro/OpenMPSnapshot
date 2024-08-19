#include <pthread.h>
#include <omp.h>
#include "matriz-operacoes-omp.h"
int main(int argc, char *argv[])
{
mymatriz mat_a, mat_b;
mymatriz *mmult_MATRIZ_SeqC;
mymatriz *mmult_MATRIZ_SeqBlC;
mymatriz *mmult_MATRIZ_OMPC;
mymatriz *mmult_MATRIZ_OMPBlC;
char filename[100];
FILE *fmat;
int nr_line;
int *vet_line = NULL;
int N, M, La, Lb;
matriz_bloco_t **Vsubmat_a = NULL;
matriz_bloco_t **Vsubmat_b = NULL;
matriz_bloco_t **Vsubmat_c = NULL;
int n_threads = 4;
int nro_submatrizes = n_threads;
int count_for = 10; 
double start_time, end_time;
double tempo_MATRIZ_SeqC = 0;
double tempo_MATRIZ_SeqBlC = 0;
double tempo_MATRIZ_OMPC = 0;
double tempo_MATRIZ_OMPBlC = 0;
double speedup_seqC;
double speedup_BlC;
if (argc < 3)
{
printf("ERRO: Numero de parametros %s <matriz_a> <matriz_b> <threads>\n", argv[0]);
exit(1);
}
if (argv[3] != NULL){
nro_submatrizes = atoi(argv[3]);
n_threads = atoi(argv[3]);
}
fmat = fopen(argv[1], "r");
if (fmat == NULL)
{
printf("Error: Na abertura dos arquivos.");
exit(1);
}
extrai_parametros_matriz(fmat, &N, &La, &vet_line, &nr_line);
mat_a.matriz = NULL;
mat_a.lin = N;
mat_a.col = La;
if (malocar(&mat_a))
{
printf("ERROR: Out of memory\n");
}
filein_matriz(mat_a.matriz, N, La, fmat, vet_line, nr_line);
free(vet_line);
fclose(fmat);
fmat = fopen(argv[2], "r");
if (fmat == NULL)
{
printf("Error: Na abertura dos arquivos.");
exit(1);
}
extrai_parametros_matriz(fmat, &Lb, &M, &vet_line, &nr_line);
mat_b.matriz = NULL;
mat_b.lin = Lb;
mat_b.col = M;
if (malocar(&mat_b))
{
printf("ERROR: Out of memory\n");
}
filein_matriz(mat_b.matriz, Lb, M, fmat, vet_line, nr_line);
free(vet_line);
fclose(fmat);
mmult_MATRIZ_SeqC = (mymatriz *)malloc(sizeof(mymatriz));
for (int count = 0; count < count_for; count++)
{   
start_time = wtime();
printf("\rMultiplicação Sequencial, teste %d...             ", count+1);
fflush(stdout);
mmult_MATRIZ_SeqC = mmultiplicar(&mat_a, &mat_b, 3);  
end_time = wtime();
tempo_MATRIZ_SeqC += end_time - start_time;
}
sprintf(filename, "MATRIZ_SeqC.result");
fmat = fopen(filename, "w");
fileout_matriz(mmult_MATRIZ_SeqC, fmat);
fclose(fmat);
printf("\n");
mmult_MATRIZ_SeqBlC = (mymatriz *)malloc(sizeof(mymatriz));
for (int count = 0; count < count_for; count++)
{
start_time = wtime();
printf("\rMultiplicação Sequencial em Bloco, teste %d...             ", count+1);
fflush(stdout);
Vsubmat_a = particionar_matriz(mat_a.matriz, N, La, 1, nro_submatrizes);
Vsubmat_a = particionar_matriz(mat_a.matriz, N, La, 1, nro_submatrizes);
Vsubmat_b = particionar_matriz(mat_b.matriz, Lb, M, 0, nro_submatrizes);
Vsubmat_c = csubmatrizv2(N, M, nro_submatrizes);
for (int i = 0; i < nro_submatrizes; i++){
multiplicar_submatriz (Vsubmat_a[i], Vsubmat_b[i], Vsubmat_c[i]);
}
mmult_MATRIZ_SeqBlC = msomar(Vsubmat_c[0]->matriz,Vsubmat_c[1]->matriz, 1);
mmult_MATRIZ_SeqBlC = msomar(Vsubmat_c[0]->matriz,Vsubmat_c[1]->matriz, 1);
for (int i = 2; i < nro_submatrizes; i++){
mmult_MATRIZ_SeqBlC = msomar(mmult_MATRIZ_SeqBlC,Vsubmat_c[i]->matriz, 1);	
}
end_time = wtime();
tempo_MATRIZ_SeqBlC += end_time - start_time;
}
sprintf(filename, "MATRIZ_SeqBlC.result");
fmat = fopen(filename, "w");
fileout_matriz(mmult_MATRIZ_SeqBlC, fmat);
fclose(fmat);
printf("\n");
mmult_MATRIZ_OMPC = (mymatriz *)malloc(sizeof(mymatriz));
mmult_MATRIZ_OMPC = malloc(sizeof(mymatriz));
mmult_MATRIZ_OMPC->matriz = NULL;
mmult_MATRIZ_OMPC->lin = mat_a.lin;
mmult_MATRIZ_OMPC->col = mat_b.col;
if (malocar(mmult_MATRIZ_OMPC)) {
printf("ERROR: Out of memory\n");
exit(1);
}else{
mzerar(mmult_MATRIZ_OMPC);
}
for (int count = 0; count < count_for; count++)
{
printf("\rMultiplicação OMP, teste %d...             ", count+1);
fflush(stdout);
mzerar(mmult_MATRIZ_OMPC);
start_time = wtime();
int tid;
int nthreads;
#pragma omp parallel num_threads(n_threads) 
{
tid = omp_get_thread_num();
nthreads = omp_get_num_threads();
multiplicarOMP(&mat_a, &mat_b, mmult_MATRIZ_OMPC, tid, nthreads);
}
end_time = wtime();
tempo_MATRIZ_OMPC += end_time - start_time;
}
sprintf(filename, "MATRIZ_OMPC.result");
fmat = fopen(filename, "w");
fileout_matriz(mmult_MATRIZ_OMPC, fmat);
fclose(fmat);
printf("\n");
mmult_MATRIZ_OMPBlC = (mymatriz *)malloc(sizeof(mymatriz));
for (int count = 0; count < count_for; count++)
{
printf("\rMultiplicação multithread em bloco, teste %d...             ", count+1);
fflush(stdout);
Vsubmat_a = particionar_matriz(mat_a.matriz, N, La, 1, nro_submatrizes);
Vsubmat_b = particionar_matriz(mat_b.matriz, Lb, M, 0, nro_submatrizes);
Vsubmat_c = csubmatrizv2(N, M, nro_submatrizes);
start_time = wtime();
int tid;
#pragma omp parallel num_threads(n_threads)   
{
tid = omp_get_thread_num();
multiplicarOMPblocos(Vsubmat_a[tid], Vsubmat_b[tid], Vsubmat_c[tid]);
}
end_time = wtime();
mmult_MATRIZ_OMPBlC = msomar(Vsubmat_c[0]->matriz,Vsubmat_c[1]->matriz, 1);
for (int i = 2; i < n_threads; i++){
mmult_MATRIZ_OMPBlC = msomar(mmult_MATRIZ_OMPBlC,Vsubmat_c[i]->matriz, 1);	
}
tempo_MATRIZ_OMPBlC += end_time - start_time;
}
sprintf(filename, "MATRIZ_OMPBlC.result");
fmat = fopen(filename, "w");
fileout_matriz(mmult_MATRIZ_OMPBlC, fmat);
fclose(fmat);
printf("\n\n\tCOMPARAR MATRIZ_SeqC c/ MATRIZ_SeqBlC\n\t");
mcomparar(mmult_MATRIZ_SeqC, mmult_MATRIZ_SeqBlC);
printf("\n\tCOMPARAR MATRIZ_SeqC c/ MATRIZ_OMPC\n\t");
mcomparar(mmult_MATRIZ_SeqC, mmult_MATRIZ_OMPC);
printf("\n\tCOMPARAR MATRIZ_SeqC c/ MATRIZ_OMPBlC\n\t");
mcomparar(mmult_MATRIZ_SeqC, mmult_MATRIZ_OMPBlC);
printf("\n\tTempo Médio MATRIZ_SeqC:\t%.6f sec \n", tempo_MATRIZ_SeqC / count_for);
printf("\tTempo Médio MATRIZ_SeqBlC:\t%.6f sec\n", tempo_MATRIZ_SeqBlC / count_for );
printf("\tTempo Médio MATRIZ_OMPC:\t%.6f sec \n", tempo_MATRIZ_OMPC / count_for);
printf("\tTempo Médio MATRIZ_OMPBlC:\t%.6f sec \n", tempo_MATRIZ_OMPBlC / count_for);
speedup_seqC = (tempo_MATRIZ_SeqC / count_for) / (tempo_MATRIZ_OMPC / count_for);
speedup_BlC = (tempo_MATRIZ_SeqBlC / count_for) / (tempo_MATRIZ_OMPBlC / count_for);
printf("\n\tSPEEDUP (MATRIZ_C): \t%.3f (%.2f %c)", speedup_seqC, speedup_seqC*100, 37 );
printf("\n\tSPEEDUP (MATRIZ_BLC): \t%.3f (%.2f %c)\n\n", speedup_BlC, speedup_BlC*100, 37 );
mliberar(mmult_MATRIZ_SeqC);
mliberar(mmult_MATRIZ_SeqBlC);
mliberar(mmult_MATRIZ_OMPC);
mliberar(mmult_MATRIZ_OMPBlC);
free(mmult_MATRIZ_SeqC);
free(mmult_MATRIZ_SeqBlC);
free(mmult_MATRIZ_OMPC);
free(mmult_MATRIZ_OMPBlC);
mliberar(&mat_a);
mliberar(&mat_b);
return 0;
}