#include "./Header/Bibliotecas.h"
#include "./Header/VariaveisGlobais.h"
#include "./Header/Funcoes.h"

int main(){

struct Fila fila;

DefineNucleos();

TransformaPalavra("Palmeiras");

IniciaFila(&fila);

#pragma omp parallel num_threads(1)
{
Enfileira(&fila);

}	

int verifica = 1;
#pragma omp parallel num_threads(nprocs-1)
{
while (verifica == 1){
verifica = Desenfileira(&fila);
}
}

ExibeOcorrencia();

fflush(stdin);

return 0;
}
