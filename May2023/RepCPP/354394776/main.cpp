

#include <omp.h>
#include <random>
#include <iostream>
#include <time.h>

using namespace std;

#define TAMANHO 500000


int vetor[TAMANHO];
int resultado = 0;

void inicia_vetor() {
random_device dev;
mt19937 rng(dev());
uniform_int_distribution<mt19937::result_type> dist6(0,TAMANHO*3);
for(int i = 0; i < TAMANHO; ++i) {
vetor[i] = dist6(rng);
cout << "Valor " << i << ": " << vetor[i] << endl;
}
}

int main(int argc, char *argv[]) {
double inicio, fim;
inicia_vetor();
omp_set_num_threads(2);

inicio = (double) clock() / CLOCKS_PER_SEC;

#pragma omp parallel for shared (vetor, resultado)
for(int i = 0; i < TAMANHO; ++i) {
#pragma omp atomic
resultado = resultado + vetor[i];
}
cout << "Resultado: " << resultado << endl;
fim = (double) clock() / CLOCKS_PER_SEC;

cout << "Tempo de execução: " << (fim - inicio) << endl;
return 0;
}