#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>

#define OMP_NUM_THREADS 32


char *texto;
char *chute;
int tamanho;
int intervalo;


int cria_palavra_secreta (char *palavra, int tam) {
srand((unsigned) time(NULL));
for (int i = 0; i < tam-1; i++)
palavra[i] = 32 + rand() % 94;
palavra[tam-1] = '\0';
};

void *t_function (void *arg) {
int j, k;
long *rank;
rank = (long *)arg;

printf("Thread %ld: começou...\n", *rank);
printf("\n");

if (*rank == (OMP_NUM_THREADS-1)) {
printf("    >> Thread do ÚLTIMO intervalo da palavra.\n");
for (j = (OMP_NUM_THREADS-1)*intervalo; j < tamanho; j++) {
for (k = 0; k < 255; k++) {
chute[j] = k;
if (chute[j] == texto[j]) {
j=0;
break;
};
};
break;
};
}
else {
printf("    >> Thread de intervalo da palavra.\n");
printf("\n");
for (j = (*rank)*intervalo; j < ((*rank)+1)*intervalo; j++) {
for (k = 0; k < 255; k++) {
chute[j] = k;
if (chute[j] == texto[j]) {
j=0;
break;
};
};
break;
};
};

printf("Thread %ld: ...terminou.\n", *rank);
pthread_exit(NULL);
}

int main (int argc, char *argv[]) {
pthread_t t[OMP_NUM_THREADS];
unsigned long tam;
long *ids[OMP_NUM_THREADS];
long i;

if (argc != 2) {
printf("Favor informar o tamanho da palavra. Por exemplo:\n");
printf("./reproduz_texto_paralelo 100\n");
return 0;
}

sscanf(argv[1], "%lu", &tam);
texto = malloc(sizeof(char)*tam);
chute = malloc(sizeof(char)*tam);
cria_palavra_secreta(texto, tam);

tamanho = tam;
intervalo =  tam/OMP_NUM_THREADS;

printf("==> NUM_THREADS: %d.\n", OMP_NUM_THREADS);


printf("==> Início (frase fora do laço FOR): começo das threads.\n");

#pragma omp parallel for 
for (i = 0; i < OMP_NUM_THREADS; i++)
{
ids[i] = malloc(sizeof(long));
*ids[i] = i;
pthread_create(&t[i], NULL, t_function, (void*)ids[i]);
}

printf("==> Fim (frase fora do laço FOR): todas as threads terminaram.\n");



return 0;
};
