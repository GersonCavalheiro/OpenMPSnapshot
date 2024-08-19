#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <sys/time.h>
struct timeval t1, t2;
void proximaPalavra();
int num_ocorrencias();
char texto[] = "Blackbird singing in the dead of night\
Take these broken wings and learn to fly\
All your life You were only waiting for this\
moment to arise Blackbird singing in the\
dead of night Take these sunken eyes and\
learn to see All your life You were only\
waiting for this moment to be free Blackbird,\
fly, blackbird, fly to the light of the dark\
black night Blackbird, fly, blackbird, fly\
Into the light of the dark black night Blackbird\
singing in the dead of night Take these broken\
wings and learn to fly All your life\
moment to arise Blackbird singing in the\
dead of night Take these sunken eyes and\
learn to see All your life You were only\
waiting for this moment to be free Blackbird,\
fly, blackbird, fly to the light of the dark\
black night Blackbird, fly, blackbird, fly\
Into the light of the dark black night Blackbird\
singing in the dead of night Take these broken\
wings and learn to fly All your life\
moment to arise Blackbird singing in the\
dead of night Take these sunken eyes and\
learn to see All your life You were only\
waiting for this moment to be free Blackbird,\
fly, blackbird, fly to the light of the dark\
black night Blackbird, fly, blackbird, fly\
Into the light of the dark black night Blackbird\
singing in the dead of night Take these broken\
wings and learn to fly All your life\
moment to arise Blackbird singing in the\
dead of night Take these sunken eyes and\
learn to see All your life You were only\
waiting for this moment to be free Blackbird,\
fly, blackbird, fly to the light of the dark\
black night Blackbird, fly, blackbird, fly\
Into the light of the dark black night Blackbird\
singing in the dead of night Take these broken\
wings and learn to fly All your life\
moment to arise Blackbird singing in the\
dead of night Take these sunken eyes and\
learn to see All your life You were only\
waiting for this moment to be free Blackbird,\
fly, blackbird, fly to the light of the dark\
black night Blackbird, fly, blackbird, fly\
Into the light of the dark black night Blackbird\
singing in the dead of night Take these broken\
wings and learn to fly All your life\
moment to arise Blackbird singing in the\
dead of night Take these sunken eyes and\
learn to see All your life You were only\
waiting for this moment to be free Blackbird,\
fly, blackbird, fly to the light of the dark\
black night Blackbird, fly, blackbird, fly\
Into the light of the dark black night Blackbird\
singing in the dead of night Take these broken\
wings and learn to fly All your life\
moment to arise Blackbird singing in the\
dead of night Take these sunken eyes and\
learn to see All your life You were only\
waiting for this moment to be free Blackbird,\
fly, blackbird, fly to the light of the dark\
black night Blackbird, fly, blackbird, fly\
Into the light of the dark black night Blackbird\
singing in the dead of night Take these broken\
wings and learn to fly All your life\
moment to arise Blackbird singing in the\
dead of night Take these sunken eyes and\
learn to see All your life You were only\
waiting for this moment to be free Blackbird,\
fly, blackbird, fly to the light of the dark\
black night Blackbird, fly, blackbird, fly\
Into the light of the dark black night Blackbird\
singing in the dead of night Take these broken\
wings and learn to fly All your life\
moment to arise Blackbird singing in the\
dead of night Take these sunken eyes and\
learn to see All your life You were only\
waiting for this moment to be free Blackbird,\
fly, blackbird, fly to the light of the dark\
black night Blackbird, fly, blackbird, fly\
Into the light of the dark black night Blackbird\
singing in the dead of night Take these broken\
wings and learn to fly All your life\
moment to arise Blackbird singing in the\
dead of night Take these sunken eyes and\
learn to see All your life You were only\
waiting for this moment to be free Blackbird,\
fly, blackbird, fly to the light of the dark\
black night Blackbird, fly, blackbird, fly\
Into the light of the dark black night Blackbird\
singing in the dead of night Take these broken\
wings and learn to fly All your life\
moment to arise Blackbird singing in the\
dead of night Take these sunken eyes and\
learn to see All your life You were only\
waiting for this moment to be free Blackbird,\
fly, blackbird, fly to the light of the dark\
black night Blackbird, fly, blackbird, fly\
Into the light of the dark black night Blackbird\
singing in the dead of night Take these broken\
wings and learn to fly All your life\
moment to arise Blackbird singing in the\
dead of night Take these sunken eyes and\
learn to see All your life You were only\
waiting for this moment to be free Blackbird,\
fly, blackbird, fly to the light of the dark\
black night Blackbird, fly, blackbird, fly\
Into the light of the dark black night Blackbird\
singing in the dead of night Take these broken\
wings and learn to fly All your life\
moment to arise Blackbird singing in the\
dead of night Take these sunken eyes and\
learn to see All your life You were only\
waiting for this moment to be free Blackbird,\
fly, blackbird, fly to the light of the dark\
black night Blackbird, fly, blackbird, fly\
Into the light of the dark black night Blackbird\
singing in the dead of night Take these broken\
wings and learn to fly All your life\
moment to arise Blackbird singing in the\
dead of night Take these sunken eyes and\
learn to see All your life You were only\
waiting for this moment to be free Blackbird,\
fly, blackbird, fly to the light of the dark\
black night Blackbird, fly, blackbird, fly\
Into the light of the dark black night Blackbird\
singing in the dead of night Take these broken\
wings and learn to fly All your life\
You were only waiting for this moment to arise\
You were only waiting for this moment to arise\
You were only waiting for this moment to arise\0";
char palavras_chaves[4][5] = {"to", "for", "fly", "night"};
int tamanho_do_texto, tamanho_da_palavra_chave, count, i, j, x;
int main () {
tamanho_do_texto = strlen(texto);
int total_de_palavras_chaves = sizeof(palavras_chaves)/sizeof(palavras_chaves[0]);
printf("Selecionamos um total de %d palavras chaves \n\n", total_de_palavras_chaves);
gettimeofday(&t1, NULL);
#pragma omp parallel for num_threads(total_de_palavras_chaves)
for (int x = 0; x < total_de_palavras_chaves; x++) {
num_ocorrencias(palavras_chaves[x]);
}
gettimeofday(&t2, NULL);
double t_total = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1000000.0);
printf("O tempo que está aplicação levou para executar foi de: %f\n", t_total);
}
int num_ocorrencias(char palavra_chave[]) {
tamanho_da_palavra_chave = strlen(palavra_chave);
for (i = 0; i < tamanho_do_texto; i++) {
if (texto[i] == palavra_chave[0]) {
for (j = 0; j < tamanho_da_palavra_chave; j++) {
if (texto[i + j] != palavra_chave[j]) {
proximaPalavra();
break;
}
}
if (j == tamanho_da_palavra_chave) {
if (isspace(texto[i + j]) || ispunct(texto[i + j]) || texto[i + j] == '\0') {
count ++;
i += j;
}
}
}
else {
proximaPalavra();
}
}
printf("Palavra chave: %s\n", palavra_chave);
printf("Quantidade de vezes que ela apareceu: %d\n\n", count);
return 0;
}
void proximaPalavra() {
while (isspace(texto[i]) == 0 && ispunct(texto[i]) == 0 && texto[i] != '\0')
i++;
}