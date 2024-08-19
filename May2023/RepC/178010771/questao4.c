#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define TAM_KERNEL 3

void aplicaKernel(float **kernel,float m[5][5],int tam,int rep);
void imprimeKernel(float **m,int tam);
int liberaMatriz(float **m,int tam);

int main(int argc, char const *argv[]){

float **kernel,entrada;
int i,j,tam,lin,col,rep,num_threads;

kernel = (float **)malloc(sizeof(float)*TAM_KERNEL);
for(i = 0; i < 3; i++){
kernel[i] = (float *)malloc(sizeof(float)*TAM_KERNEL);
}

#pragma omp parallel
{
#pragma omp master
{
num_threads = omp_get_num_threads();
}
}

for(i = 0; i < TAM_KERNEL; i++){
for(j = 0; j < TAM_KERNEL; j++){
if(i==1 && j==1){
kernel[i][j] = 0.2;
}else{
kernel[i][j] = 0.1;
}
}
}

printf("Kernel:\n");
imprimeKernel(kernel,TAM_KERNEL);

printf("\nDigite o tamanho da matriz:\n");
scanf("%d",&tam);
printf("Digite as coordenadas:\n");
scanf("%d%d",&lin,&col);
printf("Digite o valor de entrada:\n");
scanf("%f",&entrada);
printf("Digite o numero de repeticões:\n");
scanf("%d",&rep);

float matriz[tam][tam];

#pragma omp parallel for
for(i = 0; i < tam; i++){
for(j = 0; j < tam; j++){
matriz[i][j] = 0;
matriz[lin][col] = entrada;
}
}

printf("\nA:\n");
for(i = 0; i < tam; i++){
for(j = 0; j < tam; j++){
printf("%.0f ", matriz[i][j]);
}
printf("\n");
}

#pragma omp parallel sections num_threads(num_threads)
{
#pragma omp section
aplicaKernel(kernel,matriz,tam,rep);
}

liberaMatriz(kernel,TAM_KERNEL);

return 0;
}

void aplicaKernel(float **kernel,float m[5][5],int tam,int rep){
int i,j;
float mAux[tam][tam];

rep--;
while(rep > 0){
#pragma omp parallel for 
for(i = 0; i < tam; ++i){
for(j = 0; j < tam; ++j){
if(i==0 || j==0 || i==(tam-1) || j==(tam-1)){
mAux[i][j] = 0;
}else{
mAux[i][j] = m[i-1][j-1] * kernel[0][0] + m[i-1][j] * kernel[0][1] + m[i-1][j+1] * kernel[0][2] 
+ m[i][j-1] * kernel[1][0] + m[i][j] * kernel[1][1] + m[i][j+1] * kernel[1][2] 
+ m[i+1][j-1] * kernel[2][0]  + m[i+1][j] * kernel[2][1] + m[i+1][j+1]*kernel[2][2];
}
}      	
}

#pragma omp parallel for 
for(i = 0; i < tam; ++i){
for(j = 0; j < tam; ++j){
m[i][j] = mAux[i][j];
}
}

printf("\n");
for(i = 0; i < tam; i++){
for(j = 0; j < tam; j++){
printf("%.2f ", mAux[i][j]);
}
printf("\n");
}
rep--;
}
}

void imprimeKernel(float **m,int tam){
int i,j;
for(i = 0; i < tam; i++){
for(j = 0; j < tam; j++){
printf("%.1f ", m[i][j]);
}
printf("\n");
}
}

int liberaMatriz(float **m,int tam){
int i,j;
for(i = 0; i < tam; i++){
free(m[i]);
free(m); 
}
}
