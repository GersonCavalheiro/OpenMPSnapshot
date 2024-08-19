#include "omp.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include "Neuronio.h"
#include "Arquivo.h"
#include "math.h"
#include <locale>
#include <ctype.h>
#include <cstdlib>
#include "timer.h"
using namespace std;

int altura, largura;
double distancia;
char* arquivo;
Arquivo* arquivo_entrada;
int threads_count;

Neuronio** grid_de_neuronios;

struct Vencedor{
Neuronio neuronio;
};

Vencedor neuronio_vencedor;

void testeDeRede(double R, double G, double B, int num_linhas, int num_colunas, Neuronio** grid_de_neuronios);
double calcular_distancia_euclidiana(double* entrada, double* pesos, int size);
double** shuffle(double** matriz_to_shuffle, int altura, int largura);
double funcao_de_vizinhanca(double sigma, Neuronio vencedor, int iteracao, int x_neuronio_atual, int y_neuronio_atual);


void treinamento(
double n,
double sigma, 
double qsi, 
double epocas, 
double variacap_minima, 
int altura_grid,
int largura_grid,
Arquivo *arquivo_entrada,
double** matriz_de_entradas,
Neuronio** grid_de_neuronios);

int main(int argc, char* argv[] ){

setlocale(LC_ALL, "pt_BR.UTF-8");



if(argv[1] == NULL || atoi(argv[1]) < 0){
cout<< "Número de threads invalido";
return 0;
}

threads_count = atoi(argv[1]);

arquivo = new char[250];
arquivo = argv[2];

arquivo_entrada = new Arquivo(arquivo);






if(!arquivo_entrada->falha == true){
arquivo_entrada->ler_do_arquivo();
}else{ cout << "Falha"; return 0;}

cout << "Insira as dimensões do mapa"<<endl;

cin  >> altura;
while(altura < 0){
cout << "Valor inválido";
cin  >> altura;
}

cin >> largura;
while(largura < 0){
cout << "Valor inválido";
cin  >> largura;
}

grid_de_neuronios = new Neuronio*[altura];
for(int i = 0; i < altura; i++)
grid_de_neuronios[i] = new Neuronio[largura];

cout << "Set dos Neuronios" <<endl;
#pragma omp parallel for num_threads(threads_count)
for(int i = 0; i < altura*largura; i++){
grid_de_neuronios[i/largura][i%largura].setLocalizacao(i/largura, i%largura);
grid_de_neuronios[i/largura][i%largura].inicializar_vetor_de_pesos(arquivo_entrada->numero_colunas);


}

cout << "set do vetor de pesos"<<endl;

treinamento(
0.05,
1000,
0.05,
100,
0.000001,
altura,
largura,
arquivo_entrada,
arquivo_entrada->matriz_de_entradas,
grid_de_neuronios);

testeDeRede(51, 255, 153, altura, largura, grid_de_neuronios);



return 0;

}

void treinamento(
double n,
double sigma,
double qsi,
double epocas,
double variacao_minima_media,
int altura_grid,
int largura_grid,
Arquivo* arquivo_entrada,
double** matriz_de_entradas,
Neuronio** grid_de_neuronios){

int count_epoca = 0;
double sigma_k = sigma;
double inicio, fim, tempo_passado;

GET_TIME(inicio);

do{
int count_entradas = 0;

while(count_entradas < arquivo_entrada->numero_linhas){    	
int size = largura*altura;
#pragma omp parallel for num_threads(threads_count)
for(int i = 0; i < size; i++){
distancia =
calcular_distancia_euclidiana(
arquivo_entrada->matriz_de_entradas[count_entradas],
grid_de_neuronios[i/largura][i%largura].vetor_de_pesos,
arquivo_entrada->numero_colunas);
grid_de_neuronios[i/largura][i%largura].setDistancia(distancia);
}


double menor_distancia = grid_de_neuronios[0][0].getDistancia();
int menor_index_x = 0, menor_index_y = 0;

for(int i = 0; i < size; i++){
if(grid_de_neuronios[i/largura][i%largura].getDistancia() < menor_distancia){
menor_distancia = grid_de_neuronios[i/largura][i%largura].getDistancia();
menor_index_x = i/largura;
menor_index_y = i%largura;
}
}


Vencedor vencedor;
vencedor.neuronio = grid_de_neuronios[menor_index_x][menor_index_y];

double** container_de_pesos_antigos = new double*[altura*largura];
int count_neuronios = 0;
#pragma omp parallel for num_threads(threads_count)
for(int i = 0; i < size; i++){

if(i/largura == vencedor.neuronio.getX() && i%largura == vencedor.neuronio.getY()){
for(int k = 0; k < arquivo_entrada->numero_colunas; k++){
double peso = grid_de_neuronios[i/largura][i%largura].getPeso(k) + 
n*(matriz_de_entradas[count_entradas][k] - grid_de_neuronios[i/largura][i%largura].getPeso(k));
grid_de_neuronios[i/largura][i%largura].setPeso(peso, k);
}
}else{
for(int k = 0; k <  arquivo_entrada->numero_colunas; k++){
double peso = grid_de_neuronios[i/largura][i%largura].getPeso(k) + 
n*funcao_de_vizinhanca(sigma_k, vencedor.neuronio, count_epoca, i/largura, i%largura)*
((matriz_de_entradas[count_entradas][k] - grid_de_neuronios[i/largura][i%largura].getPeso(k)));
grid_de_neuronios[i/largura][i%largura].setPeso(peso, k);
}
}
}
count_entradas++;
}

cout << count_epoca <<endl;
count_epoca++;
count_entradas = 0;

double** matriz_shuffled = new double*[altura];
matriz_shuffled = shuffle(matriz_de_entradas, altura, largura);

for(int i = 0; i < altura; i++){
for(int j = 0; j < largura; j++){
matriz_de_entradas[i][j] = matriz_shuffled[i][j];
}
}

free(matriz_shuffled);

}while(count_epoca < epocas);
GET_TIME(fim);
tempo_passado = fim - inicio;
cout << "Tempo decorrido: "<< tempo_passado<<endl;
}




double calcular_distancia_euclidiana(double* entrada, double* pesos, int size){

double distancia = 0;
#pragma omp parallel for num_threads(threads_count) reduction(+: distancia)
for(int i = 0; i < size; i++)
distancia += pow((entrada[i] - pesos[i]), 2);
return distancia;
}

double funcao_de_vizinhanca(double sigma, Neuronio vencedor, int iteracao, int x_neuronio_atual, int y_neuronio_atual){

double h;
double distancia_quadrada;

distancia_quadrada = pow((neuronio_vencedor.neuronio.getX() - grid_de_neuronios[x_neuronio_atual][y_neuronio_atual].getX()), 2)+
pow((neuronio_vencedor.neuronio.getY() - grid_de_neuronios[x_neuronio_atual][y_neuronio_atual].getY()), 2);

h = 0.0;
return h;

}

void testeDeRede(double R, double G, double B, int num_linhas, int num_colunas, Neuronio** grid_de_neuronios){

double vetor_de_distancias[num_linhas*num_colunas];
int count_neuronios = 0;
double * entrada = new double[3];
entrada[0] = R;
entrada[1] = G;
entrada[2] = B;
#pragma omp parallel for num_threads(threads_count)
for(int i = 0; i < num_linhas*num_colunas; i++){
distancia =
calcular_distancia_euclidiana(entrada, grid_de_neuronios[i/num_colunas][i%num_colunas].vetor_de_pesos, 3);
vetor_de_distancias[i] = distancia;
}

double menor_distancia = vetor_de_distancias[0];
int menor_index_x = 0, menor_index_y = 0;


for(int i = 0; i < num_linhas*num_colunas; i++){
if(vetor_de_distancias[i] < menor_distancia){
menor_distancia = vetor_de_distancias[i];
menor_index_x = i/num_colunas;
menor_index_y = i%num_colunas;
}
}


for(int i = 0; i < num_linhas; i++){
for(int j = 0; j < num_colunas; j++){
for(int k = 0; k < 3; k++){
cout <<"Peso "<<k<<" do neurônio de indices: "<<i<<""<<j<<" "<<": "<<(grid_de_neuronios[i][j].get_vetor_de_pesos())[k]<<endl;
}
cout << endl;
}
}


cout<<endl;

cout<<"Neurônio vencedor: "<<"indice x: "<<menor_index_x <<", indice y: "<<menor_index_y<<endl;

}

double** shuffle(double** matriz_to_shuffle, int altura, int largura){

srand(time(NULL));
double** matriz_shuffled= new double*[altura];
for(int i = 0; i < altura; i++)
matriz_shuffled[i] = new double[largura];

bool* vetor_flags = new bool[altura];
for(int i = 0; i < altura; i++)
vetor_flags[i] = false;

int nova_posicao;

for(int i = 0; i < altura; i++){
nova_posicao = rand()%altura;
while(vetor_flags[nova_posicao]){

nova_posicao = rand()%altura;
}
matriz_shuffled[nova_posicao] = matriz_to_shuffle[i];
vetor_flags[nova_posicao] = true;

}

return matriz_shuffled;
}

