#include "omp.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include "Neuronio.h"
#include "Arquivo.h"
#include "math.h"
#include <ctype.h>
#include <cstdlib>
using namespace std;

int main(){

int matriz[3][3];
int count = 0;
int t = 3;
int my_rank;

for(int i = 0; i< 3; i++){
for(int j = 0; j < 3; j++){
matriz[i][j] = count;
count++;
}

}

#pragma omp parallel for num_threads(t)
for(int i = 0; i< 9; i++){
cout << matriz[i/3][i%3]<<"\t";
my_rank = omp_get_thread_num();
cout<<my_rank<<endl;

}

#pragma omp parallel for num_threads(t)
for(int i = 0; i< 3; i++){
my_rank = omp_get_thread_num();
}

}