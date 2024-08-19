#include <iostream>
#include <bits/stdc++.h>
#include <omp.h>



using namespace std;

template <typename A ,typename B , typename C>
int minKeyOMP(A key[], B mstSet[],C V){

A min = INT_MAX;
int index, i;
#pragma omp parallel
{

int index_local = index;
A min_local = min;
#pragma omp for nowait
for (i = 0; i < V; i++)
{
if (mstSet[i] == false && key[i] < min_local)
{
min_local = key[i];
index_local = i;
}
}
#pragma omp critical
{
if (min_local < min)
{
min = min_local;
index = index_local;
}
}
}
return index;

}


template <typename T , size_t M, size_t N>
void PrimsOMP(T (&graph)[M][N] , int Parent[]){

int V = M;
T key[V];
int u;
bool mstSet[V];

Parent[0] = -1;
key[0] = 0;
for(int i = 1 ; i<V ; ++i){
key[i] = INT_MAX;
mstSet[i] = false;
}

for(int count = 0 ; count < V ; ++count){
u = minKeyOMP(key,mstSet,V);

mstSet[u] = true;

#pragma omp parallel for schedule(static)

for(int v = 0 ; v < V ; ++v){
if( graph[u][v] < key[v] && mstSet[v] == false && graph[u][v]){
key[v] = graph[u][v];
Parent[v] = u;
}
}
}

}



