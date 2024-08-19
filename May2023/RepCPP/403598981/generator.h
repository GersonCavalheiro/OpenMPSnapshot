#pragma once
#include <limits.h>
#include <stdio.h>
#include <iostream>
#include <time.h>
using namespace std;

namespace graph {
class generator
{
private:
int V; 
int E; 
int minWeight;
int maxWeight;
void generate();
public:
generator(int V, int E, int minWeight, int maxWeight); 
~generator(); 
int **graph;
void printAdjMatrix();
};
}