

#pragma once

#include "header.h"
#include "timer.h"
#include "tools.h"
#include "mesh.h"

#define MATCHED -2

namespace trinity {

class Match {

friend class Swap;

public:

Match();
Match(const Match& other) = delete;
Match& operator=(Match other) = delete;
Match(Match&& other) noexcept = delete;
Match& operator=(Match&& other) noexcept = delete;
~Match();


void initialize(size_t capa, int* mapping, int* index);
int* computeGreedyMatching(const Graph& graph, int nb);
int* localSearchBipartite(const Graph& graph, int nb);
int  getRatio(const Graph& graph, int nb, int* count);

private:

void reset();
void matchAndUpdate(int vertex, const Graph& graph, std::stack<int>* stack);
bool lookAheadDFS(int vertex, const Graph& graph, std::stack<int>* stack);

struct {
bool found;
} param;

struct {
int  capa;         
int  depth;        
} max;

struct {
int*  matched;    
int** lists;      
int*  mapping;    
int   cardin[2];      
} task;

struct {
char* visited;    
char* degree;     
int*  off;        
} sync;
};

} 
