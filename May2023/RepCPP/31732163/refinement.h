

#pragma once

#include "tools.h"
#include "mesh.h"
#include "hashtable.h"
#include "numeric.h"

namespace trinity {

class Refine {

public:

Refine() = delete;
Refine(const Refine& other) = delete;
Refine& operator=(Refine other) = delete;
Refine(Refine&& other) noexcept = delete;
Refine& operator=(Refine&& other) noexcept = delete;
Refine(Mesh* input, int level);
~Refine();

void run(Stats* total = nullptr);

private:

void preProcess(std::vector<int>* heap);
void filterElems(std::vector<int>* heap);
void computeSteinerPoints();
void processElems(int tid);
void cutElem(int id, int* offset);

void initialize();
void saveStat(int level, int* stat, int* form);
void showStat(int level, int* form);
void recap(int* elap, int* stat, int* form, Stats* total);

Mesh* mesh;                 
Hashtable<int> steiner;     

struct {
int*  edges;              
int*  elems;              
char* pattern;            
int   level;              
} task;

struct {
int   shift;
int*  index;              
int*  off;                
char* activ;              
} sync;

struct {
int adds;
int split;
int eval;
int tasks;
int steiner;
struct { int node, elem; } old;
} nb;

struct {
Time start;
Time iter;
Time tic;
} time;

int& cores;
int& iter;
int& nb_nodes;
int& nb_elems;
int& verbose;
int& rounds;
};

} 