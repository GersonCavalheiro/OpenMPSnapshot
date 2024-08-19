

#pragma once

#include "mesh.h"
#include "hashtable.h"
#include "numeric.h"
#include "matching.h"

namespace trinity {

class Swap {

public:

Swap() = delete;
Swap(const Swap& other) = delete;
Swap& operator=(Swap other) = delete;
Swap(Swap&& other) noexcept = delete;
Swap& operator=(Swap&& other) noexcept = delete;
explicit Swap(Mesh* input);
~Swap();

void run(Stats* total = nullptr);

private:
void cacheQuality();
void filterElems(std::vector<int>* heap);
void extractDualGraph();
void processFlips();

int swap(int id1, int id2, int idx);

void initialize();
void saveStat(int level, int* stat, int* form);
void showStat(int level, int* form);
void recap(int* elap, int* stat, int* form, Stats* total);

Mesh* mesh;
Graph dual;
Match heuris;

struct {
int* match;
int* list;
int  depth;
} task;

struct {
int*  off;
char* fixes;
char* activ;
} sync;

struct {
int activ;
int tasks;
int commit;
int count;  
} nb;

struct { double* qualit; } geom;

struct {
Time start;
Time iter;
Time tic;
} time;

int& cores;
int& nb_nodes;
int& nb_elems;
int& verbose;
int& iter;
int& rounds;
};

} 