
#pragma once

#include "mesh.h"
#include "sync.h"
#include "numeric.h"
#include "partition.h"

namespace trinity {

class Coarse {

public:

Coarse() = delete;
Coarse(const Coarse& other) = delete;
Coarse& operator=(Coarse other) = delete;
Coarse(Coarse&& other) noexcept = delete;
Coarse& operator=(Coarse&& other) noexcept = delete;
~Coarse() = default;

Coarse(Mesh* input, Partit* algo);

void run(Stats* total = nullptr);

private:
void preProcess();
void filterPoints(std::vector<int>* heap);
void extractSubGraph();
void processPoints();

void identifyTarget(int source);
void collapseEdge(int source, int destin);

void initialize();
void saveStat(int level, int* stat, int* form);
void showStat(int level, int* form);
void recap(int* elap, int* stat, int* form, Stats* total);

Mesh*   mesh;
Graph   primal;
Partit* heuris;

struct {
int* target;
int* filter;
int* indep;
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
} nb;

struct {
Time start;
Time iter;
Time tic;
} time;

int& cores;
int& nb_nodes;
int& nb_elems;
int& nb_indep;
int& verbose;
int& iter;
int& rounds;
};

} 
