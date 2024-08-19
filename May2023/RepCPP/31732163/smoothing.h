

#pragma once

#include "mesh.h"
#include "hashtable.h"
#include "numeric.h"
#include "partition.h"

namespace trinity {

class Smooth {

public:

Smooth() = delete;
Smooth(const Smooth& other) = delete;
Smooth& operator=(Smooth other) = delete;
Smooth(Smooth&& other) noexcept = delete;
Smooth& operator=(Smooth&& other) noexcept = delete;
Smooth(Mesh* input, Partit* algo, int level);
~Smooth() = default;

void run(Stats* total = nullptr);

private:

void preProcess();
void cacheQuality();
void movePoints();

int moveSmartLaplacian(int id);

void initialize();
void saveStat(int level, int* stat, int* form);
void showStat(int level, int* form);
void recap(int* elap, int* stat, int* form, Stats* total);

Mesh*   mesh;
Partit* heuris;

struct { char* activ; }    sync;
struct { double* qualit; } geom;
struct { int depth; }      task;

struct {
int tasks;
int commit;
} nb;

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
