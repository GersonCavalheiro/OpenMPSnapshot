

#pragma once

#include "header.h"
#include "timer.h"
#include "tools.h"
#include "mesh.h"
#include "rmat.h"

namespace trinity {

class Partit {

friend class Coarse;
friend class Smooth;

public:

Partit() = delete;
Partit(const Partit& other) = delete;
Partit& operator=(Partit other) = delete;
Partit(Partit&& other) noexcept = delete;
Partit& operator=(Partit&& other) noexcept = delete;
~Partit();

Partit(int max_graph_size, int max_part_size);
Partit(Mesh const* mesh, int max_part_size);

void extractIndepSet(const Graph& graph, int nb);
void extractColoring(const Mesh* mesh);
void extractPartition(const Mesh* mesh);

private:

void reset();

struct {
int capa;         
int part;         
} max;

struct {
int*  cardin;
int*  mapping;
int** subset;
int** lists;     
} task;

struct {
int remain[2];    
int cores;
int parts;        
int defect;       
int rounds;       
} nb;

struct { int* offset; } sync;
};

} 
