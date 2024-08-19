

#pragma once

#include "header.h"
#include "timer.h"
#include "sync.h"
#include "numeric.h"

namespace trinity {

class Mesh {

friend class Metrics;
friend class Refine;
friend class Coarse;
friend class Swap;
friend class Smooth;
friend class Partit;

public:

Mesh() = delete;
Mesh(const Mesh& other) = delete;
Mesh& operator=(Mesh other) = delete;
Mesh(Mesh&& other) noexcept = delete;
Mesh& operator=(Mesh&& other) noexcept = delete;
Mesh(int size[2], int bucket, int depth, int verbosity, int rounds);
~Mesh();

void reallocMemory();
void doFirstTouch();
void compressStorage();  
void reorderCells();     
void extractPrimalGraph();
void extractDualGraph(Graph* dual) const;

void rebuildTopology();
void fixTagged();
void fixAll();
void initActivElems();
bool verifyTopology() const;

void load(const std::string& path, const std::string& solu);
void store(const std::string& path) const;
void storePrimalGraph(const std::string& path) const;

Patch getVicinity(int id, int deg) const;
const int* getElem(int i) const;
const int* getElemCoord(int id, double* p) const;
int getElemNeigh(int id, int i, int j) const;
bool isActiveNode(int i) const;
bool isActiveElem(int i) const;
bool isBoundary(int i) const;
bool isCorner(int i) const;
int getCapaNode() const;
int getCapaElem() const;

void replaceElem(int id, const int* v);
void eraseElem(int id);
void updateStencil(int i, int t);
void updateStencil(int i, const std::initializer_list<int>& t);
void copyStencil(int i, int j, int nb_rm);

#if DEFER_UPDATES
void initUpdates();
void commitUpdates();
void resetUpdates();
void deferredAppend(int tid, int i, int t);
void deferredAppend(int tid, int i, const std::initializer_list<int>& t);
void deferredRemove(int tid, int i, int t);
#endif

double computeLength(int i, int j) const;
double computeQuality(int i) const;
double computeQuality(const int* t) const;
bool isCounterclockwise(const int* elem) const;
void computeSteinerPoint(int edge_i, int edge_j, double* point, double* metric) const;
void computeQuality(double quality[3]);

private:

struct {
int nodes;                        
int elems;                        
int cores;                        
int activ_elem;                   
} nb;

struct {
std::vector<int> elems;           
Graph stenc;                      
Graph vicin;                      
} topo;                             

struct {
std::vector<double> points;       
std::vector<double> tensor;       
std::vector<double> solut;        
std::vector<double> qualit;       
} geom;

struct {
int depth;                        
int verb;                         
int iter;                         
int rounds;                       
} param;

struct {
int    scale;                     
size_t bucket;                    
size_t node;                      
size_t elem;                      
} capa;

struct {
int* deg      = nullptr;          
int* off      = nullptr;          
char* fixes   = nullptr;          
char* activ   = nullptr;          
uint8_t* tags = nullptr;          
Time tic;                         
} sync;

#if DEFER_UPDATES
struct Updates {
std::vector<int> add;
std::vector<int> rem;
};
std::vector<std::vector<Updates>> deferred;
const int def_scale_fact = 32;
#endif
};
} 
