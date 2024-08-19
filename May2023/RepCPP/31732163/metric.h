

#pragma once

#include "mesh.h"
#include "numeric.h"
#include "hessian.h"

namespace trinity {

class Metrics {

public:

Metrics() = delete;
Metrics(const Metrics& other) = delete;
Metrics& operator=(Metrics other) = delete;
Metrics(Metrics&& other) noexcept = delete;
Metrics& operator=(Metrics&& other) noexcept = delete;
Metrics(
Mesh* input_mesh, double target_factor, int Lp_norm,
double h_min, double h_max
);
~Metrics();

void run(Stats* total = nullptr);

private:

void recoverHessianField();
void normalizeLocally();
void computeComplexity();
void normalizeGlobally();

void computeGradient(int index);
void computeHessian(int index);

void initialize();
void recap(Stats* total);
void clear();

Mesh* mesh;

struct {
double* gradient;
double* solut;
double* tensor;
Patch*  stencil;
double  complexity;
} field;

struct {
int    chunk;
int    target;
int    norm;
double h_min;
double h_max;
struct { double min, max; }  eigen;
struct { double fact, exp; } scale;
} param;

struct { Time start; } time;

int& nb_nodes;
int& nb_elems;
int& nb_cores;
int& verbose;
int& iter;
int& rounds;

};
} 
