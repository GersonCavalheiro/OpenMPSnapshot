#pragma once


struct Block {
int t;

double * prev;
double * curr;
double * next;

double * edges[6];

int sx, ex, nx;
int sy, ey, ny;
int sz, ez, nz;

Block();
Block(int rank);
~Block();

double& get(double * layer, int i, int j, int k);

void swap();

void copyAxes(int x, int y, int z, double * from, double * to);
void prepare();

double delta(int i, int j, int k, double* curr);

void init0();
void init1();
void calcNext();

double get_error();
void print_layer();
};