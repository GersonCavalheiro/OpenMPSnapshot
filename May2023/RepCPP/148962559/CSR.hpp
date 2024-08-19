

#pragma once

#include <cstdlib>
#include <vector>
#include <string>

#include "MemoryPool.hpp"
#include "Utils.hpp"

namespace SpMP {
class CSR {
public:
int m;
int n;
offset_type *rowptr;
int *colidx;

explicit CSR(const char *file, int base = 0, bool forceSymmetric = false, int pad = 1);

CSR(const CSR &A);

CSR(int m, int n, offset_type *rowptr, int *colidx);

~CSR();


void loadBin(const char *fileName, int base = 0);

public:

void getRCMPermutation(int *perm, int *inversePerm, bool pseudoDiameterSourceSelection = true);

void getBFSPermutation(int *perm, int *inversePerm);

public:
void make0BasedIndexing();

void make1BasedIndexing();

void alloc(int m, int nnz);

void dealloc();

bool useMemoryPool_() const;

public:
int getBandwidth() const;

double getAverageWidth(bool sorted = false) const;

int getMaxDegree() const;

offset_type getNnz() const { return rowptr[m] - getBase(); }

offset_type getBase() const { return rowptr[0]; }

private:
bool ownData_;
}; 
} 
