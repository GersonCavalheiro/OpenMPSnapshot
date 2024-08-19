

#ifndef CVX_LINEARSOLVER_H
#define CVX_LINEARSOLVER_H

class CVoxelyze;
#include <string>
#include <vector>

#ifdef PARDISO_5
#ifdef _WIN32
#pragma comment (lib, "libpardiso500-WIN-X86-64.lib") 
#endif
extern "C" void pardisoinit (void* pt, int* mtype, int* solver, int* iparm, double* dparm, int* error);
extern "C" void pardiso (void* pt, int* maxfct, int* mnum, int* mtype, int* phase, int* n, double* a, int* ia, int* ja, int* perm, int* nrhs, int* iparm, int* msglvl, double* b, double* x, int* error, double* dparm);
#endif


class CVX_LinearSolver
{
public:
CVX_LinearSolver(CVoxelyze* voxelyze); 
bool solve(); 

int progressTick; 
int progressMaxTick; 
std::string progressMsg; 
std::string errorMsg; 
bool cancelFlag; 

private: 
CVoxelyze* vx;
int dof; 
std::vector<double> a, b, x;
std::vector<int> ia, ja; 

int mtype; 
int nrhs; 
void *pt[64];
int iparm[64];
double dparm[64];
int maxfct, mnum, phase, error, msglvl;

void calculateA(); 
void addAValue(int row, int column, float value);
void consolidateA(); 
void applyBX(); 
void convertTo1Base(); 
void postResults(); 
void OutputMatrices(); 

void updateProgress(float percent, std::string message) {progressTick=(int)(percent*100), progressMsg = message;} 
};



#endif 

