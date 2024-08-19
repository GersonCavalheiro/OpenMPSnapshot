#include <algorithm>
#include <cstring>

#define DIM 3
#define Npt 4


template<class T>
void ParallelEnsemble(T** glM, int** Np, T*** K, int elemsNum, int nodesNum)
{
int*  Nconc;
int** Elem_Conc;
int** No_Corresp;

Set3AdditionalArrs (Nconc, Elem_Conc, No_Corresp, elemsNum, Np,        nodesNum);
ParallelEnsemble<T>(glM,   Np,        K,          Nconc,    Elem_Conc, No_Corresp, nodesNum);

delete[] Nconc;
for (int i = 0; i < nodesNum; i++)
{
delete[] Elem_Conc [i];
delete[] No_Corresp[i];
}
delete[] Elem_Conc;
delete[] No_Corresp;
}

template<class T>
void ParallelEnsemble(T** glM, int** Np, T*** K, int* Nconc, int** Elem_Conc, int** No_Corresp, int nodesNum)
{
int node_loc_num;
int elem_glob_num;
int ii, jj;
T*  block[3];

#pragma omp parallel for private(node_loc_num, elem_glob_num, ii, jj, block)
for (int i = 0; i < nodesNum; i++)
{
for (int j = 0; j < Nconc[i]; j++)
{
node_loc_num  = No_Corresp[i][j];
elem_glob_num = Elem_Conc [i][j];

ii = DIM * Np[elem_glob_num][node_loc_num];
for (int k = 0; k < Npt; k++)
{
block[0] = &K[elem_glob_num][DIM * node_loc_num    ][DIM * k];
block[1] = &K[elem_glob_num][DIM * node_loc_num + 1][DIM * k];
block[2] = &K[elem_glob_num][DIM * node_loc_num + 2][DIM * k];

jj = DIM * Np[elem_glob_num][k];
Add3x3<T>(glM, ii, jj, block);
}
}
}
}

void Set3AdditionalArrs(int* &Nconc, int** &Elem_Conc, int** &No_Corresp, int elemsNum, int** Np, int nodesNum)
{
Nconc = new int[nodesNum];
std::memset((void*)Nconc, 0, nodesNum * sizeof(int));

for (int i = 0; i < elemsNum; i++)
{
for (int j = 0; j < Npt; j++)
{
Nconc[Np[i][j]]++;
}
}
int maxNconc = MaxVal(Nconc, nodesNum);

std::memset((void*)Nconc, 0, nodesNum * sizeof(int));
Elem_Conc  = AllocMatrix<int>(nodesNum, maxNconc);
No_Corresp = AllocMatrix<int>(nodesNum, maxNconc);

for (int i = 0, p; i < elemsNum; i++)
{
for (int j = 0; j < Npt[i]; j++)
{
p = Np[i][j];

Elem_Conc [p][Nconc[p]] = i;
No_Corresp[p][Nconc[p]] = j;
Nconc[p]++;
}
}
}

template<class T>
void Add3x3(T** glM, int ii, int jj, T* block[3])
{
glM[ii    ][jj] += block[0][0]; glM[ii    ][jj + 1] += block[0][1]; glM[ii    ][jj + 2] += block[0][2];
glM[ii + 1][jj] += block[1][0]; glM[ii + 1][jj + 1] += block[1][1]; glM[ii + 1][jj + 2] += block[1][2];
glM[ii + 2][jj] += block[2][0]; glM[ii + 2][jj + 1] += block[2][1]; glM[ii + 2][jj + 2] += block[2][2];
}

int MaxVal(int* arr, int n)
{
int res = arr[0];
for (int i = 1; i < n; i++)
{
res = std::max(res, arr[i]);
}

return res;
}

template<class T>
T** AllocMatrix(int n, int m)
{
T** res = new T*[n];
for (int i = 0; i < n; i++)
{
res[i] = new T[m];
std::memset((void*)res[i], 0, m * sizeof(T));
}

return res;
}
