



#include "dimensions.h"
#include "local_sequence_alignment.h"
#include <stdio.h>
#include <algorithm>
#include <cstdlib>
#include <string>






LSA::LSA(Dimensions d) {
printf("Creating Local Sequence Alignment Simulation\n\n");
dims=d;
if(dims.ncomponents==32) {
seq1=seq1_cstr;
seq2=seq2_cstr;
}
else {
seq1=generate_sequence(dims.ncell_x,"ATCG");
seq2=generate_sequence(dims.ncell_y,"ATCG");
}
rows = seq1.size() + 1;
cols = seq2.size() + 1;
matrix_size = rows*cols;

matrix = (int*)malloc_host_int(matrix_size);
}
int LSA::r(){
return rows;
}
int LSA::c(){
return cols;
}


void LSA::init(){
for (int x = 0; x < rows; x++)
for (int y = 0; y < cols; y++)
matrix[(x * cols) + y] = 0;
}



void LSA::run(){
create_score_matrix(rows, cols);
}


void LSA::print(){
for (int x = 1; x < rows; x++)
for (int y = 1; y < cols; y++)
printf("matrix[%d][%d] = %d\n", x, cols-1, matrix[(x * cols) + cols-1]);
}






void LSA::create_score_matrix(int rows, int cols)
{
int max_score = 0;
int nthreads=1,b=2;
char* bx_str=std::getenv("OMP_BLOCK_DIMX");
char* by_str=std::getenv("OMP_BLOCK_DIMY");
char* nthreads_str=std::getenv("OMP_NUM_THREADS");
if(nthreads_str!=nullptr)
if(strlen(nthreads_str)>0)
nthreads=std::stoi(std::string(nthreads_str));
b*=nthreads;
int bx=b,by=b;
if(bx_str!=nullptr)
if(strlen(bx_str)>0)
bx=std::stoi(std::string(bx_str));
if(by_str!=nullptr)
if(strlen(by_str)>0)
by=std::stoi(std::string(by_str));
#pragma omp dag coarsening(BLOCK,bx,by)
for (int x = 1; x < rows; x++)
for (int y = 1; y < cols; y++) {
#pragma omp dag task depend({(x+1)*cols+y+1,((x+1)<rows)&&((y+1)<cols)},{(x+1)*cols+y,(x+1)<rows},{x*cols+y+1,(y+1)<cols})
{
matrix[(x * cols) + y] = calc_score(x, y);
}
}
}



int LSA::calc_score(int x, int y)
{
int similarity;
if (seq1[x - 1] == seq2[y - 1])
similarity = match;
else
similarity = mismatch;


int diag_score = matrix[(x - 1) * cols + (y - 1)] + similarity;
int up_score   = matrix[(x - 1) * cols + y] + gap;
int left_score = matrix[x * cols + (y - 1)] + gap;


int result = 0;
if (diag_score > result)
result = diag_score;
if (up_score > result)
result = up_score;
if (left_score > result)
result = left_score;

return result;
}
