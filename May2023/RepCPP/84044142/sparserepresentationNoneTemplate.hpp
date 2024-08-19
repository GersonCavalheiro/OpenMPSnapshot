#ifndef _SPARSEREPRESENTATION_HPP_
#define _SPARSEREPRESENTATION_HPP_

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include "Eigen/Dense"

#ifdef _OPENMP

#if defined(__clang__)
#include <omp.h>
#elif defined(__GNUG__) || defined(__GNUC__)
#include <omp.h>
#endif

#endif

using namespace std;
using namespace Eigen;

class LSparseRepresentation {
public:
LSparseRepresentation(vector<vector<vector<double>>> dicts);
~LSparseRepresentation();
int SRClassify(const vector<double>& y, double min_residual, int sparsity);
bool SRClassify(const vector<vector<double>>& y, double min_residual, int sparsity, vector<int> &srres);

private:
vector<vector<double>> dic;		
vector<int> dicclassnum;			
int	classnum;						

private:
double Dot(const vector<double>& vec1, const vector<double>& vec2);

double Norm(const vector<double>& vec);

bool solve(vector<vector<double>>& phi, vector<double>& y, vector<double>& x);

bool OrthMatchPursuit(const vector<double>& y, double min_residual, int sparsity, vector<double>& x, vector<int>& patch_indices);
};

LSparseRepresentation::LSparseRepresentation(vector<vector<vector<double>>> dicts) {
classnum = 0;
dicclassnum.clear();
dic.clear();

this->classnum = dicts.size();
for(auto matrix:dicts) {
this->dicclassnum.push_back(matrix.size());
for(auto item:matrix) {
this->dic.push_back(item);
}
}
}

int LSparseRepresentation::SRClassify(const vector<double>& y, double min_residual, int sparsity) {

int fsize = y.size(); 
if (!fsize) return -1; 
int dcol = dic.size();      
if (!dcol) return -1; 
int drow = dic[0].size();   
if (!drow || drow != fsize) return -1;
int i, j;
vector<int> patch_indices;	
vector<double> coefficient;

if(!OrthMatchPursuit(y, min_residual, sparsity, coefficient, patch_indices)) {
vector<int>().swap(patch_indices);
vector<double>().swap(coefficient);
return -1;
}

vector<double> x(dcol, 0.0); 
for (i = 0; i < patch_indices.size(); ++i) {
x[patch_indices[i]] = coefficient[i];
}

int result = 0;
int start = 0;
double mindist = 100000000;
for(i = 0; i < classnum; ++i) {
vector<double> tmp(fsize, 0.0);
for(j = start; j < start + dicclassnum[i]; ++j) {
if (x[j] != 0.0) {
tmp = tmp + dic[j] * x[j];		
}
}
double dist = Norm(y - tmp);

if (mindist > dist) {
mindist = dist;
result = i;
}

start += dicclassnum[i];
}

return result; 

}

bool LSparseRepresentation::SRClassify(const vector<vector<double>>& y,
double min_residual, int sparsity,
vector<int> &srres){

int i,j;
int size = y.size();
vector<int> result(this->classnum, 0);

vector<int> results(size,0);

#pragma omp parallel for
for (i = 0; i < size; ++i) {
results[i] = SRClassify(y[i], min_residual, sparsity);
}

for(i = 0; i < size; ++i) {
if (results[i] < 0) {
return false;
}
result[results[i]]++;
}

srres.resize(this->classnum);
for(i = 0; i < this->classnum; ++i) {
srres[i] = i;
}

for(i = this->classnum-1; i >= 0; --i) { 
if(result[i] == 0) {
result.erase(result.begin() + i);
srres.erase(srres.begin() + i);
}
}

int resnum = result.size(); 
for(i = 0; i < resnum - 1; ++i) {
int index = i;
for(j = i + 1; j < resnum; ++j){
if(result[index] < result[j]){
index = j;
}
}
swap(result[index], result[i]);
swap(srres[index], srres[i]);
}
return true;
}

LSparseRepresentation::~LSparseRepresentation() {
int size = dic.size();
for(int i = 0; i < size; ++i)  {
vector<double>().swap(dic[i]);
}
vector<vector<double>>().swap(dic);
vector<int>().swap(dicclassnum);
}

double LSparseRepresentation::Dot(const vector<double>& vec1, const vector<double>& vec2) {

double sum = 0;
int len = vec1.size();
for(int i = 0; i < len; ++i) {
sum += vec1[i] * vec2[i];
}

return sum;
}

double LSparseRepresentation::Norm(const vector<double>& vec) {

double norm = 0;
int len = vec.size();
for(int i = 0; i < len; ++i) {
norm += vec[i] * vec[i];
}

return sqrt(norm);
}

bool LSparseRepresentation::solve(vector<vector<double>>& phi,
vector<double>& y, vector<double>& x) {

int col = phi.size();
if (col <= 0) return false;
int row = phi[0].size();
if (row != y.size() || col != x.size()) return false;
MatrixXd A(row, col);
VectorXd b(row);
for(int i = 0; i < row; ++i) {
b(i) = y[i];
for(int j = 0; j < col; ++j)
A(i, j) = phi[j][i];
}

VectorXd result = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
for(int i = 0; i < col; ++i) {
x[i] = result(i);
}

return true;
}

bool LSparseRepresentation::OrthMatchPursuit(vector<double>& y,
double min_residual, int sparsity,
vector<double>& x, vector<int>& patch_indices) {

int fsize = y.size(); 
if (!fsize) return false;
int dcol = dic.size();      
if (!dcol) return false;

int drow = dic[0].size();   
if (!drow || drow != fsize) return false;

vector<double> residual(y); 
vector<vector<double>> phi; 
x.clear();

double max_coefficient;
unsigned int patch_index;
vector<double> coefficient(dcol,0);

for(;;) {
max_coefficient = 0;
#pragma omp parallel for shared(coefficient)
for (int i = 0; i < dcol; ++i) {
coefficient[i] = (double)Dot(dic[i], residual);
}

for (int i = 0; i < dcol; ++i) {
if (fabs(coefficient[i]) > fabs(max_coefficient)) {
max_coefficient = coefficient[i];
patch_index = i;
}
}

patch_indices.push_back(patch_index);
phi.push_back(dic[patch_index]);
x.push_back(0.0);

if( !solve(phi, y, x) ) {
return false;
}

residual = y - phi *x;
double res_norm = (double)Norm(residual);

if (x.size() >= sparsity)
break;
}

return true;
}

#endif
