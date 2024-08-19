#include <iostream>
#include "support_vector_machine.h"
#include "utils.h"
#include <omp.h>
#include <ctime>
#include <random>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace Eigen;


SupportVectorMachine::SupportVectorMachine(int max_iter, double C, double tolerance) : max_iter(max_iter), C(C), tolerance(tolerance) {}

SupportVectorMachine::~SupportVectorMachine(){}

void SupportVectorMachine::fit(const MatrixXd& X_train, const VectorXd& y_train, VectorXd(*kernel)(MatrixXd x, VectorXd y)){
X = X_train;
y = y_train;
size = X.rows();
b = 0;
MatrixXd K(size, size);
for(int i = 0; i < size; i++){
K.col(i) = kernel(X, X.row(i));
}
alpha = VectorXd::Zero(size);
sv_idx = VectorXi::Ones(size);
int iter;
for(iter = 0; iter < max_iter; iter++){
VectorXd alpha_prev = alpha;
for(int j = 0; j < size; j++){
int i = random_index(j);
double eta = 2 * K(i,j) - K(i,i) - K(j,j);
if(eta >= 0){
continue;
}
pair<double,double> pair_LH = find_bounds(i, j);
double e_i = error(i, K.col(i));
double e_j = error(j, K.col(j));
double alpha_io = alpha(i);
double alpha_jo = alpha(j);
alpha(j) -= ((y(j) * (e_i - e_j)) / eta);
alpha(j) = clip(alpha(j), pair_LH.second, pair_LH.first);
alpha(i) = alpha(i) + y(i)*y(j)*(alpha_jo - alpha(j));

double b1 = b - e_i - y(i)*(alpha_io-alpha(i))*K(i,i) - y(j)*(alpha_jo-alpha(j))*K(i,j);
double b2 = b - e_j - y(i)*(alpha_io-alpha(i))*K(i,j) - y(j)*(alpha_jo-alpha(j))*K(j,j);
if(alpha(i) > 0 && alpha(i) < C){
b = b1;
}else if(alpha(j) > 0 && alpha(j) < C){
b = b2;
}else{
b = (b1+b2)/2;
}
}
double diff = 0.0;
for(int index = 0; index < alpha.size(); index++){
diff += abs(alpha(index) - alpha_prev(index));
}
cout << "diff: " << diff << endl;
if(diff < tolerance){
break;
}
}
cout << "Convergence has reached after " << iter << " " << endl;
sv_count = 0;
for(int i = 0; i < alpha.size(); i++){
if(alpha(i) > 0){
sv_idx[i] = 1;
sv_count++;
}else{
sv_idx[i] = 0;
}
}
X_sv = MatrixXd::Zero(sv_count, X.cols());
alpha_sv = VectorXd::Zero(sv_count);
y_sv = VectorXd::Zero(sv_count);
cout << "support vector count :" << sv_count << endl;
for(int i = 0, j = 0; i < sv_idx.size(); i++){
if(sv_idx(i) == 1){
X_sv.row(j) = X.row(i);
alpha_sv(j) = alpha(i);
y_sv(j) = y(i);
j++;
}
}
}

VectorXi SupportVectorMachine::predict(const MatrixXd& X_test){
int n = X_test.rows();
cout << "rows: " << n << endl;
VectorXi ret(n);
#pragma omp parallel for
for(int i = 0; i < n; i++){
double res = predict_row(X_test.row(i));
ret(i) = res>=0?1:0;
}
return ret;
}


double SupportVectorMachine::predict_row(const VectorXd& x_row){
VectorXd k_v = Utils::Linear(X_sv, x_row);
return ((alpha_sv.array()*y_sv.array()).matrix().transpose()*k_v)+b;
}

int SupportVectorMachine::random_index(int z){
int i = z;
uniform_int_distribution<int> u(0, size-1);
default_random_engine e(0);
while(i == z){
i = u(e);
}
return i;
}

pair<double,double> SupportVectorMachine::find_bounds(int i, int j){
double L, H;
if(y(i) != y(j)){
L = max(0.0, double(alpha(j) - alpha(i)));
H = min(C, double(C-alpha(i)+alpha(j)));
}
else{
L = max(0.0, double(alpha(i)+alpha(j)-C));
H = min(C, double(alpha(i)+alpha(j)));
}
return pair<double,double>(L,H);
}

double SupportVectorMachine::predict_row(const VectorXd& x_row, const VectorXd& k_v){
return ((alpha.array()*y.array()).matrix().transpose()*k_v)+b;
}

double SupportVectorMachine::error(int i, const VectorXd& k_v){
return predict_row(X.row(i), k_v) - y(i);
}

double SupportVectorMachine::clip(double alpha, double H, double L){
if(alpha > H){
alpha = H;
}
if(alpha < L){
alpha = L;
}
return alpha;
}