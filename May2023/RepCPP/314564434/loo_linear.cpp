#include <RcppArmadillo.h>
#include <Rcpp.h>
#include "myomp.h"
using namespace Rcpp;


Rcpp::List fastols(arma::mat const &X,
arma::vec const &y,
int const compute_se = 1,
int const compute_hat = 0) {
int ncols = X.n_cols, nrows = X.n_rows;
arma::vec hat(nrows, arma::fill::zeros);
arma::vec pred_err(nrows, arma::fill::zeros);
arma::mat R(ncols, ncols, arma::fill::zeros);
arma::mat Q;
arma::qr_econ(Q, R, X);
arma::vec Qy(ncols, arma::fill::none);
for(int k = 0; k < ncols; k++){
Qy(k) = dot(Q.col(k), y);
}
arma::vec beta = solve(R, Qy);
arma::vec fittedy = Q*Qy;
arma::vec res = y - fittedy;
if(compute_se == 1){
double sig2 = arma::as_scalar(sum(pow(res, 2))/(Q.n_rows - ncols));
R = inv(R);
R = sig2 * R * R.t();
}
if(compute_hat==1){
hat = sum(Q % Q, 1);
pred_err = res / (1 - hat);
}
List listout = List::create(Named("beta")          = beta,
Named("hatdiag")       = hat,
Named("loo_pred_err")  = pred_err,
Named("fitted.values") = fittedy,
Named("VCV")           = R);
return listout;
}


Rcpp::List fastolswt(arma::mat const &X,
arma::vec const &y,
arma::vec const &w,
int const compute_se = 1,
int const compute_hat = 0) {
int ncols = X.n_cols, nrows = X.n_rows;
arma::vec hat(nrows, arma::fill::zeros);
arma::vec pred_err(nrows, arma::fill::zeros);
arma::mat R(ncols, ncols, arma::fill::zeros);
arma::vec ws = sqrt(w);
arma::vec yw = y % ws;
arma::mat Q;
arma::qr_econ(Q, R, X.each_col() % ws);
arma::vec Qy(ncols, arma::fill::none);
for(int k = 0; k < ncols; k++){
Qy(k) = dot(Q.col(k), yw);
}
arma::vec beta = solve(R, Qy);
arma::vec fittedy = (Q*Qy)/ws;
yw = y - fittedy;
if(compute_se == 1){
double sig2 = arma::as_scalar(arma::dot(w, pow(yw, 2))/(Q.n_rows - ncols));
R = inv(R);
R = sig2 * R * R.t();
}
if(compute_hat==1){
hat = sum(Q % Q, 1);
pred_err = yw / (1 - hat);
}
List listout = List::create(Named("beta")          = beta,
Named("hatdiag")       = hat,
Named("loo_pred_err")  = pred_err,
Named("fitted.values") = fittedy,
Named("VCV")           = R);
return listout;
}


Rcpp::List fastols_by(arma::mat const &X,
arma::vec const &y,
IntegerVector const &g,
int const nthr = 1,
int const compute_se = 1,
int const compute_hat = 0) {
int nrows = X.n_rows, ncols = X.n_cols;
arma::vec beta(nrows, arma::fill::zeros),
hat(nrows, arma::fill::zeros),
pred_err(nrows, arma::fill::zeros),
start(nrows, arma::fill::zeros),
res(nrows, arma::fill::zeros);

int cur = g(0);
int numgrps = 1;
for(int i=1; i < nrows; i++){
if(g(i)!=cur){
start(numgrps) = i;
numgrps += 1;
cur = g(i);
}
}
start(numgrps) = nrows;
start = start.head(numgrps + 1);
arma::mat betamat(ncols, numgrps, arma::fill::none);
arma::mat varmat(ncols, numgrps, arma::fill::zeros);
#pragma omp parallel for schedule(dynamic, nthr)
for(int j=0; j < numgrps; j++){
int startj = start(j), endj = start(j + 1) - 1;
arma::mat Q, R;
arma::qr_econ(Q, R, X.rows(startj, endj));
arma::vec Qy(ncols, arma::fill::none);
for(int k = 0; k < ncols; k++){
Qy(k) = dot(Q.col(k), y.subvec(startj, endj));
}
betamat.col(j) = solve(R, Qy);
res.subvec(startj, endj) = y.subvec(startj, endj) - Q*Qy;
if(compute_se == 1){
R = inv(R);
varmat.col(j) = arma::sum(R % R, 1);
}
if(compute_hat==1){
hat.subvec(startj, endj) = sum(Q % Q, 1);
}
}
if(compute_se == 1){
arma::vec tmp = pow(res, 2);
arma::vec denom = arma::shift(start, -1) - start - ncols;
#pragma omp parallel for schedule(dynamic, nthr)
for(int j = 0; j < numgrps; j++){
varmat.col(j) *= arma::sum(tmp.subvec(start(j), start(j + 1) - 1))/denom(j);
}
}
if(compute_hat == 1){
pred_err = res/(1-hat);
}
List listout = List::create(Named("beta")          = betamat,
Named("hatdiag")       = hat,
Named("loo_pred_err")  = pred_err,
Named("residuals")     = res,
Named("variance")      = varmat);
return listout;
}



Rcpp::List fastols_bywt(arma::mat const &X,
arma::vec const &y,
arma::vec const &w,
IntegerVector const &g,
int const nthr = 1,
int const compute_se = 1,
int const compute_hat = 0) {
int nrows = X.n_rows, ncols = X.n_cols;
arma::vec beta(nrows, arma::fill::zeros),
hat(nrows, arma::fill::zeros),
pred_err(nrows, arma::fill::zeros),
start(nrows, arma::fill::zeros);

arma::vec ws = sqrt(w);
arma::mat Xw = X.each_col() % ws;
arma::vec yw = y % ws;

int cur = g(0);
int numgrps = 1;
for(int i=1; i < nrows; i++){
if(g(i)!=cur){
start(numgrps) = i;
numgrps += 1;
cur = g(i);
}
}
start(numgrps) = nrows;
start = start.head(numgrps + 1);
arma::mat betamat(ncols, numgrps, arma::fill::none);
arma::mat varmat(ncols, numgrps, arma::fill::zeros);
#pragma omp parallel for schedule(dynamic, nthr)
for(int j=0; j < numgrps; j++){
int startj = start(j), endj = start(j + 1) - 1;
arma::mat Q, R;
arma::qr_econ(Q, R, Xw.rows(startj, endj));
arma::vec Qy(ncols, arma::fill::none);
for(int k = 0; k < ncols; k++){
Qy(k) = dot(Q.col(k), yw.subvec(startj, endj));
}
betamat.col(j) = solve(R, Qy);
yw.subvec(startj, endj) = y.subvec(startj, endj) - (Q*Qy)/ws.subvec(startj, endj);
if(compute_se == 1){
R = inv(R);
varmat.col(j) = arma::sum(R % R, 1);
}
if(compute_hat==1){
hat.subvec(startj, endj) = sum(Q % Q, 1);
}
}
if(compute_se == 1){
arma::vec tmp = w % pow(yw, 2);
arma::vec denom = arma::shift(start, -1) - start - ncols;
#pragma omp parallel for schedule(dynamic, nthr)
for(int j = 0; j < numgrps; j++){
varmat.col(j) *= arma::sum(tmp.subvec(start(j), start(j + 1) - 1))/denom(j);
}
}
if(compute_hat == 1){
pred_err = yw/(1-hat);
}
List listout = List::create(Named("beta")          = betamat,
Named("hatdiag")       = hat,
Named("loo_pred_err")  = pred_err,
Named("residuals")     = yw,
Named("variance")      = varmat);
return listout;
}
