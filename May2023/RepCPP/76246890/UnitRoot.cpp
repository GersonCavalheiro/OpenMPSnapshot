
#include "../include/URT.hpp"

#ifdef _OPENMP 
#include <omp.h>
static const int MAX_THREADS = omp_get_max_threads();
#endif

namespace urt {


template <typename T>
UnitRoot<T>::UnitRoot(const Vector<T>& data, int lags, const std::string& trend, bool regression)
{
this->data = data; 
set_data();
#ifdef USE_ARMA
nobs = data.n_rows;
#elif defined(USE_BLAZE) || defined(USE_EIGEN) 
nobs = data.size();
#endif        
this->lags = lags;
this->trend = trend;
this->regression = regression;
}



template <typename T>
UnitRoot<T>::UnitRoot(const Vector<T>& data, const std::string& method, const std::string& trend, bool regression)
{
this->data = data;
set_data();
#ifdef USE_ARMA
nobs = data.n_rows;
#elif defined(USE_BLAZE) || defined(USE_EIGEN) 
nobs = data.size();
#endif   
this->method = method;
this->trend = trend;
this->regression = regression;
}


template <typename T>
const T& UnitRoot<T>::get_stat() const
{
return stat;
}


template <typename T>
const T& UnitRoot<T>::get_pval() const
{
return pval;
}


template <typename T>
const OLS<T>& UnitRoot<T>::get_ols() const
{
return *result;
}


template <typename T>
const std::vector<std::string>& UnitRoot<T>::get_trends() const
{
return valid_trends;
}


template <typename T>
void UnitRoot<T>::set_data()
{
ptr = &data;
}


template <typename T>
void UnitRoot<T>::set_lags()
{
if (!optim) {
if (lags < 0) {
lags = 0;
std::cout << "\n  WARNING: number of lags cannot be negative, it has been set to 0 by default.\n\n";
}
if (!lags_type.empty() && lags != prev_lags) {
lags_type = std::string();
}
} else {
if (max_lags < 0) {
max_lags = 0;
std::cout << "\n  WARNING: maximum number of lags cannot be negative, it has been set to a default value (L12-rule).\n\n";
}
if (!lags_type.empty() && (lags != prev_lags || max_lags != prev_max_lags)) {
lags_type = std::string();
}
if (lags != prev_lags) {
method = std::string();
max_lags = 0;
prev_max_lags = 0;
optim = false;
}
}

if ((optim && !max_lags) || !lags_type.empty()) {
if (lags_type == "short") {
max_lags = int(4 * pow(0.01 * nobs, 0.25));
}
else {
max_lags = int(12 * pow(0.01 * nobs, 0.25));
}
lags = max_lags; 
}

if (!optim && lags != prev_lags) {
new_test = true;
new_lags = true;
prev_lags = lags;
}
else if (optim && max_lags != prev_max_lags) {
new_test = true;
new_lags = true;
prev_max_lags = max_lags;
}
}


template <typename T>
void UnitRoot<T>::set_lags_type()
{
if (lags_type.empty() || lags_type == prev_lags_type) {
return;
}
if (lags_type != "long" && lags_type != "short") {
std::cout << "\n  WARNING: unknown default type of lags, long has been selected by default.\n\n";
lags_type = "long";
}
prev_lags_type = lags_type;
}


template <typename T>
void UnitRoot<T>::set_level()
{
if (level == prev_level) {
return;
}
if (level < 0) {
level *= -1;
}
if (level != prev_level) {
new_test = true;
new_level = true;
prev_level = level;
}
}


template <typename T>
void UnitRoot<T>::set_method()
{
if (method.empty() || method == prev_method) {
return;
}
auto p = std::find(valid_methods.begin(), valid_methods.end(), method);
if (p == valid_methods.end()) {
method = "AIC";
std::cout << "\n  WARNING: unknown method for lag length optimization, AIC has been selected by default.\n\n";
std::cout << "  Possible methods for this test are: " << valid_methods << "\n";
}
if (method != prev_method) {
new_method = true;
new_test = true;
}
optim = true;
}


template <typename T>
void UnitRoot<T>::set_IC()
{   
if (!new_method) { 
return;
}
if (method[0] != 'M') {
ICfunc = &UnitRoot<T>::IC;
} else {
ICfunc = &UnitRoot<T>::MIC;
}
if (method == "BIC" || method == "MBIC") {
ICcc = log(nobs - 1 - max_lags);
}
else if (method == "HQC" || method == "MHQC") {
ICcc = 2.0 * log(log(nobs - 1 - max_lags));
}
else {
ICcc = 2.0;
}
}


template <typename T>
void UnitRoot<T>::set_niter()
{
if (niter <= 0) {
niter = 1000;
std::cout << "\n  WARNING: in UnitRoot<T>::bootstrap(), number of iterations cannot be null or negative it has been set to 1000 by default.\n\n";
} else if (niter < 1000) {
std::cout << "\n  WARNING: in UnitRoot<T>::bootstrap(), number of iterations might be too small for computing p-value.\n\n";
}
if (niter != prev_niter) {
new_niter = true;
prev_niter = niter;
}
}


template <typename T>
void UnitRoot<T>::set_test_type()
{
if (test_type == prev_test_type) {
return;
}
if (test_type != "rho" && test_type != "tau") {
test_type = "tau";
std::cout << "\n  WARNING: unknown test type, tau has been selected by default.\n\n";
}
if (test_type != prev_test_type) {
new_test = true;
new_test_type = true;
prev_test_type = test_type;
}
}


template <typename T>
void UnitRoot<T>::set_trend()
{
if (trend == prev_trend) {
return;
}
auto p = std::find(valid_trends.begin(), valid_trends.end(), trend);
if (p == valid_trends.end()) {
trend = "c";
trend_type = "constant";
npar = 2;
std::cout << "\n  WARNING: unknown regression trend selected, regression with constant term has been selected by default.\n\n";
std::cout << "  Possible trends for this test are " << valid_trends << "\n"; 
} else {
if (trend == "c") {
trend_type = "constant";
npar = 2;
}
else if (trend == "nc") {
trend_type = "no constant";
npar = 1;
}
else if (trend == "ct") {
trend_type = "constant trend";
npar = 3;
}
else if (trend == "ctt") {
trend_type = "quadratic trend";
npar = 4;
}
}
if (trend != prev_trend) {
new_test = true;
new_trend = true;
prev_trend = trend;
}
}


template <typename T>
void UnitRoot<T>::ols_detrend()
{  
#ifdef USE_ARMA
Matrix<T> w(nobs, 1, arma::fill::ones); 
#elif USE_BLAZE
Matrix<T> w(nobs, 1);
w = forEach(w, [](T val){ return 1; });
#elif USE_EIGEN
Matrix<T> w = Matrix<T>::Ones(nobs, 1);
#endif

if (trend == "ct") {
#ifdef USE_ARMA
w.insert_cols(1, arma::cumsum(w));
#elif USE_BLAZE
w.resize(nobs, w.columns() + 1);
std::partial_sum(&w(0, 0), &w(nobs, 0), &w(0, 1));
#elif USE_EIGEN
w.conservativeResize(Eigen::NoChange, w.cols() + 1);
w.col(w.cols() - 1) = Vector<T>::LinSpaced(w.rows(), 0, w.rows() - 1);
#endif
}
else if (trend == "nc") {
throw std::invalid_argument("\n  ERROR: in UnitRoot<T>::ols_detrend(), no detrending possible when regression trend set to no constant.\n\n");
}

OLS<T> fit(*ptr, w);
z = *ptr - w * fit.param;
ptr = &z;
}


template <typename T>
void UnitRoot<T>::gls_detrend()
{
T alpha;
int nc;
if (trend == "ct") {
alpha = 1.0 - 13.5 / nobs;
nc = 2;
}
else if (trend == "c") { 
alpha = 1.0 - 7.0 / nobs;
nc = 1;
} else {
throw std::invalid_argument("\n  ERROR: in UnitRoot<T>::gls_detrend(), no detrending possible when regression trend set to no constant.\n\n");
}

Matrix<T> u(nobs, nc); 
Vector<T> v(nobs);
Matrix<T> w(nobs, nc);

u(0, 0) = 1;
v[0] = ptr->operator[](0);
w(0, 0) = 1;

if (nc == 2) {
w(0, 1) = 1;
u(0, 1) = 1;
}   

for (int i = 1; i < nobs; ++i) {
w(i, 0) = 1;
u(i, 0) = 1 - alpha;

if (nc == 2) {
w(i, 1) = i + 1;
u(i, 1) = w(i, 1) - alpha * i;
}
v[i] = ptr->operator[](i) - alpha * ptr->operator[](i - 1);
} 

OLS<T> fit(v, u);
z = *ptr - w * fit.param;
ptr = &z;
}


template <typename T>
void UnitRoot<T>::adf_regression()
{
int nr = nobs - lags - 1;
int nc = npar + lags;

if (nr  < nc) {
throw std::invalid_argument("\n  ERROR: in UnitRoot<T>::adf_regression(), more data required to compute ADF test for " + std::to_string(lags) + " lags, at least " + std::to_string(nc - nr) + " element(s) need(s) to be added or the number of lags to be reduced.\n\n");
}


#ifdef USE_ARMA
x.set_size(nr, nc);
y.set_size(nr);
#elif defined(USE_BLAZE) || defined(USE_EIGEN)
x.resize(nr, nc);
y.resize(nr);
#endif

for (int i = 0; i < nr; ++i) {
y[i] = ptr->operator[](i + lags + 1) - ptr->operator[](i + lags);
x(i, 0) = ptr->operator[](i + lags);
if (npar >= 2) { 
x(i, 1) = 1; 
if (npar >= 3) {
x(i, 2) = i + 1;
if (npar == 4) {
x(i, 3) = x(i, 2) * x(i, 2);
}
}
}   
for (int j = npar; j < nc ; ++j) {
x(i, j) = ptr->operator[](i - j + nc) - ptr->operator[](i - j + nc - 1);
}  
}

if (!regression) {
result = std::make_shared<OLS<T>>(y, x);
} else {
result = std::make_shared<OLS<T>>(y, x, true);
prev_regression = true;
}

stat = result->t_stat[0];
}


template <typename T>
void UnitRoot<T>::initialize_adf()
{
this->nrows = nobs - 1;
int ncols = npar + max_lags;

if (nrows - max_lags < ncols) {
throw std::invalid_argument("\n  ERROR: in UnitRoot<T>::adf_regression(), more data required to compute ADF test for " + std::to_string(max_lags) + " lags, at least " + std::to_string(ncols - nrows + max_lags) + " element(s) need(s) to be added or the number of lags to be reduced.\n\n");
}

#ifdef USE_ARMA
x.set_size(nrows, ncols);
y.set_size(nrows);
#elif defined(USE_EIGEN) || defined(USE_BLAZE)
x.resize(nrows, ncols);
y.resize(nrows);
#endif

#ifdef _OPENMP
#pragma omp parallel for num_threads(MAX_THREADS)
#endif
for (int i = 0; i < nrows; ++i) {
y[i] = ptr->operator[](i + 1) - ptr->operator[](i);
x(i, 0) = ptr->operator[](i);
if (npar >= 2) { 
x(i, 1) = 1; 
if (npar >= 3) {
x(i, 2) = i + 1;
if (npar == 4) {
x(i, 3) = x(i, 2) * x(i, 2);
}
}
}
for (int j = 0; j < max_lags ; ++j) {
if (j < i) {
x(i, j + npar) = ptr->operator[](i - j) - ptr->operator[](i - j - 1);
} else {
x(i, j + npar) = 0;
}
}   
}
}


template <typename T>
void UnitRoot<T>::optimize_lag()
{
if (new_method && !prev_method.empty() && prev_method[0] != 'T') {
#ifdef _OPENMP
#pragma omp parallel for num_threads(MAX_THREADS)
#endif
for (int i = max_lags; i > -1; --i) {
results[i]->IC = (this->*ICfunc)(results[i]);
}
}
else {
results.clear();
results.resize(max_lags + 1);

#ifdef _OPENMP
#pragma omp parallel for num_threads(MAX_THREADS)
#endif
for (int i = max_lags; i > -1; --i) {
#ifdef USE_ARMA
Vector<T> ysub(y.memptr() + i, nrows - i, false);
Matrix<T> xsub(x.submat(i, 0, nrows - 1, npar + i - 1));
#elif USE_BLAZE
Vector<T> ysub(subvector(y, i, nrows - i));
Matrix<T> xsub(submatrix(x, i, 0, nrows - i, npar + i));
#elif USE_EIGEN
Vector<T> ysub(y.segment(i, nrows - i));
Matrix<T> xsub(x.block(i, 0, nrows - i, npar + i));
#endif
results[i] = std::make_shared<OLS<T>>(ysub, xsub);         
results[i]->lags = i;
results[i]->IC = (this->*ICfunc)(results[i]);
}
}

auto best_ols = std::min_element(results.begin(), results.end(), [](const std::shared_ptr<OLS<T>>& ols1, const std::shared_ptr<OLS<T>>& ols2){return T(ols1->IC) < T(ols2->IC);});

result = *best_ols;
lags = result->lags;

if (regression) {
#ifdef USE_ARMA
Vector<T> ysub(y.memptr() + lags, nrows - lags, false);
Matrix<T> xsub(x.submat(lags, 0, nrows - 1, npar + lags - 1));
#elif USE_BLAZE
Vector<T> ysub(subvector(y, lags, nrows - lags));
Matrix<T> xsub(submatrix(x, lags, 0, nrows - lags, npar + lags)); 
#elif USE_EIGEN
Vector<T> ysub(y.segment(lags, nrows - lags));
Matrix<T> xsub(x.block(lags, 0, nrows - lags, npar + lags));
#endif
result->get_stats(ysub, xsub);
prev_regression = true;
}

stat = result->t_stat[0];
prev_lags = lags;
prev_method = method;
}



template <typename T>
T UnitRoot<T>::IC(const std::shared_ptr<OLS<T>>& res)
{

#ifdef USE_ARMA
int k = res->param.n_elem;
int n = res->resid.n_elem;
Vector<T> z(res->resid.memptr() + max_lags - res->lags, n - max_lags + res->lags, false);
#elif defined(USE_BLAZE) || defined(USE_EIGEN) 
int k = res->param.size();
int n = res->resid.size();
#ifdef USE_BLAZE
Vector<T> z(subvector(res->resid, max_lags - res->lags, n - max_lags + res->lags));
#elif USE_EIGEN
Vector<T> z(res->resid.segment(max_lags - res->lags, n - max_lags + res->lags));
#endif
#endif
T factor = 1.0 / (n - max_lags + res->lags);
#ifdef USE_ARMA
T sigma2 = arma::as_scalar(z.t() * z) * factor;
#elif USE_BLAZE
T sigma2 = (blaze::trans(z) * z) * factor;
#elif USE_EIGEN
T sigma2 = z.dot(z) * factor;
#endif
return log(sigma2) + ICcc * k * factor;
}


template <typename T>
T UnitRoot<T>::MIC(const std::shared_ptr<OLS<T>>& res)
{ 
#ifdef USE_ARMA
int n = res->resid.n_elem;
Vector<T> z(res->resid.memptr() + max_lags - res->lags, n - max_lags + res->lags, false);
#elif defined(USE_BLAZE) || defined(USE_EIGEN)
int n = res->resid.size();
#elif USE_BLAZE
Vector<T> z(subvector(res->resid, max_lags - res->lags, n - max_lags + res->lags));
#ifdef USE_EIGEN
Vector<T> z(res->resid.segment(max_lags - res->lags, n - max_lags + res->lags));
#endif
#endif
T factor = 1.0 / (n - max_lags + res->lags);
#ifdef USE_ARMA
T sigma2 = arma::as_scalar(z.t() * z) * factor;
Vector<T> y(ptr->memptr() + max_lags, nobs - max_lags - 1, false);
T y2 = arma::as_scalar(y.t() * y);
#elif USE_BLAZE
T sigma2 = (blaze::trans(z) * z) * factor;
Vector<T> y(subvector(*ptr, max_lags, nobs - max_lags - 1));
T y2 = blaze::trans(y) * y; 
#elif USE_EIGEN
T sigma2 = z.dot(z) * factor;
Vector<T> y(ptr->segment(max_lags, nobs - max_lags - 1));
T y2 = y.transpose() * y;
#endif
T tau = (res->param[0] * res->param[0] * y2) / sigma2; 
return log(sigma2) + ICcc * (tau + res->lags) * factor;
}


template <typename T>
void UnitRoot<T>::select_lag()
{
bool opt_lag_found = false;

results.clear();
results.resize(max_lags + 1);

for (int i = max_lags; i > -1; --i) {
#ifdef USE_ARMA
Vector<T> ysub(y.memptr() + i, nrows - i, false);
Matrix<T> xsub(x.submat(i, 0, nrows - 1, npar + i - 1));
#elif USE_BLAZE
Vector<T> ysub(subvector(y, i, nrows - i));
Matrix<T> xsub(submatrix(x, i, 0, nrows - i, npar + i)); 
#elif USE_EIGEN
Vector<T> ysub(y.segment(i, nrows - i));
Matrix<T> xsub(x.block(i, 0, nrows - i, npar + i));
#endif
results[i] = std::make_shared<OLS<T>>(ysub, xsub);
if (fabs(results[i]->t_stat[i]) > level) {
result = results[i];
if (regression) {
result->get_stats(ysub, xsub);
prev_regression = true;
}
result->lags = i;
opt_lag_found = true;
i = -1; 
}
}

if (!opt_lag_found) {
std::cout << "\n  WARNING: no optimal number of lags found with method T-STAT and level " << level << " please try again with a (positive) lower level.\n\n";
result = results[0];
if (regression) {
#ifdef USE_ARMA
Vector<T> ysub(y.memptr(), nrows, false);
Matrix<T> xsub(x.submat(0, 0, nrows - 1, npar - 1));
#elif USE_BLAZE
Vector<T> ysub(subvector(y, 0, nrows - 0));
Matrix<T> xsub(submatrix(x, 0, 0, nrows - 0, npar + 0));
#elif USE_EIGEN
Vector<T> ysub(y.segment(0, nrows));
Matrix<T> xsub(x.block(0, 0, nrows, npar));
#endif
result->get_stats(ysub, xsub);
prev_regression = true;
}
result->lags = 0;
}

lags = result->lags;
stat = result->t_stat[0];
prev_lags = lags;   
prev_method = method; 
}


template <typename T>
void UnitRoot<T>::compute_adf()
{
if (optim) {
void (UnitRoot<T>::*optim_func)();

if (method[0] != 'T') {
set_IC();
optim_func = &UnitRoot<T>::optimize_lag;
} else {
set_level();
optim_func = &UnitRoot<T>::select_lag;
}
if (new_trend || new_lags) {
initialize_adf();
new_method = false;
(this->*optim_func)();             
new_trend = false;
new_lags = false;
}
else if (new_method) { 
(this->*optim_func)();
new_method = false;
}
else if (method[0] == 'T' && new_level) {
(this->*optim_func)();
new_level = false;
}
else if (regression && !prev_regression) {
#ifdef USE_ARMA
Vector<T> ysub(y.memptr() + lags, nrows - lags, false);
Matrix<T> xsub(x.submat(lags, 0, nrows - 1, npar + lags - 1));
#elif USE_BLAZE
Vector<T> ysub(subvector(y, lags, nrows - lags));
Matrix<T> xsub(submatrix(x, lags, 0, nrows - lags, npar + lags)); 
#elif USE_EIGEN
Vector<T> ysub(y.segment(lags, nrows - lags));
Matrix<T> xsub(x.block(lags, 0, nrows - lags, npar + lags));
#endif
result->get_stats(ysub, xsub);
prev_regression = true;
}
else {
lags = prev_lags;
}
}
else {
if (new_trend || new_lags || new_data) {
adf_regression();
new_trend = false;
new_lags = false;
}
else if (regression && !prev_regression) {
result->get_stats(y, x);
prev_regression = true;
} 
}
}


template <typename T>
void UnitRoot<T>::run_bootstrap()
{  
bool saved_optim = optim;
optim = false;
T saved_stat = stat;
OLS<T> saved_result = *result;   

#ifdef USE_ARMA 
Vector<T> eps = result->resid - arma::mean(result->resid);
#elif USE_BLAZE
Vector<T> eps(result->resid);
T m = std::accumulate(&eps[0], &eps[eps.size()], 0.0) / eps.size();
eps = forEach(eps, [&m](const T& val){ return val - m; });  
#elif USE_EIGEN
T m = result->resid.mean();
Vector<T> eps = result->resid - Vector<T>::Constant(result->resid.size(), 1, m);   
#endif 

int nu = nobs - 1;

Vector<T> u, delta;

if (lags > 0) {
#ifdef USE_ARMA 
u = ptr->subvec(1, lags - 1) - ptr->subvec(0, lags - 2);
delta = arma::flipud(result->param.subvec(npar, result->param.size() - 1));
#elif USE_BLAZE
u = subvector(*ptr, 1, lags - 1) - subvector(*ptr, 0, lags - 1); 
delta = subvector(result->param, npar, result->param.size() - npar);  
std::reverse(&delta[0], &delta[delta.size()]);
#elif USE_EIGEN
u = ptr->segment(1, lags - 1) - ptr->segment(0, lags - 1); 
delta = result->param.segment(npar, result->param.size() - npar).reverse();
#endif    
}
#if defined(USE_ARMA) || defined(USE_BLAZE)
u.resize(nu);
#elif USE_EIGEN
u.conservativeResize(nu, Eigen::NoChange);
#endif

new_data = true;

#ifdef USE_ARMA 
Vector<T> new_path(ptr->memptr(), lags + 1, false);
new_path.resize(nobs);
#elif USE_BLAZE
Vector<T> new_path(subvector(*ptr, 0, lags + 1));
new_path.resize(nobs);
#elif USE_EIGEN
Vector<T> new_path(ptr->segment(0, lags + 1));
new_path.conservativeResize(nobs, Eigen::NoChange);
#endif

std::vector<T> stats(niter);

boost::uniform_int<> udistrib(0, eps.size() - 1); 
boost::variate_generator<boost::mt19937&, boost::uniform_int<>> runif(rng, udistrib);

T term = 0;

for(int k = niter; k--; ) { 
for (int i = lags; i < nu; ++i) {
switch (lags) {
case 0:
break;
default:
#ifdef USE_ARMA 
term = arma::as_scalar(arma::Col<T>(u.memptr() + i - lags, lags, false).t() * delta);
#elif USE_BLAZE
term = blaze::trans(subvector(u, i - lags, lags)) * delta;
#elif USE_EIGEN 
term = u.segment(i - lags, lags).transpose() * delta;          
#endif 
}
u[i] = term + eps[runif()];
new_path[i + 1] = new_path[i] + u[i];
}

ptr = &new_path;
stats[k] = statistic();
}

std::sort(stats.data(), stats.data() + niter);

for (int i = 0; i < 15; ++i) {
critical_values[i] = .5 * (stats[floor((niter - 1) * probas[i])] + stats[floor((niter - 1) * probas[i]) + 1]);
}

new_data = false;
set_data();
optim = saved_optim;
stat = saved_stat;
result = std::make_shared<OLS<T>>(saved_result);

}


template <typename T>
void UnitRoot<T>::compute_cv()
{
int n = nobs - lags - 1;

for (int i = 0; i < 15; ++i) {
critical_values[i] = 0;

int n0 = coeff_ptr->at(probas[i]).at(0).size();

for (int j = 0; j < n0; ++j) {
critical_values[i] += coeff_ptr->at(probas[i]).at(0).at(j) / pow(n, j);
}

int n1 = coeff_ptr->at(probas[i]).at(1).size();

for (int j = 0; j < n1; ++j) {
critical_values[i] += coeff_ptr->at(probas[i]).at(1).at(j) * pow( T(lags) / n, j + 1);
}
}
}


template <typename T>
void UnitRoot<T>::compute_pval()
{
if (stat <= critical_values[0]) {
pval = probas[0];
} else {
for (int i = 1; i < 15; ++i) {
if (stat <= critical_values[i]) {
pval = probas[i - 1] + (stat - critical_values[i - 1]) * (probas[i] - probas[i - 1]) / (critical_values[i] - critical_values[i - 1]);
break;
}
}
}

if (stat > critical_values[14]) {
pval = probas[14];
}
}


template <typename T>
const T& UnitRoot<T>::pvalue()
{
if (std::isnan(stat)) {
pval =  -std::nan("");
} 
else if (!bootstrap) {
if (new_test || prev_bootstrap) {
compute_cv();
compute_pval();

new_test = false;
}
} 
else {
set_niter();

if (new_test || new_niter || !prev_bootstrap) {
run_bootstrap();
compute_pval();

new_test = false;
new_niter = false;
}
}
prev_bootstrap = bootstrap;

return pval;
}


template <typename T>
void UnitRoot<T>::show()
{
std::cout << std::fixed << std::setprecision(3);

std::string stat_type;

if (test_type == "tau")  {      
stat_type = " (Z-tau)";
}
else if (test_type == "rho") { 
stat_type = " (Z-rho)";
}

std::cout << "\n  " + test_name + " Test Results" + stat_type + "\n";
std::cout << "  ====================================\n";

std::cout << "  Statistic" << std::setw(27) << stat << "\n";

std::string s;
(bootstrap) ? s = " (*)\n" : s = "\n";

std::cout << "  P-value";

if (pval <= probas[0]) {
std::cout << std::setw(29) << "< 0.001";
}
else if (pval >= probas[14]) {
std::cout << std::setw(29) << "> 0.999";
} else {
std::cout << std::setw(29) << pval;
}
std::cout << s;

if (optim) {
std::cout << "  Optimal Lags" << std::setw(24) << lags << "\n";

if (method == "T-STAT") {
std::cout << "  Method" << std::setw(30) << method << "\n";
} else {
std::cout << "  Criterion" << std::setw(27) << method << "\n";
}
} else {    
std::cout << "  Lags" << std::setw(32) << lags << "\n";
}

std::cout << "  Trend" << std::setw(31) << trend_type << "\n"; 
std::cout << "  ------------------------------------\n\n";

std::cout << "  Test Hypothesis\n";
std::cout << "  ------------------------------------\n";

if (test_name == "KPSS") {
std::cout << "  H0: The process is weakly stationary" << "\n";
std::cout << "  H1: The process contains a unit root" << "\n";
} else {
std::cout << "  H0: The process contains a unit root" << "\n";
std::cout << "  H1: The process is weakly stationary" << "\n";
}
std::cout << "\n";

std::cout << "  Critical Values" << s;
std::cout << "  ---------------\n";

std::vector<int> idx;

if (test_name == "KPSS") {
idx = {12, 10, 9};
} 
else {
idx = {2, 4, 5};
}

if (std::isnan(stat)) {
std::cout << "   1% " << std::setw(11) << -std::nan("") << "\n";
std::cout << "   5% " << std::setw(11) << -std::nan("") << "\n";
std::cout << "  10% " << std::setw(11) << -std::nan("") << "\n";
} 
else {
std::cout << "   1% " << std::setw(11) << critical_values[idx[0]] << "\n";
std::cout << "   5% " << std::setw(11) << critical_values[idx[1]] << "\n";
std::cout << "  10% " << std::setw(11) << critical_values[idx[2]] << "\n";
}

if (bootstrap) {
std::cout << "\n  (*) computed by bootstrap\n";
}
std::cout << "\n";

std::cout << "  Test Conclusion\n";
std::cout << "  ---------------\n";

if (pval <= 0.01) {
std::cout << "  We can reject H0 at the 1% significance level\n";
}
else if (pval <= 0.05) {
std::cout << "  We can reject H0 at the 5% significance level\n";
}
else if (pval <= 0.10) {
std::cout << "  We can reject H0 at the 10% significance level\n";
} 
else if (!std::isnan(pval)) {
std::cout << "  We cannot reject H0\n";
} 
else {
std::cout << "  We cannot conclude, nan produced\n";
}
std::cout << "\n";

if (regression) {
result->show();    
}
}


template <typename T>
std::ostream& operator<<(std::ostream& out, urt::UnitRoot<T>& test)
{
test.show();

return out;
}


}
