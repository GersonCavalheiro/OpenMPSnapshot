#include <RcppArmadillo.h>
#include <omp.h>
using namespace Rcpp;


template<typename T>
class matrix {
private:
unsigned int Nrows;
unsigned int Ncols;
std::vector<T> data;

public:
matrix(unsigned int Nrows_, unsigned int Ncols_):
Nrows(Nrows_), Ncols(Ncols_), data(Ncols_*Nrows_,(T) 0) {}



~matrix() {
}

T operator()(unsigned int i, unsigned int j) const;

void assign(unsigned int i, unsigned int j, T value);
void add(unsigned int i, unsigned int j, T value);

unsigned int nrow() const {return Nrows;}
unsigned int ncol() const {return Ncols;}
void print() const;

};



template<typename T>
inline matrix<T> Transpose(const matrix<T> & m, int ncores = 2) {
matrix<T> ans(m.ncol(), m.nrow());

omp_set_num_threads(ncores);
#pragma omp parallel for
for (int i = 0; i < m.nrow(); ++i) {
for (int j = 0; j < m.ncol(); ++j) {
ans.assign(j, i, m(i, j));
}
}

return ans;
}

#pragma omp declare simd uniform(i,j)
template<typename T>
inline T matrix<T>::operator()(unsigned int i, unsigned int j) const {
return data[i + j*Nrows];
}

#pragma omp declare simd uniform(i,j,value)
template<typename T>
inline void matrix<T>::assign(unsigned int i, unsigned int j, T value) {
data[i + j*Nrows] = value;
return;
}

#pragma omp declare simd uniform(i,j,value)
template<typename T>
inline void matrix<T>::add(unsigned int i, unsigned int j, T value) {
data[i + j*Nrows] += value;
return;
}

template<typename T>
inline void matrix<T>::print() const {
for (int i = 0; i < Nrows * Ncols; ++i) {
std::cout << data[i] << ", ";
}
std::cout << std::endl;
return;
}

template<typename T>
inline matrix<T> mat_product_pll(const matrix<T> & a, const matrix<T> & b, int ncores) {

if (a.ncol() != b.nrow())
throw std::logic_error("-a- and -b- should match the size.");

omp_set_num_threads(ncores);
matrix<T> at = Transpose<double>(a, ncores);
matrix<T> ans(a.nrow(), b.ncol());
#pragma omp parallel for 
for (int i = 0; i < a.nrow(); ++i) {
for (int j = 0; j < b.ncol(); ++j) {
#pragma omp simd simdlen(4)
for (int ii = 0; ii < a.ncol(); ++ii) 
ans.add(i, j, at(ii, i) * b(ii, j));
}
}

return ans;
}

template<typename T>
inline matrix<T> mat_product(const matrix<T> & a, const matrix<T> & b, int ncores) {

if (a.ncol() != b.nrow())
throw std::logic_error("-a- and -b- should match the size.");

matrix<T> ans(a.nrow(), b.ncol());
for (int i = 0; i < a.nrow(); ++i) {
for (int j = 0; j < b.ncol(); ++j) {
for (int ii = 0; ii < a.ncol(); ++ii) 
ans.add(i, j, a(i,ii) * b(ii, j));
}
}

return ans;
}

SEXP new_mat(int n, int m) {
Rcpp::XPtr< matrix<double> > ptr(
new matrix<double>(n, m), true
);

return ptr;
}

int set_mat(SEXP m, int i, int j, double x) {
Rcpp::XPtr<matrix<double>> ptr(m);
std::cout << ptr->operator()(i, j) << std::endl;
ptr->assign(i, j, x);
std::cout << ptr->operator()(i, j) << std::endl;
return 0;
}




int mult(int n0, int m0, int n1, int m1, bool pll = true, int ncores = 2) {

matrix<double> a(n0, m0);
matrix<double> b(n1, m1);


if (pll) {
matrix<double> ans = mat_product_pll(a,b,ncores);
} else {
matrix<double> ans = mat_product(a,b,ncores);
}


return 0;

}

int multArma(int n0, int m0, int n1, int m1) {

arma::mat a(n0, m0, arma::fill::zeros);
arma::mat b(n1, m1, arma::fill::zeros);

arma::mat ans = a * b;
return 0;

}


