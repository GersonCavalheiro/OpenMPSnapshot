#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/progress.hpp>
#include "boost/format.hpp"
#include <omp.h>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <list>
#include "JacobiEigenvalueAlgorithm.h"
#include <cmath> 
#include <omp.h>
#include < stdio.h>
#include "f2c.h"
#include "clapack.h"
#include <float.h>                 
#define omptest true
void classic_bisection(
boost::numeric::ublas::vector<real> diagonal,
boost::numeric::ublas::vector<real> offdiagonal,
boost::numeric::ublas::vector<real> &result);
void modified_bisection(
int nthreads,
boost::numeric::ublas::vector<real> diagonal,
boost::numeric::ublas::vector<real> offdiagonal,
boost::numeric::ublas::vector<real> &eigenvalues);
void estimate_eigen_interval(
boost::numeric::ublas::vector<real> diagonal,
boost::numeric::ublas::vector<real> offdiagonal,
real& left_boundary,
real& right_boundary
);
void compute_group_bisect(
boost::numeric::ublas::vector<real> diagonal,
boost::numeric::ublas::vector<real> offdiagonal,
double left_boundary,
double right_boundary,
int m1,
int m2,
int n,
double relative_tolerance,
boost::numeric::ublas::vector<real> &eigenvalues_result
);
int sturm_sequence(real d[], real off[], real x, int n);
void bisection_test(
int nthreads,
boost::numeric::ublas::matrix<double> M,
std::string isWriteToConsole,
std::ofstream fp_outs[1],
int i) {
int iter = 0;
auto begin = std::chrono::high_resolution_clock::now();
auto end = std::chrono::high_resolution_clock::now();
integer matrixSize = M.size1();
boost::numeric::ublas::vector<real> result(matrixSize);
boost::numeric::ublas::vector<real> diagonal(matrixSize);
boost::numeric::ublas::vector<real> offdiagonal(matrixSize);
for (int j = 0; j < matrixSize; j++) {
diagonal[j] = M(j, j);
}
offdiagonal[0] = 0.0;
for (int j = 1; j < matrixSize; j++) {
offdiagonal[j] = M(j - 1, j);
}
begin = std::chrono::high_resolution_clock::now();
modified_bisection(nthreads, diagonal, offdiagonal, result);
end = std::chrono::high_resolution_clock::now();
double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1000000.0;
bool writeCSV = true;
if (writeCSV) {
writeToAllStreams((boost::format("%1%,%2%")
% M.size1() % duration).str(), fp_outs);
return;
}
if (isWriteToConsole == "true") {
std::string eig = "[";
eig += std::to_string(matrixSize);
eig += "](";
for (int i = 0; i < matrixSize - 1; i++)
{
eig += std::to_string(result[i]);
eig += ",";
}
eig += std::to_string(result[matrixSize - 1]);
eig += ")";
writeToAllStreams((boost::format("Name: %1% \nEigenvalues: %2% \nElapsed(ms): %3% \nIter: %4%")
% "bisection"% eig%duration%iter).str(), fp_outs);
writeToAllStreams("============================", fp_outs);
writeToAllStreams((boost::format("#%1%: \n") % i).str(), fp_outs);
}
}
void modified_bisection(
int nthreads,
boost::numeric::ublas::vector<real> diagonal,
boost::numeric::ublas::vector<real> offdiagonal,
boost::numeric::ublas::vector<real> &eigenvalues)
{
int n = diagonal.size();
int i, j, k;
double left_boundary;
double right_boundary;
double relative_tolerance = 10e-06;
double epsilon;
double x;
offdiagonal[0] = 0.0;
right_boundary = diagonal[n - 1] + fabs(offdiagonal[n - 1]);
left_boundary = diagonal[n - 1] - fabs(offdiagonal[n - 1]);
for (i = n - 2; i >= 0; i--) {
x = fabs(offdiagonal[i]) + fabs(offdiagonal[i + 1]);
if ((diagonal[i] + x) > right_boundary) right_boundary = diagonal[i] + x;
if ((diagonal[i] - x) < left_boundary) left_boundary = diagonal[i] - x;
}
epsilon = ((right_boundary + left_boundary) > 0.0) ? left_boundary : right_boundary;
epsilon *= DBL_EPSILON;
if (relative_tolerance < epsilon) relative_tolerance = epsilon;
epsilon = 0.5 * relative_tolerance + 7.0 * epsilon;
#pragma omp parallel for
for (int i = 0; i < nthreads; i++) {
int f = i * (n / nthreads);
int l = (i + 1)*(n / nthreads) - 1;
compute_group_bisect(diagonal, offdiagonal, left_boundary, right_boundary, f, l, n, relative_tolerance, eigenvalues);
}
}
void compute_group_bisect(
boost::numeric::ublas::vector<real> diagonal,
boost::numeric::ublas::vector<real> offdiagonal,
double left_boundary,
double right_boundary,
int m1,
int m2,
int n, 
double relative_tolerance,
boost::numeric::ublas::vector<real> &eigenvalues_result
)
{
double *lowerbounds;
double tolerance;
int j;
boost::numeric::ublas::vector<real> eigenvalues(eigenvalues_result);
lowerbounds = (double*)malloc(n * sizeof(double));
double xlower, xupper, xmid;
double q;
for ( int i = 0; i < n; i++) {
eigenvalues[i] = right_boundary;
lowerbounds[i] = left_boundary;
}
xupper = right_boundary;
for (int k = m2; k >= m1; k--) {
xlower = left_boundary;
for (int i = k; i >= 0; i--)
if (xlower < lowerbounds[i]) { xlower = lowerbounds[i]; break; }
if (xupper > eigenvalues[k]) xupper = eigenvalues[k];
tolerance = 2.0 * DBL_EPSILON * (fabs(xlower) + fabs(xupper)) + relative_tolerance;
while ((xupper - xlower) > tolerance) {
xmid = 0.5 * (xupper + xlower);
j = sturm_sequence(&diagonal[0], &offdiagonal[0], xmid, n) - 1;
if (j < k) {
if (j < 0) { xlower = lowerbounds[0] = xmid; }
else {
xlower = lowerbounds[j + 1] = xmid;
if (eigenvalues[j] > xmid) eigenvalues[j] = xmid;
}
}
else xupper = xmid;
tolerance = 2.0 * DBL_EPSILON * (fabs(xlower) + fabs(xupper)) + relative_tolerance;
};
eigenvalues[k] = 0.5 * (xupper + xlower);
}
for (int i = m1; i <= m2; i++) {
eigenvalues_result[i] = eigenvalues[i];
}
}
void classic_bisection(
boost::numeric::ublas::vector<real> diagonal,
boost::numeric::ublas::vector<real> offdiagonal,
boost::numeric::ublas::vector<real> &eigenvalues)
{
int n = diagonal.size();
int i, j, k;
double tolerance;
double left_boundary;
double right_boundary;
double x;
double q;
double xlower, xupper, xmid;
double *lowerbounds;
double relative_tolerance = 10e-05;
double epsilon;
lowerbounds = (double*)malloc(n * sizeof(double));
offdiagonal[0] = 0.0;
right_boundary = diagonal[n - 1] + fabs(offdiagonal[n - 1]);
left_boundary = diagonal[n - 1] - fabs(offdiagonal[n - 1]);
for (i = n - 2; i >= 0; i--) {
x = fabs(offdiagonal[i]) + fabs(offdiagonal[i + 1]);
if ((diagonal[i] + x) > right_boundary) right_boundary = diagonal[i] + x;
if ((diagonal[i] - x) < left_boundary) left_boundary = diagonal[i] - x;
}
epsilon = ((right_boundary + left_boundary) > 0.0) ? left_boundary : right_boundary;
epsilon *= DBL_EPSILON;
if (relative_tolerance < epsilon) relative_tolerance = epsilon;
epsilon = 0.5 * relative_tolerance + 7.0 * epsilon;
for (i = 0; i < n; i++) {
eigenvalues[i] = right_boundary;
lowerbounds[i] = left_boundary;
}
xupper = right_boundary;
for (k = n - 1; k >= 0; k--) {
xlower = left_boundary;
for (i = k; i >= 0; i--)
if (xlower < lowerbounds[i]) { xlower = lowerbounds[i]; break; }
if (xupper > eigenvalues[k]) xupper = eigenvalues[k];
tolerance = 2.0 * DBL_EPSILON * (fabs(xlower) + fabs(xupper)) + relative_tolerance;
while ((xupper - xlower) > tolerance) {
xmid = 0.5 * (xupper + xlower);
j = sturm_sequence(&diagonal[0], &offdiagonal[0], xmid, n) - 1;
if (j < k) {
if (j < 0) { xlower = lowerbounds[0] = xmid; }
else {
xlower = lowerbounds[j + 1] = xmid;
if (eigenvalues[j] > xmid) eigenvalues[j] = xmid;
}
}
else xupper = xmid;
tolerance = 2.0 * DBL_EPSILON * (fabs(xlower) + fabs(xupper)) + relative_tolerance;
};
eigenvalues[k] = 0.5 * (xupper + xlower);
}
}
void estimate_eigen_interval(
boost::numeric::ublas::vector<real> diagonal,
boost::numeric::ublas::vector<real> offdiagonal,
real& left_boundary,
real& right_boundary
) 
{
int matrixSize = diagonal.size();
boost::numeric::ublas::vector<real> radius_i(matrixSize);
boost::numeric::ublas::vector<real> left_estimation(matrixSize);
boost::numeric::ublas::vector<real> right_estimation(matrixSize);
for (size_t i = 0; i < matrixSize; i++)
{
if (i != matrixSize - 1)
{
radius_i[i] = fabs(offdiagonal[i]) + fabs(offdiagonal[i+1]);
}
else
{
radius_i[i] = fabs(offdiagonal[i]);
}
}
for (size_t i = 0; i < matrixSize; i++)
{
left_estimation[i] = diagonal[i] - radius_i[i];
right_estimation[i] = diagonal[i] + radius_i[i];
}
left_boundary = *std::min_element(std::begin(left_estimation), std::end(left_estimation));
right_boundary = *std::max_element(std::begin(right_estimation), std::end(right_estimation));
}
static int sturm_sequence(real d[], real off[], real x, int n)
{
double q = 1.0;
int k = 0;
int i;
for (i = 0; i < n; i++) {
if (q == 0.0)
q = d[i] - x - fabs(off[i]) / DBL_EPSILON;
else
q = d[i] - x - off[i] * off[i] / q;
if (q < 0.0) k++;
}
return k;
}
