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
void generateDisJointPairs(boost::numeric::ublas::vector<double> &top, boost::numeric::ublas::vector<double> &bot) {
int sizeTop = top.size();
int sizeBot = bot.size();
boost::numeric::ublas::vector<double> newTop(sizeTop);
boost::numeric::ublas::vector<double> newBot(sizeBot);
for (int i = 0; i < sizeTop; i++)
{
if (i == 0) {
newTop(i) = bot(0);
}
else if (i > 0) {
newTop(i) = top(i - 1);
}
if (i == (sizeTop - 1)) {
newBot(i) = top(i);
}
else {
newBot(i) = bot(i + 1);
}
}
top = newTop;
bot = newBot;
}
void generateStartDisJointPair(boost::numeric::ublas::vector<double> &top, boost::numeric::ublas::vector<double> &bot) {
for (int i = 0, j = 0; (i + j) < top.size() * 2;) {
if ((i + j) % 2 == 0) {
bot(i) = (i + 1) * 2;
i++;
}
else {
top(j) = 1 + j * 2;
j++;
}
}
}
void cycleVector(boost::numeric::ublas::vector<double> &v) {
boost::numeric::ublas::vector<double> newV(v.size());
for (int i = 1; i < v.size(); i++) {
newV(i) = v(i - 1);
}
newV(0) = v(v.size() - 1);
v = newV;
}
bool checkConjunction(boost::numeric::ublas::vector<double> &v, int i) {
int j = i - 1;
bool result = (v(i) == 0);
if (i - 1 >= 0) {
result = result && (v(i - 1) == 0);
}
if (i + 1 < v.size() - 1) {
result = result && (v(i + 1) == 0);
}
return result;
}
void setConjunction(boost::numeric::ublas::vector<double> &v, int i) {
v(i) = 1;
if (i - 1 >= 0) {
v(i - 1) = 1;
}
if (i + 1 < v.size() - 1) {
v(i + 1) = 1;
}
}
int jacobiAsync(boost::numeric::ublas::matrix<double> &S, boost::numeric::ublas::vector<double> &e, boost::numeric::ublas::matrix<double>  &U, int &iter) {
iter = 0;
int col, row;
bool iterating = true;
const int n = S.size1();
if (S.size2() != n)
{
return -1;
}
boost::numeric::ublas::matrix<double> M(n, n);
U = boost::numeric::ublas::identity_matrix<double>(n, n);
boost::numeric::ublas::vector<double> top(S.size1() / 2);
boost::numeric::ublas::vector<double> bot(S.size2() / 2);
generateStartDisJointPair(top, bot);
while (iterating)
{
iter++;
boost::numeric::ublas::vector<double> distr_status(n);
boost::numeric::ublas::matrix<double> toProcess(n, 2);
int processPointer = 0;
for (int j = 0; j < n - 1; j++)
{
distr_status.clear();
toProcess.clear();
processPointer = 0;
#pragma omp parallel for shared(distr_status, top, bot, S, U, processPointer) private(row,col) 
for (int i = 0; i < S.size1() / 2; i++)
{
std::string num_thread_str = "[" + std::to_string(omp_get_thread_num()) + "]";
row = std::max(top(i), bot(i)) - 1;
col = std::min(top(i), bot(i)) - 1;
if (checkConjunction(distr_status, row) && checkConjunction(distr_status, col)) {
setConjunction(distr_status, row);
setConjunction(distr_status, col);
rotateRowCol(S, U, col, row);
}
else {
toProcess(processPointer, 1) = row;
toProcess(processPointer, 0) = col;
#pragma omp atomic
processPointer++;
}
}
for (int i = 0; (toProcess(i, 0) != 0 && (toProcess(i, 1) != 0)); i++) {
rotateRowCol(S, U, toProcess(i, 0), toProcess(i, 1));
}
generateDisJointPairs(top, bot);
}
cycleVector(bot);
if (sumOffDiagonal(S) < _EPS) iterating = false;
}
for (int i = 0; i < n; i++) e(i) = S(i, i);
return 0;
}
