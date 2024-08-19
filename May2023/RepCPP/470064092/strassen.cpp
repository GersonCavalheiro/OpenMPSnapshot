#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <random>
#include <string>
#include <omp.h>

using namespace std;

const string QUERY_PRINT = "print";
const string QUERY_GET_NUM_OF_THREADS = "get_threads";
const string QUERY_SET_NUM_OF_THREADS = "set_threads";
const string QUERY_STRASSEN_SEQ = "strassen_seq";
const string QUERY_STRASSEN_PAR = "strassen_par";
const string QUERY_STRASSEN_SEQ_N_TIMES = "strassen_seq_n_times";
const string QUERY_STRASSEN_PAR_N_TIMES = "strassen_par_n_times";
const string QUERY_GET_MATRICES_FROM_TERMINAL = "init";
const string QUERY_GET_MATRICES_FROM_FILE = "load";
const string QUERY_SAVE_MATRICES_TO_FILE = "save";
const string QUERY_GENERATE_RANDOM_MATRICES = "gen_random";
const string QUERY_HELP = "help";
const string QUERY_CLEAR = "clear";
const string QUERY_EXIT = "exit";

const int MAX_NUM_OF_THREADS = 100;
const int MIN_NUM_OF_THREADS = 1;
const int MIN_TIMES = 1;
const int MIN_N = 0;
const int MAX_N = 12;
const int MIN_VAL = -50000;
const int MAX_VAL = 50000;

const int AD_HOC = 64;

int numOfThreads = 8;

void printHelp();
void clearScreen();
void run();

void getMatricesFromTerminal(int**& a, int**& b, int**& c, int& n);
void saveMatrices(int** a, int** b, int** c, int n, char path[300]);
void loadMatrices(int**& a, int**& b, int**& c, int& n, char path[300]);
void generateRandomMatrices(int**& a, int**& b, int**& c, int& n);

double strassenSequentialProduct(int** a, int** b, int**& c, int n
, int xa = 0, int ya = 0, int xb = 0, int yb = 0, bool computeTime = false);

void sequentialSquareMatrixAddition(int** a, int** b, int**& c, int n
, int xa, int ya, int xb, int yb, int xc = 0, int yc = 0);

void sequentialSquareMatrixSubtraction(int** a, int** b, int**& c, int n
, int xa, int ya, int xb, int yb, int xc = 0, int yc = 0);


double strassenParallelProduct(int** a, int** b, int**& c, int n
, int xa = 0, int ya = 0, int xb = 0, int yb = 0, bool computeTime = false);

void parallelSquareMatrixAddition(int** a, int** b, int**& c, int n
, int xa, int ya, int xb, int yb, int xc = 0, int yc = 0);

void parallelSquareMatrixSubtraction(int** a, int** b, int**& c, int n
, int xa, int ya, int xb, int yb, int xc = 0, int yc = 0);


void printMatrices(int** a, int** b, int** c, int n, bool computed);

void deleteAllIfNeeded(int**& a, int**& b, int**& c, int& n);

int main()
{
run();
}

void printHelp()
{
printf("\n_______________HELP_______________\n");

printf("\n\t* To input the matrices from terminal enter: \"%s\" without the quotations.\n"
, QUERY_GET_MATRICES_FROM_TERMINAL.c_str());

printf("\n\t* To read the matrices from a file with path <PATH> enter: \"%s <PATH>\" without the quotations.\n"
, QUERY_GET_MATRICES_FROM_FILE.c_str());

printf("\n\t* To save the matrices to a file with path <PATH> enter: \"%s <PATH>\" without the quotations.\n"
, QUERY_SAVE_MATRICES_TO_FILE.c_str());

printf("\n\t* To generate random matrices with size 2^<N> enter: \"%s <N>\" without the quotations.\n"
, QUERY_GENERATE_RANDOM_MATRICES.c_str());


printf("\n\t* To get the number of threads used for parallel computation enter: \"%s\" without the quotations.\n"
, QUERY_GET_NUM_OF_THREADS.c_str());


printf("\n\t* To set the number of threads used for parallel computation enter: \"%s <numberOfThreads>\" without the quotations.\n"
, QUERY_SET_NUM_OF_THREADS.c_str());

printf("\n\t* To print the matrices enter: \"%s\" without the quotations.\n", QUERY_PRINT.c_str());

printf("\n\t* To compute the produxt of input matrices sequentially enter: \"%s\" without the quotations.\n"
, QUERY_STRASSEN_SEQ.c_str());

printf("\n\t* To compute the produxt of input matrices in parallel enter: \"%s\" without the quotations.\n"
, QUERY_STRASSEN_PAR.c_str());

printf("\n\t* To compute the produxt of input matrices <N> times sequentially enter: \"%s <N>\" without the quotations.\n"
, QUERY_STRASSEN_SEQ_N_TIMES.c_str());

printf("\n\t* To compute the produxt of input matrices <N> times in parallel enter: \"%s <N>\" without the quotations.\n"
, QUERY_STRASSEN_PAR_N_TIMES.c_str());

printf("\n\t* To print these information again enter: \"%s\" without the quotations.\n", QUERY_HELP.c_str());

printf("\n\t* To clear the screen enter: \"%s\" without the quotations.\n", QUERY_CLEAR.c_str());

printf("\n\t* To exit enter: \"%s\" without the quotations.\n", QUERY_EXIT.c_str());

printf("\n");

printf("\n+ You should first use \"%s\" or \"%s\" commands to initialize input matrices.\n"
, QUERY_GET_MATRICES_FROM_TERMINAL.c_str(), QUERY_GET_MATRICES_FROM_FILE.c_str());
printf("\n+ While using \"%s\", or \"%s\" queries, <PATH> should not contain any whitespace.\n"
, QUERY_GET_MATRICES_FROM_FILE.c_str(), QUERY_SAVE_MATRICES_TO_FILE.c_str());
printf("\n+ When the matrices are too large don't use the \"%s\" query.\n"
, QUERY_PRINT.c_str());
printf("\n+ This matrix product is implementation of Strassen algorithm.\n");
}

void clearScreen() {
system("cls");
}

void run()
{
char c_query[50] = "";
string query(c_query);
printHelp();

bool computed = false;
int n = -1;
int **a = nullptr, **b = nullptr, **c = nullptr;

double sigmaTime = 0.0;

while (query.compare(QUERY_EXIT))
{
printf("\nEnter any command(to see all possible commands type help): ");

scanf("%s", c_query);
query.assign(c_query);

if (!query.compare(QUERY_GET_MATRICES_FROM_TERMINAL))
{
deleteAllIfNeeded(a, b, c, n);
getMatricesFromTerminal(a, b, c, n);
computed = false;
}
else if (!query.compare(QUERY_GET_MATRICES_FROM_FILE))
{
char path[300] = "";
scanf("%s", path);

deleteAllIfNeeded(a, b, c, n);
loadMatrices(a, b, c, n, path);
computed = false;
}
else if (!query.compare(QUERY_SAVE_MATRICES_TO_FILE))
{
if (n == -1)
{
printf("\nError: Matrices are not initialized!\n\n");
continue;
}

char path[300] = "";
scanf("%s", path);

saveMatrices(a, b, c, n, path);
}
else if (!query.compare(QUERY_GENERATE_RANDOM_MATRICES))
{
int temp;
scanf("%d", &temp);
if (temp < MIN_N || temp > MAX_N)
{
printf("\nError: <N> should be at least %d and at most %d!\n\n", MIN_N, MAX_N);
continue;
}
deleteAllIfNeeded(a, b, c, n);
n = temp;
generateRandomMatrices(a, b, c, n);
computed = false;
}
else if (!query.compare(QUERY_GET_NUM_OF_THREADS))
{
printf("\nNumber of threads = %d\n", numOfThreads);
}
else if (!query.compare(QUERY_SET_NUM_OF_THREADS))
{
int nt;
scanf("%d", &nt);
if (nt < MIN_NUM_OF_THREADS || nt > MAX_NUM_OF_THREADS)
printf("\nError: The number of threads should be at least %d and at most %d!\n\n"
, MIN_NUM_OF_THREADS, MAX_NUM_OF_THREADS);
else
numOfThreads = nt;
}
else if (!query.compare(QUERY_PRINT))
{
if (n == -1)
{
printf("\nError: Matrices are not initialized!\n\n");
continue;
}

printMatrices(a, b, c, n, computed);
}
else if (!query.compare(QUERY_STRASSEN_SEQ))
{
if (n == -1)
{
printf("\nError: Matrices are not initialized!\n\n");
continue;
}

sigmaTime = strassenSequentialProduct(a, b, c, n, 0, 0, 0, 0, true);
computed = true;
printf("\tsequential strassen for matrices of size %d took %f seconds to compute.\n", n, sigmaTime);
sigmaTime = 0.0;
}
else if (!query.compare(QUERY_STRASSEN_PAR))
{
if (n == -1)
{
printf("\nError: Matrices are not initialized!\n\n");
continue;
}

sigmaTime = strassenParallelProduct(a, b, c, n, 0, 0, 0, 0, true);
computed = true;
printf("\tparallel strassen for matrices of size %d took %f seconds to compute.\n", n, sigmaTime);
sigmaTime = 0.0;
}
else if (!query.compare(QUERY_STRASSEN_SEQ_N_TIMES))
{
if (n == -1)
{
printf("\nError: Matrices are not initialized!\n\n");
continue;
}

int t;
scanf("%d", &t);
if (t < MIN_TIMES)
{
printf("\nError: N should be at least %d!\n\n", MIN_TIMES);
continue;
}

for (int i = 0; i < t; ++i)
{
sigmaTime += strassenSequentialProduct(a, b, c, n, 0, 0, 0, 0, true);
}

printf("\tsequential strassen for matrices of size %d took %f seconds in average to compute.\n", n, sigmaTime / t);
sigmaTime = 0.0;
}
else if (!query.compare(QUERY_STRASSEN_PAR_N_TIMES))
{
if (n == -1)
{
printf("\nError: Matrices are not initialized!\n\n");
continue;
}

int t;
scanf("%d", &t);
if (t < MIN_TIMES)
{
printf("\nError: N should be at least %d!\n\n", MIN_TIMES);
continue;
}

for (int i = 0; i < t; ++i)
{
sigmaTime += strassenParallelProduct(a, b, c, n, 0, 0, 0, 0, true);
}

printf("\tparallel strassen for matrices of size %d took %f seconds in average to compute.\n", n, sigmaTime / t);
sigmaTime = 0.0;
}
else if (!query.compare(QUERY_CLEAR))
{
clearScreen();
}
else if (!query.compare(QUERY_HELP))
{
printHelp();
}
else if (query.compare(QUERY_EXIT))
{
printf("\nError: Command not defined!\n\n");
}
}

deleteAllIfNeeded(a, b, c, n);
}

void getMatricesFromTerminal(int**& a, int**& b, int**& c, int& n)
{
while (n < MIN_N || n > MAX_N) {
printf("Enter the log base 2 of matrices\' size(it should be an integer not lower than %d or more than %d): "
, MIN_N, MAX_N);
scanf("%d", &n);
}

n = 1 << n;

a = new int* [n];
b = new int* [n];
c = new int* [n];

printf("\n__________First Matrix__________\n\n");
for (int i = 0; i < n; i++)
{
a[i] = new int[n];
c[i] = new int[n];

printf("+ %dth row of the first matrix:\n", i+1);
for (int j = 0; j < n; j++)
{
printf("\t- enter the %dth element: ", j+1);
scanf("%d", a[i] + j);
c[i][j] = 0;
}
}

printf("\n__________Second Matrix__________\n\n");
for (int i = 0; i < n; i++)
{
b[i] = new int[n];

printf("+ %dth row of the second matrix:\n", i+1);
for (int j = 0; j < n; j++)
{
printf("\t- enter the %dth element: ", j+1);
scanf("%d", b[i] + j);
}
}

printf("\n#### Matrices initialized succesfully! ####\n\n");
}

void saveMatrices(int** a, int** b, int** c, int n, char path[300])
{
int pow = 0;
int tempN = n;
while (tempN >>= 1) pow++;

FILE* fp = fopen(path, "w+");
fprintf(fp, "%d\n", pow);

for (int i = 0; i < n; ++i)
{
for (int j = 0; j < n; ++j)
fprintf(fp, j == n - 1 ? "%d" : "%d ", a[i][j]);
fprintf(fp, i == n - 1 ? "\n\n" : "\n");
}

for (int i = 0; i < n; ++i)
{
for (int j = 0; j < n; ++j)
fprintf(fp, j == n - 1 ? "%d" : "%d ", b[i][j]);
fprintf(fp, i == n - 1 ? "" : "\n");
}

fclose(fp);

printf("\n#### Matrices saved succesfully! ####\n\n");
}

void loadMatrices(int**& a, int**& b, int**& c, int& n, char path[300])
{

FILE* fp = fopen(path, "r");
fscanf(fp, "%d", &n);
n = 1 << n;

a = new int* [n];
b = new int* [n];
c = new int* [n];

for (int i = 0; i < n; ++i)
{
a[i] = new int[n];
c[i] = new int[n];
for (int j = 0; j < n; ++j)
{
fscanf(fp, "%d", a[i] + j);
c[i][j] = 0;
}
}

for (int i = 0; i < n; ++i)
{
b[i] = new int[n];
for (int j = 0; j < n; ++j)
fscanf(fp, "%d", b[i] + j);
}

fclose(fp);

printf("\n#### Matrices loaded succesfully! ####\n\n");
}

void generateRandomMatrices(int**& a, int**& b, int**& c, int& n)
{
n = 1 << n;

random_device rd;
mt19937 gen(rd());
uniform_int_distribution<> distrib(MIN_VAL, MAX_VAL);

a = new int* [n];
b = new int* [n];
c = new int* [n];

for (int i = 0; i < n; ++i)
{
a[i] = new int[n];
b[i] = new int[n];
c[i] = new int[n];
for (int j = 0; j < n; ++j)
{
a[i][j] = distrib(gen);
b[i][j] = distrib(gen);
c[i][j] = 0;
}
}

printf("\n#### Random matrices generated succesfully! ####\n\n");
}

double strassenSequentialProduct(int** a, int** b, int**& c, int n
, int xa, int ya, int xb, int yb, bool computeTime)
{
double startTime = 0.0;
if (computeTime)
startTime = omp_get_wtime();

if (n <= AD_HOC)
{
for (int i = 0; i < n; ++i)
{
for (int j = 0; j < n; ++j)
{
c[i][j] = 0;
for (int k = 0; k < n; ++k)
{
c[i][j] += a[i + xa][k + ya] * b[k + xb][j + yb];
}
}
}
}
else
{
int dividedSize = n / 2;
int ***s = new int** [10], ***p = new int** [7];
for (int i = 0; i < 10; ++i)
{
if (i < 7)
p[i] = new int* [dividedSize];
s[i] = new int* [dividedSize];
for (int j = 0; j < dividedSize; ++j)
{
if (i < 7)
p[i][j] = new int[dividedSize];
s[i][j] = new int[dividedSize];
}
}

sequentialSquareMatrixSubtraction(b, b, s[0], dividedSize, xb, yb + dividedSize
, xb + dividedSize, yb + dividedSize);

sequentialSquareMatrixAddition(a, a, s[1], dividedSize, xa, ya, xa, ya + dividedSize);

sequentialSquareMatrixAddition(a, a, s[2], dividedSize, xa + dividedSize, ya
, xa + dividedSize, ya + dividedSize);

sequentialSquareMatrixSubtraction(b, b, s[3], dividedSize, xb + dividedSize, yb, xb, yb);

sequentialSquareMatrixAddition(a, a, s[4], dividedSize, xa, ya, xa + dividedSize, ya + dividedSize);

sequentialSquareMatrixAddition(b, b, s[5], dividedSize, xb, yb, xb + dividedSize, yb + dividedSize);

sequentialSquareMatrixSubtraction(a, a, s[6], dividedSize, xa, ya + dividedSize
, xa + dividedSize, ya + dividedSize);

sequentialSquareMatrixAddition(b, b, s[7], dividedSize, xb + dividedSize, yb
, xb + dividedSize, yb + dividedSize);

sequentialSquareMatrixSubtraction(a, a, s[8], dividedSize, xa, ya, xa + dividedSize, ya);

sequentialSquareMatrixAddition(b, b, s[9], dividedSize, xb, yb, xb, yb + dividedSize);

strassenSequentialProduct(a, s[0], p[0], dividedSize, xa, ya, 0, 0);
strassenSequentialProduct(s[1], b, p[1], dividedSize, 0, 0, xb + dividedSize, yb + dividedSize);
strassenSequentialProduct(s[2], b, p[2], dividedSize, 0, 0, xb, yb);
strassenSequentialProduct(a, s[3], p[3], dividedSize, xa + dividedSize, ya + dividedSize, 0, 0);
strassenSequentialProduct(s[4], s[5], p[4], dividedSize);
strassenSequentialProduct(s[6], s[7], p[5], dividedSize);
strassenSequentialProduct(s[8], s[9], p[6], dividedSize);

sequentialSquareMatrixAddition(p[4], p[3], c, dividedSize, 0, 0, 0, 0, 0, 0);
sequentialSquareMatrixSubtraction(c, p[1], c, dividedSize, 0, 0, 0, 0, 0, 0);
sequentialSquareMatrixAddition(c, p[5], c, dividedSize, 0, 0, 0, 0, 0, 0);

sequentialSquareMatrixAddition(p[0], p[1], c, dividedSize, 0, 0, 0, 0, 0, dividedSize);

sequentialSquareMatrixAddition(p[2], p[3], c, dividedSize, 0, 0, 0, 0, dividedSize, 0);

sequentialSquareMatrixAddition(p[4], p[0], c, dividedSize, 0, 0, 0, 0, dividedSize, dividedSize);
sequentialSquareMatrixSubtraction(c, p[2], c, dividedSize
, dividedSize, dividedSize, 0, 0, dividedSize, dividedSize);
sequentialSquareMatrixSubtraction(c, p[6], c, dividedSize
, dividedSize, dividedSize, 0, 0, dividedSize, dividedSize);

for (int i = 0; i < 10; ++i)
{
for (int j = 0; j < dividedSize; ++j)
{
delete[] s[i][j];
s[i][j] = nullptr;
if (i < 7)
{
delete[] p[i][j];
p[i][j] = nullptr;
}
}
delete[] s[i];
s[i] = nullptr;
if (i < 7) 
{
delete[] p[i];
p[i] = nullptr;
}
}
delete[] s;
s = nullptr;
delete[] p;
p = nullptr;
}

if (computeTime)
startTime = omp_get_wtime() - startTime;
return startTime;
}

void sequentialSquareMatrixAddition(int** a, int** b, int**& c, int n
, int xa, int ya, int xb, int yb, int xc, int yc)
{
for (int i = 0; i < n; ++i)
for (int j = 0; j < n; ++j)
c[i + xc][j + yc] = a[i + xa][j + ya] +b[i + xb][j + yb];
}

void sequentialSquareMatrixSubtraction(int** a, int** b, int**& c, int n
, int xa, int ya, int xb, int yb, int xc, int yc)
{
for (int i = 0; i < n; ++i)
for (int j = 0; j < n; ++j)
c[i + xc][j + yc] = a[i + xa][j + ya] - b[i + xb][j + yb];
}

double strassenParallelProduct(int** a, int** b, int**& c, int n
, int xa, int ya, int xb, int yb, bool computeTime)
{
double startTime = 0.0;
if (computeTime)
startTime = omp_get_wtime();

if (n <= AD_HOC)
{
#		pragma omp parallel for num_threads(numOfThreads)
for (int i = 0; i < n; ++i)
{
for (int j = 0; j < n; ++j)
{
c[i][j] = 0;
for (int k = 0; k < n; ++k)
{
c[i][j] += a[i + xa][k + ya] * b[k + xb][j + yb];
}
}
}
}
else
{
int dividedSize = n / 2;
int ***s = new int** [10], ***p = new int** [7];
for (int i = 0; i < 10; ++i)
{
if (i < 7)
p[i] = new int* [dividedSize];
s[i] = new int* [dividedSize];
for (int j = 0; j < dividedSize; ++j)
{
if (i < 7)
p[i][j] = new int[dividedSize];
s[i][j] = new int[dividedSize];
}
}

#		pragma omp parallel
{
#			pragma omp sections
{
#				pragma omp section
parallelSquareMatrixSubtraction(b, b, s[0], dividedSize, xb, yb + dividedSize
, xb + dividedSize, yb + dividedSize);

#				pragma omp section
parallelSquareMatrixAddition(a, a, s[1], dividedSize, xa, ya, xa, ya + dividedSize);

#				pragma omp section
parallelSquareMatrixAddition(a, a, s[2], dividedSize, xa + dividedSize, ya
, xa + dividedSize, ya + dividedSize);

#				pragma omp section
parallelSquareMatrixSubtraction(b, b, s[3], dividedSize, xb + dividedSize, yb, xb, yb);

#				pragma omp section
parallelSquareMatrixAddition(a, a, s[4], dividedSize, xa, ya, xa + dividedSize, ya + dividedSize);

#				pragma omp section
parallelSquareMatrixAddition(b, b, s[5], dividedSize, xb, yb, xb + dividedSize, yb + dividedSize);

#				pragma omp section
parallelSquareMatrixSubtraction(a, a, s[6], dividedSize, xa, ya + dividedSize
, xa + dividedSize, ya + dividedSize);

#				pragma omp section
parallelSquareMatrixAddition(b, b, s[7], dividedSize, xb + dividedSize, yb
, xb + dividedSize, yb + dividedSize);

#				pragma omp section
parallelSquareMatrixSubtraction(a, a, s[8], dividedSize, xa, ya, xa + dividedSize, ya);

#				pragma omp section
parallelSquareMatrixAddition(b, b, s[9], dividedSize, xb, yb, xb, yb + dividedSize);
}

#			pragma omp sections
{
#				pragma omp section
strassenParallelProduct(a, s[0], p[0], dividedSize, xa, ya, 0, 0);
#				pragma omp section
strassenParallelProduct(s[1], b, p[1], dividedSize, 0, 0, xb + dividedSize, yb + dividedSize);
#				pragma omp section
strassenParallelProduct(s[2], b, p[2], dividedSize, 0, 0, xb, yb);
#				pragma omp section
strassenParallelProduct(a, s[3], p[3], dividedSize, xa + dividedSize, ya + dividedSize, 0, 0);
#				pragma omp section
strassenParallelProduct(s[4], s[5], p[4], dividedSize);
#				pragma omp section
strassenParallelProduct(s[6], s[7], p[5], dividedSize);
#				pragma omp section
strassenParallelProduct(s[8], s[9], p[6], dividedSize);
}

#			pragma omp sections
{
#				pragma omp section
{
parallelSquareMatrixAddition(p[4], p[3], c, dividedSize, 0, 0, 0, 0, 0, 0);
parallelSquareMatrixSubtraction(c, p[1], c, dividedSize, 0, 0, 0, 0, 0, 0);
parallelSquareMatrixAddition(c, p[5], c, dividedSize, 0, 0, 0, 0, 0, 0);
}

#				pragma omp section
parallelSquareMatrixAddition(p[0], p[1], c, dividedSize, 0, 0, 0, 0, 0, dividedSize);

#				pragma omp section
parallelSquareMatrixAddition(p[2], p[3], c, dividedSize, 0, 0, 0, 0, dividedSize, 0);

#				pragma omp section
{
parallelSquareMatrixAddition(p[4], p[0], c, dividedSize, 0, 0, 0, 0, dividedSize, dividedSize);
parallelSquareMatrixSubtraction(c, p[2], c, dividedSize
, dividedSize, dividedSize, 0, 0, dividedSize, dividedSize);
parallelSquareMatrixSubtraction(c, p[6], c, dividedSize
, dividedSize, dividedSize, 0, 0, dividedSize, dividedSize);
}
}
}

for (int i = 0; i < 10; ++i)
{
for (int j = 0; j < dividedSize; ++j)
{
delete[] s[i][j];
s[i][j] = nullptr;
if (i < 7)
{
delete[] p[i][j];
p[i][j] = nullptr;
}
}
delete[] s[i];
s[i] = nullptr;
if (i < 7) 
{
delete[] p[i];
p[i] = nullptr;
}
}
delete[] s;
s = nullptr;
delete[] p;
p = nullptr;
}

if (computeTime)
startTime = omp_get_wtime() - startTime;
return startTime;
}

void parallelSquareMatrixAddition(int** a, int** b, int**& c, int n, int xa, int ya, int xb, int yb, int xc, int yc)
{
#	pragma omp parallel for num_threads(numOfThreads)
for (int i = 0; i < n; ++i)
for (int j = 0; j < n; ++j)
c[i + xc][j + yc] = a[i + xa][j + ya] + b[i + xb][j + yb];
}

void parallelSquareMatrixSubtraction(int** a, int** b, int**& c, int n, int xa, int ya, int xb, int yb, int xc, int yc)
{
#	pragma omp parallel for num_threads(numOfThreads)
for (int i = 0; i < n; ++i)
for (int j = 0; j < n; ++j)
c[i + xc][j + yc] = a[i + xa][j + ya] - b[i + xb][j + yb];
}

void printMatrices(int** a, int** b, int** c, int n, bool computed)
{
int pow = 0;
int tempN = n;
while (tempN >>= 1) pow++;

printf("Size = 2^%d = %d\n", pow, n);

printf("A:\n");
for (int i = 0; i < n; ++i)
{
for (int j = 0; j < n; ++j)
printf("\t%d ", a[i][j]);
printf("\n");
}

printf("\nB:\n");
for (int i = 0; i < n; ++i)
{
for (int j = 0; j < n; ++j)
printf("\t%d ", b[i][j]);
printf("\n");
}

if (computed)
{
printf("\nC:\n");
for (int i = 0; i < n; ++i)
{
for (int j = 0; j < n; ++j)
printf("\t%d ", c[i][j]);
printf("\n");
}
}
else
printf("\nProduct is not computed!\n");

printf("\n#### Done ####\n\n");
}

void deleteAllIfNeeded(int**& a, int**& b, int**& c, int& n)
{
if (n != -1)
{
for (int i = 0; i < n; ++i)
{
delete[] a[i];
delete[] b[i];
delete[] c[i];
a[i] = nullptr;
b[i] = nullptr;
c[i] = nullptr;
}
delete[] a;
delete[] b;
delete[] c;
a = nullptr;
b = nullptr;
c = nullptr;

n = -1;
}
}