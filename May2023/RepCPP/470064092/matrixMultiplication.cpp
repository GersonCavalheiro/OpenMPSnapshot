#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <random>
#include <string>
#include <omp.h>

using namespace std;

const string QUERY_PRINT = "print";
const string QUERY_GET_NUM_OF_THREADS = "get_threads";
const string QUERY_SET_NUM_OF_THREADS = "set_threads";
const string QUERY_GET_CHUNK_SIZE = "get_chunksize";
const string QUERY_SET_CHUNK_SIZE = "set_chunksize";
const string QUERY_GET_SCHEDULE = "get_schedule";
const string QUERY_SET_SCHEDULE = "set_schedule";
const string QUERY_PRODUCT_SEQ = "product_seq";
const string QUERY_PRODUCT_PAR = "product_par";
const string QUERY_PRODUCT_SEQ_N_TIMES = "product_seq_n_times";
const string QUERY_PRODUCT_PAR_N_TIMES = "product_par_n_times";
const string QUERY_GET_MATRICES_FROM_TERMINAL = "init";
const string QUERY_GET_MATRICES_FROM_FILE = "load";
const string QUERY_SAVE_MATRICES_TO_FILE = "save";
const string QUERY_GENERATE_RANDOM_MATRICES = "gen_random";
const string QUERY_HELP = "help";
const string QUERY_CLEAR = "clear";
const string QUERY_EXIT = "exit";

const string SCHEDULE_DEFAULT = "default";
const string SCHEDULE_STATIC = "static";
const string SCHEDULE_DYNAMIC = "dynamic";
const string SCHEDULE_GUIDED = "guided";

const int MAX_NUM_OF_THREADS = 100;
const int MIN_NUM_OF_THREADS = 1;
const int MAX_CHUNK_SIZE = 100000;
const int MIN_CHUNK_SIZE = 1;
const int MIN_TIMES = 1;
const int MIN_N = 1;
const int MAX_N = 4096;
const int MIN_VAL = -50000;
const int MAX_VAL = 50000;

int numOfThreads = 8;
int chunkSize = 1;
string schedule = "default";

void printHelp();
void clearScreen();
void run();

void getMatricesFromTerminal(int**& a, int**& b, int**& c, int& n, int& k, int& m);
void saveMatrices(int** a, int** b, int** c, int n, int k, int m, char path[300]);
void loadMatrices(int**& a, int**& b, int**& c, int& n, int& k, int& m, char path[300]);
void generateRandomMatrices(int**& a, int**& b, int**& c, int& n, int& k, int& m);

double sequentialMatrixProduct(int** a, int** b, int**& c, int n, int k, int m);
double parallelMatrixProduct(int** a, int** b, int**& c, int n, int k, int m);

void printMatrices(int** a, int** b, int** c, int n, int k, int m, bool computed);
void deleteAllIfNeeded(int**& a, int**& b, int**& c, int& n, int& k, int& m);

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

printf("\n\t* To generate random matrices with size <N> x <K> and <K> x <M> enter: \"%s <N> <K> <M>\" without the quotations.\n"
, QUERY_GENERATE_RANDOM_MATRICES.c_str());

printf("\n\t* To get the chunkSize used for scheduling parallel computation enter: \"%s\" without the quotations.\n"
, QUERY_GET_CHUNK_SIZE.c_str());

printf("\n\t* To get the number of threads used for parallel computation enter: \"%s\" without the quotations.\n"
, QUERY_GET_NUM_OF_THREADS.c_str());

printf("\n\t* To set the chunkSize used for scheduling parallel computation enter: \"%s <chunkSize>\" without the quotations.\n"
, QUERY_SET_CHUNK_SIZE.c_str());

printf("\n\t* To set the number of threads used for parallel computation enter: \"%s <numberOfThreads>\" without the quotations.\n"
, QUERY_SET_NUM_OF_THREADS.c_str());

printf("\n\t* To get the scheduling algorithm for parallel computation enter: \"%s\" without the quotations.\n"
, QUERY_GET_SCHEDULE.c_str());

printf("\n\t* To set the scheduling algorithm for parallel computation enter: \"%s <schedule>\" without the quotations.\n"
, QUERY_SET_SCHEDULE.c_str());
printf("\t\t Where schedule can be either \"%s\", \"%s\", \"%s\" or \"%s\".\n"
, SCHEDULE_DEFAULT.c_str(), SCHEDULE_STATIC.c_str(), SCHEDULE_DYNAMIC.c_str(), SCHEDULE_GUIDED.c_str());

printf("\n\t* To print the matrices enter: \"%s\" without the quotations.\n", QUERY_PRINT.c_str());

printf("\n\t* To compute the product of input matrices sequentially enter: \"%s\" without the quotations.\n"
, QUERY_PRODUCT_SEQ.c_str());

printf("\n\t* To compute the product of input matrices in parallel enter: \"%s\" without the quotations.\n"
, QUERY_PRODUCT_PAR.c_str());

printf("\n\t* To compute the product of input matrices <N> times sequentially enter: \"%s <N>\" without the quotations.\n"
, QUERY_PRODUCT_SEQ_N_TIMES.c_str());

printf("\n\t* To compute the product of input matrices <N> times in parallel enter: \"%s <N>\" without the quotations.\n"
, QUERY_PRODUCT_PAR_N_TIMES.c_str());

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
int n = -1, k = -1, m = -1;
int** a = nullptr, ** b = nullptr, ** c = nullptr;

double sigmaTime = 0.0;

while (query.compare(QUERY_EXIT))
{
printf("\nEnter any command(to see all possible commands type help): ");

scanf("%s", c_query);
query.assign(c_query);

if (!query.compare(QUERY_GET_MATRICES_FROM_TERMINAL))
{
deleteAllIfNeeded(a, b, c, n, k, m);
getMatricesFromTerminal(a, b, c, n, k, m);
computed = false;
}
else if (!query.compare(QUERY_GET_MATRICES_FROM_FILE))
{
char path[300] = "";
scanf("%s", path);

deleteAllIfNeeded(a, b, c, n, k, m);
loadMatrices(a, b, c, n, k, m, path);
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

saveMatrices(a, b, c, n, k, m, path);
}
else if (!query.compare(QUERY_GENERATE_RANDOM_MATRICES))
{
int tempN;
scanf("%d", &tempN);
if (tempN < MIN_N || tempN > MAX_N)
{
printf("\nError: <N> should be at least %d and at most %d!\n\n", MIN_N, MAX_N);
continue;
}

int tempK;
scanf("%d", &tempK);
if (tempK < MIN_N || tempK > MAX_N)
{
printf("\nError: <K> should be at least %d and at most %d!\n\n", MIN_N, MAX_N);
continue;
}

int tempM;
scanf("%d", &tempM);
if (tempM < MIN_N || tempM > MAX_N)
{
printf("\nError: <M> should be at least %d and at most %d!\n\n", MIN_N, MAX_N);
continue;
}
deleteAllIfNeeded(a, b, c, n, k, m);
n = tempN;
k = tempK;
m = tempM;
generateRandomMatrices(a, b, c, n, k, m);
computed = false;
}
else if (!query.compare(QUERY_GET_CHUNK_SIZE))
{
printf("\nchunkSize = %d\n", chunkSize);
}
else if (!query.compare(QUERY_GET_NUM_OF_THREADS))
{
printf("\nNumber of threads = %d\n", numOfThreads);
}
else if (!query.compare(QUERY_GET_SCHEDULE))
{
printf("\nScheduling algorithm is %s\n", schedule);
}
else if (!query.compare(QUERY_SET_CHUNK_SIZE))
{
int n;
scanf("%d", &n);
if (n < MIN_CHUNK_SIZE || n > MAX_CHUNK_SIZE)
printf("\nError: ChunkSize should be at least %d and at most %d!\n\n", MIN_CHUNK_SIZE, MAX_CHUNK_SIZE);
else
chunkSize = n;
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
else if (!query.compare(QUERY_SET_SCHEDULE))
{
char tempS[7] = "";
scanf("%s", tempS);
string inS(tempS);
if (!inS.compare(SCHEDULE_DEFAULT)) {
schedule = SCHEDULE_DEFAULT;
}
else if (!inS.compare(SCHEDULE_STATIC)) {
schedule = SCHEDULE_STATIC;
}
else if (!inS.compare(SCHEDULE_DYNAMIC)) {
schedule = SCHEDULE_DYNAMIC;
}
else if (!inS.compare(SCHEDULE_GUIDED)) {
schedule = SCHEDULE_GUIDED;
}
else {
printf("\nError: Undefined Schedule!\n\n");
}
}
else if (!query.compare(QUERY_PRINT))
{
if (n == -1)
{
printf("\nError: Matrices are not initialized!\n\n");
continue;
}

printMatrices(a, b, c, n, k, m, computed);
}
else if (!query.compare(QUERY_PRODUCT_SEQ))
{
if (n == -1)
{
printf("\nError: Matrices are not initialized!\n\n");
continue;
}

sigmaTime = sequentialMatrixProduct(a, b, c, n, k, m);
computed = true;
printf("\sequential product for matrices of size %d x %d and %d x %d took %f seconds to compute.\n"
, n, k, k, m, sigmaTime);
sigmaTime = 0.0;
}
else if (!query.compare(QUERY_PRODUCT_PAR))
{
if (n == -1)
{
printf("\nError: Matrices are not initialized!\n\n");
continue;
}

sigmaTime = parallelMatrixProduct(a, b, c, n, k, m);
computed = true;
printf("\parallel product for matrices of size %d x %d and %d x %d took %f seconds to compute.\n"
, n, k, k, m, sigmaTime);
sigmaTime = 0.0;
}
else if (!query.compare(QUERY_PRODUCT_SEQ_N_TIMES))
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
sigmaTime += sequentialMatrixProduct(a, b, c, n, k, m);
}

printf("\sequential product for matrices of size %d x %d and %d x %d took %f seconds in average to compute.\n"
, n, k, k, m, sigmaTime / t);
sigmaTime = 0.0;
}
else if (!query.compare(QUERY_PRODUCT_PAR_N_TIMES))
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
sigmaTime += parallelMatrixProduct(a, b, c, n, k, m);
}

printf("\parallel product for matrices of size %d x %d and %d x %d took %f seconds in average to compute.\n"
, n, k, k, m, sigmaTime / t);
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

deleteAllIfNeeded(a, b, c, n, k , m);
}

void getMatricesFromTerminal(int**& a, int**& b, int**& c, int& n, int& k, int& m)
{
while (n < MIN_N || n > MAX_N) {
printf("Enter the height of the first matrix (it should be an integer not lower than %d or more than %d): "
, MIN_N, MAX_N);
scanf("%d", &n);
}

while (k < MIN_N || k > MAX_N) {
printf("Enter the width of the first matrix(it should be an integer not lower than % d or more than % d) : "
, MIN_N, MAX_N);
scanf("%d", &k);
}

while (m < MIN_N || m > MAX_N) {
printf("Enter the width of the second matrix(it should be an integer not lower than % d or more than % d) : "
, MIN_N, MAX_N);
scanf("%d", &m);
}

a = new int* [n];
b = new int* [k];
c = new int* [n];

printf("\n__________First Matrix__________\n\n");
for (int i = 0; i < n; i++)
{
a[i] = new int[k];
c[i] = new int[m];

printf("+ %dth row of the first matrix:\n", i + 1);
for (int j = 0; j < k; j++)
{
printf("\t- enter the %dth element: ", j + 1);
scanf("%d", a[i] + j);
c[i][j] = 0;
}
}

printf("\n__________Second Matrix__________\n\n");
for (int i = 0; i < k; i++)
{
b[i] = new int[m];

printf("+ %dth row of the second matrix:\n", i + 1);
for (int j = 0; j < m; j++)
{
printf("\t- enter the %dth element: ", j + 1);
scanf("%d", b[i] + j);
}
}

printf("\n#### Matrices initialized succesfully! ####\n\n");
}

void saveMatrices(int** a, int** b, int** c, int n, int k, int m, char path[300])
{
FILE* fp = fopen(path, "w+");
fprintf(fp, "%d %d %d\n", n, k, m);

for (int i = 0; i < n; ++i)
{
for (int j = 0; j < k; ++j)
fprintf(fp, j == k - 1 ? "%d" : "%d ", a[i][j]);
fprintf(fp, i == n - 1 ? "\n\n" : "\n");
}

for (int i = 0; i < k; ++i)
{
for (int j = 0; j < m; ++j)
fprintf(fp, j == k - 1 ? "%d" : "%d ", b[i][j]);
fprintf(fp, i == m - 1 ? "" : "\n");
}

fclose(fp);

printf("\n#### Matrices saved succesfully! ####\n\n");
}

void loadMatrices(int**& a, int**& b, int**& c, int& n, int& k, int& m, char path[300])
{

FILE* fp = fopen(path, "r");
fscanf(fp, "%d%d%d", &n, &k, &m);

a = new int* [n];
b = new int* [k];
c = new int* [n];

for (int i = 0; i < n; ++i)
{
a[i] = new int[k];
c[i] = new int[m];
for (int j = 0; j < k; ++j)
{
fscanf(fp, "%d", a[i] + j);
c[i][j] = 0;
}
}

for (int i = 0; i < k; ++i)
{
b[i] = new int[m];
for (int j = 0; j < m; ++j)
fscanf(fp, "%d", b[i] + j);
}

fclose(fp);

printf("\n#### Matrices loaded succesfully! ####\n\n");
}

void generateRandomMatrices(int**& a, int**& b, int**& c, int& n, int& k, int& m)
{
random_device rd;
mt19937 gen(rd());
uniform_int_distribution<> distrib(MIN_VAL, MAX_VAL);

a = new int* [n];
b = new int* [k];
c = new int* [n];

for (int i = 0; i < n; ++i)
{
a[i] = new int[k];
c[i] = new int[m];
for (int j = 0; j < k; ++j)
{
a[i][j] = distrib(gen);
}
for (int j = 0; j < m; ++j)
{
c[i][j] = 0;
}
}

for (int i = 0; i < k; ++i)
{
b[i] = new int[m];
for (int j = 0; j < m; ++j)
{
b[i][j] = distrib(gen);
}
}

printf("\n#### Random matrices generated succesfully! ####\n\n");
}

double sequentialMatrixProduct(int** a, int** b, int**& c, int n, int k, int m)
{
double startTime = omp_get_wtime();
for (int i = 0; i < n; ++i)
{
for (int j = 0; j < m; ++j)
{
c[i][j] = 0;
for (int t = 0; t < k; ++t)
{
c[i][j] += a[i][t] * b[t][j];
}
}
}

startTime = omp_get_wtime() - startTime;
return startTime;
}


double parallelMatrixProduct(int** a, int** b, int**& c, int n, int k, int m)
{
double startTime = omp_get_wtime();

if (!schedule.compare(SCHEDULE_DEFAULT)) {
#	pragma omp parallel for num_threads(numOfThreads)
for (int i = 0; i < n; ++i)
{
for (int j = 0; j < m; ++j)
{
c[i][j] = 0;
for (int t = 0; t < k; ++t)
{
c[i][j] += a[i][t] * b[t][j];
}
}
}
}
else if (!schedule.compare(SCHEDULE_STATIC)) {
#	pragma omp parallel for num_threads(numOfThreads) schedule(static, chunkSize)
for (int i = 0; i < n; ++i)
{
for (int j = 0; j < m; ++j)
{
c[i][j] = 0;
for (int t = 0; t < k; ++t)
{
c[i][j] += a[i][t] * b[t][j];
}
}
}
}
else if (!schedule.compare(SCHEDULE_DYNAMIC)) {
#	pragma omp parallel for num_threads(numOfThreads) schedule(dynamic, chunkSize)
for (int i = 0; i < n; ++i)
{
for (int j = 0; j < m; ++j)
{
c[i][j] = 0;
for (int t = 0; t < k; ++t)
{
c[i][j] += a[i][t] * b[t][j];
}
}
}
}
else {
#	pragma omp parallel for num_threads(numOfThreads) schedule(guided, chunkSize)
for (int i = 0; i < n; ++i)
{
for (int j = 0; j < m; ++j)
{
c[i][j] = 0;
for (int t = 0; t < k; ++t)
{
c[i][j] += a[i][t] * b[t][j];
}
}
}
}

startTime = omp_get_wtime() - startTime;
return startTime;
}

void printMatrices(int** a, int** b, int** c, int n, int k, int m, bool computed)
{
printf("Size = %d x %d || %d x %d\n", n, k, k, m);

printf("A:\n");
for (int i = 0; i < n; ++i)
{
for (int j = 0; j < k; ++j)
printf("\t%d ", a[i][j]);
printf("\n");
}

printf("\nB:\n");
for (int i = 0; i < k; ++i)
{
for (int j = 0; j < m; ++j)
printf("\t%d ", b[i][j]);
printf("\n");
}

if (computed)
{
printf("\nC:\n");
for (int i = 0; i < n; ++i)
{
for (int j = 0; j < m; ++j)
printf("\t%d ", c[i][j]);
printf("\n");
}
}
else
printf("\nProduct is not computed!\n");

printf("\n#### Done ####\n\n");
}

void deleteAllIfNeeded(int**& a, int**& b, int**& c, int& n, int& k, int& m)
{
if (n != -1)
{
for (int i = 0; i < n; ++i)
{
delete[] a[i];
delete[] c[i];
a[i] = nullptr;
c[i] = nullptr;
}
for (int i = 0; i < k; ++i)
{
delete[] b[i];
b[i] = nullptr;
}
delete[] a;
delete[] b;
delete[] c;
a = nullptr;
b = nullptr;
c = nullptr;

n = -1;
k = -1;
m = -1;
}
}