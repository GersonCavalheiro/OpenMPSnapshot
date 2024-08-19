#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <getopt.h>
#include <algorithm>
#include <climits>
#include <iostream>

#include "adf.h"
#include "list.h"
#include <omp.h>

#define OPTIMAL

#define MAXINT  INT_MAX
#define MAXLONG LONG_MAX

using namespace std;

bool  print_results;
char *input_file_name;
int   num_threads;



class Settings
{
public:
char* dwarfName;
char* inputFile;
char* resultFile;

Settings(char *input_file);
~Settings();
void PrintStringSettings();

private:
static const int maxArraySize = 1<<12;
};


Settings::Settings(char *input_file)
{
dwarfName = new char[maxArraySize];
inputFile = new char[maxArraySize];
resultFile = new char[maxArraySize];

dwarfName[0] = inputFile[0] = resultFile[0] = '\0';

sprintf(dwarfName, "%s", "BranchAndBound");
sprintf(inputFile, "%s", input_file);
sprintf(resultFile, "result_omp.txt");
}

Settings::~Settings()
{
delete[] dwarfName;
delete[] inputFile;
delete[] resultFile;
}

void Settings::PrintStringSettings()
{
char* stringSettings = new char[maxArraySize];
stringSettings[0] = '\0';

sprintf(stringSettings, "%s%s", stringSettings, "Kernel settings summary: ");
sprintf(stringSettings, "%s%s", stringSettings, "\nDwarf name        : ");
sprintf(stringSettings, "%s%s", stringSettings, dwarfName);
sprintf(stringSettings, "%s%s", stringSettings, "\nInputFile         : ");
sprintf(stringSettings, "%s%s", stringSettings, inputFile);
sprintf(stringSettings, "%s%s", stringSettings, "\nResultFile        : ");
sprintf(stringSettings, "%s%s", stringSettings, resultFile);

printf("%s", stringSettings);

delete[] stringSettings;
}





class Configurator
{
public:
Configurator(char* name, char *input_file);
~Configurator();
int** GetData(int &cityCount);
void WriteSettings();
void Close(int &cityCount, int *cities);

private:
char *dwarfName;
Settings* settings;
FILE *profile;
static const int infinity = MAXINT;
void AddNextChar(int &value, int source);
static const int BUFFER_SIZE = 1024;
};


Configurator::Configurator(char* name, char *input_file)
{
dwarfName = name;
settings = new Settings(input_file);
}

Configurator::~Configurator()
{
dwarfName = NULL;
delete settings;
}

void Configurator :: WriteSettings()
{
settings -> PrintStringSettings();
}

void Configurator :: AddNextChar(int &value, int source) 	
{
value = value * 10 + ((int)source - (int)'0');			
}

int** Configurator :: GetData(int &cityCount)
{
FILE *content = fopen(settings -> inputFile, "rb");		
if(!content) {
printf("Input file couldn't be openned!\n");
exit(1);
}

int ch = 0;
ch = getc(content);
cityCount = 0;
while (feof(content) == 0 && ch != '\n')
{
if (!(ch == ' ' || ch == '\r'))
{
AddNextChar(cityCount, ch);
}
ch = getc(content);
}

int** distances = new int*[cityCount];
for (int i = 0; i < cityCount; i++)
{
int* row = new int[cityCount];
for (int j = 0; j < cityCount; j++)
{
row[j] = 0;								
}
distances[i] = row;							
}

for (int i = 0; i < cityCount; i++)             
{
int position = 0;                           
ch = getc(content);
while ((feof(content) == 0) && (ch != '\n'))
{
if (ch == ' ')                 			
{
position++;                         
}
else if (ch == '\r')
{
ch = getc(content);
continue;
}
else
{                              			
if (ch == '-')             			
{
distances[i][position] = infinity;
}
else
{
AddNextChar(distances[i][position], ch);
}
}
ch = getc(content);
}
}
for (int i = 0; i < cityCount; i++)             
{
distances[i][i] = infinity;
}
fclose(content);
return distances;
}

void Configurator :: Close(int &cityCount, int *cities)
{
if  (print_results == true) {
FILE *result = fopen(settings -> resultFile, "wb");
for (int i = 0; i < cityCount; i++)
fprintf(result, "Edge: (%i;%i)\r\n", i, cities[i]);

fclose(result);
}
}




typedef struct Cycle {
int StartNode;   
int Length;      
int FreeLength;  
bool operator<(const struct Cycle &b) const { return Length < b.Length; }
} Cycle;

class AssignmentProblem {
private:
static const int infinity = MAXINT;

int cityCount;                          
int** distances;                       	

int* constraintMap;						

int* exclusions;						
int exclusionsCount;
int exclusionsLength;

int childIndex;
AssignmentProblem *self;
AssignmentProblem *parent;
AssignmentProblem(AssignmentProblem *parent, int index);

int* toCities;               
int* fromCities;             
long lowerBound;             
int* patchedToCities;		
int* patchedFromCities;     

int* u;
int* v;

int* childFromCities;
int* childToCities;

std::vector<Cycle> cycles;
int minCycle;
int refCount;

bool IsAllowed(int row, int column);
bool IsAllowedConstraint(int value, int column);
void Include(int from, int to);
void Exclude(int from, int to);
void AddConstraints();
bool SolveFromScratch();
bool SolveIncremental();

int* pathOrigins;
int* reducedCosts;
int* labeledRow;
int* unlabeledColumn;
void Dump(int* a);
void AllocateAugmentingPathTemporaries();
void FreeAugmentingPathTemporaries();
void FindAugmentingPath(int row);

void ComputeCycles();

void Patch(int from, int to, int from2, int to2);
long Patch(int startCity, int startCity2);



public:
static int ID;
int myID;

AssignmentProblem(int cityCount, int** distances);
~AssignmentProblem();

long getLowerBound() const { return lowerBound; }
long getParentLowerBound() { return parent == NULL ? infinity : parent->lowerBound; }
int* GetCities();

long Solve();

int getCycleCount() const { return cycles.size(); }
int getChildCount() const { return cycles[minCycle].FreeLength; }

long Patch(long upperBound);

bool operator<(AssignmentProblem const &y);
void GenerateSubproblems(list<AssignmentProblem*> &problemList, list<AssignmentProblem*> &deleteList);
void Set_self(AssignmentProblem *ptr) { self = ptr; }
};


int AssignmentProblem::ID = 0;

AssignmentProblem::AssignmentProblem(int cityCount, int** distances) {
myID = ++ID;
this->distances = NULL;
this->constraintMap = NULL;
this->exclusions = NULL;
this->toCities = NULL;
this->fromCities = NULL;
this->patchedToCities = NULL;
this->patchedFromCities = NULL;
this->childFromCities = NULL;
this->childToCities = NULL;
this->u = NULL;
this->v = NULL;

this->cityCount = cityCount;
this->distances = distances;
this->exclusionsCount = 0;
this->lowerBound = infinity;

this->constraintMap = new int[cityCount];
for (int i = 0; i < cityCount; i++) {
constraintMap[i] = 0;
}
}

AssignmentProblem::AssignmentProblem(AssignmentProblem *parent, int index) {
myID = ++ID;
this->distances = NULL;
this->constraintMap = NULL;
this->exclusions = NULL;
this->toCities = NULL;
this->fromCities = NULL;
this->patchedToCities = NULL;
this->patchedFromCities = NULL;
this->childFromCities = NULL;
this->childToCities = NULL;
this->u = NULL;
this->v = NULL;

this->parent = parent;

this->childIndex = index;
this->lowerBound = infinity;
}

AssignmentProblem::~AssignmentProblem() {
self = NULL;

delete[] this->constraintMap;
delete[] this->exclusions;
delete[] this->toCities;
delete[] this->fromCities;
delete[] this->patchedToCities;
delete[] this->patchedFromCities;
delete[] this->childFromCities;
delete[] this->childToCities;
delete[] this->u;
delete[] this->v;
this->u = NULL;
this->v = NULL;
}

int* AssignmentProblem::GetCities() {
return patchedToCities == NULL ? toCities : patchedToCities;
}

long AssignmentProblem::Solve() {
if (lowerBound < infinity)
return lowerBound;
bool success;
if (parent == NULL)
success = SolveFromScratch();
else {
AddConstraints();
success = SolveIncremental();
}

if (!success)
return infinity;

this->lowerBound = 0;
for (int i = 0; i < cityCount; i++)
lowerBound += u[i] + v[i];

ComputeCycles();
return lowerBound;
}

bool AssignmentProblem::IsAllowed(int row, int column) {
int value = constraintMap[row];
return (row != column) && ((value == 0) ? (row != column) : IsAllowedConstraint(value, column));
}
bool AssignmentProblem::IsAllowedConstraint(int value, int column) {
if (value > 0)
return column == (value - 1);
else {
value = -1 - value;
while (exclusions[value] >= 0) {
if (exclusions[value] == column)
return false;
value++;
}
return true;
}
}

void AssignmentProblem::Include(int from, int to) {
this->constraintMap[from] = 1 + to;
}

void AssignmentProblem::Exclude(int from, int to) {
int index = this->constraintMap[from];
if (index == 0) {
this->constraintMap[from] = -1 - this->exclusionsCount;
exclusions[exclusionsCount++] = to;
exclusions[exclusionsCount++] = -1;
} else if (index < 0) {
for (int i = -1 - index; this->exclusions[i] != -1; i++) {
if (this->exclusions[i] == to) {
return;
}
}
int first = -1 - index;
memmove(this->exclusions + (first + 1), this->exclusions + first,
(this->exclusionsCount - first) * sizeof(int));
this->exclusions[first] = to;
for (int i = 0; i < this->cityCount; i++) {
if (i == from)
continue;
if (this->constraintMap[i] < index)
this->constraintMap[i]--;
}
}
}

bool AssignmentProblem::SolveFromScratch() {
delete[] this->toCities;
this->toCities = new int[cityCount];
delete[] this->fromCities;
this->fromCities = new int[cityCount];
delete[] this->u;
this->u = new int[cityCount];
delete[] this->v;
this->v = new int[cityCount];

int assignedCities = 0;
for (int i = 0; i < cityCount; i++) {
toCities[i] = -1;
fromCities[i] = -1;
u[i] = 0;
v[i] = 0;
}

for (int column = 0; column < cityCount; column++) {
int minDistance = infinity;
int minRow = -1;
for (int row = 0; row < cityCount; row++) {
if (!IsAllowed(row, column))
continue;
int distance = distances[row][column];
if (distance < minDistance || (distance == minDistance && toCities[row] < 0)) {
minRow = row;
minDistance = distance;
}
}
if (minRow < 0)
return false;

v[column] = minDistance;

if (toCities[minRow] < 0) {
assignedCities++;
fromCities[column] = minRow;
toCities[minRow] = column;
u[minRow] = 0;
}
}

for (int row = 0; row < cityCount; row++) {
if (toCities[row] >= 0)
continue;

int minReducedCost = infinity;
int minColumn = -1;
for (int column = 0; column < cityCount; column++) {
if (!IsAllowed(row, column))
continue;
int reducedCost = distances[row][column] - v[column];
if (reducedCost < minReducedCost
|| (reducedCost == minReducedCost && fromCities[column] < 0 && fromCities[minColumn] >= 0)) {
minReducedCost = reducedCost;
minColumn = column;
}
}

if (minColumn < 0)
return false;

u[row] = minReducedCost;

if (fromCities[minColumn] < 0) {
assignedCities++;
toCities[row] = minColumn;
fromCities[minColumn] = row;
}
}

if (assignedCities < cityCount) {
AllocateAugmentingPathTemporaries();
for (int row = 0; row < cityCount; row++) {
if (toCities[row] >= 0)
continue;
FindAugmentingPath(row);
}
FreeAugmentingPathTemporaries();
}
return true;
}

void AssignmentProblem::AddConstraints() {
this->cityCount = parent->cityCount;
this->distances = parent->distances;

delete[] this->constraintMap;
this->constraintMap = new int[this->cityCount];
memcpy(this->constraintMap, parent->constraintMap, this->cityCount * sizeof(int));

delete[] this->exclusions;
this->exclusions = new int[parent->exclusionsCount + 4];
this->exclusionsLength = parent->exclusionsCount + 4;
this->exclusionsCount = parent->exclusionsCount;
if (parent->exclusionsCount > 0) {
memcpy(this->exclusions, parent->exclusions, parent->exclusionsCount * sizeof(int));
}

for (int i = 0; i < this->childIndex; i++)
Include(parent->childFromCities[i], parent->childToCities[i]);
Exclude(parent->childFromCities[this->childIndex], parent->childToCities[this->childIndex]);
if (this->childIndex > 0) {
Exclude(parent->toCities[this->childIndex - 1], parent->fromCities[0]);
}
}

bool AssignmentProblem::SolveIncremental() {

delete[] this->toCities;
this->toCities = new int[cityCount];
memcpy(this->toCities, parent->toCities, cityCount * sizeof(int));
delete[] this->fromCities;
this->fromCities = new int[cityCount];
memcpy(this->fromCities, parent->fromCities, cityCount * sizeof(int));
delete[] this->u;
this->u = new int[cityCount];
memcpy(this->u, parent->u, cityCount * sizeof(int));
delete[] this->v;
this->v = new int[cityCount];
memcpy(this->v, parent->v, cityCount * sizeof(int));

int row = parent->childFromCities[this->childIndex];
int column = toCities[row];
fromCities[column] = -1;
toCities[row] = -1;

AllocateAugmentingPathTemporaries();
FindAugmentingPath(row);
FreeAugmentingPathTemporaries();
return true;
}

void AssignmentProblem::AllocateAugmentingPathTemporaries() {
pathOrigins = new int[cityCount];
reducedCosts = new int[cityCount];
labeledRow = new int[cityCount];
unlabeledColumn = new int[cityCount];
}
void AssignmentProblem::FreeAugmentingPathTemporaries() {
delete[] pathOrigins;
delete[] reducedCosts;
delete[] labeledRow;
delete[] unlabeledColumn;
}

void AssignmentProblem::FindAugmentingPath(int row) {
labeledRow[0] = row;
for (int column = 0; column < cityCount; column++) {
if (!IsAllowed(row, column))
reducedCosts[column] = infinity;
else
reducedCosts[column] = distances[row][column] - u[row] - v[column];
pathOrigins[column] = row;
unlabeledColumn[column] = column;
}
int unlabeledColumnCount = cityCount;
int labeledRowCount = 1;

for (; ; ) {
bool foundZero = false;
for (int l = 0; l < unlabeledColumnCount; l++) {
int column = unlabeledColumn[l];
if (reducedCosts[column] == 0) {
foundZero = true;

if (fromCities[column] < 0) {
do {
int origin = pathOrigins[column];
fromCities[column] = origin;
int hold = toCities[origin];
toCities[origin] = column;
column = hold;
} while (column >= 0);
return;
}

labeledRow[labeledRowCount++] = fromCities[column];
unlabeledColumn[l] = unlabeledColumn[--unlabeledColumnCount];
break;
}
}
if (foundZero) {
int r = labeledRow[labeledRowCount - 1];
for (int l = 0; l < unlabeledColumnCount; l++) {
int column = unlabeledColumn[l];
if (!IsAllowed(r, column))
continue;
int reducedCost = distances[r][column] - u[r] - v[column];
if (reducedCost < reducedCosts[column]) {
reducedCosts[column] = reducedCost;
pathOrigins[column] = r;
}
}
} else {
int minReducedCost = infinity;
for (int l = 0; l < unlabeledColumnCount; l++) {
int column = unlabeledColumn[l];
if (reducedCosts[column] < minReducedCost)
minReducedCost = reducedCosts[column];
}
for (int l = 0; l < labeledRowCount; l++) {
int r = labeledRow[l];
u[r] += minReducedCost;
}
for (int column = 0; column < cityCount; column++) {
if (reducedCosts[column] == 0)
v[column] -= minReducedCost;
else
reducedCosts[column] -= minReducedCost;
}
}

}
}

void AssignmentProblem::ComputeCycles() {
bool* visited = new bool[cityCount];
for (int i = 0; i < cityCount; i++)
visited[i] = false;
this->cycles.clear();
this->cycles = std::vector<Cycle>();
int cycleCount = 0;
for (int i = 0; i < cityCount; i++) {
if (visited[i])
continue;
cycleCount++;
visited[i] = true;
int next = toCities[i];
int length = 1;
int freeLength = constraintMap[i] > 0 ? 0 : 1;
while (next != i) {
visited[next] = true;
length++;
if (constraintMap[next] <= 0)
freeLength++;
next = toCities[next];
}
Cycle cycle;
cycle.Length = length;
cycle.FreeLength = freeLength;
cycle.StartNode = i;
this->cycles.push_back(cycle);
}

delete [] visited;

sort(this->cycles.begin(), this->cycles.end());

int minFreeLength = MAXINT;
for (unsigned int i = 0; i < cycles.size(); i++)
if (cycles[i].FreeLength < minFreeLength) {
minCycle = i;
minFreeLength = cycles[i].FreeLength;
}
}

void AssignmentProblem::Patch(int from, int to, int from2, int to2) {
patchedToCities[from] = to2;
patchedFromCities[to2] = from;
patchedToCities[from2] = to;
patchedFromCities[to] = from2;
}

long AssignmentProblem::Patch(int startCity, int startCity2) {
int bestLast = -1;
int bestLast2 = -1;
int last = startCity;
int next = toCities[last];
long minPenalty = MAXLONG;
while (next != startCity) {

long basePenalty = -distances[last][next];
int last2 = startCity2;
int next2 = toCities[last2];
while (next2 != startCity2) {
if (IsAllowed(last, next2) && IsAllowed(last2, next)) {
long penalty = basePenalty - distances[last2][next2]
+ distances[last][next2] + distances[last2][next];
if (penalty < minPenalty) {
if (penalty == 0) {
Patch(last, next, last2, next2);
return 0;
}
minPenalty = penalty;
bestLast = last;
bestLast2 = last2;
}
}
last2 = next2;
next2 = toCities[last2];
}
last = next;
next = toCities[last];
}

if (bestLast < 0 || bestLast2 < 0)
return infinity;
Patch(bestLast, patchedToCities[bestLast], bestLast2, patchedToCities[bestLast2]);
return minPenalty;
}

long AssignmentProblem::Patch(long upperBound) {
delete[] patchedToCities;
patchedToCities = new int[cityCount];
memcpy(patchedToCities, toCities, cityCount * sizeof(int));
delete[] patchedFromCities;
patchedFromCities = new int[cityCount];
memcpy(patchedFromCities, fromCities, cityCount * sizeof(int));

long penalty = 0;
long maxPenalty = upperBound - this->lowerBound;
for (size_t cycle = cycles.size() - 1; cycle > 0; --cycle) {
int startCity = cycles[cycle].StartNode;
int startCity2 = cycles[cycle - 1].StartNode;
penalty += Patch(startCity, startCity2);
if (penalty >= maxPenalty)
break;
}

return penalty;
}

bool AssignmentProblem::operator<(AssignmentProblem const &y) {
if (getLowerBound() < y.getLowerBound())
return true;
if (getLowerBound() > y.getLowerBound())
return false;
if (getChildCount() < y.getChildCount())
return true;
if (getChildCount() > y.getChildCount())
return false;
return false;
}

void AssignmentProblem::GenerateSubproblems(list<AssignmentProblem*>  &problemList, list<AssignmentProblem*> &deleteList)
{
if (this->cycles.size() == 0)
return;

int minLength = cycles[minCycle].FreeLength;
delete[] childFromCities;
childFromCities = new int[minLength];
delete[] childToCities;
childToCities = new int[minLength];

int last = cycles[minCycle].StartNode;
int next;
int index = 0;
do {
next = toCities[last];
if (constraintMap[last] <= 0) {
childFromCities[index] = last;
childToCities[index++] = next;
}
last = next;
} while (last != cycles[minCycle].StartNode);

for (int i = 0; i < minLength; i++) {
AssignmentProblem *newProblem = new AssignmentProblem(this->self, i);
deleteList.push_back(newProblem);
newProblem->Set_self(newProblem);
problemList.push_back(newProblem);
}
}






class Solver
{
public:
Solver(Configurator *configurator);
~Solver();
void Solve();
void PrintSummery();
void Finish();

private:
static const int infinity = MAXINT;

Configurator				 *dwarfConfigurator;

int 						  cityCount;			
int 						**distances;			

long 			  			  upperBound;

AssignmentProblem   		  *best;
list<AssignmentProblem*>      problemList;
list<AssignmentProblem*> 	  solvedList;
list<AssignmentProblem*>      deleteList;


void ArrayDestroyer(int **arrayForDel, int dimension);
TRANSACTION_SAFE void AddSolvedProblem(AssignmentProblem *problem);
TRANSACTION_SAFE AssignmentProblem* NextSolvedProblem();
};



Solver :: Solver(Configurator *configurator)
{
dwarfConfigurator = configurator;					
dwarfConfigurator -> WriteSettings();
distances = dwarfConfigurator -> GetData(cityCount);

best = new AssignmentProblem(cityCount, distances);
best->Set_self(best);
deleteList.push_back(best);

}

Solver :: ~Solver()
{
AssignmentProblem *problem;
while (!deleteList.empty())
{
problem = deleteList.front();
deleteList.pop_front();
delete problem;
}
}

void Solver :: ArrayDestroyer(int **arrayForDel, int dimension)
{
for (int i = 0; i < dimension; i++)
delete[] arrayForDel[i];
delete[] arrayForDel;
}



void Solver::AddSolvedProblem(AssignmentProblem *problem) {
list<AssignmentProblem*>::iterator it = solvedList.begin();
while (it != solvedList.end()) {
if (*(*it) < *problem)
break;
it++;
}
solvedList.insert(it, problem);
}


AssignmentProblem* Solver::NextSolvedProblem() {
while (!solvedList.empty() && solvedList.front()->getLowerBound() >= upperBound) {
solvedList.pop_front();
}
if (!solvedList.empty()) {
AssignmentProblem *problem = solvedList.back();
solvedList.pop_back();
return problem;
} else {
return NULL;
}
}

void Solver::Solve() {
bool done = false;

best->Solve();
if (best->getCycleCount() == 1)
upperBound = best->getLowerBound();
else {
upperBound = best->getLowerBound() + best->Patch(infinity);
best->GenerateSubproblems(problemList, deleteList);
}

#pragma omp parallel num_threads(num_threads)
{
while (!done) {

#pragma omp single
{
while (problemList.size() > 0) {
AssignmentProblem *problem = problemList.front();
problemList.pop_front();

#pragma omp task firstprivate(problem)
{
STAT_COUNT_EVENT(task_exec, omp_get_thread_num());

AssignmentProblem *problem_tx = problem;

#ifndef OPTIMAL
TRANSACTION_BEGIN(10,RW)
#endif
if (problem_tx->getParentLowerBound() < upperBound) {
long lowerBound = problem_tx->Solve();
if (lowerBound < upperBound) {
if (problem_tx->getCycleCount() == 1) {
#ifdef OPTIMAL
TRANSACTION_BEGIN(11,RW)
#endif
best = problem_tx;
upperBound = lowerBound;
#ifdef OPTIMAL
TRANSACTION_END
#endif
}
else {
long penalty = problem_tx->Patch(upperBound);
long newUpperBound = penalty == infinity ? infinity : lowerBound + penalty;
if (newUpperBound < upperBound) {
#ifdef OPTIMAL
TRANSACTION_BEGIN(12,RW)
#endif
best = problem_tx;
upperBound = newUpperBound;
#ifdef OPTIMAL
TRANSACTION_END
#endif
}
#ifdef OPTIMAL
TRANSACTION_BEGIN(13,RW)
#endif
AddSolvedProblem(problem_tx);
#ifdef OPTIMAL
TRANSACTION_END
#endif
}
}
}
#ifndef OPTIMAL
TRANSACTION_END
#endif
} 


} 
} 

#pragma omp single
{
if (!solvedList.empty()) {
AssignmentProblem *problem = NextSolvedProblem();
if (problem != NULL) {
if (problem->getLowerBound() < upperBound) {
problem->GenerateSubproblems(problemList, deleteList);
}
}
} else
done = true;
}
} 
} 
}


void Solver::PrintSummery()
{
printf("\nTotal Distance    : %ld\r\n", this->upperBound);
}


void Solver::Finish() {
dwarfConfigurator->Close(this->cityCount, best->GetCities());
ArrayDestroyer(distances, cityCount);
}





void ParseCommandLine(int argc, char **argv)
{
char c;

input_file_name = (char*) "smallcontent.txt";
num_threads = 1;
print_results = false;

while ((c = getopt (argc, argv, "hf:n:r")) != -1)
switch (c) {
case 'h':
printf("\nBranch and Bound - ADF benchmark application\n"
"\n"
"Usage:\n"
"   branch_bound_omp [options ...]\n"
"\n"
"Options:\n"
"   -h\n"
"        Print this help message.\n"
"   -f <filename>\n"
"        Input file name.\n"
"   -n <long>\n"
"        Number of worker threads. (default 1)\n"
"   -r \n"
"        print results to a result file\n"
"\n"
);
exit (0);
case 'f':
input_file_name = optarg;
break;
case 'n':
num_threads = strtol(optarg, NULL, 10);
break;
case 'r':
print_results = true;
break;
case '?':
if (optopt == 'c')
fprintf (stderr, "Option -%c requires an argument.\n", optopt);
else if (isprint (optopt))
fprintf (stderr, "Unknown option `-%c'.\n", optopt);
else
fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
exit(1);
default:
exit(1);
}

if(input_file_name == NULL) {
printf("missing input file!\n");
exit(1);
}

printf ("\nStarting openmp Branch and Bound.\n");
printf ("Running with %d threads. Input file %s\n", num_threads, input_file_name);
printf ("=====================================================\n\n");
}






int main(int argc, char** argv)
{
ParseCommandLine(argc, argv);

Configurator dwarfConfigurator((char *) "BranchAndBound, unmanaged serial kernel", input_file_name);
Solver solver(&dwarfConfigurator);					

NonADF_init(num_threads);

omp_set_num_threads(num_threads);
omp_set_nested(1);

solver.Solve();										

solver.PrintSummery();

NonADF_terminate();

solver.Finish();									

return 0;
}

