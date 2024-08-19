#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <vector>
#include <omp.h>
#include <string>
#include <ctime>
#include <cstdlib>

using namespace std;

class Graph 
{
private:
class Node 
{
public:
int value;
vector<int> children;
bool seen;

omp_lock_t lock;
};

const int maxNumOfNodes;
vector<Node> nodes;
double sigmaTime;


public:
Graph(int);
~Graph();

inline void addNode(int);
inline void addChild(int, int);
inline void fitNode(int);
inline void fitGraph();
inline void setSeenFalse();
inline int getNumberOfNodes();
inline int getCapacity();
inline int getValue(int);
inline int getNumOfChildren(int);
inline int getChildIndex(int, int);
inline void printGraph();

inline vector<int> traverseSequential(); 
inline vector<int> traverseParallel(); 

inline double getSigmaTime();
inline void setTimerToZero();
};

const string QUERY_PRINT = "print";
const string QUERY_GET_CHUNK_SIZE = "get_chunksize";
const string QUERY_SET_CHUNK_SIZE = "set_chunksize";
const string QUERY_GET_NUM_OF_THREADS = "get_threads";
const string QUERY_SET_NUM_OF_THREADS = "set_threads";
const string QUERY_TRAVERSE_SEQ = "traverse_seq";
const string QUERY_TRAVERSE_PAR = "traverse_par";
const string QUERY_TRAVERSE_SEQ_N_TIMES = "traverse_seq_n_times";
const string QUERY_TRAVERSE_PAR_N_TIMES = "traverse_par_n_times";
const string QUERY_GET_GRAPH_FROM_TERMINAL = "init_graph_by_terminal";
const string QUERY_GET_GRAPH_FROM_FILE = "init_graph_from_file";
const string QUERY_BULID_RANDOM_GRAPH = "build_and_save_random_graph";
const string QUERY_HELP = "help";
const string QUERY_CLEAR = "clear";
const string QUERY_EXIT = "exit";

const int MAX_NUM_OF_THREADS = 100;
const int MIN_NUM_OF_THREADS = 1;
const int MAX_CHUNK_SIZE = 100000;
const int MIN_CHUNK_SIZE = 1;
const int MIN_TIMES = 1;
const int MIN_NODES = 1;
const int MAX_NODES = 20000;

int chunkSize = 1;
int numOfThreads = 8;


Graph* graph = nullptr;

void getGraph(); 
void readGraph(); 
void run(); 
void printHeader();
void clearScreen();
void bulidRandomGraph(int); 


int main()
{
run();
}

void getGraph()
{
if (graph != nullptr)
{
delete graph;
graph = nullptr;
}

int graphSize = -1;
while (graphSize <= 0) {
printf("Enter the number of nodes(it should be a positive non_zero integer): ");
scanf("%d", &graphSize);
}
graph = new Graph(graphSize);

for (register int i = 0; i < graphSize; ++i) {
bool inRange = false;
int numOfNodeChildren, value;
while (!inRange) {
inRange = true;
printf("Enter the \"value\" and the \"number of children\" for node number %d: ", i);
scanf("%d%d", &value, &numOfNodeChildren);

if (numOfNodeChildren > graphSize) {
printf("\nError: The number of children can not be more than the number of nodes in the graph!\n\n");
inRange = false;
}
else if (numOfNodeChildren < 0) {
printf("\nError: The number of children can not be less than 0!\n\n");
inRange = false;
}
}

graph->addNode(value);
for (register int j = 0; j < numOfNodeChildren; ++j) {
bool acceptable = false;
int childIndex;
while (!acceptable) {
acceptable = true;
printf("Enter index of child number %d for node number %d: ", j, i);
scanf("%d", &childIndex);

if (childIndex < 0 || childIndex >= graphSize) {
printf("\nError: Node with index %d does not exists.\n\n", childIndex);
acceptable = false;
continue;
}
for (int k = 0; k < j; ++k) {
if (graph->getChildIndex(i, k) == childIndex) {
printf("\nError: Node number %d is already a child of node number %d. choose another node.\n\n", graph->getChildIndex(i, k), i);
acceptable = false;
break;
}
}
}
graph->addChild(i, childIndex);
}
graph->fitNode(i);
printf("__________________________________\n");
}

graph->fitGraph();
}

void readGraph()
{
if (graph != nullptr)
{
delete graph;
graph = nullptr;
}

int graphSize = -1;
char path[200] = "";
scanf("%s", path);

FILE* fp = fopen(path, "r");

fscanf(fp, "%d", &graphSize);
graph = new Graph(graphSize);

for (register int i = 0; i < graphSize; ++i) {
bool inRange = false;
int numOfNodeChildren, value;
fscanf(fp, "%d%d", &value, &numOfNodeChildren);

if (numOfNodeChildren > graphSize || numOfNodeChildren < 0) {
numOfNodeChildren = 0;
}

graph->addNode(value);
for (register int j = 0; j < numOfNodeChildren; ++j) {
int childIndex;
fscanf(fp, "%d", &childIndex);

if (childIndex < 0 || childIndex >= graphSize) {
continue;
}
graph->addChild(i, childIndex);
}
graph->fitNode(i);
}

graph->fitGraph();
fclose(fp);
}

void bulidRandomGraph(int n) 
{
if (graph != nullptr)
{
delete graph;
graph = nullptr;
}

char path[200] = "";
scanf("%s", path);

srand(time(0));

FILE* fp = fopen(path, "w+");
graph = new Graph(n);

fprintf(fp, "%d ", n);
for (int i = 0; i < n; ++i)
{
int val = rand();
fprintf(fp, "%d ");
graph->addNode(val);

vector<int> children;
for (int j = 0; j < n; ++j)
{	
if (rand() % 2)
{
children.push_back(j);
graph->addChild(i, j);
}
}
graph->fitNode(i);

fprintf(fp, "%d ", children.size());
for (int j = 0; j < children.size(); ++j)
{
fprintf(fp, "%d ", children[j]);
}
}
graph->fitGraph();
fclose(fp);
}

void printHeader()
{
printf("\n_______________HELP_______________\n");

printf("\n\t* To input the graph from terminal enter: \"%s\" without the quotations.\n"
, QUERY_GET_GRAPH_FROM_TERMINAL.c_str());

printf("\n\t* To read the graph from a file with path <PATH> enter: \"%s <PATH>\" without the quotations.\n"
, QUERY_GET_GRAPH_FROM_FILE.c_str());

printf("\n\t* To bulid and save a random graph with <N> nodes in <PATH> enter: \"%s <N> <PATH>\" without the quotations.\n"
, QUERY_BULID_RANDOM_GRAPH.c_str());

printf("\n\t* To get the chunkSize used for scheduling parallel traverse enter: \"%s\" without the quotations.\n"
, QUERY_GET_CHUNK_SIZE.c_str());

printf("\n\t* To get the number of threads used for parallel traverse enter: \"%s\" without the quotations.\n"
, QUERY_GET_NUM_OF_THREADS.c_str());

printf("\n\t* To set the chunkSize used for scheduling parallel traverse enter: \"%s <chunkSize>\" without the quotations.\n"
, QUERY_SET_CHUNK_SIZE.c_str());

printf("\n\t* To set the number of threads used for parallel traverse enter: \"%s <numberOfThreads>\" without the quotations.\n"
, QUERY_SET_NUM_OF_THREADS.c_str());

printf("\n\t* To print the graph enter: \"%s\" without the quotations.\n", QUERY_PRINT.c_str());

printf("\n\t* To traverse the whole graph sequentially starting from root enter: \"%s\" without the quotations.\n"
, QUERY_TRAVERSE_SEQ.c_str());

printf("\n\t* To traverse the whole graph starting from root in parallel enter: \"%s\" without the quotations.\n"
, QUERY_TRAVERSE_PAR.c_str());

printf("\n\t* To traverse the whole graph <N> times sequentially starting from root enter: \"%s <N>\" without the quotations.\n"
, QUERY_TRAVERSE_SEQ_N_TIMES.c_str());

printf("\n\t* To traverse the whole graph <N> times starting from root in parallel enter: \"%s <N>\" without the quotations.\n"
, QUERY_TRAVERSE_PAR_N_TIMES.c_str());

printf("\n\t* To print these information again enter: \"%s\" without the quotations.\n", QUERY_HELP.c_str());

printf("\n\t* To clear the screen enter: \"%s\" without the quotations.\n", QUERY_CLEAR.c_str());

printf("\n\t* To exit enter: \"%s\" without the quotations.\n", QUERY_EXIT.c_str());

printf("\n");

printf("\n+ You should first use \"%s\" or \"%s\" commands to initialize a graph.\n"
, QUERY_GET_GRAPH_FROM_TERMINAL.c_str(), QUERY_GET_GRAPH_FROM_FILE.c_str());
printf("\n+ While using \"%s\", or \"%s\", or \"%s\" queries, <PATH> should not contain any whitespace.\n"
, QUERY_GET_GRAPH_FROM_TERMINAL.c_str(), QUERY_GET_GRAPH_FROM_FILE.c_str(), QUERY_BULID_RANDOM_GRAPH.c_str());
printf("\n+ \"%s\" will save and initialize the graph.\n", QUERY_BULID_RANDOM_GRAPH.c_str());
printf("\n+ When the graph is too large don't use \"%s\", \"%s\" and \"%s\" queries.\n"
, QUERY_PRINT.c_str(), QUERY_TRAVERSE_SEQ.c_str(), QUERY_TRAVERSE_PAR.c_str());
printf("\n+ Traverse algorithms are DFS implementations.\n");
}

void clearScreen() {
system("cls");
}

void run()
{
char c_query[100] = "";
string query(c_query);
printHeader();

while (query.compare(QUERY_EXIT))
{
printf("\nEnter any command(to see all possible commands type help): ");

scanf("%s", c_query);
query.assign(c_query);

if (!query.compare(QUERY_GET_GRAPH_FROM_TERMINAL))
{
getGraph();
}
else if (!query.compare(QUERY_GET_GRAPH_FROM_FILE))
{
readGraph();
}
else if (!query.compare(QUERY_BULID_RANDOM_GRAPH))
{
int n;
scanf("%d", &n);
if (n < MIN_NODES || n > MAX_NODES)
{
printf("\nError: number of nodes should be at least %d and at most %d!\n\n", MIN_NODES, MAX_NODES);
continue;
}

bulidRandomGraph(n);
}
else if (!query.compare(QUERY_GET_CHUNK_SIZE))
{
printf("\nchunkSize = %d\n", chunkSize);
}
else if (!query.compare(QUERY_GET_NUM_OF_THREADS))
{
printf("\nNumber of threads = %d\n", numOfThreads);
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
int n;
scanf("%d", &n);
if (n < MIN_NUM_OF_THREADS || n > MAX_NUM_OF_THREADS)
printf("\nError: The number of threads should be at least %d and at most %d!\n\n"
, MIN_NUM_OF_THREADS, MAX_NUM_OF_THREADS);
else
numOfThreads = n;
}
else if (!query.compare(QUERY_PRINT))
{
if (graph == nullptr) 
{
printf("\nError: Graph should be initialized!\n\n");
continue;
}

graph->printGraph();
}
else if (!query.compare(QUERY_TRAVERSE_SEQ))
{
if (graph == nullptr)
{
printf("\nError: Graph should be initialized!\n\n");
continue;
}

vector<int> indexes = graph->traverseSequential();
printf("\n________Sequential Traverse________\n\n");
for (int index : indexes)
{
printf("%d ", index);
}
printf("\nthis methode took %f seconds to compute.\n", graph->getSigmaTime());
graph->setTimerToZero();
}
else if (!query.compare(QUERY_TRAVERSE_PAR))
{
if (graph == nullptr)
{
printf("\nError: Graph should be initialized!\n\n");
continue;
}

vector<int> indexes = graph->traverseParallel();
printf("\n________Parallel Traverse________\n\n");
for (int index : indexes)
{
printf("%d ", index);
}
printf("\nthis methode took %f seconds to compute.\n", graph->getSigmaTime());
graph->setTimerToZero();
}
else if (!query.compare(QUERY_TRAVERSE_SEQ_N_TIMES))
{
if (graph == nullptr)
{
printf("\nError: Graph should be initialized!\n\n");
continue;
}

int n;
scanf("%d", &n);
if (n < MIN_TIMES)
{
printf("\nError: N should be at least %d!\n\n", MIN_TIMES);
continue;
}

for (int i = 0; i < n; ++i)
{
graph->traverseSequential();
}

printf("\nthis methode took %f seconds in average to compute.\n", graph->getSigmaTime() / n);
graph->setTimerToZero();
}
else if (!query.compare(QUERY_TRAVERSE_PAR_N_TIMES))
{
if (graph == nullptr)
{
printf("\nError: Graph should be initialized!\n\n");
continue;
}

int n;
scanf("%d", &n);
if (n < MIN_TIMES)
{
printf("\nError: N should be at least %d!\n\n", MIN_TIMES);
continue;
}

for (int i = 0; i < n; ++i)
{
graph->traverseParallel();
}

printf("\nthis methode took %f seconds in average to compute.\n", graph->getSigmaTime() / n);
graph->setTimerToZero();
}
else if (!query.compare(QUERY_CLEAR)) 
{
clearScreen();
}
else if (!query.compare(QUERY_HELP))
{
printHeader();
}
else if (query.compare(QUERY_EXIT)) 
{
printf("\nError: Command not defined!\n\n");
}
}

if (graph != nullptr)
{
delete graph;
graph = nullptr;
}
}

Graph::Graph(int maxNumOfNodes) :maxNumOfNodes(maxNumOfNodes)
{
nodes.reserve(maxNumOfNodes);
sigmaTime = 0.0;
}

Graph::~Graph()
{
for (int i = 0; i < maxNumOfNodes; ++i)
omp_destroy_lock(&(nodes[i].lock));
}

inline void Graph::addNode(int value)
{
if (nodes.size() < maxNumOfNodes)
{
Node newNode;
newNode.children.reserve(maxNumOfNodes);
newNode.value = value;
newNode.seen = false;
omp_init_lock(&(newNode.lock));
nodes.push_back(newNode);
}
}

inline void Graph::addChild(int index, int child)
{
if (index >= 0 && index < nodes.size() && nodes[index].children.size() < maxNumOfNodes)
{

nodes[index].children.push_back(child);
}
}

inline void Graph::fitNode(int index)
{
nodes[index].children.shrink_to_fit();
}

inline void Graph::fitGraph()
{
nodes.shrink_to_fit();
}

inline void Graph::setSeenFalse()
{
for (int i = 0; i < maxNumOfNodes; ++i)
{
nodes[i].seen = false;
}
}

inline int Graph::getNumberOfNodes()
{
return nodes.size();
}

inline int Graph::getCapacity()
{
return maxNumOfNodes;
}

inline int Graph::getValue(int index)
{
return nodes[index].value;
}

inline int Graph::getNumOfChildren(int index)
{
return nodes[index].children.size();
}

inline int Graph::getChildIndex(int nodeIndex, int child)
{
return nodes[nodeIndex].children[child];
}

inline void Graph::printGraph()
{
printf("\n__________Graph__________\n\n");
for (int i = 0; i < nodes.size(); ++i)
{
printf("Value of the %dth node is %d, and it has %d children that are listed below:\n\t"
, i, nodes[i].value, nodes[i].children.size());
for (int j = 0; j < nodes[i].children.size(); ++j)
{
printf("%d ", nodes[i].children[j]);
}
printf("\n\n");
}
}

inline double Graph::getSigmaTime()
{
return sigmaTime;
}

inline void Graph::setTimerToZero()
{
sigmaTime = 0.0;
}

inline vector<int> Graph::traverseSequential()
{
double startTime;
startTime = omp_get_wtime();

register int n = maxNumOfNodes, lastNodeChecked = 1;
vector<int> stack;
stack.reserve(n);

stack.push_back(0);
nodes[0].seen = true;

vector<int> res;

while (!stack.empty())
{
register int index = stack.back();
Node& node = nodes[index];
stack.pop_back();
res.push_back(index);

register int numOfChildren = node.children.size();

for (register int i = 0; i < numOfChildren; ++i)
{
int childIndex = node.children[i];

if (!nodes[childIndex].seen) {
stack.push_back(childIndex);
nodes[childIndex].seen = true;
}
}

if (stack.empty()) {
for (; lastNodeChecked < n; ++lastNodeChecked) {
if (!nodes[lastNodeChecked].seen) {
stack.push_back(lastNodeChecked);
nodes[lastNodeChecked].seen = true;
break;
}
}
}
}

sigmaTime += (omp_get_wtime() - startTime);
setSeenFalse(); 

return res;
}

inline vector<int> Graph::traverseParallel()
{
double startTime;
startTime = omp_get_wtime();

register int n = maxNumOfNodes;
vector<int> stack;
stack.reserve(n);

vector<int> res;
omp_lock_t resultLock;
omp_init_lock(&resultLock);

#	pragma omp parallel for num_threads(numOfThreads) private(stack) schedule(dynamic, chunkSize)
for (register int i = 0; i < n; ++i)
{
omp_set_lock(&(nodes[i].lock));
if (!nodes[i].seen)
{
stack.push_back(i);
nodes[i].seen = true;
}
omp_unset_lock(&(nodes[i].lock));

while (!stack.empty())
{
register int index = stack.back();
Node& node = nodes[index];
stack.pop_back();

omp_set_lock(&resultLock);
res.push_back(index);
omp_unset_lock(&resultLock);

register int numOfChildren = node.children.size();

for (register int j = 0; j < numOfChildren; ++j)
{
int childIndex = node.children[j];

omp_set_lock(&(nodes[childIndex].lock));
if (!nodes[childIndex].seen) {
stack.push_back(childIndex);
nodes[childIndex].seen = true;
}
omp_unset_lock(&(nodes[childIndex].lock));
}
}
}

omp_destroy_lock(&resultLock);
sigmaTime += omp_get_wtime() - startTime;
setSeenFalse(); 

return res;
}









