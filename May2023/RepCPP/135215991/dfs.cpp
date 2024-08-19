#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <queue>
#include <sstream>
#include <omp.h>
#define NODES 67108863            


using namespace std;

typedef struct node{
bool visited;
int n;
struct node *left;
struct node *right;
struct node *parent;
}node;

node *tree = NULL;

void toArray(node **tree, int nodes){
*tree = (node*)malloc(nodes * sizeof(**tree));
}

int lvlNumber(int nodes){
return log2 (nodes + 1);
}

int lastLvlNodes(int levels){
return pow(2, levels - 1);
}

void makeEdges(int length, int hilos){
int limit = (length/2) - 1;
omp_set_num_threads(hilos);
#pragma omp parallel num_threads(hilos)
{
#pragma omp for
for(int x = 0; x <= limit; x++){
tree[x].left = &tree[(2 * x) + 1];
tree[x].right = &tree[(2 * x) + 2];
}
}
#pragma omp parallel num_threads(hilos)
{
#pragma omp for
for(int y = (limit + 1); y < NODES; y++){
tree[y].left = NULL;
tree[y].right = NULL;
}
}
}

void findParents(int hilos){
omp_set_num_threads(hilos);
#pragma omp parallel num_threads(hilos)
{
#pragma omp for
for(int x = 0; x < NODES; x++){
if(x == 0){
tree[x].parent = NULL;
}else{
int par = (x - 1)/2;
tree[x].parent = &tree[par];
}
}
}
}

void makeTree(int hilos){
toArray(&tree, NODES);
omp_set_num_threads(hilos);
#pragma omp parallel num_threads(hilos)
{
#pragma omp for
for(int x = 0; x < NODES; x++){
tree[x].n = x + 1;
tree[x].visited = false;
}
}
}

bool hasChildren(node n){
if(n.left != NULL && n.right != NULL){
return true;
}else{
return false;
}
}

void DFS(int element, int hilos){
bool found = false;
queue <int> path;
node *temp = &tree[0];
do{
if(temp->n != element){
path.push(temp->n);
if (hasChildren(*temp)){
if(temp->left->visited == false){
temp = temp->left;
}else if(temp->right->visited == false){
temp = temp->right;
}else{
temp->visited = true;
temp = temp->parent;
}
}else{
temp->visited = true;
temp = temp->parent;
}
}else{
found = true;
path.push(temp->n);
printf("Element Found!: %d \n", temp->n);
}
}while(found == false);

printf("\n");
}

int main(int argc, char **argv){
int to_find = 0;
int hilos = 0;
stringstream ss;
ss << argv[1];
ss >> hilos;
makeTree(hilos);

makeEdges(NODES, hilos);
findParents(hilos);

ss.clear();
ss << argv[2];
ss >> to_find;
DFS(to_find, hilos);
ss.clear();
free(tree);
return(0);
}
