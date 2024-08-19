#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include "app-desc.h"
#include "bots.h"
#include "uts.h"
unsigned long long nLeaves = 0;
int maxTreeDepth = 0;
double b_0   = 4.0; 
int   rootId = 0;   
int    nonLeafBF   = 4;            
double nonLeafProb = 15.0 / 64.0;  
int computeGranularity = 1;
unsigned long long  exp_tree_size = 0;
int        exp_tree_depth = 0;
unsigned long long  exp_num_leaves = 0;
double rng_toProb(int n)
{
if (n < 0) {
printf("*** toProb: rand n = %d out of range\n",n);
}
return ((n<0)? 0.0 : ((double) n)/2147483648.0);
}
void uts_initRoot(Node * root)
{
root->height = 0;
root->numChildren = -1;      
rng_init(root->state.state, rootId);
bots_message("Root node at %p\n", root);
}
int uts_numChildren_bin(Node * parent)
{
int    v = rng_rand(parent->state.state);	
double d = rng_toProb(v);
return (d < nonLeafProb) ? nonLeafBF : 0;
}
int uts_numChildren(Node *parent)
{
int numChildren = 0;
if (parent->height == 0) numChildren = (int) floor(b_0);
else numChildren = uts_numChildren_bin(parent);
if (parent->height == 0) {
int rootBF = (int) ceil(b_0);
if (numChildren > rootBF) {
bots_debug("*** Number of children of root truncated from %d to %d\n", numChildren, rootBF);
numChildren = rootBF;
}
}
else {
if (numChildren > MAXNUMCHILDREN) {
bots_debug("*** Number of children truncated from %d to %d\n", numChildren, MAXNUMCHILDREN);
numChildren = MAXNUMCHILDREN;
}
}
return numChildren;
}
unsigned long long parallel_uts ( Node *root )
{
unsigned long long num_nodes = 0 ;
root->numChildren = uts_numChildren(root);
bots_message("Computing Unbalance Tree Search algorithm ");
#pragma omp parallel  
#pragma omp single nowait
#pragma omp task untied
num_nodes = parTreeSearch( 0, root, root->numChildren );
bots_message(" completed!");
return num_nodes;
}
unsigned long long parTreeSearch(int depth, Node *parent, int numChildren) 
{
Node n[numChildren], *nodePtr;
int i, j;
unsigned long long subtreesize = 1, partialCount[numChildren];
for (i = 0; i < numChildren; i++) {
nodePtr = &n[i];
nodePtr->height = parent->height + 1;
for (j = 0; j < computeGranularity; j++) {
rng_spawn(parent->state.state, nodePtr->state.state, i);
}
nodePtr->numChildren = uts_numChildren(nodePtr);
#pragma omp task untied firstprivate(i, nodePtr) shared(partialCount)
partialCount[i] = parTreeSearch(depth+1, nodePtr, nodePtr->numChildren);
}
#pragma omp taskwait
for (i = 0; i < numChildren; i++) {
subtreesize += partialCount[i];
}
return subtreesize;
}
void uts_read_file ( char *filename )
{
FILE *fin;
if ((fin = fopen(filename, "r")) == NULL) {
bots_message("Could not open input file (%s)\n", filename);
exit (-1);
}
fscanf(fin,"%lf %lf %d %d %d %llu %d %llu",
&b_0,
&nonLeafProb,
&nonLeafBF,
&rootId,
&computeGranularity,
&exp_tree_size,
&exp_tree_depth,
&exp_num_leaves
);
fclose(fin);
computeGranularity = max(1,computeGranularity);
bots_message("\n");
bots_message("Root branching factor                = %f\n", b_0);
bots_message("Root seed (0 <= 2^31)                = %d\n", rootId);
bots_message("Probability of non-leaf node         = %f\n", nonLeafProb);
bots_message("Number of children for non-leaf node = %d\n", nonLeafBF);
bots_message("E(n)                                 = %f\n", (double) ( nonLeafProb * nonLeafBF ) );
bots_message("E(s)                                 = %f\n", (double) ( 1.0 / (1.0 - nonLeafProb * nonLeafBF) ) );
bots_message("Compute granularity                  = %d\n", computeGranularity);
bots_message("Random number generator              = "); rng_showtype();
}
void uts_show_stats( void )
{
int nPes = atoi(bots_resources);
int chunkSize = 0;
bots_message("\n");
bots_message("Tree size                            = %llu\n", (unsigned long long)  bots_number_of_tasks );
bots_message("Maximum tree depth                   = %d\n", maxTreeDepth );
bots_message("Chunk size                           = %d\n", chunkSize );
bots_message("Number of leaves                     = %llu (%.2f%%)\n", nLeaves, nLeaves/(float)bots_number_of_tasks*100.0 ); 
bots_message("Number of PE's                       = %.4d threads\n", nPes );
bots_message("Wallclock time                       = %.3f sec\n", bots_time_program );
bots_message("Overall performance                  = %.0f nodes/sec\n", (bots_number_of_tasks / bots_time_program) );
bots_message("Performance per PE                   = %.0f nodes/sec\n", (bots_number_of_tasks / bots_time_program / nPes) );
}
int uts_check_result ( void )
{
int answer = BOTS_RESULT_SUCCESSFUL;
if ( bots_number_of_tasks != exp_tree_size ) {
answer = BOTS_RESULT_UNSUCCESSFUL;
bots_message("Incorrect tree size result (%llu instead of %llu).\n", bots_number_of_tasks, exp_tree_size);
}
return answer;
}
