#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
struct tree
{
float d;
struct tree* left;
struct tree* right;
};
struct tree* newNode()
{
struct tree* node = (struct tree*)malloc(sizeof(struct tree));
node->d = (float) rand()/RAND_MAX;
node->left = NULL;
node->right = NULL;
return(node);
}
struct tree* seqaddNode(struct tree* root, int depth)
{
if(depth >= 20)
{
return NULL; 
}
if(root == NULL)
{
root = newNode();
}
root->left = seqaddNode(root->left, depth+1);
root->right = seqaddNode(root->right, depth+1);
return root;
}
struct tree* plladdNode(struct tree* root, int depth)
{
if(depth >= 24)
{
return NULL;
}
if(root == NULL)
{
root = newNode();
}
if(depth < 5)
{
#pragma omp task
{
root->left = plladdNode(root->left, depth+1);
}
#pragma omp task
{
root->right = plladdNode(root->right, depth+1);
}
}
else
{
root->left = plladdNode(root->left, depth+1);
root->right = plladdNode(root->right, depth+1);
if(depth == 5) printf("TID = %02d\n", omp_get_thread_num());
}
return root;
}
void traverseNode(struct tree* root, int depth)
{
if(root == NULL)
{
return;
}
traverseNode(root->left, depth+1);
printf("TREE depth = %02d, value = %0f\n", depth, root->d);
traverseNode(root->right, depth+1);
}
int seqCount(struct tree* root, int depth)
{
int lcount;
int rcount;
lcount = 0;
rcount = 0;
if(root->left) lcount = seqCount(root->left, depth+1);
if(root->right) rcount = seqCount(root->right, depth+1); 
if(root->d < 0.50)
{
return (lcount + rcount + 1);
}
else
{
return (lcount + rcount);
}
}
int pllCount(struct tree* root, int depth)
{
int lcount;
int rcount;
lcount = 0;
rcount = 0;
if(depth < 5)
{
#pragma omp task shared(lcount)
{
if(root->left) lcount = pllCount(root->left, depth+1);
}
#pragma omp task shared(rcount)
{
if(root->right) rcount = pllCount(root->right, depth+1);
}
#pragma omp taskwait
{
if(root->d < 0.50)
{
return (lcount + rcount + 1);
}
else
{
return (lcount + rcount);
}
}
}
else
{
if(root->left) lcount = pllCount(root->left, depth+1);
if(root->right) rcount = pllCount(root->right, depth+1);
if(depth == 5) printf("TID = %02d\n", omp_get_thread_num());
if(root->d < 0.50)
{
return (lcount + rcount + 1);
}
else
{
return (lcount + rcount);
}
}
}
int main (int argc, char *argv[]) {
int nt, i=0;
double time1, time2;
struct tree* root1 = NULL;
struct tree* root2 = NULL;
srand(2);
time2 = omp_get_wtime();
#pragma omp parallel
{
#pragma omp single
{
printf("Threads running in parallel at depth == 5\n");
root2 = plladdNode(root2, 0);
}
}
time2 = omp_get_wtime() - time2;
root1 = root2; 
printf("\n\nElapsed Time (CREATE Parallely) = %0f\n\n", time2);
time2 = omp_get_wtime();
#pragma omp parallel
{
#pragma omp single
{
printf("Threads running in parallel at depth == 5\n");
i = pllCount(root2, 0);
}
}
time2 = omp_get_wtime() - time2;
printf("\n\nElapsed Time (COUNT Parallely) = %0f\n\n", time2);
printf("Number of Items with value < 0.50 is %d\n\n", i);
i = 0;
time1 = omp_get_wtime();
i = seqCount(root1, 0);
time1 = omp_get_wtime() - time1;
printf("\n\nElapsed Time (COUNT Sequentialy) = %0f\n\n", time1);
printf("Number of Items with value < 0.50 is %d\n\n", i);
printf("\n\nSPEEDUP (PARALLEL COUNTING vs SEQUENTIAL COUNTING) = %0f\n\n", time1/time2);
free(root1);
#pragma omp parallel
{
nt = omp_get_num_threads();
}
printf("Total Threads Running = %02d\n", nt);
return 0;
}
