#define _GNU_SOURCE
#include<stdlib.h>
#include<sched.h>
#include<stdio.h>
#include<omp.h>
#define SIZE 1000000
struct bin_tree {
int data;
struct bin_tree * right, * left;
};
typedef struct bin_tree node;
void insert(node ** tree, int val)
{
node *temp = NULL;
if(!(*tree))
{
temp = (node *)malloc(sizeof(node));
temp->left = temp->right = NULL;
temp->data = val;
*tree = temp;
return;
}
if(val < (*tree)->data)
{
insert(&(*tree)->left, val);
}
else if(val > (*tree)->data)
{
insert(&(*tree)->right, val);
}
}
void process(node * tree)
{
}
void print_postorder_serial(node * tree)
{
if (tree->left)
{
print_postorder_serial(tree->left);
}
if (tree->right)
{
print_postorder_serial(tree->right);
}
process(tree);
}
void print_postorder_parellel(node * tree, int threads)
{
if(threads == 1){
int thread_num = omp_get_thread_num();
int cpu_num = sched_getcpu();
print_postorder_serial(tree);
return;
}
else
{
#pragma omp parallel sections
{
#pragma omp section
{
if (tree->left)
print_postorder_parellel(tree->left, threads/2);
}
#pragma omp section
{
if (tree->right)
print_postorder_parellel(tree->right, threads - threads/2);
process(tree);
}
}
}	
}
int main()
{
node *root;
node *tmp;
int i;
int sz = SIZE; 
int threads;
double start_time, run_time_serial, run_time_parallel;
root = NULL;
srand(5); 
for (i=0; i<sz; i++){
insert(&root, (1+(rand()%1000000000)));
}
#pragma omp parallel
{
#pragma omp single
{
threads =  omp_get_num_threads();
}
}
printf("Post Order Display using parallel\n");
start_time = omp_get_wtime();
print_postorder_parellel(root, threads);		
run_time_parallel = omp_get_wtime() - start_time;
printf("Post Order Display using serial\n");
start_time = omp_get_wtime();
print_postorder_serial(root);
run_time_serial = omp_get_wtime() - start_time;
printf("\n");
printf("Time to traverse tree (in serial) is %f seconds \n",  run_time_serial);
printf("Time to traverse tree (in parellel) is %f seconds \n",  run_time_parallel);
printf("\n");
return 0;
}
