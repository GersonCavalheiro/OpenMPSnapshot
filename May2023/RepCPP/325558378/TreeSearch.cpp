
#include "stdafx.h"
#include <conio.h>
#include <iostream>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _OPENMP
int thread_id = omp_get_thread_num();
int threads_num = omp_get_num_threads();
#else
int thread_id = 0;
int threads_num = 1;
#endif

struct tree
{
int value;
struct tree *left;
struct tree *right;
};

typedef struct tree node;

void insert(node** tree, node* item);
void printout(node* tree, int delta);
node* serial_search(node* tree, int value);
node* parallel_search(node* tree, int value);
bool output(node* temp, node* root, int value);
void run_algorithm(node* temp, node* root, int value, int algorithm);

int _tmain(int argc, _TCHAR* argv[])
{
node* current, * temp = NULL;
int i, value, number = 0;
node* root = NULL;

printf("\tNum processors: %d\n\n", omp_get_num_procs());

printf("Random numbers:");
for (i = 1; i <= 10; i++) {
current = (node*)malloc(sizeof(node));
current->left = NULL;
current->right = NULL;
current->value = rand() % 100;
printf("\t%d", current->value);
insert(&root, current);
}

printf("\n\n\n");
printout(root, 0);
printf("\n");
printf("\nValue = ");
scanf_s("%d", &value);

int algorithm = 0;
printf("\tChoose the algorithm for search: (enter the number)");
printf("\n1. Serial");
printf("\n2. Parallel\n");
scanf_s("%d", &algorithm);

run_algorithm(temp, root, value, algorithm);

return 0;
}

void insert(node** tree, node* item) {
if (!(*tree)) {
*tree = item;
return;
}
if (item->value < (*tree)->value) {
insert(&(*tree)->left, item);
} 
else if (item->value >(*tree)->value) {
insert(&(*tree)->right, item);
}
else
return;
}

void printout(node *tree, int delta) {
if (tree->right) {
printout(tree->right, delta + 1);
}

for (int i = 1; i <= delta; i++)
printf("\t");

printf("%d\t", tree->value);
printf("\n");

if (tree->left) {
printf("\n");
printout(tree->left, delta + 1);
}
}

node* serial_search(node* tree, int value) {

while (tree && tree->value != value) {

if (value > tree->value) {
tree = tree->right;
}

else if (value < tree->value) {
tree = tree->left;
}
}

if (!tree)
return NULL;  
else
return tree;   


}

node* parallel_search(node *tree, int value) {

while (tree && tree->value != value) {

#pragma omp parallel
threads_num = omp_get_num_threads();

#pragma omp parallel sections num_threads(threads_num) 
{
#pragma omp section
{
if (value > tree->value)
tree = tree->right;
}
#pragma omp section
{
if (value < tree->value)
tree = tree->left;
}
}
}

#pragma omp master    
if (!tree)
return NULL;  
else
return tree;  


}

bool output(node *temp, node *root, int value) {
if (temp) {
if (temp-> value < root->value)
printf("\tSuccess! Value %d is found in the left subtree.\n", temp->value);
else if (temp->value > root->value)
printf("\tSuccess! Value %d is found in the right subtree.\n", temp->value);
else
printf("\tSuccess! Value %d is the root of the tree.\n", temp->value);
return true;
}
else {
printf("There is no such value!\nWould you like to enter another value to serach for?(y/n) ");
char is_yes_or_no = _getch();
if (is_yes_or_no == 'y') {
printf("\nValue = ");
scanf_s("%d", &value);
int algorithm = 0;
printf("\tChoose the algorithm for search: (enter the number)");
printf("\n1. Serial");
printf("\n2. Parallel\n");
scanf_s("%d", &algorithm);
run_algorithm(temp, root, value, algorithm);
}
printf("\n");
return false;
}
}

void run_algorithm(node *temp, node *root, int value, int algorithm) {
if (algorithm == 1) {
double time_before = omp_get_wtime();
temp = serial_search(root, value);
if (output(temp, root, value)) {
double time_after = omp_get_wtime();
double diff = time_after - time_before;
printf("Time for serial search: %.5g\n", diff);
}

}
else if (algorithm == 2) {
double time_before = omp_get_wtime();
temp = parallel_search(root, value);
if (output(temp, root, value)) {
double time_after = omp_get_wtime();
double diff = time_after - time_before;
printf("Time for parallel search: %.5g\n", diff);
}

}
else {
printf("Invalid number!\n");
}
}

