#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 16
typedef struct List{
int data;
struct List *next;
}List;
List* addItem(List *list, int data);
int main(){
int i, r;
List *list = NULL;
List *p = NULL;
srand(time(NULL));
for(i = 0; i < N; i++){
r = 1 + rand() % 100;
list = addItem(list, r);
printf("Added %d to list\n", r);
}
printf("\nSerial Processing:\n");
p = list;
while (p != NULL){
printf("Processing %d...\n", p->data);
p = p->next;
}
printf("\nParallel Processing:\n");
p = list;
#pragma omp parallel
{
#pragma omp single
{
while(p != NULL){
#pragma omp task
{
printf("Thread %d processes %d\n", omp_get_thread_num(), p->data);
}
p = p->next;             
}
}
}
return 0;
}
List* addItem(List *list, int data){
List *newList = NULL;
newList = (List*) malloc(1 * sizeof(List));
newList->next = list;
newList->data = data;
return newList;
}
