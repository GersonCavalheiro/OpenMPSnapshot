#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include "ompvv.h"
#define SIZE_LIST 10
typedef struct node {
int data;
struct node *next;
} node_t;
void map_ll(node_t * head) {
OMPVV_INFOMSG("Entering map_ll");
node_t * temp = head;
if (!temp) {
OMPVV_ERROR("Head was null");
return;
}
#pragma omp target enter data map(to:temp[:1])
#pragma omp target 
{
temp->data += 1;  
}
while(temp->next) {
node_t * cur = temp->next;
#pragma omp target enter data map(to:cur[:1])
#pragma omp target 
{
cur->data += 1;
temp->next = cur;
}
temp=temp->next;
}
}
void unmap_ll(node_t * head) {
OMPVV_INFOMSG("Entering unmap_ll");
node_t * temp = head, *tempNext;
if (!temp) {
OMPVV_ERROR("Head was null");
return;
}
tempNext = temp->next;
#pragma omp target exit data map(from:temp[0:1])
temp->next = tempNext;
while(temp->next) {
temp = temp->next;
tempNext = temp->next;
#pragma omp target exit data map(from: temp[0:1])
temp->next = tempNext;
}
}
void push(node_t * head, int data) {
node_t * current = head;
while (current->next != NULL) {
current = current->next;
}
current->next = (node_t *) malloc(sizeof(node_t));
current->next->data = data;
current->next->next = NULL;
}
void display(node_t * head)
{
node_t * temp=head;
while(temp!=NULL)
{
printf("%d\n",temp->data);
temp=temp->next;
}
}
int check(node_t * head)
{
OMPVV_INFOMSG("Entering check");
int error = 0, i=0;
node_t * temp=head;
while(temp!=NULL)
{
OMPVV_TEST_AND_SET_VERBOSE(error, temp->data != i+1);
i++;  
temp=temp->next;
}
return error;
}
int main() {
OMPVV_TEST_OFFLOADING;
int i, error = 0;
node_t * head = NULL;
head = (node_t *) malloc(sizeof(node_t));
if (head == NULL) {
OMPVV_ERROR("There was a problem allocating the head node");
return 1;
}
head->data = 0;
head->next = NULL;
for(i=1; i < SIZE_LIST; i++)
push(head,i);
map_ll(head);
unmap_ll(head);
OMPVV_TEST_AND_SET_VERBOSE(error, check(head));
while (head) {
node_t * next = head->next;
free (head);
head = next;
}
OMPVV_REPORT_AND_RETURN(error);
return 0;
}
