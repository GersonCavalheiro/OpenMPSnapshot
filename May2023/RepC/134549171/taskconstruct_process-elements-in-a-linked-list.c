#include <stdio.h> 
#include <stdlib.h> 
struct Node 
{ 
int data; 
struct Node *next; 
}; 
void push(struct Node** head_ref, int new_data) 
{ 
struct Node* new_node = (struct Node*) malloc(sizeof(struct Node)); 
new_node->data  = new_data; 
new_node->next = (head_ref); 
*(head_ref)    = new_node; 
} 
void process(struct Node* p)
{
p->data = (p->data) * (p->data);
}
void increment_list_items(struct Node** head)
{
#pragma omp parallel
{
#pragma omp single
{
Node * p = *head;
while (p) {
#pragma omp task
process(p);
p = p->next;
}
}
}
int main() 
{ 
struct Node* head = NULL; 
push(&head, 6);      
push(&head, 7); 
push(&head, 1); 
push(&head, 4); 
increment_list_items(&head);          
return 0; 
} 
