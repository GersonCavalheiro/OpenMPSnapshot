
#ifndef TREE_TRAVERSALS_TREE_UTILS_H
#define TREE_TRAVERSALS_TREE_UTILS_H

#include <malloc.h>
#include <iostream>
#include <omp.h>

using namespace std;

struct node{
int data;
int children;
struct node** pointers;

};
typedef struct node Node;

Node* createNode(int Data, int children, int depth)
{
Node* node= new Node();
node->data = Data;
node->children = children;
if(depth<7)
{
Node* arr = (Node*) malloc(children*sizeof(Node));
node->pointers = &arr;
} else
{
node->pointers= nullptr;
}
return  node;
}

void populateChildren(Node* node, int depth)
{
if (depth==7)
{
node = nullptr;
return;
}
#pragma omp parallel for
for (int i=0; i< node->children; i++)
{
int current_depth = depth;
node->pointers[i] = createNode(int(rand()), 3, current_depth+1);
populateChildren(node->pointers[i], current_depth+1);
}
}

Node* createTree()
{
Node* head = createNode(8, 5, 0);
populateChildren(head, 0);
return head;
}

void parallel_tree_search(Node* head, int query)
{
if (head== nullptr)
return;

if(head->data==query)
{
cout<<"Query found at"<<head;
cout<<'\n';
}
#pragma omp parallel for
for (int i=0;i<head->children;i++)
{
parallel_tree_search(head->pointers[i], query);
}

}
#endif 