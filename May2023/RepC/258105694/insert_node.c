#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
struct node {
int data;
struct node *left;
struct node *right;
struct node *parent;
};
typedef struct node* Node;
void insertNode(Node *q, int data) {
if((*q) == NULL) {
Node p = (Node)malloc(sizeof(struct node));
p->left = NULL;
p->right = NULL;
p->parent = NULL;
p->data = data;
(*q) = p;
return;
}
else{
if(data < (*q)->data && (*q)->left != NULL) {
insertNode(&((*q)->left),data);
}
else if(data > (*q)->data && (*q)->right != NULL) {
insertNode(&((*q)->right),data);
}
else if(data == (*q)->data){
printf("ya existe el valor");
return;
}
if((*q)->left == NULL && data < (*q)->data) {
Node p = (Node)malloc(sizeof(struct node));
p->data = data;
p->right = NULL;
p->left = NULL;
(*q)->left = p;
p->parent = (*q);
return;
}
else if((*q)->right == NULL  && data > (*q)->data) {
Node p = (Node)malloc(sizeof(struct node));
p->data = data;
p->right = NULL;
p->left = NULL;
(*q)->right = p;
p->parent = (*q);
return;
}
}
}
void traverse(Node ptrArbol )
{  
if ( ptrArbol != NULL ) {
#pragma omp task
traverse(ptrArbol->left);
#pragma omp task
traverse(ptrArbol->right);
#pragma omp taskwait
printf("%d,",ptrArbol->data);
}
}
int main()
{ 
int i; 
Node ptrRaiz = NULL;
srand( time( NULL ) ); 
printf( "\n\nLos numeros colocados en el arbol son: 5,2,3,0,1,7,6,9" );
insertNode( &ptrRaiz, 5 );
insertNode( &ptrRaiz, 2 );
insertNode( &ptrRaiz, 3 );
insertNode( &ptrRaiz, 0 );
insertNode( &ptrRaiz, 1 );
insertNode( &ptrRaiz, 7 );
insertNode( &ptrRaiz, 6 );
insertNode( &ptrRaiz, 9 );
printf( "\n\nEl recorrido en traverse es:\n" );
traverse( ptrRaiz );
return 0;
}
