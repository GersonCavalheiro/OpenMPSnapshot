#ifndef __ELIMINATE_BRANCHES_H__
#define __ELIMINATE_BRANCHES_H__

#include <omp.h>

#include "../data-structures/data-structures.h"
#include "../matrix-formats/matrix-formats.h"
#include "elimitate-variables.h"

namespace __core__ {
namespace __linear_algebra__ {
namespace __cholesky__ {
template <typename T,typename IT,typename Allocator>
void eliminateBranch(MatrixCXSHandler<T,IT> L,MatrixCXSHandler<T,IT> A,MatrixCXSHandler<IT,IT> RP,LRTree<IT,IT,Allocator,int>& tree,Node<IT,IT,int>& root,ArrayHandler<IT,IT> p,
ArrayHandler<T,IT> c){
typedef Node<IT,IT,int> NT;
int node=root.left_child;
if(node!=tree.invalidPos) {
bool down=true;
size_t i=0;
while((i++)<tree.size()) {
if(down)
while(tree[node].left_child!=tree.invalidPos)
node=tree[node].left_child;
down=false;
eliminateVariable(L,A,RP,p,c,tree[node].key);
if(tree[node].right_sibling!=tree.invalidPos) {
node=tree[node].right_sibling;
down=true;
}
else
node=tree[node].parent;
if(node==0||node==tree.invalidPos||node==root.self)
break;
}
}
eliminateVariable(L,A,RP,p,c,root.key);
#pragma omp critical
tree.eraseBranch(root);
}
template <typename T,typename IT,typename Allocator>
void eliminateBranch(MatrixCXSHandler<T,IT> L,MatrixCXSHandler<T,IT> A,MatrixCXSHandler<IT,IT> RP,LRTree<IT,IT,Allocator,int>& tree,Node<IT,IT,int>& root,ArrayHandler<IT,IT> p,
std::vector<ArrayHandler<T,IT>> &c){
typedef Node<IT,IT,int> NT;
int node=root.left_child;
if(node!=tree.invalidPos) {
bool down=true;
size_t i=0;
while((i++)<tree.size()) {
if(down)
while(tree[node].left_child!=tree.invalidPos)
node=tree[node].left_child;
down=false;
eliminateVariable(L,A,RP,p,c.at(omp_get_thread_num()),tree[node].key);
if(tree[node].right_sibling!=tree.invalidPos) {
node=tree[node].right_sibling;
down=true;
}
else
node=tree[node].parent;
if(node==0||node==tree.invalidPos||node==root.self)
break;
}
}
eliminateVariable(L,A,RP,p,c.at(omp_get_thread_num()),root.key);
#pragma omp critical
tree.eraseBranch(root);
}
template <typename T,typename IT,typename Allocator>
void eliminateBranch(MatrixCXSHandler<T,IT> L,MatrixCXSHandler<T,IT> A,MatrixCXSHandler<IT,IT> RP,LRTree<IT,IT,Allocator,int>& tree,Node<IT,IT,int>& root,ArrayHandler<IT,IT> p,
std::vector<ArrayHandler<T,IT>> &c,int tid){
typedef Node<IT,IT,int> NT;
int node=root.left_child;
if(node!=tree.invalidPos) {
bool down=true;
size_t i=0;
while((i++)<tree.size()) {
if(down)
while(tree[node].left_child!=tree.invalidPos)
node=tree[node].left_child;
down=false;
eliminateVariable(L,A,RP,p,c.at(tid),tree[node].key);
if(tree[node].right_sibling!=tree.invalidPos) {
node=tree[node].right_sibling;
down=true;
}
else
node=tree[node].parent;
if(node==0||node==tree.invalidPos||node==root.self)
break;
}
}
eliminateVariable(L,A,RP,p,c.at(tid),root.key);
#pragma omp critical
tree.eraseBranch(root);
}
template <typename T,typename IT,typename Allocator>
void eliminateBranch(MatrixCXSHandler<T,IT> L,MatrixCXSHandler<T,IT> A,MatrixCXSHandler<IT,IT> RP,LRTree<IT,IT,Allocator,int>& tree,Node<IT,IT,int>& root,ArrayHandler<IT,IT> p,
std::vector<ArrayHandler<T,IT>> &c,double& time){
typedef Node<IT,IT,int> NT;
int node=root.left_child;
cpu_timer timer;
if(node!=tree.invalidPos) {
bool down=true;
size_t i=0;
while((i++)<tree.size()) {
if(down)
while(tree[node].left_child!=tree.invalidPos)
node=tree[node].left_child;
down=false;
time+=eliminateVariableT(L,A,RP,p,c.at(omp_get_thread_num()),tree[node].key);
if(tree[node].right_sibling!=tree.invalidPos) {
node=tree[node].right_sibling;
down=true;
}
else
node=tree[node].parent;
if(node==0||node==tree.invalidPos||node==root.self)
break;
}
}
time+=eliminateVariableT(L,A,RP,p,c.at(omp_get_thread_num()),root.key);
#pragma omp critical
tree.eraseBranch(root);
}
}
}
}
#endif
