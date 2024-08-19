#ifndef __PARALLEL_LEFT_LOOKING_H__
#define __PARALLEL_LEFT_LOOKING_H__

#include <omp.h>

#include "../matrix-formats/matrix-formats.h"
#include "elimitate-variables.h"
#include "eliminate-branches.h"
#include "super-nodes.h"

namespace __core__ {
namespace __linear_algebra__ {
namespace __cholesky__ {
template <typename T=void,typename IT=void,typename Allocator=void>
void choleskyLeftLookingP(MatrixCXSHandler<T,IT> L,MatrixCXSHandler<T,IT> A,MatrixCXSHandler<IT,IT> RP,LRTree<IT,IT,Allocator,int>& tree,LRTree<IT,SuperNode,Allocator,int>& supernodetree,
ArrayHandler<IT,IT> p,std::vector<ArrayHandler<T,IT>>& c,int threadnum=4){
std::vector<int> __dummy__(supernodetree.size()+10,0);
int *ptr=__dummy__.data();
#pragma omp parallel num_threads(threadnum) shared(L,A,RP,tree,supernodetree,p,c)
{
#pragma omp single
{
int node=(*supernodetree).left_child;
bool down=true;
size_t i=0;
while((i++)<supernodetree.size()) {
if(down)
while(supernodetree[node].left_child!=supernodetree.invalidPos)
node=supernodetree[node].left_child;
down=false;
#pragma omp task depend(out:ptr[supernodetree[node].key]) depend(in:ptr[supernodetree[node].value.parent])
{
for(auto j=supernodetree[node].value.nodes.begin();j!=supernodetree[node].value.nodes.end();++j)
eliminateBranch(L,A,RP,tree,tree[*j],p,c);
}
if(supernodetree[node].right_sibling!=supernodetree.invalidPos) {
node=supernodetree[node].right_sibling;
down=true;
}
else
node=supernodetree[node].parent;
if(node==0||node==supernodetree.invalidPos)
break;
}
}
}
}
template <typename T=void,typename IT=void,typename Allocator=void>
void choleskyLeftLookingPT(MatrixCXSHandler<T,IT> L,MatrixCXSHandler<T,IT> A,MatrixCXSHandler<IT,IT> RP,LRTree<IT,IT,Allocator,int>& tree,LRTree<IT,SuperNode,Allocator,int>& supernodetree,
ArrayHandler<IT,IT> p,std::vector<ArrayHandler<T,IT>>& c,int threadnum=4){
std::vector<int> __dummy__(supernodetree.size()+10,0);
int *ptr=__dummy__.data();
#pragma omp parallel num_threads(threadnum) shared(L,A,RP,tree,supernodetree,p,c)
{
#pragma omp single
{
int node=(*supernodetree).left_child;
bool down=true;
size_t i=0;
while((i++)<supernodetree.size()) {
if(down)
while(supernodetree[node].left_child!=supernodetree.invalidPos)
node=supernodetree[node].left_child;
down=false;
#pragma omp task depend(out:ptr[supernodetree[node].key]) depend(in:ptr[supernodetree[node].value.parent])
{
supernodetree[node].value.time=0;
supernodetree[node].value.processor=omp_get_thread_num();
cpu_timer timer;
timer.start();
for(auto j=supernodetree[node].value.nodes.begin();j!=supernodetree[node].value.nodes.end();++j)
eliminateBranch(L,A,RP,tree,tree[*j],p,c,supernodetree[node].value.processor);
timer.stop();
supernodetree[node].value.time=timer.elapsed_time();
}
if(supernodetree[node].right_sibling!=supernodetree.invalidPos) {
node=supernodetree[node].right_sibling;
down=true;
}
else
node=supernodetree[node].parent;
if(node==0||node==supernodetree.invalidPos)
break;
}
}
}
}
}
}
}
#endif
