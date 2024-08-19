#ifndef __OMP_GRAPH_RUNTIME_H__
#define __OMP_GRAPH_RUNTIME_H__

#include "../data-structures/data-structures.h"

namespace __core__ {
namespace __wavefront__ {
namespace __runtime__ {
enum __ompGraphVersion__ {
OMPOrderedGraph,
OMPTopologicalSort,
OMPUserOrder,
OMPUserOrderDebug
};
typedef __ompGraphVersion__ ompGraphVersion;

template <ompGraphVersion version=OMPOrderedGraph,typename FT=void,typename VertexType=void,typename EdgeType=void,typename Allocator=void,typename IT=int,typename...Args,enable_IT<eq_CE(version,OMPOrderedGraph)> = 0>
void ompGraph(GraphCXS<VertexType,EdgeType,Allocator,IT>& graph,FT& function,int threadnum,Args...args) {
IT* ptr=graph.ptr();
#pragma omp parallel  num_threads(threadnum)
{
#pragma omp single
{
for(size_t i=0;i<graph.v();++i) {
IT pos=graph.ptr(i),size=graph.ptr(i+1)-graph.ptr(i);
if(size>0) {
#pragma omp task depend(iterator(it=0:size), in:ptr[graph.indxs(pos+it)]) depend(out:ptr[i])
{
function(i,omp_get_thread_num(),args...);
}
}
else
break;
}
}
}
}
template <ompGraphVersion version=OMPOrderedGraph,typename FT=void,typename VertexType=void,typename EdgeType=void,typename Allocator=void,typename IT=int,typename...Args,enable_IT<eq_CE(version,OMPTopologicalSort)> = 0>
void ompGraph(GraphCXS<VertexType,EdgeType,Allocator,IT>& graph,FT& function,int threadnum,Args...args) {
IT* ptr=graph.ptr();
std::vector<int> indegree(graph.v(),0);
for(size_t i=0;i<graph.e();++i)
indegree[graph.indxs(i)]+=1;
std::stack<IT> s;
for(size_t i=0;i<graph.v();++i)
if(indegree[i]==0)
s.push(i);
#pragma omp parallel  num_threads(threadnum)
{
#pragma omp single
{
while(!s.empty()) {
IT v=s.top();
s.pop();
IT pos=graph.ptr(v),size=graph.ptr(v+1)-graph.ptr(v);
if(size>0) {
#pragma omp task depend(iterator(it=0:size), in:ptr[graph.indxs(pos+it)]) depend(out:ptr[v])
{
function(v,omp_get_thread_num(),args...);
}
fn(v,graph,args...);
for(IT j=graph.ptr(v);j<graph.ptr(v+1);++j) {
indegree[graph.indxs(j)]-=1;
if(indegree[graph.indxs(j)]==0)
s.push(graph.indxs(j));
}
}
}
}
}
}
template <ompGraphVersion version=OMPOrderedGraph,typename FT=void,typename VertexType=void,typename EdgeType=void,typename Allocator=void,typename IT=int,typename...Args,enable_IT<eq_CE(version,OMPUserOrder)> = 0>
void ompGraph(GraphCXS<VertexType,EdgeType,Allocator,IT>& graph,FT& function,const std::vector<int>& order,int threadnum,Args...args) {
IT* ptr=graph.ptr();
#pragma omp parallel num_threads(threadnum)
{
#pragma omp single
{
for(size_t i=0;i<order.size();++i) {
int v=order[i];
IT pos=graph.ptr(v),size=graph.ptr(v+1)-graph.ptr(v);
if(size>0) {
#pragma omp task depend(iterator(it=0:size), in:ptr[graph.indxs(pos+it)]) depend(out:ptr[v])
function(v,omp_get_thread_num(),args...);
}
else
#pragma omp task depend(out:ptr[v])
function(v,omp_get_thread_num(),args...);
}
}
}
}
template <ompGraphVersion version=OMPOrderedGraph,typename FT=void,typename VertexType=void,typename EdgeType=void,typename Allocator=void,typename IT=int,typename...Args,enable_IT<eq_CE(version,OMPUserOrder)> = 0>
void ompGraph(GraphCXS<VertexType,EdgeType,Allocator,IT>& graph,FT& function,const std::vector<int>& order,Args...args) {
IT* ptr=graph.ptr();
#pragma omp parallel
{
#pragma omp single
{
for(size_t i=0;i<order.size();++i) {
int v=order[i];
IT pos=graph.ptr(v),size=graph.ptr(v+1)-graph.ptr(v);
if(size>0) {
#pragma omp task depend(iterator(it=0:size), in:ptr[graph.indxs(pos+it)]) depend(out:ptr[v])
function(v,omp_get_thread_num(),args...);
}
else
#pragma omp task depend(out:ptr[v])
function(v,omp_get_thread_num(),args...);
}
}
}
}
template <ompGraphVersion version=OMPOrderedGraph,typename FT=void,typename VertexType=void,typename EdgeType=void,typename Allocator=void,typename IT=int,typename...Args,enable_IT<eq_CE(version,OMPUserOrderDebug)> = 0>
void ompGraph(GraphCXS<VertexType,EdgeType,Allocator,IT>& graph,FT& function,const std::vector<int>& order,int threadnum,Args...args) {
IT* ptr=graph.ptr();
#pragma omp parallel num_threads(threadnum)
{
#pragma omp single
{
for(size_t i=0;i<order.size();++i) {
int v=order[i];
IT pos=graph.ptr(v),size=graph.ptr(v+1)-graph.ptr(v);
if(size>0){
#pragma omp task depend(iterator(it=0:size), in:ptr[graph.indxs(pos+it)]) depend(out:ptr[v])
{
cpu_timer timer;
int tid=omp_get_thread_num();
timer.start();
function(v,tid,args...);
timer.stop();
graph[v].etime=timer.elapsed_time();
graph[v].tid=tid;
}
}
else {
#pragma omp task depend(out:ptr[v])
{
cpu_timer timer;
int tid=omp_get_thread_num();
timer.start();
function(v,tid,args...);
timer.stop();
graph[v].etime=timer.elapsed_time();
graph[v].tid=tid;
}
}
}
}
}
}
template <ompGraphVersion version=OMPOrderedGraph,typename FT=void,typename VertexType=void,typename EdgeType=void,typename Allocator=void,typename IT=int,typename...Args,enable_IT<eq_CE(version,OMPUserOrderDebug)> = 0>
void ompGraph(GraphCXS<VertexType,EdgeType,Allocator,IT>& graph,FT& function,const std::vector<int>& order,Args...args) {
IT* ptr=graph.ptr();
#pragma omp parallel
{
#pragma omp single
{
for(size_t i=0;i<order.size();++i) {
int v=order[i];
IT pos=graph.ptr(v),size=graph.ptr(v+1)-graph.ptr(v);
if(size>0){
#pragma omp task depend(iterator(it=0:size), in:ptr[graph.indxs(pos+it)]) depend(out:ptr[v])
{
cpu_timer timer;
int tid=omp_get_thread_num();
timer.start();
function(v,tid,args...);
timer.stop();
graph[v].etime=timer.elapsed_time();
graph[v].tid=tid;
}
}
else {
#pragma omp task depend(out:ptr[v])
{
cpu_timer timer;
int tid=omp_get_thread_num();
timer.start();
function(v,tid,args...);
timer.stop();
graph[v].etime=timer.elapsed_time();
graph[v].tid=tid;
}
}
}
}
}
}


}
}
}

#endif
