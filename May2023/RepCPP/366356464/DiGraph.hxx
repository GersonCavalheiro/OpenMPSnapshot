#pragma once
#include <vector>
#include "_main.hxx"

using std::vector;




template <class V=NONE, class E=NONE>
class DiGraph {
public:
using TVertex = V;
using TEdge   = E;

private:
vector<int>  none;
vector<bool> vex;
vector<vector<int>> vto;
vector<vector<E>> edata;
vector<V>         vdata;
int N = 0, M = 0;

private:
int s() const { return vto.size(); }
int ei(int u, int v) const { return findIndex(vto[u], v); }

public:
int span()  const { return s(); }
int order() const { return N; }
int size()  const { return M; }

bool hasVertex(int u)      const { return u < s() && vex[u]; }
bool hasEdge(int u, int v) const { return u < s() && ei(u, v) >= 0; }
auto edges(int u)          const { return u < s()? iterable(vto[u]) : iterable(none); }
int degree(int u)          const { return u < s()? vto[u].size()    : 0; }
auto vertices()      const { return filter(range(s()), [&](int u) { return  vex[u]; }); }
auto nonVertices()   const { return filter(range(s()), [&](int u) { return !vex[u]; }); }
auto inEdges(int v)  const { return filter(range(s()), [&](int u) { return ei(u, v) >= 0; }); }
int inDegree(int v) const { return countIf(range(s()), [&](int u) { return ei(u, v) >= 0; }); }

V vertexData(int u)   const { return hasVertex(u)? vdata[u] : V(); }
void setVertexData(int u, V d) { if (hasVertex(u)) vdata[u] = d; }
E edgeData(int u, int v)   const { return hasEdge(u, v)? edata[u][ei(u, v)] : E(); }
void setEdgeData(int u, int v, E d) { if (hasEdge(u, v)) edata[u][ei(u, v)] = d; }

public:
void addVertex(int u, V d=V()) {
if (hasVertex(u)) return;
if (u >= s()) {
vex.resize(u+1);
vto.resize(u+1);
edata.resize(u+1);
vdata.resize(u+1);
}
vex[u] = true;
vdata[u] = d;
N++;
}

void addEdge(int u, int v, E d=E()) {
if (hasEdge(u, v)) return;
addVertex(u);
addVertex(v);
vto[u].push_back(v);
edata[u].push_back(d);
M++;
}

void removeEdge(int u, int v) {
if (!hasEdge(u, v)) return;
int o = ei(u, v);
eraseIndex(vto[u], o);
eraseIndex(edata[u], o);
M--;
}

void removeEdges(int u) {
if (!hasVertex(u)) return;
M -= degree(u);
vto[u].clear();
edata[u].clear();
}

void removeInEdges(int v) {
if (!hasVertex(v)) return;
for (int u : inEdges(v))
removeEdge(u, v);
}

void removeVertex(int u) {
if (!hasVertex(u)) return;
removeEdges(u);
removeInEdges(u);
vex[u] = false;
N--;
}

public:
template <class T>
auto vertexContainer(T _) const { return vector<T>(s()); }
};
