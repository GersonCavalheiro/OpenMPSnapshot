#include "pagerank.hpp"

void pagerank(Graph &g, double alpha, int maxiteration){


size_t vertexNum = num_vertices(g);
const double tolerance = 5e-8;
double bias = (1.0 - alpha) / vertexNum;
std::vector<int> outdegree(vertexNum);
std::vector<double> memory(vertexNum);



int stdvalue= 1;

#pragma omp parallel for 
for(size_t i = 0; i < vertexNum; i++){
outdegree[i]=boost::out_degree(i,g);
g[i].value=stdvalue;
}


int iter=0;
double update;
while (iter++ < maxiteration){

#pragma omp parallel for 
for (size_t i = 0; i < vertexNum; ++i){
memory[i]=g[i].value;
g[i].value=0;
}

#pragma omp parallel for private(update)
for (size_t i = 0; i < vertexNum; ++i){
if(outdegree[i]>0){
update = alpha * (memory[i]/outdegree[i]);
std::pair<adjacency_iterator, adjacency_iterator> neighbors = boost::adjacent_vertices(i, g);
for(; neighbors.first != neighbors.second; ++neighbors.first){
#pragma omp atomic
g[*neighbors.first].value += update;   
}
}
} 

double error = 0.0;
#pragma omp parallel for reduction(+:error)
for (size_t i = 0; i < vertexNum; i++){
g[i].value += bias;  
error += fabs(g[i].value - memory[i]);
}

if (error < tolerance)
break;

std::cout << "Pagerank step "<< iter << " ends with error " << error << "\n";
} 
}
