#include <iostream>
#include <algorithm>
#include "kd_tree.hpp"
#include <vector>
#include <omp.h>

#define NDIM 2

template<typename T>
kpoint<T>::kpoint(){
ndim = N_DIM;
}

template<typename T>
T kpoint<T>::get_n_point(int axis){
return points[axis];
}

template<typename T>
void kpoint<T>::set_point(int axis, T value){
points[axis] = value;
}

template<typename T>
void kpoint<T>::print_kpoints(int axis){
std::cout<<"( ";
for(auto i=0;i<ndim;++i){
std::cout<<points[i]<<", ";
}
std::cout<<"axis="<<axis<<") ";

return;
}



template<typename T>
kdnode<T>::kdnode(){
axis = -1; 
}

template<typename T>
void kdnode<T>::in_order(){

if( axis != -1) {

left -> in_order();
this -> split.print_kpoints(axis);
right -> in_order();

}
}




template<typename T>
void print_array(kpoint<T> &arr, int n, int axis){
for(auto i=0; i<n; ++i){
for(auto j=0; j<axis; ++j){
std::cout<<arr.get_n_point(j)<<" ";
}
std::cout<<std::endl;
}
}

template<typename T>
std::vector<struct kpoint<T>> choose_splitting_point(std::vector<struct kpoint<T>> points, int n, int myaxis){


auto beg = points.begin();
int half = n/2;

std::nth_element(beg, beg + half, points.end(), 
[&myaxis](kpoint<T>& a, kpoint<T>& b){return a.get_n_point(myaxis) < b.get_n_point(myaxis);}
);


return points; 

}

template<typename T>
struct kdnode<T> * build_kdtree( std::vector<struct kpoint<T>> points, int ndim, int axis ) {



struct kdnode<T>* node = new kdnode<T>; 
const int N = points.size();
int myaxis = (axis+1) % ndim;

if( N == 1 ) { 

node->left = new kdnode<T>;
node->right = new kdnode<T>;
node->axis = myaxis;
node->split = points.at(0);

}else{

auto mypoint = choose_splitting_point( points, N, myaxis);

int half = N/2;

std::vector<struct kpoint<T>> left_points(mypoint.begin(), mypoint.begin() + half);
std::vector<struct kpoint<T>> right_points(mypoint.begin() + half + 1, mypoint.end());

node->axis = myaxis;

node->split = mypoint.at(half);

#pragma omp task shared(ndim) firstprivate(left_points, myaxis)
{

#ifdef DEBUG
std::cout<<"Start thread "<<omp_get_thread_num()<<std::endl;
#endif

node->left = build_kdtree( left_points, ndim, myaxis );
}

#pragma omp task shared(ndim) firstprivate(right_points, myaxis)
{

#ifdef DEBUG
std::cout<<"Start thread "<<omp_get_thread_num()<<std::endl;
#endif

if( N != 2)
node -> right = build_kdtree( right_points, ndim, myaxis );
else
node -> right = new kdnode<T>;

}
}


return node;

}
