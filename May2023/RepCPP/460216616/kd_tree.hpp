#ifndef KD_TREE_H
#define KD_TREE_H


#include <vector>

#define N_DIM 2


template<typename T>
struct kpoint{
T points[N_DIM];
int ndim;

kpoint();
kpoint(T &points_) : points(points_), ndim(N_DIM) {}

T get_n_point(int axis);

void set_point(int axis, T value);

void print_kpoints(int axis);




};

template<typename T>
struct kdnode {
int axis; 
struct kpoint<T> split; 
struct kdnode<T> *left, *right; 

kdnode();

void in_order();

};



template<typename T>
struct kdnode<T> * build_kdtree( std::vector<struct kpoint<T>> points, int ndim, int axis );


#endif

