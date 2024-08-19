#include <iostream>
#include <algorithm>
#include "kd_tree.hpp"
#include <math.h>
#include <string>
#include <sstream>
#include <chrono>

#define N_DIM 2

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
std::string kpoint<T>::save_kpoints(int axis){
std::string s ("");
s += "[";

for(auto i=0;i<ndim;++i){
s += std::to_string(points[i]);
s += ",";
}
s += ";";
s += std::to_string(axis);
s += "]";

return s;
}



template<typename T>
kdnode<T>::kdnode(){
axis = -1; 
left = NULL;
right = NULL;
}

template<typename T>
void kdnode<T>::in_order(){

if( axis != -1) {

if ( left != NULL)
left -> in_order();

this -> split.print_kpoints(axis);

if ( right != NULL)
right -> in_order();

}
}

template<typename T>
void kdnode<T>::pre_order(){

#ifdef DEBUG
std::cout<<"Axis = "<<axis;
#endif

if( axis != -1) {

this -> split.print_kpoints(axis);
if (left != NULL)
left -> pre_order();
if (right != NULL)
right -> pre_order();

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
struct kdnode<T> * build_serial_kdtree( std::vector<struct kpoint<T>> points, int ndim, int axis ) {



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

node->left = build_serial_kdtree( left_points, ndim, myaxis );



if( N != 2)
node -> right = build_serial_kdtree( right_points, ndim, myaxis );
else
node -> right = new kdnode<T>;

}

return node;

}


template<typename T>
std::string serialize_node(struct kdnode<T> *kdtree){

std::string s ("");

try{

if( kdtree -> axis != -1) {

s = kdtree -> split.save_kpoints( kdtree -> axis);

if ( kdtree -> left != NULL)
s += "(" + serialize_node(kdtree -> left) + ")";

if ( kdtree -> right != NULL)
s += "(" + serialize_node(kdtree -> right) + ")";

}
}catch(const std::exception& e){
std::cout<<e.what();
}

return s;
}

template<typename T>
struct kdnode<T> * deserialize_node(std::string data){

std::string s = data;

if ( s.size() == 0 )
return new kdnode<T>; 

if ( s[0] == ')' ) 
return new kdnode<T>; 

int j = 0;

while ( j < s.size() && s[j] != '(' ) 
j ++;

T arr[N_DIM];
std::string temp_str;
if ( s[1] == '[' ){

temp_str = s.substr(2, j-5);

}else{
temp_str = s.substr(1, j-5);

}

int temp_val;
std::istringstream iss(temp_str);
for(int i =0; i<N_DIM; ++i) {
iss >> temp_val;
arr[i] = temp_val;
if (iss.peek() == ',')
iss.ignore();
}

struct kpoint<T> point(arr);
struct kdnode<T> * root = new kdnode<T>;
root -> split = point;

temp_val = (int)s[j-2] - 48 ;
root -> axis = temp_val; 

int left = 0, i = j;

while ( i < s.size() )  {

if ( s[i] == '(' ) 
left ++;
else if ( s[i] == ')' ) 
left --;

if ( left == 0 ) {
break;
}
i ++;
}

if ( j < s.size() - 1 ) {
root->left = deserialize_node<T>(s.substr(j + 1, i - 1 - j));
}
if ( i + 1 < s.size() - 1 ) {
root->right = deserialize_node<T>(s.substr(i + 2, s.size() - i - 2));   
}
return root;



}






template<typename T>
struct kdnode<T> * build_parallel_kdtree4(std::vector<struct kpoint<T>> points, int ndim, int axis, int np, int level, MPI_Comm comm, int which){



std::chrono::time_point<std::chrono::high_resolution_clock> start;
std::chrono::time_point<std::chrono::high_resolution_clock> end;
std::chrono::duration<double> diff;


int size, irank;
MPI_Comm_rank(comm, &irank);

MPI_Status status;
MPI_Request request;

struct kdnode<T>* node = new kdnode<T>; 
const int N = points.size();
int myaxis = (axis+1) % ndim;

if ( N == 1 ){

#ifdef DEBUG
#endif

node->left = new kdnode<T>;
node->right = new kdnode<T>;
node->axis = myaxis;
node->split = points.at(0);

}else {


auto mypoint = choose_splitting_point( points, N, myaxis);
int half = N/2;
node->axis = myaxis;
node->split = mypoint.at(half);


if ( irank != 0){



if ( np/2 != pow(2, level)) {

std::vector<struct kpoint<T>> left_points(mypoint.begin(), mypoint.begin() + half);
std::vector<struct kpoint<T>> right_points(mypoint.begin() + half + 1, mypoint.end());

level = level + 1;
node->left = build_parallel_kdtree4( left_points, ndim, myaxis, np, level, comm, which);

which = which + 2;
if( N != 2)
node -> right = build_parallel_kdtree4( right_points, ndim, myaxis, np, level, comm, which);
else
node -> right = new kdnode<T>;



}else{



if( irank == which ){

std::vector<struct kpoint<T>> left_points(mypoint.begin(), mypoint.begin() + half);

#ifdef DEBUG
#endif

#pragma omp task shared(ndim) firstprivate(left_points, myaxis)
{
node -> left = build_serial_kdtree(left_points, ndim, myaxis);
}

std::string kdtree_str = serialize_node<T>(node -> left);
MPI_Send( kdtree_str.c_str() , kdtree_str.length() , MPI_CHAR , 0 , 10 , comm );


#ifdef DEBUG
std::cout<<"\n Processor n: "<<irank<<" AFTER SEND";
#endif
}

if ( which < np-1 ) 
which = which + 1;
else 
which = 1;


if( irank == which ){

std::vector<struct kpoint<T>> right_points(mypoint.begin() + half + 1, mypoint.end());

#ifdef DEBUG
#endif
if( N != 2)
#pragma omp task shared(ndim) firstprivate(left_points, myaxis)
{
node -> right = build_serial_kdtree(right_points, ndim, myaxis);
}
else
node -> right = new kdnode<T>;

std::string kdtree_str = serialize_node<T>(node -> right);
MPI_Send( kdtree_str.c_str() , kdtree_str.length() , MPI_CHAR , 0 , 20 , comm );  




#ifdef DEBUG
std::cout<<"\n Processor n: "<<irank<<" AFTER SEND";
#endif
}




}




} else if ( irank == 0) {

if ( np/2 != pow(2, level)){

std::vector<struct kpoint<T>> left_points(mypoint.begin(), mypoint.begin() + half);
std::vector<struct kpoint<T>> right_points(mypoint.begin() + half + 1, mypoint.end());

level = level + 1;
node->left = build_parallel_kdtree4( left_points, ndim, myaxis, np, level, comm, which);

which = which + 2; 
if( N != 2)
node -> right = build_parallel_kdtree4( right_points, ndim, myaxis, np, level, comm, which);
else
node -> right = new kdnode<T>;




}else{ 

#ifdef DEBUG
std::cout<<"\n Processor n "<<irank<<" BEFORE RECV ";
#endif
int flag = 0, count;

MPI_Probe(which, 10, comm, &status);  

MPI_Get_count( &status, MPI_CHAR, &count);
char *buf1 = new char[count];
MPI_Recv(buf1, count, MPI_CHAR, which, 10, comm, &status);
std::string bla1(buf1, count);
delete [] buf1;

node -> left = deserialize_node<T>(bla1);
diff = end - start;



#ifdef DEBUG
std::cout<<"\n string "<<bla1<<" \nAFTER 1 RECV ";
#endif


if ( which < np-1 ) 
which = which + 1;
else 
which = 1;


flag = 0;

MPI_Probe(which, 20, comm, &status);

MPI_Get_count( &status, MPI_CHAR, &count); 
char *buf2 = new char[count];
MPI_Recv(buf2, count, MPI_CHAR, which, 20, comm, &status);  
std::string bla2(buf2, count);
delete [] buf2;

node -> right = deserialize_node<T>(bla2);


#ifdef DEBUG
std::cout<<"\n string "<<bla2<<" \nAFTER 2 RECV ";
#endif






}


}



}

return node;

}



