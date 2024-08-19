#include<iostream>
#include<algorithm> 
#include<vector> 
#include<bits/stdc++.h> 
#include<random>
#include<omp.h>

using namespace std; 

const int NDIM= 2;


struct Node
{

int point[NDIM]; 

Node *left,*right; 
};


Node *kd_tree( std::vector<std::vector<double>> vect, bool myaxis, int* compt){


struct Node *newnode = new Node;
if (vect.size()==1){

newnode-> point[0] = vect[0][0];
newnode->point[1] =vect[0][1];
newnode->right = newnode->left=NULL;
*compt = *compt+1;
return newnode;
}
else{ 
int m=vect.size(); 
int l= m/2;




if(myaxis==true){ 
#pragma omp parallel shared(vect,l,m)
{
#pragma omp for ordered					
for(int i=0; i<m; i++){
#pragma omp odered 
swap(vect[i][0],vect[i][1]);}				

sort(vect.begin(),vect.end());
#pragma omp for ordered 

for(int i=0; i<m; i++){
#pragma omp ordered 
swap(vect[i][0],vect[i][1]);}

} 
}
else{ 

sort(vect.begin(),vect.end());
}

newnode->point[0]=vect[l][0];
newnode->point[1] = vect[l][1];

vector<vector<double>> left;
vector<vector<double>> right; 


for (int  i=0; i<l; i++)
{
left.push_back(vect[i]);}

for(int i=l+1; i<m;i++)
{
right.push_back(vect[i]);}	

#pragma omp parallel
{	
#pragma omp single nowait 
{
#pragma omp task
newnode->left = kd_tree(left,!myaxis, compt);
#pragma omp task
{
if(right.size()>0) 
newnode->right= kd_tree(right,!myaxis, compt);
}

}
}
return newnode;

}

}


int main(){

struct Node* root= new Node;

vector<vector<double>> vect{};
int m;
int compt{0};

for(int m =4 ; m<7;m++){
int n= pow(10,m);


#pragma omp parallel shared(vect,n)
{
int rank= omp_get_thread_num(); 
int nthrds = omp_get_num_threads(); 
int count= n/nthrds;
int start = start*count;
int stop = start + count;

#pragma omp for  
for(int i=start; i<stop; i++){

srand(i);
double a = double( rand())/double(RAND_MAX);
srand(i+10);
double b = double( rand())/double(RAND_MAX);
#pragma omp critical	
{ 
vect.push_back({a,b});
}
}
}

bool myaxis=false;
double start = omp_get_wtime(); 
root = kd_tree(vect,myaxis, &compt);
double endtime = omp_get_wtime();

cout<< m << ";" << endtime - start<<std::endl;
}
cout<< " Done .."<<endl;	
return 0;
}
