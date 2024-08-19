#include<iostream>
#include<algorithm> 
#include<vector> 
#include<bits/stdc++.h> 
#include<random>
#include<mpi.h>
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
for(int i=0; i<m; i++){
swap(vect[i][0],vect[i][1]);}				

sort(vect.begin(),vect.end());

for(int i=0; i<m; i++){
swap(vect[i][0],vect[i][1]);}

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


int main(int argc, char **argv){

int required; 

omp_set_num_threads(16);
MPI_Init_thread(&argc,&argv, MPI_THREAD_MULTIPLE,&required);

int rank =MPI::COMM_WORLD.Get_rank();
int nprc= MPI::COMM_WORLD.Get_size();
MPI_Status stats[nprc];

int m=7;

int n= pow(10,m);
struct Node* root= new Node;
vector<vector<double>> vect{};

int compt{0};
double tab[2];

double starttime{0}, endtime{0};
int count= n/nprc;
int start = rank*count;
int stop = start + count;
bool myaxis=true;
int th;	
if(rank==0){
#pragma omp parallel shared(vect,n, th)
{
int thrank= omp_get_thread_num(); 
int nthrds = omp_get_num_threads(); 
th = nthrds;
#pragma omp for  
for(int i=0; i<n; i++){				 	
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
starttime = MPI_Wtime();
struct Node* ndo = new Node;
sort(vect.begin(),vect.end()); 
int l= vect.size()/2;
std::vector<std::vector<double>> vet{};

MPI_Send(&l,1,MPI_INT,1,1,MPI_COMM_WORLD);

for(int i=l+1;i<n;i++)
{
tab[0] = vect[i][0];
tab[1] = vect[i][1];
MPI_Send(&tab,2,MPI_DOUBLE,rank+1,rank+1,MPI_COMM_WORLD); 
}


for(int i=0;i<l;i++)
vet.push_back(vect[i]);

root->point[0] = vect[l][0];
root->point[1] = vect[l][1];

vect ={};

l= vet.size()/2;
MPI_Send(&l,1,MPI_INT,3,3,MPI_COMM_WORLD);

for(int i=l+1;i<vet.size();i++)
{
tab[0] = vet[i][0];
tab[1] = vet[i][1];
MPI_Send(&tab,2,MPI_DOUBLE,rank+3,rank+3,MPI_COMM_WORLD); 
}


for(int i=0;i<l;i++)
vect.push_back(vet[i]);


ndo->point[0] = vet[l][0];
ndo->point[1] = vet[l][1];

bool myaxis=true;		
ndo->left = kd_tree(vect,myaxis, &compt); 


root->left = ndo;

cout<< " Number of leaves  " << compt<<endl;
endtime = MPI_Wtime();
cout<< rank <<","<<th<<","<<m<<","<<endtime-starttime<<endl;

}

if(rank==1){
starttime = MPI_Wtime();		
int  l;

MPI_Recv(&l,1,MPI_INT,rank-1,rank,MPI_COMM_WORLD,&stats[rank]);

for(int i=1; i<l; i++){
MPI_Recv(&tab,2,MPI_DOUBLE,rank-1,rank,MPI_COMM_WORLD,&stats[rank]);
vect.push_back({tab[0],tab[1]});
}

myaxis= true;

for(int i=0; i<vect.size(); i++){
swap(vect[i][0],vect[i][1]);}				

sort(vect.begin(),vect.end());

for(int i=0; i<vect.size(); i++){
swap(vect[i][0],vect[i][1]);}

int h= vect.size()/2;
struct Node* nd = new Node;
nd->point[0] = vect[h][0];
nd->point[1] = vect[h][0];


MPI_Send(&h,1,MPI_INT,rank+1,rank+1,MPI_COMM_WORLD);

for(int i=h+1;i<vect.size();i++)
{
tab[0] = vect[i][0];
tab[1] = vect[i][1];
MPI_Send(&tab,2,MPI_DOUBLE,rank+1,rank+1,MPI_COMM_WORLD); 
}

std::vector<std::vector<double>> vet{};

for(int i=0;i<h;i++)
vet.push_back(vect[i]);

nd->left = kd_tree(vet,myaxis, &compt);
root->left=nd;
cout<< " Number of leaves  " << compt<<endl;
endtime = MPI_Wtime();
cout<< rank<<","<<m<<","<<endtime-starttime<<endl;

}

if(rank==3){
starttime = MPI_Wtime();		
int  l;

MPI_Recv(&l,1,MPI_INT,0,rank,MPI_COMM_WORLD,&stats[rank]);

for(int i=1; i<l; i++){
MPI_Recv(&tab,2,MPI_DOUBLE,0,rank,MPI_COMM_WORLD,&stats[rank]);
vect.push_back({tab[0],tab[1]});
}

myaxis= true;

for(int i=0; i<vect.size(); i++){
swap(vect[i][0],vect[i][1]);}				

sort(vect.begin(),vect.end());

for(int i=0; i<vect.size(); i++){
swap(vect[i][0],vect[i][1]);}

root = kd_tree(vect,myaxis, &compt);
cout<< " Number of leaves  " << compt<<endl;
endtime = MPI_Wtime();
cout<< rank<<","<<m<<","<<endtime-starttime<<endl;

}


if(rank==2 ){
starttime = MPI_Wtime();		
int  l;		
MPI_Recv(&l,1,MPI_INT,1,2,MPI_COMM_WORLD,&stats[rank]);

for(int i=1; i<l; i++){
MPI_Recv(&tab,2,MPI_DOUBLE,1,rank,MPI_COMM_WORLD,&stats[rank]);
vect.push_back({tab[0],tab[1]});
}

myaxis = false;
sort(vect.begin(),vect.end());

root = kd_tree(vect,myaxis, &compt);
cout<< " Number of leaves  " << compt<<endl;
endtime = MPI_Wtime();
cout<< rank<<","<<m<<","<<endtime-starttime<<endl;

}

MPI::Finalize();

cout<< " Done .."<<endl;

return 0;
}
