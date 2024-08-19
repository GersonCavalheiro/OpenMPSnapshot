#include<iostream>
#include<algorithm> 
#include<vector> 
#include<bits/stdc++.h> 
#include<random>
#include<mpi.h>

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
{
for(int i=0; i<m; i++){
swap(vect[i][0],vect[i][1]);}				

sort(vect.begin(),vect.end());

for(int i=0; i<m; i++){
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

{	
{
newnode->left = kd_tree(left,!myaxis, compt);
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
MPI_Init(&argc,&argv);
int rank =MPI::COMM_WORLD.Get_rank();
int nprc= MPI::COMM_WORLD.Get_size();
MPI_Status stats[nprc];

for(int m =4 ; m<5;m++){

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
if(rank==0){
for(int i=0; i<n; i++){				 	
srand(i);
double a = double( rand())/double(RAND_MAX);
srand(i+10);
double b = double( rand())/double(RAND_MAX);
vect.push_back({a,b});
}
starttime = MPI_Wtime();

sort(vect.begin(),vect.end()); 
int l= vect.size()/2;
MPI_Send(&l,1,MPI_INT,1,1,MPI_COMM_WORLD);

for(int i=l+1;i<n;i++)
{
tab[0] = vect[i][0];
tab[1] = vect[i][1];
MPI_Send(&tab,2,MPI_DOUBLE,1,1,MPI_COMM_WORLD); 
}

std::vector<std::vector<double>> vet{};

for(int i=0;i<l;i++)
vet.push_back(vect[i]);
bool myaxis=false;		
root = kd_tree(vet,myaxis, &compt);
cout<< " Number of leaves  " << compt<<endl;
endtime = MPI_Wtime();
cout<< rank<<","<<m<<","<<endtime-starttime<<endl;

}

if(rank ==1){
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
MPI_Send(&h,1,MPI_INT,2,2,MPI_COMM_WORLD);

for(int i=h+1;i<vect.size();i++)
{
tab[0] = vect[i][0];
tab[1] = vect[i][1];
MPI_Send(&tab,2,MPI_DOUBLE,2,2,MPI_COMM_WORLD); 
}

std::vector<std::vector<double>> vet{};

for(int i=0;i<h;i++)
vet.push_back(vect[i]);

root = kd_tree(vet,myaxis, &compt);
cout<< " Number of leaves  " << compt<<endl;
endtime = MPI_Wtime();
cout<< rank<<","<<m<<","<<endtime-starttime<<endl;

}

if(rank ==2){
starttime = MPI_Wtime();		
int  h;		
MPI_Recv(&h,1,MPI_INT,rank-1,rank,MPI_COMM_WORLD,&stats[rank]);

for(int i=1; i<h; i++){
MPI_Recv(&tab,2,MPI_DOUBLE,1,2,MPI_COMM_WORLD,&stats[rank]);
vect.push_back({tab[0],tab[1]});
}

myaxis = false;
sort(vect.begin(),vect.end());




root = kd_tree(vect,myaxis, &compt);
cout<< " Number of leaves  " << compt<<endl;
endtime = MPI_Wtime();
cout<< rank<<","<<m<<","<<endtime-starttime<<endl;

}
} 
MPI::Finalize();

cout<< " Done .."<<endl;	
return 0;
}
