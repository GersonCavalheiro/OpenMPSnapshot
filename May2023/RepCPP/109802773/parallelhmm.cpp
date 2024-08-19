#include <stdio.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <omp.h>

using namespace std;

double forward(string[100],int,double[100][100],double[100][100],double[100]);
double backward(string[100],int,double[100][100],double[100][100],double[100]);
void forward_backward(string[100],int,int,double[100][100],double[100][100],double[100]);
void viterbi(string[100],int,double[100][100],double[100][100],double[100]);
void read(int &,int &,double[100][100],double[100][100],double[100]);
void display(int ,int ,double[100][100],double[100][100],double[100]);

double fwd[100][100];
double bwk[100][100];
int length = 0;


int main()
{
double a[100][100],b[100][100],pi[100];
int i,j,n,m;
read(n,m,a,b,pi);


cout<<n<<" "<<m<<endl;
cout<<endl;

for(i=0;i<n;i++)
{
for(j=0;j<n;j++)
{

cout<<a[i][j]<<'\t';
}
cout<<endl;			
}

cout<<endl;

for(i=0;i<n;i++)
{
for(j=0;j<m;j++)
{

cout<<b[i][j]<<'\t';
}
cout<<endl;
}

cout<<endl;

for(i=0;i<n;i++)
{
cout<<pi[i]<<"\t";
}

cout<<endl<<endl;

ifstream f("observables.txt");
string line;
string observations[100];
char ch;

while(!f.eof())
{
getline(f,line);

length = 0;

for(i=0;i<line.length();i++)
{

ch=line[i];

j=(int)ch;

if(j==49)
{
observations[length++]="Heads";
}

else if(j==48)
{

observations[length++]="Tails";
}

}

viterbi(observations,n,a,b,pi);
clock_t begin = clock();
forward_backward(observations,n,m,a,b,pi);
clock_t end = clock();
display(n,m,a,b,pi);
double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
cout<< "time elapsed is - " << elapsed_secs;
cout<<endl;

}
cout<<endl;
f.close();


return 0;
}

void display(int n,int m,double a[100][100],double b[100][100],double pi[100])
{
int i,j;

cout<< "New A is"<<endl;
for(i=0;i<n;i++)
{
for(j=0;j<n;j++)
{

cout<<a[i][j]<<'\t';
}
cout<<endl;			
}

cout<<endl;
cout<< "New B is"<<endl;
for(i=0;i<n;i++)
{
for(j=0;j<m;j++)
{

cout<<b[i][j]<<'\t';
}
cout<<endl;
}

cout<<endl;
cout<< "New pi is"<<endl;
for(i=0;i<n;i++)
{
cout<<pi[i]<<"\t";
}

cout<<endl;
}

void read(int &n,int &m,double a[100][100],double b[100][100],double pi[100])
{
int i,j;

string line;

ifstream read("input.txt");


while(!read.eof())
{
getline(read,line);

if(line=="a")
{
read>>n>>m;

}

else if(line=="b")
{
for(i=0;i<n;i++)
{
for(j=0;j<n;j++)
{
read>>a[i][j];
}			
}
}

else if(line=="c")
{
for(i=0;i<n;i++)
{
for(j=0;j<m;j++)
{
read>>b[i][j];
}
}
}

else if(line=="d")
{
for(i=0;i<n;i++)
{
read>>pi[i];
}
}
}

read.close();
}

void forward_backward(string observations[100],int n,int m,double a[100][100],double b[100][100],double pi[100])
{
double gamma[100][100];
double zi[100][100][100];
int i,j,k;

double p_obs,b_obs,val,sum,sum1;

sum = 0.0;
sum1 = 0.0;

p_obs=forward(observations,n,a,b,pi);

b_obs=backward(observations,n,a,b,pi);
#pragma omp parallel for private(j,k)
for(i=0;i<length;i++)
{
for(j=0;j<n;j++)
{
gamma[i][j] = ( fwd[i][j] * bwk[i][j] ) / (p_obs );

if(i==0)
{
pi[j] = gamma[i][j];
}

if(i==length-1)
{
continue;
}



for(k=0;k<n;k++)
{
if(observations[i+1]=="Heads")
{
zi[i][j][k] = fwd[i][j] * a[j][k] * b[k][0] * bwk[i+1][k] / p_obs ;
}
else
{
zi[i][j][k] = fwd[i][j] * a[j][k] * b[k][1] * bwk[i+1][k] / p_obs ;
}
}
}
}

#pragma omp parallel for reduction(+:sum,sum1)
for(i=0;i<n;i++)
{
for(j=0;j<n;j++)
{
sum=0.0;
sum1=0.0;


for(k=0;k<length-1;k++)
{
sum = sum + zi[k][i][j];
sum1 = sum1 + gamma[k][i];
}


a[i][j] = sum/sum1;
}
}

sum = 0.0;
int t;

for(i=0;i<n;i++)
{
for(j=0;j<m;j++)
{
val= 0.0;
sum=0.0;
for(k=0;k<length;k++)
{
if(observations[k]=="Heads")
t=0;
else
t=1;

if(t == j)
{
val = val + gamma[k][i];
}
sum = sum + gamma[k][i];
}


b[i][j] = val/sum;
}
}

}

double forward(string observations[100],int n,double a[100][100],double b[100][100],double pi[100])
{
int i,j,k;
double value,sum,prob;

value = 0.0;
sum = 0.0;
#pragma omp parallel for shared(observations)
for(i=0;i<n;i++)
{
if(observations[0]=="Heads")
{
fwd[0][i] = pi[i] * b[i][0];
}

else
{
fwd[0][i] = pi[i] * b[i][1];
}

}

for(i=1;i<length;i++)
{
for(j=0;j<n;j++)
{
value=0.0;

for(k=0;k<n;k++)
{
if(observations[i]=="Heads")
{
value = value + fwd[i-1][k] * a[k][j] * b[j][0];
}

else
{
value = value + fwd[i-1][k] * a[k][j] * b[j][1];

}


}


fwd[i][j]=value;
}
}
#pragma omp parallel for reduction(+:sum)
for(i=0;i<n;i++)
{
sum = sum + fwd[length-1][i];

}

prob = sum;


return prob;

}

double backward(string observations[100],int n,double a[100][100],double b[100][100],double pi[100])
{
int i,j,k;
double value,prob,sum;

sum = 0.0;
#pragma omp parallel for
for(i=0;i<n;i++)
{
bwk[length-1][i]=1;
}

for(i=length-2;i>=0;i--)
{
for(j=0;j<n;j++)
{
value = 0.0;

for(k=0;k<n;k++)
{
if(observations[i+1]=="Heads")
{
value = value + bwk[i+1][k] * a[j][k] * b[k][0];
}

else
{
value = value + bwk[i+1][k] * a[j][k] * b[k][1];
}
}

bwk[i][j] = value;
}
}
#pragma omp parallel for shared(observations) reduction(+:sum)
for(i=0;i<n;i++)
{
if(observations[0]=="Heads")
{
sum = sum + pi[i] * b[i][0] * bwk[0][i];
}

else
{
sum = sum + pi[i] * b[i][1] * bwk[0][i];
}
}


prob = sum;

return prob;

}

void viterbi(string observations[100],int n,double a[100][100],double b[100][100],double pi[100])
{

int *seq = new int[length];
for (int i = 0; i < length; i++)
seq[i] = 0;

double **prob = new double*[length];
int **prevs = new int*[length];

for (int i = 0; i < length; i++) 
{
prob[i] = new double[n];
prevs[i] = new int[n];

}

for (int i = 0; i < n; i++) 
{
if(observations[0]=="Heads")
{
prob[0][i] = pi[i] * b[i][0];
}

else
{
prob[0][i] = pi[i] * b[i][1];
}
}

for (int i = 1; i < length; i++) 
{
for (int j = 0; j < n; j++) 
{
double pmax = 0, p; int dmax;
for (int k = 0; k < n; k++) 
{
p = prob[i-1][k] * a[k][j];
if (p > pmax) 
{
pmax = p;
dmax = k;
}
}

if(observations[i]=="Head")
{
prob[i][j] = b[j][0] * pmax;
}

else
{
prob[i][j] = b[j][1] * pmax;
}
prevs[i-1][j] = dmax;
}
}

double pmax = 0; int dmax;
for (int i = 0; i < n; i++) 
{
if (prob[length-1][i] > pmax) 
{
pmax = prob[length-1][i];
dmax = i;
}
}
seq[length-1] = dmax;

for (int i = length-2; i >= 0; i--) 
{
seq[i] = prevs[i][ seq[i+1] ];
}


cout << endl;
for (int i = 0; i < length; i++) 
cout << seq[i] << ' ';
cout << endl;


for (int i = 0; i < length; i++) 
{
delete[] prob[i];
delete[] prevs[i];
}
delete[] prob;
delete[] prevs;

}

