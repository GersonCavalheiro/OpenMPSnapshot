#include<algorithm>
#include <fstream>
#include <string>

#include <iostream>
#include <sstream>
#include "Utils.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

const char rc='2';
const char bc='1';
const char cv='0';


traffico::traffico(unsigned int r,unsigned int c,const char *filename_X)
{
ifstream file_X;
string line;
rig=r;
cols=c;
v=new char[rig*cols];
unsigned int contatore=0;

file_X.open(filename_X);
if (file_X.is_open())
{
unsigned int i=0;
unsigned int length=0;
unsigned int n=0;
string line1;
getline(file_X, line1);
length=line1.length();

while(i<length)
{
n=atoi(line1.substr(0,line1.find(',')).c_str());
vec.push_back(n);
i+=line1.substr(0,line1.find(',')).length()+1;
line1=line1.substr(line1.find(',')+1);
}

sort(vec.begin(),vec.end());    
while(!file_X.eof())
{
getline(file_X, line);
i=0;
while(i<line.size())
{
v[contatore]=line[i];
switch(line[i])
{
case '0':
break;
case '1':
blucar.push_back(((contatore-(contatore/cols)*cols)*rig)+(contatore/cols));
break;
case '2':
redcar.push_back(contatore);
break;
default:
cout<<"dati non validi";
exit(1);
}
i+=2;
++contatore;
}
}
file_X.close();
}
sort(blucar.begin(),blucar.end());
unsigned int n=1;
unsigned int i=0;
unsigned int cont=0;
numsurig.push_back(0);
while(i<redcar.size())
{
while(i<redcar.size() && redcar[i]<n*cols)
{
++cont;
++i;
}
numsurig.push_back(cont);
++n;
}
while(numsurig.size()<rig+1)
{
numsurig.push_back(cont);
}



n=1;i=0;cont=0;
numcol.push_back(0);
while(i<blucar.size())
{
while(i<blucar.size()&&blucar[i]<n*rig )
{
++cont;
++i;
}
numcol.push_back(cont);
++n;
}
while(numcol.size()<cols+1)
{
numcol.push_back(cont);
}
}

traffico::~traffico()
{
delete [] v;
}

void traffico::stampa()
{
for(unsigned int i=0;i<rig;i++)
{
for(unsigned int j=0;j<cols;j++)
{
cout<<v[j+i*cols];cout<<' ';
}
cout<<std::endl;
}
}

void traffico::ordinared(unsigned int ri)
{
for(unsigned int contpercol=numsurig[ri/cols+1]-1;contpercol>numsurig[ri/cols];contpercol--)
{
swap(redcar[contpercol],redcar[contpercol-1]);
}
}

bool traffico::red()
{
bool Ftot=false;

#pragma omp parallel for schedule(guided)

for(unsigned int j=0;j<rig;++j)
{
int prig=-1;
if(numsurig[j]<redcar.size() && redcar[numsurig[j]]%cols==0)
{
prig=redcar[numsurig[j]];
}
for(unsigned int i=numsurig[j];i<numsurig[j+1];++i)
{
if((redcar[i]+1)%cols!=0)
{
if(v[redcar[i]+1]==cv)
{
if(!Ftot)
{
Ftot=true;
}
v[redcar[i]]=cv;
v[redcar[i]+1]=rc;
++redcar[i];
}
}
else
{
if(v[redcar[i]-cols+1]==cv&&redcar[i]+1!=prig+cols)
{
if(!Ftot)
{
Ftot=true;
}
v[redcar[i]]=cv;
v[redcar[i]-cols+1]=rc;
redcar[i]-=cols-1;
ordinared(redcar[i]);
}
}
}

}
return Ftot;
}

void traffico::ordinablu(unsigned int bi)
{
for(unsigned int contperrig=numcol[bi/rig+1]-1;contperrig>numcol[bi/rig];contperrig--)
{
swap(blucar[contperrig],blucar[contperrig-1]);
}
}

bool traffico::blu()
{
bool Ttot=false;

#pragma omp parallel for schedule(guided)

for(unsigned int j=0;j<cols;++j)
{
int prig=-1;
if(numcol[j]<blucar.size()&&blucar[numcol[j]]%rig==0)
{
prig=blucar[numcol[j]];
}
unsigned int trasf=0;
for(unsigned int i=numcol[j];i<numcol[j+1];++i)
{
trasf=(blucar[i]-(blucar[i]/rig)*rig)*cols+(blucar[i]/rig);
if((blucar[i]+1)%rig!=0)
{
if(v[trasf+cols]==cv)
{
if(!Ttot)
{
Ttot=true;
}
v[trasf]=cv;
v[trasf+cols]=bc;
++blucar[i];
}
}
else
{
if(v[trasf-cols*(rig-1)]==cv&&blucar[i]+1!=prig+rig)
{
if(!Ttot)
{
Ttot=true;
}
v[trasf]=cv;
v[trasf-cols*(rig-1)]=bc;
blucar[i]-=rig-1;
ordinablu(blucar[i]);
}
}
}    
}
return Ttot;
}


void traffico::scambi(unsigned int o,unsigned int n)
{
bool B=false;bool R=false;unsigned int cont=0;
if(o==0||vec[o-1]%2==0)
{
while(cont<n)
{
B=blu();
++cont;
if(cont==n)
break;
R=red();
++cont;
if(B==false && R==false)
{
cout<<"Matrix stopped at step "<<cont<<std::endl;
break;
}
}
}

else
{
while(cont<n)
{
R=red();
++cont;
if(cont==n)
break;
B=blu();
++cont;
if(B==false && R==false)
{
cout<<"Matrix stopped at step "<<cont<<std::endl;
break;
}
}
}
}



void traffico::stampafile(unsigned int n)
{
stringstream c;
string s;
c<<vec[n];
c<<".csv";
s=c.str();
cout<<c.str();
ofstream outfile(s.c_str());
for(unsigned int h=0;h<rig*cols;++h)
{
outfile<<v[h];
if((h+1)%cols!=0)
{
outfile<<',';
}
else if (h!=rig*cols-1)
{
outfile<<std::endl;
}
}
outfile.close();
}

void traffico::fai()
{
int pre=0;
for(unsigned int c=0;c<vec.size();++c)
{
scambi(c,vec[c]-pre);
stampafile(c);
cout<<std::endl;
pre=vec[c];
}
}



dimension::dimension(const char *filename_X)
{  
ifstream file_X;
file_X.open(filename_X);
if(file_X.is_open())
{
string line;
rig=0;
cols=0;
getline(file_X,line);
getline(file_X, line);
cols=(line.size()+1)/2;
unsigned int lunghezza=line.size();
while(!file_X.eof())
{
if(lunghezza!=line.size())
{
cout<<"righe di diversa lunghezza o vuote";
exit(1);
}
++rig;
getline(file_X, line);

}

}
file_X.close();
}




inline unsigned int dimension:: colonne()
{
return cols;
}

inline unsigned int dimension:: righe()
{
return rig;
}

int main()
{

dimension b("problem.csv");
traffico a(b.righe(),b.colonne(),"problem.csv");
a.fai();
return 0;
}
