
#include <omp.h>
#define N 100
#define ENABLE_SHOW 1
void ShowMatrix(float A[N][N],int n)
{
int i,j;
if (ENABLE_SHOW)
for (i=0;i<n;i++)
{
for (j=0;j<n;j++)
printf("%.2f ",A[i][j]);
printf("\n");
}
}
void ShowMatrixCol(float *b,int n)
{
int i;
if (ENABLE_SHOW)
for (i=0;i<n;i++)
printf("%.2f\n",b[i]);
}
void ShowMatrixCol_int(int *b,int n)
{
int i;
if (ENABLE_SHOW)
{
for (i=0;i<n;i++)
printf("%d  ",b[i]);
printf("\n");
}
}
void VerificaEgalitate(float a[N],float b[N],int n)
{
int i;
if (ENABLE_SHOW)
for (i=0;i<n;i++)
printf("%d  :  %.2f == %.2f \n",i,a[i],b[i]);
}
int Allflags_eq_2(int f[N],int n)
{
int i;
for (i=0;i<n;i++)
if (f[i]!=2)
return 0;
return 1;
}
int main()
{
float A[N][N],Ai[N][N];
float b[N],bi[N],y[N];
float x[N],v[N];
float AA[16]={1,2,1,5,
3,-1,2,-2,
2,3,-4,1,
-1,8,1,-2};
float bb[4]={9,2,2,6};
int n,i,j,k;
int flags[N];
float s_l[N]; 
float l_d[N]; 
float s_v_r; 
int s_index; 
int t,divizat=0;
int tv; 
int ild;
int did_something; 
printf("Numarul de threaduri : %d", omp_get_max_threads());
printf("\nDati n < %d :",N);
scanf("%d",&n);
for (i=0;i<n;i++)
{
for (j=0;j<n;j++){
A[i][j]=AA[i*4+j];
Ai[i][j]=A[i][j];
}
b[i]=bb[i];
bi[i]=b[i];
y[i]=0;
}



ShowMatrix(A,n);
ShowMatrixCol(b,n);
s_index=-1;
for (i=0;i<n;i++)
{
s_l[i]=1;
flags[i]=0;
}
ild=0;
flags[0]=1;
do
{
#pragma omp parallel for schedule(static,1) shared(A,b,y,n,s_l, flags,s_v_r , s_index) private(j,t,did_something)
for (k=0;k<n;k++)
{
t=omp_get_thread_num();
did_something=0;
do
{
if (flags[k]==1) 
{
printf("\nThread %d folosit..   k= %d .. divizare\n",t,k);
for (j=k+1;j<n;j++)
{
A[k][j]=A[k][j]/A[k][k];
s_l[j]=A[k][j];
}
s_l[k]=1;
y[k]=b[k]/A[k][k];
A[k][k]=1;
s_v_r=y[k];
s_index=k;
flags[k]=2;
for (j=k+1;j<n;j++)
flags[j]=3; 
ShowMatrixCol_int(flags,n);
did_something = 1;
flags[k+1]=4; 
} 
if (flags[k]==3) 
{
printf("\nThread %d folosit..   k = %d .. eliminare cu linia %d\n",t,k,s_index);
for (j=s_index+1;j<n;j++)
{
A[k][j]=A[k][j]-A[k][s_index]*s_l[j];
}
b[k]=b[k]-A[k][s_index]*s_v_r;
A[k][s_index]=0;
flags[k]=0; 
did_something = 1;
}
if (flags[k]==4) 
{
printf("\nThread %d folosit..   k = %d .. eliminare cu linia %d\n",t,k,s_index);
for (j=s_index+1;j<n;j++)
{
A[k][j]=A[k][j]-A[k][s_index]*s_l[j];
}
b[k]=b[k]-A[k][s_index]*s_v_r;
A[k][s_index]=0;
flags[k]=1; 
did_something = 1;
}
if (flags[k]==2)
did_something=1;
}
while (!did_something);
} 
ild++;
}
while (!Allflags_eq_2(flags,n)); 
ShowMatrix(A,n);
ShowMatrixCol(y,n);
for (k=n-1;k>=0;k--)
{
x[k]=y[k];
for(j=k+1;j<n;j++)
{
x[k]-=A[k][j]*x[j];
}
}
printf("\nSolutiile sunt:\n");
ShowMatrixCol(x,n);
for (k=0;k<n;k++)
{
v[k]=0;
for (j=0;j<n;j++)
{
v[k]+=Ai[k][j]*x[j];
}
}
printf("\nVerificare:\n");
VerificaEgalitate(bi,v,n);
return 0;
}
