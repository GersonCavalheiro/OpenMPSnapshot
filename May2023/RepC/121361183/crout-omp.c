int calculez_crout_normal_omp(int dim,int thread,double **mat,double *x,double *libre)
{
long i,k,j,p;
double **U,**L;
double *pmat;
omp_set_num_threads(thread);
U=(double **)calloc(dim,sizeof(double *));
pmat=(double *)calloc(dim*dim,sizeof(double));
for(i=0;i<dim;i++)
{
U[i]=pmat;
pmat+=dim;
}
L=(double **)calloc(dim,sizeof(double *));
pmat=(double *)calloc(dim*dim,sizeof(double));
for(i=0;i<dim;i++)
{
L[i]=pmat;
pmat+=dim;
}
for(k=0;k<dim;k++)
{
for(i=k;i<dim;i++)
{
L[i][k]=mat[i][k];
U[k][i]=mat[k][i];
#pragma omp for
for(p=0;p<k;p++) 
{
L[i][k]-=L[i][p]*U[p][k];
U[k][i]-=L[k][p]*U[p][i];
}
if(L[k][k]==0.0)
{
printf("Impartire prin zero %d\n",k);
fflush(stdout);
}
U[k][i]=U[k][i]/L[k][k];
}
}
for(i=0;i<dim;i++)
{
x[i]=libre[i];
for(j=0;j<i;j++)  x[i]-=L[i][j]*x[j];
x[i]=x[i]/L[i][i];
}
for(i=dim-1;i>=0;i--)
{
for(j=dim-1;j>=i+1;j--) x[i]-=U[i][j]*x[j];
x[i]=x[i]/U[i][i];
}
free(*U);
free(*L);
free(U);
free(L);
return(0);
}
