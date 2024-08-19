#include <iostream>
#include <cstdlib>
#include <math.h>
#include <omp.h>

#include "kMeans.cpp"
#include "con_hull.cpp"

using namespace std;

int main()
{
int n=15000,k=3;
double t1,t2;
int x[n],y[n],convx[n],convy[n],kc[n];
srand(time(0));
cout<<"Generating Random Numbers...!!!"<<endl;
t1=omp_get_wtime();
for(int i=0;i<n;i++){
x[i]=rand()%100;
y[i]=rand()%100;
}


cout<<"Obtaining the clusters...!!"<<endl;
find_clusters(n,k,x,y,kc);

int yc[k][n],xc[k][n],clust_size[k];
for(int p=0;p<k;p++){
clust_size[p]=0;
}
{
for(int j=0;j<k;j++){
printf("Cluster %d; tid=%d\n", j+1,omp_get_thread_num());
for(int i=0;i<n;i++){
if(kc[i]==j){
printf("point %d: (%d,%d)\n",i,x[i],y[i]);
clust_size[j]+=1;
xc[j][clust_size[j]-1]=x[i];
yc[j][clust_size[j]-1]=y[i];
}
}
}
}
cout<<"Finding the convex hulls for each cluster obtained..!!"<<endl;
int con_x[n],con_y[n],con_n=0,fin_x[n],fin_y[n],fin_n=0;
for(int j=0;j<k;j++){
findHull(xc[j],yc[j],clust_size[j],con_x,con_y,&con_n);
printf("\n\nCluster %d; tid=%d\n", j+1,omp_get_thread_num());
for(int i=0;i<con_n;i++){
fin_x[fin_n+i]=con_x[i];
fin_y[fin_n+i]=con_y[i];
printf("(%d,%d)\n",con_x[i],con_y[i]);
}
fin_n+=con_n;
printf("l_n=%d,g_n=%d\n",con_n,fin_n );
}

printf("Finding the final Hull..!!\n");
for(int i=0;i<fin_n;i++)
printf("(%d,%d)\n", fin_x[i],fin_y[i]);
printf("%d\n",fin_n);
findHull(fin_x,fin_y,fin_n,con_x,con_y,&con_n);
printf("***\n");
for(int i=0;i<con_n;i++){
printf("(%d,%d)\n",con_x[i],con_y[i]);
}
t2=omp_get_wtime();
printf("Time Taken:%lf\n", t2-t1);
return 0;
}