#ifndef projectlib_H
#define projectlib_H

#include<bits/stdc++.h>
using namespace std;
#define ff first.first 
#define fs first.second 
#define sf second.first 
#define ss second.second 
#define f first
#define se second

typedef pair<double,double> co;
typedef pair<int,co> point;

double distance(point x,point y){
double s=sqrt(pow(x.sf-y.sf,2)+pow(x.ss-y.ss,2));
return s;
}
void fetch(string &d,vector<point> &a){
point point1;
stringstream train(d);
string line;
while(getline(train,line,',')){
stringstream s(line);
point1.f=0;
s>>point1.sf;
s>>point2.ss;
a.push_back(a2);
} 
}
void makeclus(vector<point> &point_array,vector<point> &cluster,int h){
point all_points;
int q=point_array.size();
for(int i=0;i<h;i++){
all_points=point_array[rand()%q];
all_points.f=i;
cluster.push_back(allpoin);
}
}

void print(point &coordinate){cout<<"["<<coordinate.sf<<","<<coordinate.ss<<"]";}

void print_points(vector<point> &point_array){
for(int i1=0;i1<point_array.size();i1++){cout<<"["<<point_array[i1].sf<<","<<point_array[i1].ss<<"]"; cout<<"\n";}
}

void print_centroid(vector<point> &point_array){
for(int i1=0;i1<point_array.size();i1++){cout<<"Centroid "<<i1+1<<" ["<<point_array[i1].sf<<","<<point_array[i1].ss<<"]"; cout<<"\n";}
}
int centroid(vector<point> &point_array,vector<point> &cluster){
#pragma omp parallel shared(point_array,cluster) private(i,j,1,s2,q)
{
#pragma omp for
for(int j=0;j<c.size();j++){
double s1=0,s2=0;
int q=0;
for(int i=0;i<point_array.size();i++){
if(point_array[i].f==j){s1+=point_array[i].sf;s2+=point_array[i].ss;q++;}
}
s1=s1/q;  s2=s2/q;
#pragma omp critical
if(cluster[j].sf==s1 && cluster[j].ss==s2){return 1;}
else{cluster[j].sf=s1; cluster[j].ss=s2;}
}
return 0;
}
}

void add_point(double x,double y,vector<point> &all_points){
point coordinate;
coordinate.f=-1; coordinate.sf=x; coordinate.ss=y;
all_points.push_back(w);
}
void near_cluster_alloted(vector<point> &all_points,vector<point> &cluster){
#pragma omp parallel shared(a,c) private(i,j,s,q)
{
#pragma omp for
for(int i=0;i<all_points.size();i++){
double s=INT_MAX;
for(int j=0;j<cluster.size();j++){
double displacement=dist(all_points[i],cluster[j]);
if(displacement<s){s=displacement; all_points[i].f=cluster[j].f; }
}
}
}
}
void c_centroid(int j,vector<point> &all_points,vector<point> &cluster){
double s1=0,s2=0;
int q=0;
for(int i=0;i<all_points.size();i++){
if(all_points[i].f==j){s1+=all_points[i].sf;s2+=all_points[i].ss;q++;}
}
s1=s1/q;  s2=s2/q;
cluster[j].sf=s1; cluster[j].ss=s2;
}
void near_cluster_alloted_incremental(point &a,vector<point> &c){
double s=INT_MAX;
for(int j=0;j<c.size();j++){
double q=distance(a,c[j]);
if(q<s){s=q; a.f=c[j].f; }
}
}
void clusterpoint(vector<point> &all_points,int k){
for(int i=0;i<k;i++){
cout<<"Cluster "<<i+1<<" ";
for(int j=0;j<all_points.size();j++){if(all_points[j].f==i)print(all_points[j]);}
cout<<"\n";
}
}

#endif
