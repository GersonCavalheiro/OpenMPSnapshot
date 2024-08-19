#include <cluster.h>
#include<pthread.h>
#include<omp.h>
pthread_mutex_t lockSet;
omp_lock_t writelock;

int numThreads =2;
int k;
int pointsChange;
vector<int> clusters;
vector<point> vec;
vector<point> centroids;

cluster::cluster(int val,int mode){
k = val; 
pointsChange = -1; 
dataPoints = -1; 
mod = mode;
if(mode==1){
cout << "setup for pthreads \n";
}else if(mode==2){
cout << "setup for openmp\n";
}
}

void cluster::readData(){
cout <<"start\n";
freopen("./dataset/input1.csv", "r", stdin);
string s,delimiter=",";
getline(cin,s);
getline(cin,s);
while(getline(cin,s)){
size_t pos = 0;
std::string token;
vector<float> v;
while ((pos = s.find(delimiter)) != std::string::npos) {
token = s.substr(0, pos);
v.push_back(stod(token));
s.erase(0, pos + delimiter.length());
}
v.push_back(stod(s));
point pnew(v[0],v[1],v[2]);
vec.push_back(pnew);
clusters.push_back(-1);
dataPoints++;
}
cout << "data Points Collected :: "<<dataPoints+1<<endl;
}

void cluster::printData(){
for(int i=0;i<dataPoints;i++){
cout << vec[i].x <<" "<<vec[i].y<<" "<<vec[i].z<<endl;
}
}

cluster::~cluster(){
clusters.clear();
centroids.clear();
vec.clear();
cout << " Destroyer Called "<<endl;
}

float getDistance(point p1,point p2){
return p1.distance(p2);
}

void cluster::init_random(){
random_shuffle(vec.begin(),vec.end());
if(centroids.size() > 0)
throw invalid_argument("centroids are already fetched\n");
if(k>vec.size())
throw invalid_argument("groups is more than number of elements");

for(int i=0;i<k;i++){
centroids.push_back(vec[i]);
}
}

void cluster::train(){
pthread_mutex_init(&lockSet, NULL);
bool run = true;
int iterationNum=0;
init_random();
while(run){
cout << "Iteration Number ::"<< ++iterationNum <<endl;
getClusters();
updateCentroids();
if(Converge(iterationNum)){
break;
}
}

printCentroids();
}

int classChecker(int i){
int clas = -1;
float distance = 1e7;
for(int j=0;j<k;j++){
float dist = getDistance(vec[i],centroids[j]);
if(dist < distance){
distance = dist;
clas = j;
}
}
return clas;
}

void* clusterSetter(void* tid){
int *ti = (int*)(tid);
int intialDis = *ti*(vec.size()/numThreads);
int i = 0 + intialDis;
int end;
int changedPoints =0;
if(*ti!=numThreads-1)
end = (*ti+1)*(vec.size()/numThreads);
else
end = vec.size();
for(;i<end;i++){
int clas = clusters[i];
clas = classChecker(i);
if(clusters[i] != clas)
changedPoints++;
clusters[i] = clas;
}
pthread_mutex_lock(&lockSet);
pointsChange += changedPoints;
pthread_mutex_unlock(&lockSet);
return NULL;
}
void clusterSetterOMP(){
int tid = omp_get_thread_num();	
int intialDis = tid*(vec.size()/numThreads);
int i = 0 + intialDis;
int end;
int changedPoints =0;
if(tid!=numThreads-1)
end = (tid+1)*(vec.size()/numThreads);
else
end = vec.size();
for(;i<end;i++){
int clas = clusters[i];
clas = classChecker(i);
if(clusters[i] != clas)
changedPoints++;
clusters[i] = clas;
}
#pragma omp critical
{
pointsChange += changedPoints;
}

}

void cluster::getClusters(){

if(mod == 0){
int changedPoints = 0;
for(int i=0;i<vec.size();i++){
int clas = clusters[i];
clas = classChecker(i);
if(clas == -1){
cout << "exited due to class -1\n"; 
}
if(clusters[i] != clas)
changedPoints++;
clusters[i] = clas;	
}
pointsChange = changedPoints;	
cout <<"Points Changed :: "<< pointsChange<<endl;
}else if(mod == 1){
pointsChange = 0;
pthread_t clusterSet[numThreads];
int  static tid[10];
for(int i=0;i<numThreads;i++){
tid[i] = i;
pthread_create(&clusterSet[i],NULL,&clusterSetter,&tid[i]);
}
for(int i=0;i<numThreads;i++){
pthread_join(clusterSet[i],NULL);
}
}else{
pointsChange = 0;
omp_init_lock(&writelock);
#pragma omp parallel num_threads(numThreads)
{
clusterSetterOMP();
}
}
}

void* centroidUpdate(void* tid){
int *ti = (int*)(tid);

point pmean(0.0,0.0,0.0);
int numPoints = 1;
for(int j=0;j<vec.size();j++){
if(clusters[j]==*ti){
pmean.runningMean(vec[j],numPoints);
numPoints++;
}
}
centroids[*ti] = pmean;

return NULL;
}

void cluster::updateCentroids(){
if(true){
for(int i=0;i<k;i++){
point pmean(0.0,0.0,0.0);
int numPoints = 1;
for(int j=0;j<dataPoints;j++){
if(clusters[j]==i){
pmean.runningMean(vec[j],numPoints);
numPoints++;
}
}
centroids[i] = pmean;
}
}else{
pthread_t centroidUp[numThreads];
int tid[k];
for(int i=0;i<k;i++){
tid[i] = i;
pthread_create(&centroidUp[i],NULL,&centroidUpdate,&tid[i]);
}
for(int i=0;i<numThreads;i++){
pthread_join(centroidUp[i],NULL);
}
}
}

void cluster::printCentroids(){
for(int i=0;i<k;i++){
centroids[i].print();
}
}


bool cluster::Converge(int iterationNumber){
if(pointsChange <= dataPoints*0.00)
return true;
if(iterationNumber > 100)
return true;
return false;

}

double runningDistance(double tillNow,long num,float newdist){

return ((float)(num-1)/num)*tillNow+ ( (float)(1)/(float)num )*newdist;
}
double cluster::kScore(){
double kscore = 0;
double subScore = 0;
vector<int> clusterIdMap[k];
for(int i=0;i<dataPoints;i++){
clusterIdMap[clusters[i]].push_back(i);
}
for(int i=0;i<k;i++){
cout << "Cluster ID "<<i+1<<" :: "<<clusterIdMap[i].size()<<endl;
float dist =0;
long count=0;
for(int j=0;j<clusterIdMap[i].size();j++){
for(int k=j+1;k<clusterIdMap[i].size();k++){
dist = getDistance( vec.at(clusterIdMap[i][j]),vec[ clusterIdMap[i][k] ]);
count++;
subScore = runningDistance(subScore,count,dist);				
}
}
subScore *= (float)(clusterIdMap[i].size())/4;
cout << "SubScore :: "<<subScore<< endl;
kscore += subScore;
}
cout << "KScore for K "<<k<<" :: "<<kscore<<endl;

}



