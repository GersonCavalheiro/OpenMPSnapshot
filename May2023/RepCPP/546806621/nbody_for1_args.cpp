#include <vector>
#include <random>
#include "omp.h"
#include <iostream>
#include <chrono> 

struct particle {
float x, y, z; 
float vx, vy, vz; 
float ax, ay, az; 
};

typedef std::vector<particle> particles;

void forces(particles &plist) {
int n = plist.size();
#pragma omp parallel for 
for(int i=0; i<n; ++i) { 
plist[i].ax = plist[i].ay = plist[i].az = 0; 
for(int j=0; j<n; ++j) { 
if (i==j) continue; 
auto dx = plist[j].x - plist[i].x;
auto dy = plist[j].y - plist[i].y;
auto dz = plist[j].z - plist[i].z;
auto r = sqrt(dx*dx + dy*dy + dz*dz);
auto ir3 = 1 / (r*r*r);
plist[i].ax += dx * ir3;
plist[i].ay += dy * ir3;
plist[i].az += dz * ir3;
}
}
}

void ic(particles &plist, int n) {
std::random_device rd; 
std::mt19937 gen(rd()); 
std::uniform_real_distribution<float> dis(0.0, 1.0);

plist.clear(); 
plist.reserve(n); 
for( auto i=0; i<n; ++i) {
particle p { dis(gen),dis(gen),dis(gen),0,0,0,0,0,0 };
plist.push_back(p);
}
}

int main(int argc, char *argv[]) {

int i;
printf("number of arguments: %d\n",argc);
for(i=0;i<argc;i++)
{
printf("arg %d: %s \n",i+1,argv[i]);
}

if (argc!=2) printf("wrong number of aguments provided \n");
else if (atoi(argv[1])<1) printf("Provide positive integer\n");
else 
{
auto start = std::chrono::high_resolution_clock::now();
int N=atoi(argv[1]); 
std::cout<<"N="<<N<<"\n";
particles plist; 
ic(plist,N); 
forces(plist); 

auto end = std::chrono::high_resolution_clock::now();
auto diff = std::chrono::duration_cast<std::chrono::seconds>(end - start);
auto diff_milli = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
std::cout<<"Total ellapsed time: "<<diff.count()<<"."<<diff_milli.count()<< "\n";
}
return 0;
}
