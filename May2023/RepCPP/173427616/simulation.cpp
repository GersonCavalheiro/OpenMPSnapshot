#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

#define MASS 1      
#define WIDTH 200
#define HEIGHT 100
#define DEPTH 400
#define TOTALSTEP 720000
#define PRINTSTEP 100   
#define DELTA 0.01    
#define NUMBALLS 1000 
#define RADIUS 0.5    
#define SKIPLINE 8    

int numthreads;

struct X{
double a, b, c;
};

class Body{

public:
int collison; 
double rx, ry, rz;  
double vx, vy, vz;  
double fx, fy, fz;  
double hvx, hvy, hvz; 
double tvx, tvy, tvz; 

Body(): rx(0.0), ry(0.0), rz(0.0), vx(0.0), vy(0.0), vz(0.0), fx(0.0), fy(0.0), fz(0.0), hvx(0.0), hvy(0.0), hvz(0.0), collison(0), tvx(0.0), tvy(0.0), tvz(0.0){}
void calculateforce(Body& b);
void updatevelocity();
void updateposition();
void resetforce();
void checkcollision(Body& b);
void resettempvelocity();
void updateVEL();
void save(ofstream& of);
void load(ifstream& inf);
friend ostream& operator<<(ostream &out, const Body& b);
};





ostream& operator<<(ostream& out, const Body& b){
cout << b.rx << " " << b.ry << " " << b.rz;
return out;
}

double calculatedistance(Body& b1, Body& b2){
double disx = b2.rx - b1.rx;
double disy = b2.ry - b1.ry;
double disz = b2.rz - b1.rz;
return disx*disx + disy*disy + disz*disz;
}

void Body::checkcollision(Body& b){
if(calculatedistance(*this, b) <= 2*RADIUS){
this->collison = 1;
this->tvx += b.vx;
this->tvy += b.vy;
this->tvz += b.vz;
}
}


void Body::resettempvelocity(){
this->tvx = this->tvy = this->tvz = 0.0;
this->collison = 0;
}


void Body::updateVEL(){
if(this->collison){
this->vx = this->tvx;
this->vy = this->tvy;
this->vz = this->tvz;
}
}


void Body::calculateforce(Body& b){
double moddist = calculatedistance(*this, b);

double modforce = 0.0;
modforce = (MASS * MASS)/ (moddist + 1e-7);
moddist += 1e-7;
double disx = b.rx - this->rx;
double disy = b.ry - this->ry;
double disz = b.rz - this->rz;
this->fx += (modforce * disx)/ sqrt(moddist);
this->fy += (modforce * disy)/ sqrt(moddist);
this->fz += (modforce * disz)/ sqrt(moddist);


}



void Body::updatevelocity(){
this->hvx = this->vx + (this->fx * DELTA)/ (2 * MASS);
this->hvy = this->vy + (this->fy * DELTA)/ (2 * MASS);
this->hvz = this->vz + (this->fz * DELTA)/ (2 * MASS);


this->vx = this->hvx + (this->fx * DELTA) / (2 * MASS);
this->vy = this->hvy + (this->fy * DELTA) / (2 * MASS);
this->vz = this->hvz + (this->fz * DELTA) / (2 * MASS);
}

void Body::resetforce(){
this->fx = this->fy = this->fz = 0.0;
}

void Body::updateposition(){
this->rx = this->rx + this->hvx * DELTA;
this->ry = this->ry + this->hvy * DELTA;
this->rz = this->rz + this->hvz * DELTA;


if((this->rx + RADIUS) >= WIDTH){
this->rx = WIDTH - RADIUS;
this->vx = -this->vx;
}else if((this->rx - RADIUS) <= 0){
this->rx = RADIUS;
this->vx = -this->vx;
}

if((this->ry + RADIUS) >= HEIGHT){
this->ry = HEIGHT - RADIUS;
this->vy = -this->vy;
}else if(this->ry - RADIUS <= 0){
this->ry = RADIUS;
this->vy = -this->vy;
}

if(this->rz + RADIUS >= DEPTH){
this->rz = DEPTH - RADIUS;
this->vz = -this->vz;
}else if(this->rz - RADIUS <= 0){
this->rz = RADIUS;
this->vz = -this->vz;
}

}


void readfile(const char* filename, Body* bodies){
fstream fin(filename);
string line;
for(int i = 0 ; i < SKIPLINE ; i++){
getline(fin, line);
}

double rx, ry, rz;
int i = 0;
while(fin >> rx >> ry >> rz){
bodies[i].rx = rx;
bodies[i].ry = ry;
bodies[i].rz = rz;
i++;
}


}


void run_simulation(Body* bodies){
int i,j;
#pragma omp parallel for num_threads(numthreads) private(i,j)
for(i = 0 ; i < NUMBALLS ; i++){
for(j = 0 ; j < NUMBALLS ; j++){
if(j != i)
bodies[i].calculateforce(bodies[j]);
}
}

#pragma omp parallel for num_threads(numthreads) private(i)
for(i = 0 ; i < NUMBALLS ; i++)
bodies[i].updatevelocity();

#pragma omp parallel for num_threads(numthreads) private(i)
for(i = 0 ; i < NUMBALLS ; i++)
bodies[i].updateposition();

#pragma omp parallel for num_threads(numthreads) private(i)
for(i = 0 ; i < NUMBALLS ; i++)
bodies[i].resetforce();



}


void writefile(const char* filename, Body* bodies){
ofstream outfile;
outfile.open(filename, ios::binary | ios::out | ios::app);
for(int i = 0 ; i < NUMBALLS ; i++){
X x;
x.a = bodies[i].rx;
x.b = bodies[i].ry;
x.c = bodies[i].rz;
outfile.write(reinterpret_cast<char*>(&x), sizeof(x));
outfile.write("\n", 1);
}
}

void Body::save(ofstream& of){
of.write((char*)&rx, sizeof(rx));
of.write((char*)&ry, sizeof(ry));
of.write((char*)&rz, sizeof(rz));
}


void Body::load(ifstream& inf){
inf.read((char*)&rx, sizeof(rx));
inf.read((char*)&ry, sizeof(ry));
inf.read((char*)&rz, sizeof(rz));
}

void writeBinary(ofstream& of, Body* bodies){
for(int i = 0 ; i < NUMBALLS ; i++){
bodies[i].save(of);
}
}

void readBinary(ifstream& inf, Body* bodies){
for(int i = 0 ; i < NUMBALLS ; i++){
bodies[i].load(inf);
}
}

void printBodies(Body* bodies){
for(int i = 0 ; i < NUMBALLS ; i++){
cout << bodies[i] << endl;
cout << bodies[i].vx << " " << bodies[i].vy << " " << bodies[i].vz << endl;
}
}


int main(int argc, char* argv[]){

if(argc != 2){
cout << "Usage: ./a.out <numthreads>" << endl;
exit(EXIT_FAILURE);
}

numthreads = atoi(argv[1]);
cout << numthreads << endl;

Body* bodies = new Body[NUMBALLS];
readfile("Trajectory.txt", bodies);

ofstream myfile;
myfile.open("Output_3.dat", ios::binary | ios::out | ios::app);
writeBinary(myfile, bodies);

double start = omp_get_wtime();

for(int i = 0 ; i < TOTALSTEP ; i++){
double st = omp_get_wtime();
run_simulation(bodies);
cout << omp_get_wtime()-st << endl;
if((i+1)%PRINTSTEP == 0){
writeBinary(myfile, bodies);
cout << "Writing Done..." << endl;
}
}
myfile.close();
cout << "Complete" << endl;

cout << "Time: " << omp_get_wtime()-start << endl;




delete[] bodies;





}
