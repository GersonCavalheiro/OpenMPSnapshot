

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <random>
#include <sstream>
#include <fstream>
#include <iomanip>

using namespace std;

constexpr double MIN_RADIUS = 1.0e1;
constexpr double MAX_RADIUS = 1.0e2;

std::uniform_real_distribution<double> distribution(MIN_RADIUS, MAX_RADIUS);
std::default_random_engine generator;

class Particle
{
public:
double pos[3], vel[3];

Particle()
{
pos[0] = 0; 
pos[1] = 0; 
pos[2] = 0; 

vel[0] = distribution(generator); 
vel[1] = distribution(generator); 
vel[2] = distribution(generator); 

makeItASphere(pos[0], pos[1], pos[2]);
}

void makeItASphere(double &x, double &y, double &z)
{
double r = sqrt(x * x + y * y + z * z);

if (r > MAX_RADIUS)
{
x = x * MAX_RADIUS / r;
y = y * MAX_RADIUS / r;
z = z * MAX_RADIUS / r;
}
else if (r < MIN_RADIUS)
{
x = distribution(generator);
y = distribution(generator);
z = distribution(generator);

makeItASphere(x, y, z);
}
}
};

class Problem
{

public:
Problem(double mass, double dt, unsigned numParticles) : mMass(mass),
mInverseMass(1.0 / mass),
mDt(dt),
mNumParticles(numParticles),
mParticles(new Particle[numParticles])
{
}

~Problem()
{
delete[] mParticles;
}

void integrate();

const Particle *getParticles() const { return mParticles; }

private:
const double mG = 6.6743e-11;
const double mMass;
const double mInverseMass;
const double mDt;
const unsigned mNumParticles;
Particle *const mParticles;
};

void Problem::integrate()
{

const double Const = mG * mMass * mMass;

#pragma omp parallel for
for (int pi = 0; pi < mNumParticles; pi++)
{
double force[3] = {};

for (int pj = 0; pj < mNumParticles; pj++)
{
if (pj != pi)
{
const double dij[3] = {
mParticles[pj].pos[0] - mParticles[pi].pos[0],
mParticles[pj].pos[1] - mParticles[pi].pos[1],
mParticles[pj].pos[2] - mParticles[pi].pos[2]};

const double dist2 = dij[0] * dij[0] +
dij[1] * dij[1] +
dij[2] * dij[2];

const double ConstDist2 = Const / dist2;
const double idist = 1 / sqrt(dist2);

force[0] += ConstDist2 * dij[0] * idist;
force[1] += ConstDist2 * dij[1] * idist;
force[2] += ConstDist2 * dij[2] * idist;
}
}

mParticles[pi].vel[0] += force[0] * mInverseMass * mDt;
mParticles[pi].vel[1] += force[1] * mInverseMass * mDt;
mParticles[pi].vel[2] += force[2] * mInverseMass * mDt;
}

#pragma omp parallel for
for (int pi = 0; pi < mNumParticles; pi++)
{
mParticles[pi].pos[0] += mParticles[pi].vel[0] * mDt;
mParticles[pi].pos[1] += mParticles[pi].vel[1] * mDt;
mParticles[pi].pos[2] += mParticles[pi].vel[2] * mDt;
}
}
void writeVTKFile(const std::string &filename, const Particle *particles, unsigned numParticles)
{
std::ofstream vtkFile(filename);

if (!vtkFile)
{
std::cerr << "Error: Could not open VTK file for writing: " << filename << std::endl;
return;
}

vtkFile << "# vtk DataFile Version 4.0\n";
vtkFile << "Particle positions\n";
vtkFile << "ASCII\n";
vtkFile << "DATASET POLYDATA\n";
vtkFile << "POINTS " << numParticles << " float\n";

for (unsigned i = 0; i < numParticles; ++i)
{
vtkFile << std::setprecision(10) << particles[i].pos[0] << " "
<< particles[i].pos[1] << " "
<< particles[i].pos[2] << "\n";
}

vtkFile << "VERTICES " << numParticles << " " << numParticles * 2 << "\n";
for (unsigned i = 0; i < numParticles; ++i)
{
vtkFile << "1 " << i << "\n";
}

vtkFile.close();
}

int main()
{
int a = omp_get_max_threads();
printf("%d", a);
omp_set_num_threads(32);
const int nTimeSteps = 500;
const double Mass = 1e12;
const double dt = 1e-2;
const unsigned numParticles = 10000;
Problem problem(Mass, dt, numParticles);

double start_time = omp_get_wtime();
for (int ts = 0; ts < nTimeSteps; ts++)
{


problem.integrate();

}
double time = omp_get_wtime() - start_time;
std::cout << "Elapsed time: " << time << endl;
return 0;
}