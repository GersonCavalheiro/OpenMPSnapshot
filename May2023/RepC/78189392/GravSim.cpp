#include "GravSim.h"
GravSim::GravSim()
{
}
GravSim::GravSim(int count, float pointMass, float G, float minDist, glm::vec3 center, glm::vec3 radius)
{
Init(count, pointMass, G, minDist, center, radius);
}
GravSim::~GravSim()
{
delete[] points;
delete[] tempForces;
}
void GravSim::Init(int count, float pointMass, float G, float minDist, glm::vec3 center, glm::vec3 radius)
{
#pragma omp parallel
{
#pragma omp single
{
numThreads = omp_get_num_threads();
}
}
pointsCount = count;
points = new Particle[count];
tempForces = new float[3 * numThreads * count];
for (int i = 0; i < count; i++)
{
Particle* point = &points[i];
for (int j = 0; j < 3; j++)
point->Force[j] = 0;
for (int j = 0; j < 3; j++)
point->Speed[j] = 0;
point->mass = pointMass;
point->Pos[0] = (float)rand() / RAND_MAX * 2 * radius.x - radius.x + center.x;
point->Pos[1] = (float)rand() / RAND_MAX * 2 * radius.y - radius.y + center.y;
point->Pos[2] = (float)rand() / RAND_MAX * 2 * radius.z - radius.y + center.z;
}
for (int i = 0; i < 3 * numThreads * count; i++)
tempForces[i] = 0;
this->G = G;
this->minDist = minDist;
}
void GravSim::CalcFrameSingleThread(float dt)
{
for (int i = 0; i < pointsCount; i++)
{
for (int j = i + 1; j < pointsCount; j++)
{
float dx = points[j].Pos[0] - points[i].Pos[0];
float dy = points[j].Pos[1] - points[i].Pos[1];
float dz = points[j].Pos[2] - points[i].Pos[2];
float r_2 = (dx*dx + dy*dy + dz*dz);
if (r_2 < minDist)
continue;
r_2 = 1 / r_2;
float r_1 = sqrt(r_2);
float f = G*points[i].mass * points[j].mass * r_2;
float fx = f*dx*r_1;
float fy = f*dy*r_1;
float fz = f*dz*r_1;
points[i].Force[0] += fx;
points[i].Force[1] += fy;
points[i].Force[2] += fz;
points[j].Force[0] -= fx;
points[j].Force[1] -= fy;
points[j].Force[2] -= fz;
}
}
for (int i = 0; i < pointsCount; i++)
{
points[i].Speed[0] += points[i].Force[0] / points[i].mass;
points[i].Speed[1] += points[i].Force[1] / points[i].mass;
points[i].Speed[2] += points[i].Force[2] / points[i].mass;
points[i].Pos[0] += points[i].Speed[0];
points[i].Pos[1] += points[i].Speed[1];
points[i].Pos[2] += points[i].Speed[2];
points[i].Force[0] = 0;
points[i].Force[1] = 0;
points[i].Force[2] = 0;
}
}
void GravSim::CalcFrameOpenMP(float dt)
{
#pragma omp parallel for
for (int i = 0; i < pointsCount; i++)
{
int thread = omp_get_thread_num();
for (int j = 0; j < pointsCount; j++)
{
if (i == j)
continue;
float dx = points[j].Pos[0] - points[i].Pos[0];
float dy = points[j].Pos[1] - points[i].Pos[1];
float dz = points[j].Pos[2] - points[i].Pos[2];
float r_2 = (dx*dx + dy*dy + dz*dz);
if (r_2 < minDist)
continue;
r_2 = 1 / r_2;
float r_1 = sqrt(r_2);
float f = G*points[i].mass * points[j].mass * r_2;
float fx = f*dx*r_1;
float fy = f*dy*r_1;
float fz = f*dz*r_1;
points[i].Force[0] += fx;
points[i].Force[1] += fy;
points[i].Force[2] += fz;
}
}
float force[3] = { 0,0,0 };
#pragma omp parallel for
for (int i = 0; i < pointsCount; i++)
{
points[i].Speed[0] += points[i].Force[0] / points[i].mass;
points[i].Speed[1] += points[i].Force[1] / points[i].mass;
points[i].Speed[2] += points[i].Force[2] / points[i].mass;
points[i].Pos[0] += points[i].Speed[0];
points[i].Pos[1] += points[i].Speed[1];
points[i].Pos[2] += points[i].Speed[2];
points[i].Force[0] = 0;
points[i].Force[1] = 0;
points[i].Force[2] = 0;
}
}
void GravSim::CalcFrameOpenMPOptimize(float dt)
{
int block = pointsCount / numThreads / 2;
#pragma omp parallel for schedule(dynamic, block) 
for (int i = 0; i < pointsCount; i++)
{
int thread = omp_get_thread_num();
for (int j = i+1; j < pointsCount; j++)
{
float dx = points[j].Pos[0] - points[i].Pos[0];
float dy = points[j].Pos[1] - points[i].Pos[1];
float dz = points[j].Pos[2] - points[i].Pos[2];
float r_2 = (dx*dx + dy*dy + dz*dz);
if (r_2 < minDist)
continue;
r_2 = 1 / r_2;
float r_1 = sqrt(r_2);
float f = G*points[i].mass * points[j].mass * r_2;
float fx = f*dx*r_1;
float fy = f*dy*r_1;
float fz = f*dz*r_1;
tempForces[i * 3 * numThreads + thread * 3 + 0] += fx;
tempForces[i * 3 * numThreads + thread * 3 + 1] += fy;
tempForces[i * 3 * numThreads + thread * 3 + 2] += fz;
tempForces[j * 3 * numThreads + thread * 3 + 0] -= fx;
tempForces[j * 3 * numThreads + thread * 3 + 1] -= fy;
tempForces[j * 3 * numThreads + thread * 3 + 2] -= fz;
}
}
float force[3] = { 0,0,0 };
#pragma omp parallel for firstprivate (force)
for (int i = 0; i < pointsCount; i++)
{
for (int j = 0; j < numThreads; j++)
{
force[0] += tempForces[i * numThreads * 3 + j * 3 + 0];
force[1] += tempForces[i * numThreads * 3 + j * 3 + 1];
force[2] += tempForces[i * numThreads * 3 + j * 3 + 2];
tempForces[i * numThreads * 3 + j * 3 + 0] = 0;
tempForces[i * numThreads * 3 + j * 3 + 1] = 0;
tempForces[i * numThreads * 3 + j * 3 + 2] = 0;
}
points[i].Speed[0] += force[0] / points[i].mass;
points[i].Speed[1] += force[1] / points[i].mass;
points[i].Speed[2] += force[2] / points[i].mass;
points[i].Pos[0] += points[i].Speed[0];
points[i].Pos[1] += points[i].Speed[1];
points[i].Pos[2] += points[i].Speed[2];
force[0] = 0;
force[1] = 0;
force[2] = 0;
}
}
Particle* GravSim::GetPoints()
{
return points;
}
int GravSim::GetPointsCount()
{
return pointsCount;
}
