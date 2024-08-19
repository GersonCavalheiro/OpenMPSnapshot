#include "MpiGravSim.h"
MpiGravSim::MpiGravSim()
{
}
MpiGravSim::MpiGravSim(int count, float pointMass, float G, float minDist, glm::vec3 center, glm::vec3 radius)
{
Init(count, pointMass, G, minDist, center, radius);
}
MpiGravSim::~MpiGravSim()
{
if (ProcessRank != RootProc)
{
delete[] points.mass;
delete[] partForces;
delete[] tempForces;
delete[] points.forces;
delete[] points.speed;
}
else
delete[] tasks;
delete[] points.pos;
}
void MpiGravSim::Init(int count, float pointMass, float G, float minDist, glm::vec3 center, glm::vec3 radius)
{
if (ProcessRank == RootProc)
MPI_Comm_split(MPI_COMM_WORLD, 0, ProcessRank, &workersComm);
else
MPI_Comm_split(MPI_COMM_WORLD, 1, ProcessRank, &workersComm);
MPI_Comm_rank(workersComm, &workersProcessRank);
MPI_Comm_size(workersComm, &workersProcessNum);
points.count = count;
this->G = G;
this->minDist = minDist;
#pragma omp parallel
{
#pragma omp single
{
numThreads = omp_get_num_threads();
}
}
if (ProcessRank != RootProc)
{
pointsPerProcess = points.count / workersProcessNum;
pointsDisplacement = pointsPerProcess * workersProcessRank;
if (points.count % workersProcessNum != 0)
{
std::cout << "WARNING: points per process not equal for all processes\n";
std::cout.flush();
}
}
points.pos = new float[points.count * 3];
if (ProcessRank != RootProc)
{
points.mass = new float[points.count];
points.forces = new float[points.count * 3];
points.speed = new float[points.count * 3];
partForces = new float[points.count * 3];
tempForces = new float[pointsPerProcess * numThreads * 3];
for (int i = 0; i < points.count; i++)
points.mass[i] = pointMass;
for (int i = 0; i < points.count * 3; i++)
{
points.forces[i] = 0;
points.speed[i] = 0;
partForces[i] = 0;
}
memset(tempForces, 0, sizeof(float) * pointsPerProcess* numThreads * 3);
}
if (ProcessRank == RootProc)
{
for (int i = 0; i < points.count * 3; i += 3)
{
points.pos[i + 0] = (float)rand() / RAND_MAX * 2 * radius.x - radius.x + center.x;
points.pos[i + 1] = (float)rand() / RAND_MAX * 2 * radius.y - radius.y + center.y;
points.pos[i + 2] = (float)rand() / RAND_MAX * 2 * radius.z - radius.y + center.z;
}
}
MPI_Bcast(points.pos, points.count * 3, MPI_FLOAT, RootProc, MPI_COMM_WORLD);
if (ProcessRank == RootProc)
{
int workers = ProcessNum - 1;
numTasks = workers*(workers + 1) / 2 + workers;
tasks = new int[numTasks * 2];
int taskCnt = 0;
for (int i = 0; i < workers; i++)
{
for (int j = i; j < workers; j++)
{
tasks[taskCnt++] = i;
tasks[taskCnt++] = j;
}
}
for (int i = 0; i < workers; i++)
{
tasks[taskCnt++] = -1;
tasks[taskCnt++] = -1;
}
}
}
void MpiGravSim::CalcStep(float dt)
{
if (ProcessRank == RootProc)
rootCalcStep();
else
workerCalcStep(dt);
}
Points* MpiGravSim::GetPoints()
{
return &points;
}
void MpiGravSim::rootCalcStep()
{
int wtf;
MPI_Status status;
currentTask = 0;
while (currentTask < numTasks)
{
MPI_Recv(&wtf, 1, MPI_INT, MPI_ANY_SOURCE, TAG_WORK_REQUEST, MPI_COMM_WORLD, &status);
MPI_Send(&tasks[currentTask * 2], 2, MPI_INT, status.MPI_SOURCE, TAG_WORK_RESPONSE, MPI_COMM_WORLD);
currentTask++;
}
}
void MpiGravSim::workerCalcStep(float dt)
{
while (true)
{
int wtf;
MPI_Status status;
int task[2];
MPI_Send(&wtf, 1, MPI_INT, RootProc, TAG_WORK_REQUEST, MPI_COMM_WORLD);
MPI_Recv(task, 2, MPI_INT, RootProc, TAG_WORK_RESPONSE, MPI_COMM_WORLD, &status);
if (task[0] == -1 && task[1] == -1)
break;
int block = pointsPerProcess / numThreads / 2;
if (task[0] == task[1])
{
int start = task[0] * pointsPerProcess;
#pragma omp parallel for schedule(dynamic, block) 
for (int i = start; i < start + pointsPerProcess; i++)
{
int thread = omp_get_thread_num();
for (int j = i+1; j < start + pointsPerProcess; j++)
{
calcForces(i, j, j-start, thread);
}
}
for (int i = 0; i < pointsPerProcess; i++)
{
for (int j = 0; j < numThreads; j++)
{
partForces[(start + i) * 3 + 0] += tempForces[i * numThreads * 3 + j * 3 + 0];
partForces[(start + i) * 3 + 1] += tempForces[i * numThreads * 3 + j * 3 + 1];
partForces[(start + i) * 3 + 2] += tempForces[i * numThreads * 3 + j * 3 + 2];
}
}
}
else
{
int start1 = task[0] * pointsPerProcess;
int start2 = task[1] * pointsPerProcess;
#pragma omp parallel for schedule(dynamic, block) 
for (int i = start1; i < start1 + pointsPerProcess; i++)
{
int thread = omp_get_thread_num();
for (int j = start2; j < start2 + pointsPerProcess; j++)
{
calcForces(i, j, j-start2, thread);
}
}
for (int i = 0; i < pointsPerProcess; i++)
{
for (int j = 0; j < numThreads; j++)
{
partForces[(start2 + i) * 3 + 0] += tempForces[i * numThreads * 3 + j * 3 + 0];
partForces[(start2 + i) * 3 + 1] += tempForces[i * numThreads * 3 + j * 3 + 1];
partForces[(start2 + i) * 3 + 2] += tempForces[i * numThreads * 3 + j * 3 + 2];
}
}
}
memset(tempForces, 0, sizeof(float) * pointsPerProcess* numThreads * 3);
}
MPI_Allreduce(partForces, points.forces, points.count * 3, MPI_FLOAT, MPI_SUM, workersComm);
#pragma omp parallel for
for (int i = pointsDisplacement; i < pointsDisplacement + pointsPerProcess; i++)
{
points.speed[i * 3 + 0] += points.forces[i * 3 + 0] / points.mass[i];
points.speed[i * 3 + 1] += points.forces[i * 3 + 1] / points.mass[i];
points.speed[i * 3 + 2] += points.forces[i * 3 + 2] / points.mass[i];
points.pos[i * 3 + 0] += points.speed[i * 3 + 0];
points.pos[i * 3 + 1] += points.speed[i * 3 + 1];
points.pos[i * 3 + 2] += points.speed[i * 3 + 2];
}
memset(partForces, 0, sizeof(float) * 3 * points.count);
for (int i = 0; i < workersProcessNum; i++)
{
MPI_Bcast(points.speed + i * pointsPerProcess * 3, pointsPerProcess * 3, MPI_FLOAT, i, workersComm);
MPI_Bcast(points.pos + i * pointsPerProcess * 3, pointsPerProcess * 3, MPI_FLOAT, i, workersComm);
}
}
bool MpiGravSim::calcForces(int pi, int pj, int tfj, int thread)
{
float dx = points.pos[pj * 3 + 0] - points.pos[pi * 3 + 0];
float dy = points.pos[pj * 3 + 1] - points.pos[pi * 3 + 1];
float dz = points.pos[pj * 3 + 2] - points.pos[pi * 3 + 2];
float r_2 = (dx*dx + dy*dy + dz*dz);
if (r_2 < minDist)
return false;
r_2 = 1 / r_2;
float r_1 = sqrt(r_2);
float f = G*points.mass[pi] * points.mass[pj] * r_2;
float fx =  f*dx*r_1;
float fy =  f*dy*r_1;
float fz =  f*dz*r_1;
tempForces[tfj * 3 * numThreads + thread * 3 + 0] -= fx;
tempForces[tfj * 3 * numThreads + thread * 3 + 1] -= fy;
tempForces[tfj * 3 * numThreads + thread * 3 + 2] -= fz;
partForces[pi * 3 + 0] += fx;
partForces[pi * 3 + 1] += fy;
partForces[pi * 3 + 2] += fz;
return true;
}
