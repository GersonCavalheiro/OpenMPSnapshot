#include "Flock.hpp"  
#include "Tracer.hpp" 
#include "Utils.hpp"  
#include "Vec.hpp"    
#include <chrono>     
#include <omp.h>      
#include <string>     
#include <vector>     

class Simulator
{
public:
Simulator()
{
Params = GlobalParams.SimulatorParams;
std::cout << "Running on " << Params.NumBoids << " boids for " << Params.NumIterations << " iterations in a ("
<< GlobalParams.ImageParams.WindowX << ", " << GlobalParams.ImageParams.WindowY << ") world with "
<< Params.NumThreads << " threads" << std::endl;

std::string ParAxis = "FLOCKS";
if (!Params.ParallelizeAcrossFlocks)
ParAxis = "BOIDS";
std::string NeighMode = "LOCAL";
if (!GlobalParams.FlockParams.UseLocalNeighbourhoods)
NeighMode = "GLOBAL";
std::cout << "Parallelizing across " << ParAxis << " with a " << NeighMode << " neighbourhood layout"
<< std::endl;

Flock::InitNeighbourhoodLayout();
for (size_t i = 0; i < Params.NumBoids; i++)
{
AllFlocks[i] = Flock(i, 1);
}

Tracer::InitFlockMatrix(AllFlocks.size());

if (Params.RenderingMovie)
{
I.Init();
}
}
static SimulatorParamsStruct Params;
std::unordered_map<size_t, Flock> AllFlocks;
Image I;

void Finish()
{
for (auto It = AllFlocks.begin(); It != AllFlocks.end(); It++)
{
assert(It != AllFlocks.end());
Flock &F = It->second;
F.Destroy();
assert(F.Size() == 0);
}
}

void Simulate()
{
double ElapsedTime = 0;
for (size_t i = 0; i < Params.NumIterations; i++)
{
ElapsedTime += Tick();
std::cout << "Tick: " << i << "\r" << std::flush; 
}
std::cout << "Finished simulation! Took " << ElapsedTime << "s" << std::endl;
}

double Tick()
{
auto StartTime = std::chrono::system_clock::now();

#ifndef NDEBUG
size_t BoidCount = 0;
for (auto It = AllFlocks.begin(); It != AllFlocks.end(); It++)
{
assert(It != AllFlocks.end());
const Flock &F = It->second;
BoidCount += F.Size();
for (const Boid *B : F.Neighbourhood.GetBoids())
{
assert(B->IsValid());
}
}
assert(Params.NumBoids == BoidCount);
#endif
std::vector<Flock *> AllFlocksVec = GetAllFlocksVector();

if (!Params.ParallelizeAcrossFlocks)
ParallelBoids(AllFlocksVec);
else
ParallelFlocks(AllFlocksVec);
UpdateFlocks(AllFlocksVec);

auto EndTime = std::chrono::system_clock::now();
std::chrono::duration<double> ElapsedTime = EndTime - StartTime;
Tracer::AddTickT(ElapsedTime.count());

if (Params.RenderingMovie)
{
Render();
}

return ElapsedTime.count(); 
}

std::vector<Flock *> GetAllFlocksVector() const
{
std::vector<Flock *> AllFlocksVec;
for (auto It = AllFlocks.begin(); It != AllFlocks.end(); It++)
{
assert(It != AllFlocks.end());
const Flock &F = It->second;
Tracer::AddFlockSize(F.Size());
AllFlocksVec.push_back(const_cast<Flock *>(&F));
}
assert(AllFlocksVec.size() == AllFlocks.size());
return AllFlocksVec;
}

void ParallelBoids(std::vector<Flock *> AllFlocksVec)
{
#pragma omp parallel num_threads(Params.NumThreads) 
{
if (!GlobalParams.FlockParams.UseLocalNeighbourhoods)
{
std::vector<Boid> &AllBoids = *(AllFlocks.begin()->second.Neighbourhood.GetAllBoidsPtr());
#pragma omp for schedule(dynamic)
for (size_t i = 0; i < AllBoids.size(); i++)
{
AllBoids[i].SenseAndPlan(omp_get_thread_num(), AllFlocks);
}
#pragma omp barrier
#pragma omp for schedule(dynamic)
for (size_t i = 0; i < AllBoids.size(); i++)
{
AllBoids[i].Act(Params.DeltaTime);
}
}
else
{
std::vector<Boid *> AllBoids;
for (const Flock *F : AllFlocksVec)
{
#pragma omp critical
{
std::vector<Boid *> LocalBoids = F->Neighbourhood.GetBoids();
AllBoids.insert(AllBoids.end(), LocalBoids.begin(), LocalBoids.end());
}
}
#pragma omp barrier
#pragma omp for schedule(dynamic)
for (size_t i = 0; i < AllBoids.size(); i++)
{
AllBoids[i]->SenseAndPlan(omp_get_thread_num(), AllFlocks);
}
#pragma omp barrier
#pragma omp for schedule(dynamic)
for (size_t i = 0; i < AllBoids.size(); i++)
{
AllBoids[i]->Act(Params.DeltaTime);
}
}
}
}

void ParallelFlocks(std::vector<Flock *> AllFlocksVec)
{
#pragma omp parallel num_threads(Params.NumThreads) 
{
#pragma omp for schedule(dynamic)
for (size_t i = 0; i < AllFlocksVec.size(); i++)
{
AllFlocksVec[i]->SenseAndPlan(omp_get_thread_num(), AllFlocks);
}
#pragma omp barrier
#pragma omp for schedule(dynamic)
for (size_t i = 0; i < AllFlocksVec.size(); i++)
{
AllFlocksVec[i]->Act(Params.DeltaTime);
}
}
}

void UpdateFlocks(std::vector<Flock *> AllFlocksVec)
{
#pragma omp parallel num_threads(Params.NumThreads) 
{
if (GlobalParams.FlockParams.UseFlocks)
{
#pragma omp for schedule(dynamic)
for (size_t i = 0; i < AllFlocksVec.size(); i++)
{
AllFlocksVec[i]->Delegate(omp_get_thread_num(), AllFlocksVec);
}
#pragma omp barrier
#pragma omp for schedule(dynamic)
for (size_t i = 0; i < AllFlocksVec.size(); i++)
{
AllFlocksVec[i]->AssignToFlock(omp_get_thread_num());
}
}
#pragma omp barrier
#pragma omp for schedule(dynamic)
for (size_t i = 0; i < AllFlocksVec.size(); i++)
{
AllFlocksVec[i]->ComputeBB();
}
}
Tracer::SaveFlockMatrix(AllFlocks);
Tracer::ComputeFlockAverageSize();
Flock::CleanUp(AllFlocks);
}

void Render()
{
std::vector<Flock *> AllFlocksVec;
for (auto It = AllFlocks.begin(); It != AllFlocks.end(); It++)
{
assert(It != AllFlocks.end());
Flock &F = It->second;
AllFlocksVec.push_back(&F);
}
#pragma omp parallel for num_threads(Params.NumThreads) schedule(dynamic)
for (size_t i = 0; i < AllFlocksVec.size(); i++)
{
AllFlocksVec[i]->Draw(I);
}
I.ExportPPMImage();
I.Blank();
}
};

SimulatorParamsStruct Simulator::Params;
ImageParamsStruct Image::Params;
TracerParamsStruct Tracer::Params;

ParamsStruct GlobalParams;

void RunSimulation()
{
Tracer::Initialize();
Simulator Sim;
Sim.Simulate();
Tracer::Dump();
Sim.Finish();
}

int main(int argc, char *argv[])
{
std::srand(0); 
if (argc == 1)
{
ParseParams("params/params.ini");
}
else
{
const std::string ParamFile(argv[1]);
ParseParams("params/" + ParamFile);
}
if (GlobalParams.SimulatorParams.NumThreads > 0)
{
RunSimulation();
}
else
{
const std::vector<size_t> AllProcTests = {2, 4, 8, 12, 16, 24, 32};
for (const size_t P : AllProcTests)
{
GlobalParams.SimulatorParams.NumThreads = P;
RunSimulation();
std::cout << std::endl;
}
}
return 0;
}

