#include "Drawing.hpp"
#include "Utils.hpp"
#include <chrono>
#include <cstdlib>
#include <omp.h>
#include <string>
#include <vector>

class FourierDraw
{
public:
FourierDraw()
{
std::cout << "Running on \"" << Params.Simulator.FileName << "\" for " << Params.Simulator.NumIters
<< " iterations with " << Params.Simulator.NumArrows << " moons and " << Params.Simulator.NumThreads
<< " threads at " << Params.Simulator.UpdatesPerTick << " updates per tick" << std::endl;
InitializeDrawings();
}

std::vector<Drawing> Drawings;

void InitializeDrawings()
{
for (size_t i = 0; i < Params.Simulator.NumThreads; i++)
{
Drawings.push_back(Drawing(i));
}
const std::vector<Complex> Cumulative = Drawing::ReadInputFile();
#pragma omp parallel for num_threads(Params.Simulator.NumThreads)
for (size_t i = 0; i < Params.Simulator.NumThreads; i++)
{
Drawings[i].FourierInit(Cumulative);
}
}

void Run()
{
double ElapsedTime = 0;
auto StartTime = std::chrono::system_clock::now();
for (size_t i = 0; i < Params.Simulator.NumIters; i += Params.Simulator.NumThreads)
{
ElapsedTime += Tick();
}
std::cout << std::endl << "Epicycle updating took " << ElapsedTime << "s" << std::endl;
auto EndTime = std::chrono::system_clock::now();
std::chrono::duration<double> TotalTime = EndTime - StartTime;
std::cout << "Total simulation took " << TotalTime.count() << "s" << std::endl;
}

double Tick()
{
auto StartTime = std::chrono::system_clock::now();
#pragma omp parallel for num_threads(Params.Simulator.NumThreads)
for (size_t i = 0; i < Drawings.size(); i++)
{
Drawings[i].Update();
}
auto EndTime = std::chrono::system_clock::now();
std::chrono::duration<double> ElapsedTime = EndTime - StartTime;
if (Params.Simulator.Render)
{
#pragma omp parallel for num_threads(Params.Simulator.NumThreads)
for (size_t i = 0; i < Drawings.size(); i++)
{
Drawings[i].Render();
}
}
return ElapsedTime.count(); 
}
};

ParamsStruct Params;

int main(int argc, char *argv[])
{
if (argc == 1)
{
ParseParams("Params/params.ini");
}
else
{
const std::string ParamFile(argv[1]);
ParseParams("Params/" + ParamFile);
}
FourierDraw FD;
FD.Run();
return 0;
}
