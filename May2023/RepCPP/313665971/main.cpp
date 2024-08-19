#include <omp.h>
#include <math.h>
#include <time.h>

#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream> 
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <vector>

#include "GA.h"
#include "Environment.h"
#include "Player.h"
#include "Vector2.hpp"

using namespace AI;
using namespace std;

#define VELKOST_POPULACIE						   (256)
#define POCET_GENOV						             (2)
#define POCET_GENERACII						        (100000)
#define MUTATION_DECAY						      (0.99995f)

#define POCET_KROKOV							   (100)
#define SIRKA_BLUDISKA                                                      (10)     
#define VYSKA_BLUDISKA                                                      (10)

Environment* _env = new Environment(
SIRKA_BLUDISKA,
VYSKA_BLUDISKA,
new int[SIRKA_BLUDISKA * VYSKA_BLUDISKA] { 
0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
0, 1, 0, 0, 0, 1, 0, 1, 0, 1,
0, 0, 0, 1, 1, 1, 1, 1, 0, 0,
1, 0, 0, 1, 0, 1, 0, 1, 0, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
0, 1, 0, 1, 0, 1, 0, 0, 1, 0,
0, 1, 1, 1, 1, 1, 0, 0, 1, 0,
0, 1, 0, 1, 0, 1, 1, 1, 1, 0,
0, 1, 1, 1, 0, 1, 0, 0, 2, 0
});

vector<Vector2*> start_positions;

GA* _ai = new GA(VELKOST_POPULACIE, (SIRKA_BLUDISKA*VYSKA_BLUDISKA), POCET_GENOV, MUTATION_DECAY);

void DoProgress(const char *label, int step, int total);
void trainig_loop(int individual_index);
void testing_loop(int t, FILE *f);
void read_start_points();

int main(int argc, char** argv)
{
start_positions.push_back(new Vector2(0, 1));
start_positions.push_back(new Vector2(9, 1));
start_positions.push_back(new Vector2(0, 5));

srand((unsigned)time(NULL));

for (int generation = 0; generation < POCET_GENERACII; generation++)
{
_ai->clearFitness();

DoProgress("Training: ", generation, POCET_GENERACII);

#pragma omp parallel for
for (int individual_index = 0; individual_index < VELKOST_POPULACIE; individual_index++)
{
trainig_loop(individual_index);
}

_ai->updateBest();

_ai->Crossover();

_ai->Mutation();
}

read_start_points();

FILE *f_fitness = fopen("statistics.txt", "w");

for (int time = 0; time < 20; time++)
{
testing_loop(time, f_fitness);
}

fclose(f_fitness);

return 0;
}

void DoProgress(const char *label, int step, int total)
{
const int pwidth = 72;

float percent = (float)step / (float)total;

int width = pwidth - strlen(label);
int pos = (int) round(percent * width);

cout << label << " [";
for (int i = 0; i < pos - 1; i++)  
cout << "=";
cout << ">>";

cout << std::setfill(' ') << std::setw(width - pos + 1);
cout << "] ";
cout << std::fixed;
cout << std::setprecision(2);
cout << percent * 100.0f;
cout << "%\r";
}

void trainig_loop(int individual_index)
{
Player _robot;
int done;

auto idx = rand() % 3;
_robot.Move(start_positions[idx]->x, start_positions[idx]->y);

for (int step = 0; step < POCET_KROKOV; step++)
{
auto oldX = _robot.getX();
auto oldY = _robot.getY();

_robot.Move(_ai->getGeneOfIndividual(individual_index, _env->getState(_robot.getX(), _robot.getY())));

if (_robot.getX() < 0 || _robot.getX() > (_env->getWidth()-1) || _robot.getY() < 0 || _robot.getY() > (_env->getHeight()-1))
{
_robot.setReward(-50.0);
_robot.Move(oldX, oldY);
done = -1;
}
else
_robot.setReward(_env->getReward(_robot.getX(), _robot.getY(), &done));

_ai->setFitnessOfIndividual(individual_index, _robot.getReward());

if (done != 0) break;
}
}

void testing_loop(int t, FILE *f)
{
Player _robot;
int done, step, best_is_end = 0;

auto t_start = chrono::high_resolution_clock::now(); 

_ai->clearFitness();

_robot.Move(start_positions[t]->x, start_positions[t]->y);

for (step = 0; step < POCET_KROKOV; step++)
{
auto oldX = _robot.getX();
auto oldY = _robot.getY();

_robot.Move(_ai->getGeneOfIndividual(_ai->getBest(), _env->getState(_robot.getX(), _robot.getY())));

if (_robot.getX() < 0 || _robot.getX() > (_env->getWidth()-1) || _robot.getY() < 0 || _robot.getY() > (_env->getHeight()-1))
{
_robot.setReward(-50.0);
_robot.Move(oldX, oldY);
done = -1;
}
else
_robot.setReward(_env->getReward(_robot.getX(), _robot.getY(), &done));

_ai->setFitnessOfIndividual(_ai->getBest(), _robot.getReward());

if (done == 1)
best_is_end = 100;

if (done != 0) break;
}

auto t_stop = chrono::high_resolution_clock::now(); 
chrono::duration<float, milli> fp_ms = t_stop - t_start;

fprintf(f, "%d;%f;%d;%f;%d\n", t, _ai->getFitnessOfIndividual(_ai->getBest()), step, (fp_ms.count() * 1000), best_is_end);
}

void read_start_points()
{
fstream log_startPos;

start_positions.clear();

log_startPos.open("log_start.txt", ios::in);
if (log_startPos.is_open())
{
string line;
string intermediate;

while (getline(log_startPos, line))
{
stringstream stream1(line);

getline(stream1, intermediate, ';');
auto x = atof(intermediate.c_str());

getline(stream1, intermediate, ';');
auto y = atof(intermediate.c_str());

start_positions.push_back(new Vector2(x, y));
}

log_startPos.close();
}
else
{
cout << "Error in reading file.\n";
exit(-1);
}
}
