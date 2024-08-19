#include "MontecarloHeuristic.h"
#include "City.h"
#include "Route.h"
#include "Problem.h"
#include <algorithm>
#include <ctime>
#include <time.h>
#include <iostream>
#include <random>
#include <chrono>

using namespace std;

Route MontecarloHeuristic::solveMontecarlo(Problem problem, int iterations) {
vector<int> routeInt;
int totalCities = problem.getNumberOfCities();
double totalCost = 0;
vector<vector<int > > routeOrderMatrix;
Route lessCostRoute = Route(0);
srand (time(NULL));

for (int i = 0; i < totalCities; i++)
routeInt.push_back(i);

for (int i = 0; i < iterations; i++){
for (int j = 0; j < totalCities; j++) {
int randomInt = j + rand() % (totalCities - j);
swap(routeInt[j], routeInt[randomInt]);
}
routeOrderMatrix.push_back(routeInt);
}

#pragma omp parallel for firstprivate(problem) shared(routeOrderMatrix, totalCities, totalCost, iterations) schedule(static)
for (int i = 0; i < iterations; i++){
vector <int> currentRouteOrder = routeOrderMatrix[i];
Route route = Route(totalCities);

for (int j = 0; j < totalCities; j++)
route.addCity(currentRouteOrder[j]);

double currentCost = problem.cost(route);

if (i == 0 || totalCost > currentCost) {
totalCost = currentCost;
lessCostRoute = route;
}
}
lessCostRoute.setCost(totalCost);
return lessCostRoute;
}
