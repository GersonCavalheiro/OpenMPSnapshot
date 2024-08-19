#pragma once

#include <vector>

#include "Team.h"
#include "Match.h"
#include "Root.h"

class Event {
private:
std::vector<Match> matches;
std::vector<Team> teams;
unsigned short maxRank;

Root root;

void calculatePossibility(unsigned long i);

void winTeam(
std::vector<Team> &teams,
const unsigned short number);

void updateSummary(
std::vector<Team> &teams,
double probability
);

void updateMonteCarloSummary(
std::vector<Team> &teams,
double probability
);

public:
Event(unsigned short maxRank);
~Event();

void addMatch(
unsigned short red1,
unsigned short red2,
unsigned short red3,
unsigned short blue1,
unsigned short blue2,
unsigned short blue3,
double redWinProbability = .5);

void addTeam(
unsigned short number,
unsigned char qs,
unsigned short firstSort = 0,
unsigned short secondSort = 0,
unsigned short thirdSort = 0,
unsigned short fourthSort = 0);

void calculate();

void calculate(size_t numSims);
};