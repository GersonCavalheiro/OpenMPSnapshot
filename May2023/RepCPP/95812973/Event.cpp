#include <algorithm>
#include <exception>
#include <iostream>
#include <random>
#include <math.h>

#include "Event.h"

Event::Event(unsigned short maxRank) {
this->maxRank = maxRank;
}

Event::~Event() {};

void Event::addMatch(
unsigned short red1,
unsigned short red2,
unsigned short red3,
unsigned short blue1,
unsigned short blue2,
unsigned short blue3,
double redWinProbability) {

matches.push_back(Match(
red1,
red2,
red3,
blue1,
blue2,
blue3,
redWinProbability));
}

void Event::addTeam(
unsigned short number,
unsigned char qs,
unsigned short firstSort,
unsigned short secondSort,
unsigned short thirdSort,
unsigned short fourthSort) {
teams.push_back(Team(
number,
qs,
firstSort,
secondSort,
thirdSort,
fourthSort));
}

void Event::winTeam(
std::vector<Team> &teams,
unsigned short number) {

for (auto &team : teams) {
if (team.getNumber() == number) {
team.win();
return;
}
}

std::cerr << "Couldn't find " << number << " in the rankings." << std::endl;
std::cerr << "Are you sure that the input files are correct?" << std::endl;

throw std::exception();
}

void Event::calculate() {
size_t numBranches = pow(2, matches.size());

#pragma omp parallel for
for (size_t i = 0; i < numBranches; i++) {
this->calculatePossibility(i);
}

this->root.log();
std::cout << std::flush;
}

void Event::calculate(size_t numSims) {
std::random_device rand_dev;
std::mt19937_64 generator(rand_dev());
std::uniform_int_distribution<unsigned long> distribution(0x00000000, 0xffffffff);

unsigned long randomPossibility = 0;

for (size_t i = 0; i < numSims; i++) {
do {
randomPossibility = distribution(generator);
randomPossibility << 32;
randomPossibility |= distribution(generator);
} while (randomPossibility >= pow(2, matches.size()));

this->calculatePossibility(randomPossibility);
}

this->root.logMonteCarlo();
std::cout << std::flush;
}

void Event::calculatePossibility(unsigned long i) {
auto percent = (i + 1) / (pow(2, matches.size()));
std::cerr.precision(2);
std::cerr << "\r" << (percent * 100) << "%" << std::flush;

auto currentNumber = i;
auto teams = this->teams;

double probability = 1;

for (size_t m = 0; m < matches.size(); m++) {
auto outcomeNumber = currentNumber % 2;

if (outcomeNumber == 0) {
probability = probability * this->matches[m].getRedWinProbability();
this->winTeam(teams, this->matches[m].getRed1());
this->winTeam(teams, this->matches[m].getRed2());
this->winTeam(teams, this->matches[m].getRed3());
} else {
probability = probability * this->matches[m].getBlueWinProbability();
this->winTeam(teams, this->matches[m].getBlue1());
this->winTeam(teams, this->matches[m].getBlue2());
this->winTeam(teams, this->matches[m].getBlue3());
}

currentNumber /= 2;
}

std::sort(teams.begin(), teams.end());

#pragma omp critical
{
this->updateMonteCarloSummary(teams, probability);
}
}

void Event::updateSummary(
std::vector<Team> &teams,
double probability) {

root.addChild(teams[0].getNumber(), 0);
Node* currentNode = root.getChild(teams[0].getNumber());
currentNode->updateProbability(probability);

for (size_t i = 1; i < this->maxRank; i++) {
currentNode->addChild(teams[i].getNumber(), 0);
currentNode = currentNode->getChild(teams[i].getNumber());
currentNode->updateProbability(probability);
}
}

void Event::updateMonteCarloSummary(
std::vector<Team> &teams,
double probability) {

root.addChild(teams[0].getNumber(), 0);
root.updateProbability(probability);
Node* currentNode = root.getChild(teams[0].getNumber());
currentNode->updateProbability(probability);

for (size_t i = 1; i < this->maxRank; i++) {
currentNode->addChild(teams[i].getNumber(), 0);
currentNode = currentNode->getChild(teams[i].getNumber());
currentNode->updateProbability(probability);
}
}
