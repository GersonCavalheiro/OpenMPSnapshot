#pragma once

#include <iostream>
#include <math.h>

#include "Individual.h"

namespace AI
{
class GA
{
private:
float mutation_decay;
float mutation_rate;
int population_size;
int gene_set;
int chromosome_size;

Individual** populations;
int best;
public:
GA(int _population_size, int _chromosome_size, int _gene_set, float _mutation_decay);
~GA();

void Crossover();
void Mutation();
void clearFitness();
int getGeneOfIndividual(int idxIndiv, int idxGene);
void setFitnessOfIndividual(int idxIndiv, float reward);
float getFitnessOfIndividual(int idxIndiv);
void updateBest();
int getBest();
void RouletteWheelMechanism(int* idxA, int* idxB);
float getMutationRate();
};

GA::GA(int _population_size, int _chromosome_size, int _gene_set, float _mutation_decay)
{
this->population_size = _population_size;
this->chromosome_size = _chromosome_size;
this->gene_set = _gene_set;
this->mutation_decay = _mutation_decay;

this->populations = new Individual*[_population_size];
for (int i = 0; i < this->population_size; i++)
this->populations[i] = new Individual(_chromosome_size, _gene_set);

this->mutation_rate = 1.0f;

this->best = 0;
}

void GA::Crossover()
{
Individual** new_populations;
int slicing_point, _parrentA=0, _parrentB=0;
int i, j;

new_populations = new Individual*[this->population_size];
for (int i = 0; i < this->population_size; i++)
new_populations[i] = new Individual(this->chromosome_size, this->gene_set);

for (i = 0; i < this->population_size; i++)
{
if (i != this->best)
{
RouletteWheelMechanism(&_parrentA, &_parrentB);

slicing_point = (rand() % this->populations[i]->getChromosomeSize());

for (j = 0; j < slicing_point; j++)
{
new_populations[i]->setGene(j, this->populations[_parrentA]->getGene(j));
}
for (j = slicing_point; j < this->populations[i]->getChromosomeSize(); j++)
{
new_populations[i]->setGene(j, this->populations[_parrentB]->getGene(j));
}
}
else
{
for (j = 0; j < this->populations[i]->getChromosomeSize(); j++)
{
new_populations[i]->setGene(j, this->populations[i]->getGene(j));
}
}	
}

for (i = 0; i < this->population_size; i++)
delete this->populations[i];		
delete[] this->populations;
this->populations = new_populations;
}

void GA::Mutation()
{
int i, j;

for (i = 0; i < this->population_size; i++)
{
if (i != this->best)
{
for (j = 0; j < this->populations[i]->getChromosomeSize(); j++)
{
if (((float)rand() / (float)RAND_MAX) < this->mutation_rate)
{
this->populations[i]->setGene(j, (rand() % (1 << this->populations[i]->getGeneSet())));
}
}
}
}

if (this->mutation_rate >= 0.01f)
this->mutation_rate *= mutation_decay;
}

void GA::clearFitness()
{
int i;

for (i = 0; i < this->population_size; i++)
{
this->populations[i]->clrFitness();
}
}

int GA::getGeneOfIndividual(int idxIndiv, int idxGene)
{
return this->populations[idxIndiv]->getGene(idxGene);
}

void GA::setFitnessOfIndividual(int idxIndiv, float reward)
{
this->populations[idxIndiv]->setFitness(reward);
}

float GA::getFitnessOfIndividual(int idxIndiv)
{
return this->populations[idxIndiv]->getFitness();
}

void GA::updateBest()
{
int i;

this->best = 0;
for (i = 1; i < this->population_size; i++)
{
if (this->populations[i]->getFitness() > this->populations[this->best]->getFitness())
this->best = i;
}
}

int GA::getBest()
{
return this->best;
}

void GA::RouletteWheelMechanism(int* idxA, int* idxB)
{
float* probability = new float[this->population_size];
float sum = 0.0;


for (int i = 0; i < this->population_size; i++)
{
sum += exp(this->populations[i]->getFitness());
}
for (int i = 0; i < this->population_size; i++)
{
probability[i] = exp(this->populations[i]->getFitness()) / sum;
}



float a = ((float)rand() / (float)RAND_MAX);
float b = ((float)rand() / (float)RAND_MAX);

float min = 0, max = 0;
for (int i = 0; i < this->population_size; i++)
{
min = max;
max += probability[i];

if (min <= a && a < max)
{
*idxA = i;
}

if (min <= b && b < max)
{
*idxB = i;
}
}

delete[] probability;
}

GA::~GA()
{
delete populations;
}

float GA::getMutationRate()
{
return this->mutation_rate;
}
}
