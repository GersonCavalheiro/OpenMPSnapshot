#pragma once

#include <iostream>
#include <math.h>

namespace AI
{
class Individual
{
private:
int chromosome_size;
int gene_set;

bool** chromosome;
float fitness;
public:
Individual(int _chromosome_size, int _gene_set, bool init_random=true);
~Individual();

void setFitness(float val);
float getFitness();
void clrFitness();
int getChromosomeSize();
int getGeneSet();
int getGene(int idx);
void setGene(int idx, int val);
};

int Individual::getGeneSet()
{
return this->gene_set;
}

void Individual::setFitness(float val)
{
this->fitness += val;
}

float Individual::getFitness()
{
return this->fitness;
}

void Individual::clrFitness()
{
this->fitness = 0.0f;
}

int Individual::getChromosomeSize()
{
return this->chromosome_size;
}

Individual::Individual(int _chromosome_size, int _gene_set, bool init_random)
{
this->chromosome_size = _chromosome_size;
this->gene_set = _gene_set;

this->chromosome = new bool*[this->chromosome_size];
for (int i = 0; i < this->chromosome_size; i++)
{
this->chromosome[i] = new bool[this->gene_set];
if (init_random)
for (int j = 0; j < this->gene_set; j++)
{
this->chromosome[i][j] = (bool)(rand() % 2);
}
}


this->clrFitness();
}

int Individual::getGene(int idx)
{
return (((int)this->chromosome[idx][1] << 1) + (int)this->chromosome[idx][0]);
}

void Individual::setGene(int idx, int val)
{
this->chromosome[idx][0] = (bool)(val & 0x01);
this->chromosome[idx][1] = (bool)((val>>1) & 0x01);
}

Individual::~Individual()
{
for (int i = 0; i < this->chromosome_size; i++)
delete this->chromosome[i];
delete[] this->chromosome;
}
}