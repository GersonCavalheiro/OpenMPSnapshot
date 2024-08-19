#pragma once

struct Block;

struct Comm
{
int rank;

int neigh[6];

#ifdef FAKEMPI
static Comm ** array;
#endif
Block * block;

Comm();
Comm(int rank, int size, Block * block);
~Comm();

void send(int to, int size, double * buff, int tag);    
void recv(int from, int size, double * buff, int tag);

void exchange();
};