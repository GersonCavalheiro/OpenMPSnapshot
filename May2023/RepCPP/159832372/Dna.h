
#pragma once

#include <cstdio>
#include <cstring>
#include <cassert>
#include <cstdint>
#include <vector>
#include <zlib.h>

#include "Threefry.h"

constexpr int8_t CODON_SIZE = 3;

constexpr const char* PROM_SEQ = "0101011001110010010110";
constexpr const char* SHINE_DAL_SEQ = "011011000";
constexpr const char* PROTEIN_END = "001"; 

class ExpManager;

class Dna {

public:
Dna() = default;

Dna(const Dna& clone);

Dna(int length, Threefry::Gen& rng);

Dna(char* genome, int length);

Dna(int length);

~Dna() = default;

int length() const;

void save(gzFile backup_file);
void load(gzFile backup_file);

void set(int pos, char c);

void do_switch(int pos);

int promoter_at(int pos);

int terminator_at(int pos);

bool shine_dal_start(int pos);

bool protein_stop(int pos);

int codon_at(int pos);

std::vector<char> seq_;
};
