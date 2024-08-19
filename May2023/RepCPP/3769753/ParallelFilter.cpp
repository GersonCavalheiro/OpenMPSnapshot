#include <iomanip>
#include <iostream>
#include <string>

#include "BloomFilter.hpp"
#include "vendor/ntHashIterator.hpp"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace opt {

size_t bloomBits = 1024 * 1024 * 8;
unsigned kmerLen = 64;
unsigned ibits = 64;
unsigned nhash = 5;
}

using namespace std;

static const unsigned char b2r[256] = { 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 
'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N',
'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 
'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N',
'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 
'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N',
'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 
'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N',
'N', 'T', 'N', 'G', 'N', 'N', 'N', 'C', 
'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N',
'N', 'N', 'N', 'N', 'A', 'N', 'N', 'N', 
'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N',
'N', 'T', 'N', 'G', 'N', 'N', 'N', 'C', 
'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N',
'N', 'N', 'N', 'N', 'A', 'N', 'N', 'N', 
'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N',
'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 
'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N',
'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 
'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N',
'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 
'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N',
'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 
'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N',
'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 
'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N',
'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 
'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N',
'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 
'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N',
'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 
'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N' };

void
getCanon(std::string& bMer)
{
int p = 0, hLen = (opt::kmerLen - 1) / 2;
while (bMer[p] == b2r[(unsigned char)bMer[opt::kmerLen - 1 - p]]) {
++p;
if (p >= hLen)
break;
}
if (bMer[p] > b2r[(unsigned char)bMer[opt::kmerLen - 1 - p]]) {
for (int lIndex = p, rIndex = opt::kmerLen - 1 - p; lIndex <= rIndex; ++lIndex, --rIndex) {
char tmp = b2r[(unsigned char)bMer[rIndex]];
bMer[rIndex] = b2r[(unsigned char)bMer[lIndex]];
bMer[lIndex] = tmp;
}
}
}

void
loadSeq(BloomFilter& BloomFilterFilter, const string& seq)
{
if (seq.size() < opt::kmerLen)
return;

ntHashIterator insertIt(seq, BloomFilterFilter.getHashNum(), BloomFilterFilter.getKmerSize());
while (insertIt != insertIt.end()) {
BloomFilterFilter.insert(*insertIt);
++insertIt;
}
}

void
loadSeqr(BloomFilter& BloomFilterFilter, const string& seq)
{
if (seq.size() < opt::kmerLen)
return;
string kmer = seq.substr(0, opt::kmerLen);
ntHashIterator itr(seq, opt::kmerLen, opt::nhash);
while (itr != itr.end()) {
BloomFilterFilter.insert(*itr);
++itr;
}
}

void
loadBf(BloomFilter& BloomFilterFilter, const char* faqFile)
{
ifstream uFile(faqFile);
bool good = true;
#pragma omp parallel
for (string line, hline; good;) {
#pragma omp critical(uFile)
{
good = static_cast<bool>(getline(uFile, hline));
good = static_cast<bool>(getline(uFile, line));
}
if (good)
loadSeqr(BloomFilterFilter, line);
}
uFile.close();
}

int
main(int argc, const char* argv[])
{

if (argc < 2)
cerr << "error!\n";
#ifdef _OPENMP
double sTime = omp_get_wtime();
#endif

BloomFilter myFilter(opt::bloomBits, opt::nhash, opt::kmerLen);
loadBf(myFilter, argv[1]);
cerr << "|popBF|=" << myFilter.getPop() << " ";
#ifdef _OPENMP
cerr << setprecision(4) << fixed << omp_get_wtime() - sTime << "\n";
#else
cerr << "\n";
#endif



return 0;
}
