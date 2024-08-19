
#include "../headers/Block.h"
#include "../headers/sha256.h"

Block::Block(uint32_t nIndexIn, const string &sDataIn) : _nIndex(nIndexIn), _sData(sDataIn)
{
_nNonce = 0;
_tTime = time(nullptr);

sHash = _CalculateHash();
}

void Block::MineBlock(uint32_t nDifficulty)
{
char cstr[nDifficulty + 1];

#pragma omp target map(tofrom:cstr)
#pragma omp teams distribute parallel for 
for (uint32_t i = 0; i < nDifficulty; ++i)
{
cstr[i] = '0';
}
cstr[nDifficulty] = '\0';

string str(cstr);

#pragma omp parallel
{
#pragma omp single nowait
while (sHash.substr(0, nDifficulty) != str)
{
#pragma omp task firstprivate(sHash)
_nNonce++;
sHash = _CalculateHash();
}
}

cout << "Block mined: " << sHash << endl;
}

inline string Block::_CalculateHash() const
{
stringstream ss;

#pragma critical 
{
ss << _nIndex << sPrevHash << _tTime << _sData << _nNonce;
}

return sha256(ss.str());
}
