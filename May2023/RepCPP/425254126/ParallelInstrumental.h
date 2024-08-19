#pragma once

#include "algo/interfaces/Instrumental.h"


class ParallelInstrumental : public Instrumental {
private:
static bool isPrime(int num);

static vec findDivisors(int num);

protected:
size_t threadNum, blockSize, interSize;

public:
ParallelInstrumental() : ParallelInstrumental(5, 0, -1, -1) {}

ParallelInstrumental(size_t n, size_t tN) : Instrumental(n) {
this->prepareData(n, tN);
}

ParallelInstrumental(size_t n, size_t threadNum, size_t blockSize, size_t interSize) : Instrumental(n),
threadNum(threadNum), blockSize(blockSize), interSize(interSize) {
this->setParallelOptions();
}

void setParallelOptions() const;

void prepareData(size_t n, size_t threadNum);

void prepareData() override;

bool checkData() const override;


matr createThirdDiagMatrI();

matr createThirdDiagMatrRand();

vec createVecN();

vec createVecRand();

matr createNewMatr(vec a_, vec c_, vec b_, pairs kappa_, pairs gamma_);

vec createNewVec(vec phi_, pairs mu_);
};