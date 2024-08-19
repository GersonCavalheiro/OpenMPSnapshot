#pragma once

#include <utility>

#include "algo/interfaces/parallel/ParallelInstrumental.h"
#include "algo/interfaces/AbstractSweepMethod.h"
#include "algo/interfaces/serial/SerialSweepMethod.h"


class ParallelSweepMethod : public ParallelInstrumental, public AbstractSweepMethod {
protected:
matr A;
vec b, y;

void preULR(matr& R);

void preLRR(matr& R);

public:
ParallelSweepMethod() = default;

ParallelSweepMethod(size_t n, size_t threadNum) : ParallelInstrumental(n, threadNum) {
this->A = createThirdDiagMatrI();
this->b = createVecN();
this->y.assign(N, 0.);
}

ParallelSweepMethod(size_t n, vec a_, vec c_, vec b_, vec phi_, pairs kappa_, pairs mu_, pairs gamma_, size_t threadNum) : ParallelInstrumental(n, threadNum) {
this->A = createNewMatr(a_, c_, b_, kappa_, gamma_);
this->b = createNewVec(phi_, mu_);
this->y.assign(N, 0.);
}

ParallelSweepMethod(matr A_, vec b_) : A(std::move(A_)), b(std::move(b_)) {
this->prepareData();
}

std::tuple<size_t, size_t, size_t, size_t, matr, vec, vec> getAllFields() const;

void setAllFields(size_t N, size_t threadNum, size_t blockSize, size_t classicSize, const matr& A_, const vec& b_, const vec& y_);


void transformation();


std::pair<matr, vec> collectInterferElem();


vec collectPartY(const matr& R, const vec& partB);


void collectNotInterferElem();


vec collectFullY(const vec& partY);


vec run() override;
};