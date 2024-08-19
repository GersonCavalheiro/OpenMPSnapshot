
#include <cmath>
#include <iostream>

#include "DfPopulation.h"
#include "Fl_Geometry.h"
#include "TlUtils.h"
#include "tl_dense_symmetric_matrix_lapack.h"
#include "tl_dense_vector_lapack.h"

DfPopulation::DfPopulation(TlSerializeData* pPdfParam)
: DfObject(pPdfParam),
orbitalInfo_((*pPdfParam)["coordinates"], (*pPdfParam)["basis_set"]) {
this->setNucleiCharges();
}

DfPopulation::~DfPopulation() {}

void DfPopulation::exec(const int iteration) {
this->getAtomPopulation<TlDenseSymmetricMatrix_Lapack,
TlDenseVector_Lapack>(iteration);
}


void DfPopulation::calcPop(const int iteration) {
this->calcPop<TlDenseSymmetricMatrix_Lapack>(iteration);
}

void DfPopulation::setNucleiCharges() {
const Fl_Geometry geom((*this->pPdfParam_)["coordinates"]);

const int numOfAtoms = this->m_nNumOfAtoms;
this->nucleiCharges_.resize(numOfAtoms);

for (int i = 0; i < numOfAtoms; ++i) {
this->nucleiCharges_[i] = geom.getCharge(i);
}
}

double DfPopulation::getSumOfNucleiCharges() const {
return this->nucleiCharges_.sum();
}

std::valarray<double> DfPopulation::getGrossAtomPop(
const std::valarray<double>& grossOrbPop) {
std::string output = "";

const index_type numOfAtoms = this->m_nNumOfAtoms;
const index_type numOfAOs = this->m_nNumOfAOs;

std::valarray<double> answer(0.0, numOfAtoms);

#pragma omp parallel for
for (index_type aoIndex = 0; aoIndex < numOfAOs; ++aoIndex) {
const index_type atomIndex = this->orbitalInfo_.getAtomIndex(aoIndex);

#pragma omp critical(DfPopulation__getGrossAtomPop)
{ answer[atomIndex] += grossOrbPop[aoIndex]; }
}

return answer;
}

double DfPopulation::getCharge(int atomIndex) {
const Fl_Geometry flGeom((*(this->pPdfParam_))["coordinates"]);
const double nucCharge = flGeom.getCharge(atomIndex);
const double grossAtomPop = this->grossAtomPopA_[atomIndex];
return nucCharge - grossAtomPop;
}
