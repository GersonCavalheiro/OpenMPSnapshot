
#include "TlMoField.h"
#include <algorithm>
#include <vector>
#include "TlMsgPack.h"
#include "TlOrbitalInfo.h"
#include "TlPosition.h"
#include "TlSerializeData.h"
#include "TlUtils.h"

#define AU_PER_ANG 1.889762
const double TlMoField::INV_SQRT12 = 1.0 / std::sqrt(12.0);

TlMoField::TlMoField(const TlSerializeData& param)
: param_(param), orbInfo_(param["coordinates"], param["basis_set"]) {}

TlMoField::~TlMoField() {}

std::vector<double> TlMoField::makeMoFld(const TlDenseVector_Lapack& MO,
const std::vector<TlPosition>& grids) {
const std::size_t numOfGrids = grids.size();
std::vector<double> values(numOfGrids, 0.0);

const int numOfAOs = this->orbInfo_.getNumOfOrbitals();
assert(MO.getSize() == numOfAOs);
for (std::size_t gridIndex = 0; gridIndex < numOfGrids; ++gridIndex) {
const TlPosition grid = grids[gridIndex];
double gridValue = 0.0;

#pragma omp parallel for schedule(runtime)
for (int aoIndex = 0; aoIndex < numOfAOs; ++aoIndex) {
const TlPosition atomPos = this->orbInfo_.getPosition(aoIndex);
const TlPosition r = grid - atomPos;
const double r2 = r.squareDistanceFrom();

if (r2 * this->orbInfo_.getMinExponent(aoIndex) > 20.0) {
continue;
}

const double preFactor =
this->getPreFactor(this->orbInfo_.getBasisType(aoIndex), r);
double buf = 0.0;
const int numOfContract =
this->orbInfo_.getCgtoContraction(aoIndex);
for (int pGtoIndex = 0; pGtoIndex < numOfContract; ++pGtoIndex) {
const double coef =
this->orbInfo_.getCoefficient(aoIndex, pGtoIndex);
const double exponent =
this->orbInfo_.getExponent(aoIndex, pGtoIndex);
double value = coef * std::exp(-exponent * r2);
buf += value;
}

const double tmp = preFactor * buf * MO.get(aoIndex);
#pragma omp atomic
gridValue += tmp;
}

values[gridIndex] = gridValue;
}

return values;
}

double TlMoField::getPreFactor(const int nType, const TlPosition& pos) {
double prefactor = 1.0;
switch (nType) {
case 0:
break;
case 1:
prefactor = pos.x();
break;
case 2:
prefactor = pos.y();
break;
case 3:
prefactor = pos.z();
break;
case 4:
prefactor = pos.x() * pos.y();
break;
case 5:
prefactor = pos.z() * pos.x();
break;
case 6:
prefactor = pos.y() * pos.z();
break;
case 7:
prefactor = 0.5 * (pos.x() * pos.x() - pos.y() * pos.y());
break;
case 8:
prefactor = INV_SQRT12 * (2.0 * pos.z() * pos.z() -
(pos.x() * pos.x() + pos.y() * pos.y()));
break;
default:
std::cout << "Basis Type is Wrong." << std::endl;
break;
}

return prefactor;
}
