
#include "TlEspField.h"
#include <algorithm>
#include <vector>
#include "DfHpqX.h"
#include "Fl_Geometry.h"
#include "TlPosition.h"
#include "TlSerializeData.h"
#include "TlUtils.h"
#include "tl_dense_symmetric_matrix_lapack.h"

#define AU_PER_ANG 1.889762

TlEspField::TlEspField(const TlSerializeData& param) : param_(param) {}

TlEspField::~TlEspField() {}

std::vector<double> TlEspField::makeEspFld(
const TlDenseSymmetricMatrix_Lapack& P,
const std::vector<TlPosition>& grids) {
DfHpqX dfHpq(&this->param_);

const std::size_t numOfGrids = grids.size();
std::vector<double> values(numOfGrids);

values = dfHpq.getESP(P, grids);

const Fl_Geometry flGeom(this->param_["coordinates"]);
const std::size_t numOfAtoms = flGeom.getNumOfAtoms();
for (std::size_t atomIndex = 0; atomIndex < numOfAtoms; ++atomIndex) {
const std::string atomSymbol = flGeom.getAtomSymbol(atomIndex);
if (atomSymbol == "X") {
continue;
}

const TlPosition pos = flGeom.getCoordinate(atomIndex);
const double charge = flGeom.getCharge(atomIndex);

#pragma omp parallel for schedule(runtime)
for (std::size_t gridIndex = 0; gridIndex < numOfGrids; ++gridIndex) {
const TlPosition grid = grids[gridIndex];
const double distance = pos.distanceFrom(grid);
const double esp = charge / distance;

#pragma omp critical(TlEspField__makeEspFld)
{ values[gridIndex] += esp; }
}
}

return values;
}

std::vector<double> TlEspField::makeEspFld(
const std::vector<TlPosition>& grids) {
const std::size_t numOfGrids = grids.size();
std::vector<double> values(numOfGrids);

const Fl_Geometry flGeom(this->param_["coordinates"]);
const std::size_t numOfAtoms = flGeom.getNumOfAtoms();
for (std::size_t atomIndex = 0; atomIndex < numOfAtoms; ++atomIndex) {
const std::string atomSymbol = flGeom.getAtomSymbol(atomIndex);
if (atomSymbol == "X") {
continue;
}

const TlPosition pos = flGeom.getCoordinate(atomIndex);
const double charge = flGeom.getCharge(atomIndex);

#pragma omp parallel for schedule(runtime)
for (std::size_t gridIndex = 0; gridIndex < numOfGrids; ++gridIndex) {
const TlPosition grid = grids[gridIndex];
const double distance = pos.distanceFrom(grid);
const double esp = charge / distance;

#pragma omp critical(TlEspField__makeEspFld)
{ values[gridIndex] += esp; }
}
}

return values;
}
