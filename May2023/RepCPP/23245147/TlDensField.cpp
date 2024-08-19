
#include "TlDensField.h"
#include <algorithm>
#include <vector>
#include "DfCalcGridX.h"
#include "Fl_Geometry.h"
#include "TlPosition.h"
#include "TlSerializeData.h"
#include "TlUtils.h"
#include "tl_dense_symmetric_matrix_lapack.h"

#define AU_PER_ANG 1.889762

TlDensField::TlDensField(const TlSerializeData& param) : param_(param) {}

TlDensField::~TlDensField() {}

std::vector<double> TlDensField::makeDensFld(
const TlDenseSymmetricMatrix_Lapack& P,
const std::vector<TlPosition>& grids) {
const std::size_t numOfGrids = grids.size();
std::vector<double> values(numOfGrids, 0.0);

DfCalcGridX dfCalcGrid(&this->param_);

#pragma omp parallel for schedule(runtime)
for (std::size_t gridIndex = 0; gridIndex < numOfGrids; ++gridIndex) {
const TlPosition grid = grids[gridIndex];
double rho = 0.0;
dfCalcGrid.gridDensity(P, grid, &rho);

#pragma omp critical(TlDensField__makeDensFld)
{ values[gridIndex] += rho; }
}

return values;
}
