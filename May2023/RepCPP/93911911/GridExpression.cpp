
#include "GridExpression.h"
#include "../io/ExprTk.h"

edge_v::models::GridExpression::GridExpression( io::Grid    const * i_grid,
std::string const & i_expr ) {
m_grid = i_grid;

if( i_expr != "" ) m_expr.str = i_expr;
}

void edge_v::models::GridExpression::free() {
if( m_speedsMin != nullptr ) delete[] m_speedsMin;
if( m_speedsMax != nullptr ) delete[] m_speedsMax;
}

edge_v::models::GridExpression::~GridExpression() {
free();
}

void edge_v::models::GridExpression::init( t_idx           i_nPts,
double const (* i_pts)[3] ) {

free();

m_speedsMin = new double[i_nPts];
m_speedsMax = new double[i_nPts];

#ifdef PP_USE_OMP
#pragma omp parallel
#endif
{
io::ExprTk l_expr;
double l_crds[3] = {0, 0, 0};
double l_data = 0;
double l_speeds[2] = {0, 0};

l_expr.addVar( m_expr.xName,
l_crds[0] );
l_expr.addVar( m_expr.yName,
l_crds[1] );
l_expr.addVar( m_expr.zName,
l_crds[2] );
l_expr.addVar( m_expr.dataName,
l_data );
l_expr.addVar( m_expr.minSpeedName,
l_speeds[0] );
l_expr.addVar( m_expr.maxSpeedName,
l_speeds[1] );

l_expr.compile( m_expr.str );

#ifdef PP_USE_OMP
#pragma omp for
#endif
for( t_idx l_pt = 0; l_pt < i_nPts; l_pt++ ) {
for( unsigned short l_di = 0; l_di < 3; l_di++ ) {
l_crds[l_di] = i_pts[l_pt][l_di];
}
l_data = m_grid->getData()[l_pt];
l_expr.eval();
m_speedsMin[l_pt] = l_speeds[0];
m_speedsMax[l_pt] = l_speeds[1];
}

}
}

double edge_v::models::GridExpression::getMinSpeed( t_idx i_pt ) const {
return m_speedsMin[i_pt];
}

double edge_v::models::GridExpression::getMaxSpeed( t_idx i_pt ) const {
return m_speedsMax[i_pt];
}