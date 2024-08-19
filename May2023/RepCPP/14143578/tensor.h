#pragma once

#include "grid.h"
#include "operator.h"
#include "evaluation.h"
#include "evaluationX.h"
#include "dg/functors.h"
#include "dg/blas1.h"



namespace dg
{


template<class container>
struct SparseTensor
{
using container_type = container;
SparseTensor( ):m_mat_idx(3,-1) {}


template<class Topology>
SparseTensor( const Topology& grid){
construct(grid);
}


SparseTensor( const container& copyable ){
construct(copyable);
}

template<class Topology>
void construct( const Topology& grid){
m_mat_idx.resize(3,0);
for( int i=0; i<3; i++)
m_mat_idx( i,i) = 1;
m_values.resize(2);
dg::assign( dg::evaluate( dg::zero, grid), m_values[0]);
dg::assign( dg::evaluate( dg::one, grid), m_values[1]);
}

void construct( const container& copyable ){
m_mat_idx.resize(3,0);
for( int i=0; i<3; i++)
m_mat_idx( i,i) = 1;
m_values.assign(2,copyable);
dg::blas1::copy( 0., m_values[0]);
dg::blas1::copy( 1., m_values[1]);
}


template<class OtherContainer>
SparseTensor( const SparseTensor<OtherContainer>& src): m_mat_idx(3,-1), m_values(src.values().size()){
for(unsigned i=0; i<3; i++)
for(unsigned j=0; j<3; j++)
m_mat_idx(i,j)=src.idx(i,j);

for( unsigned i=0; i<src.values().size(); i++)
dg::assign( src.values()[i], m_values[i]);
}


int idx(unsigned i, unsigned j)const{
return m_mat_idx(i,j);
}

int& idx(unsigned i, unsigned j){
return m_mat_idx(i,j);
}


const container& value(size_t i, size_t j)const{
int k = m_mat_idx(i,j);
return m_values[k];
}

std::vector<container>& values() {
return m_values;
}

const std::vector<container>& values()const{
return m_values;
}


SparseTensor transpose()const{
SparseTensor tmp(*this);
tmp.m_mat_idx = m_mat_idx.transpose();
return tmp;
}

private:
dg::Operator<int> m_mat_idx;
std::vector<container> m_values;
};

}
