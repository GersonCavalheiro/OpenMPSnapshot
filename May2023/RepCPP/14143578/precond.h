#pragma once

#include <vector>

namespace dg
{

namespace create
{

namespace detail
{
template<class T>
void sparsify( cusp::array1d<int, cusp::host_memory>& row_offsets,
cusp::array1d<int, cusp::host_memory>& column_indices,
cusp::array1d<T, cusp::host_memory>& values,
const int i,
const thrust::host_vector<T>& zw,
const std::vector<int>& iz_zw,
unsigned nnzmax, T threshold)
{

std::vector<std::pair<double, int>> pairs;
std::vector<std::pair<int, double>> accept;
accept.push_back( {i, zw[i]});
for( auto idx : iz_zw)
if( idx != i) 
pairs.push_back( { zw[idx], idx});
std::sort( pairs.begin(), pairs.end(), std::greater<>());
for( int k=0; k<(int)nnzmax-1; k++)
{
if( k < (int)pairs.size() && fabs(pairs[k].first) > threshold)
{
accept.push_back({pairs[k].second, pairs[k].first});
}
}
std::sort( accept.begin(), accept.end());


row_offsets.push_back(row_offsets[i]);
for( auto pair : accept)
{
column_indices.push_back( pair.first);
values.push_back( pair.second);
row_offsets[i+1]++;
}
}
}


template<class T>
void sainv_precond(
const cusp::csr_matrix<int, T, cusp::host_memory>& a,
cusp::csr_matrix<int, T, cusp::host_memory>& s,
thrust::host_vector<T>& d,
const thrust::host_vector<T>& weights,
unsigned nnzmax,
T threshold)
{
unsigned n = a.num_rows;

d.resize( n, 0.);

for( int j = a.row_offsets[0]; j<a.row_offsets[1]; j++)
{
if( a.column_indices[j] == 0)
d[0] = a.values[j]*weights[0];
}
if( fabs( d[0] ) < threshold)
d[0] = threshold;
cusp::array1d<int, cusp::host_memory> row_offsets, column_indices;
cusp::array1d<T, cusp::host_memory> values;

row_offsets.push_back(0);
row_offsets.push_back(1);
column_indices.push_back( 0);
values.push_back( 1.0);

for( int i = 1; i<(int)n; i++)
{
thrust::host_vector<T> zw( n, 0.);
std::vector<int> iz_zw; 
zw[i] = 1.0;
iz_zw.push_back(i);
std::vector<int> s;
for( int k = a.row_offsets[i]; k<a.row_offsets[i+1]; k++)
{
if( a.column_indices[k] < i )
s.push_back( a.column_indices[k]);
}
while( !s.empty())
{
auto it = std::min_element( s.begin(), s.end());
int j = *it; 
s.erase( it);
d[i] = 0.0;
for( int k = a.row_offsets[j]; k<a.row_offsets[j+1]; k++)
{
d[i] += weights[j]*a.values[k]*zw[ a.column_indices[k]];
}
T alpha = d[i]/d[j];
if( fabs( alpha) > threshold)
{
for( int k = row_offsets[j]; k<row_offsets[j+1]; k++)
{
int zkk = column_indices[k];
zw[ zkk] -= alpha * values[k];
if (std::find(iz_zw.begin(), iz_zw.end(), zkk) == iz_zw.end()) {
iz_zw.push_back( zkk);
}
for( int l = a.row_offsets[zkk]; l < a.row_offsets[zkk+1]; l++)
{
if ( (std::find(s.begin(), s.end(), l) == s.end()) && (j<l) && (l < i) ) {
s.push_back( l);
}
}
}
}
}
d[i] = 0.0;
for( int k = a.row_offsets[i]; k<a.row_offsets[i+1]; k++)
d[i] += a.values[k]*zw[ a.column_indices[k]]*weights[i];
if( fabs(d[i]) < threshold)
d[i] = threshold;
detail::sparsify( row_offsets, column_indices, values, i, zw, iz_zw, nnzmax, threshold);
}
s.resize( n, n, values.size());
s.column_indices = column_indices;
s.row_offsets = row_offsets;
s.values = values;


}
} 

}
