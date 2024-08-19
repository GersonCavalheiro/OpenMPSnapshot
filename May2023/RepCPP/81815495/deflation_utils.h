
#if !defined(KRATOS_DEFLATION_UTILS )
#define  KRATOS_DEFLATION_UTILS






#include "includes/define.h"
#include "includes/model_part.h"
#include "includes/global_pointer_variables.h"
#include "utilities/atomic_utilities.h"

namespace Kratos
{




























class DeflationUtils
{
public:


typedef boost::numeric::ublas::compressed_matrix<double> SparseMatrixType;

typedef boost::numeric::ublas::vector<double> SparseVectorType;













void VisualizeAggregates(ModelPart::NodesContainerType& rNodes, Variable<double>& rVariable, const int max_reduced_size)
{
SparseMatrixType A(rNodes.size(),rNodes.size());
SparseMatrixType Adeflated;

std::vector< std::vector<int> > index_list(rNodes.size());

std::size_t total_size = 0;

int new_id = 1;
for(ModelPart::NodesContainerType::iterator in = rNodes.begin();  in!=rNodes.end(); in++)
in->SetId(new_id++);


std::size_t index_i;
for(ModelPart::NodesContainerType::iterator in = rNodes.begin();
in!=rNodes.end(); in++)
{
index_i = (in)->Id()-1;
auto& neighb_nodes = in->GetValue(NEIGHBOUR_NODES);

std::vector<int>& indices = index_list[index_i];
indices.reserve(neighb_nodes.size()+1);

indices.push_back(index_i);
for( auto i =	neighb_nodes.begin();
i != neighb_nodes.end(); i++)
{

int index_j = i->Id()-1;
indices.push_back(index_j);
}

std::sort(indices.begin(),indices.end());
std::vector<int>::iterator new_end = std::unique(indices.begin(),indices.end());

indices.erase(new_end,indices.end());

total_size += indices.size();

}

A.reserve(total_size,false);

for(unsigned int i=0; i<A.size1(); i++)
{
std::vector<int>& indices = index_list[i];
for(unsigned int j=0; j<indices.size(); j++)
{
A.push_back(i,indices[j] , 0.00);
}
}

std::vector<int> w(rNodes.size());
ConstructW(max_reduced_size, A, w, Adeflated);

int counter = 0;
for(ModelPart::NodesContainerType::iterator in=rNodes.begin(); in!=rNodes.end(); in++)
{
in->FastGetSolutionStepValue(rVariable) = w[counter++];
}
}

static void ConstructW(const std::size_t max_reduced_size, SparseMatrixType& rA, std::vector<int>& w, SparseMatrixType&  deflatedA)
{
KRATOS_TRY

std::size_t full_size = rA.size1();
w.resize(full_size,0);

std::size_t reduced_size = standard_aggregation<int>(rA.size1(),rA.index1_data().begin(), rA.index2_data().begin(), &w[0]);


std::vector<std::set<std::size_t> > deflatedANZ(reduced_size);

SparseMatrixType::iterator1 a_iterator = rA.begin1();

for (std::size_t i = 0; i < full_size; i++)
{
#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION
for (SparseMatrixType::iterator2 row_iterator = a_iterator.begin() ;
row_iterator != a_iterator.end() ; ++row_iterator)
{
#else
for (typename SparseMatrixType::iterator2 row_iterator = begin(a_iterator,
boost::numeric::ublas::iterator1_tag());
row_iterator != end(a_iterator,
boost::numeric::ublas::iterator1_tag()); ++row_iterator )
{
#endif
deflatedANZ[w[a_iterator.index1()]].insert(w[row_iterator.index2()]);
}

a_iterator++;
}


std::size_t NZ = 0;
for (std::size_t i = 0; i < reduced_size; i++)
NZ += deflatedANZ[i].size();

deflatedA = SparseMatrixType(reduced_size, reduced_size,NZ);

for(std::size_t i = 0 ; i < reduced_size ; i++)
{
for(std::set<std::size_t>::iterator j = deflatedANZ[i].begin() ; j != deflatedANZ[i].end() ; j++)
{
deflatedA.push_back(i,*j, 0.00);
}
}

if(reduced_size > max_reduced_size)
{
SparseMatrixType Areduced;
std::vector<int> wsmaller;
ConstructW(max_reduced_size, deflatedA, wsmaller, Areduced);
for(unsigned int i=0; i<full_size; i++)
{
int color = w[i];
int new_color = wsmaller[color];
w[i] = new_color;
}
deflatedA.clear();
deflatedA = Areduced;
reduced_size = wsmaller.size();
}



KRATOS_CATCH("")
}





static void ConstructW(const int max_reduced_size, SparseMatrixType& rA, std::vector<int>& w, SparseMatrixType&  deflatedA, const std::size_t block_size)
{
if(block_size == 1)
{
ConstructW(max_reduced_size,rA, w, deflatedA);
}
else
{
if(rA.size1()%block_size != 0 || rA.size2()%block_size != 0)
KRATOS_THROW_ERROR(std::logic_error,"the number of rows is not a multiple of block_size. Can not use the block deflation","")
if(rA.nnz()%block_size != 0)
KRATOS_THROW_ERROR(std::logic_error,"the number of non zeros is not a multiple of block_size. Can not use the block deflation","")

SparseMatrixType Ascalar;
ConstructScalarMatrix(rA.size1(),block_size,rA.index1_data().begin(), rA.index2_data().begin(), Ascalar);

SparseMatrixType deflatedAscalar;
std::vector<int> wscalar;
ConstructW(max_reduced_size/block_size,Ascalar, wscalar, deflatedAscalar);


std::vector<int> w(wscalar.size()*block_size);
for(std::size_t i=0; i<wscalar.size(); i++)
{
for(std::size_t j=0; j<block_size; j++)
{
w[i*block_size + j] = wscalar[i]*block_size+j;
}
}

SparseMatrixType deflatedA(deflatedAscalar.size1()*block_size,deflatedAscalar.size2()*block_size);
deflatedA.reserve(deflatedAscalar.nnz()*block_size*block_size);
ExpandScalarMatrix(rA.size1(),block_size,rA.index1_data().begin(), rA.index2_data().begin(), deflatedA);



}
}



static void ApplyW(const std::vector<int>& w, const SparseVectorType& x, SparseVectorType& y)
{
#pragma omp parallel for
for(int i=0; i<static_cast<int>(w.size()); i++)
{
y[i] = x[w[i]];
}
}


static void ApplyWtranspose(const std::vector<int>& w, const SparseVectorType& x, SparseVectorType& y)
{
#pragma omp parallel for
for(int i=0; i<static_cast<int>(y.size()); i++)
y[i] = 0.0;

#if(_MSC_FULL_VER == 190023506)
for(int i=0; i<static_cast<int>(w.size()); i++)
{
y[w[i]] += x[i];
}
#else
#pragma omp parallel for
for(int i=0; i<static_cast<int>(w.size()); i++) {
AtomicAdd(y[w[i]], x[i]);
}
#endif
}

/




















private:














template <class I>
static I standard_aggregation(const I n_row,
const std::size_t Ap[],
const std::size_t Aj[],
I  x[])
{
std::fill(x, x + n_row, 0);

I next_aggregate = 1; 

for(I i = 0; i < n_row; i++)
{
if(x[i])
{
continue;    
}

const I row_start = Ap[i];
const I row_end   = Ap[i+1];

bool has_aggregated_neighbors = false;
bool has_neighbors            = false;
for(I jj = row_start; jj < row_end; jj++)
{
const I j = Aj[jj];
if( i != j )
{
has_neighbors = true;
if( x[j] )
{
has_aggregated_neighbors = true;
break;
}
}
}

if(!has_neighbors)
{
x[i] = -n_row;
}
else if (!has_aggregated_neighbors)
{
x[i] = next_aggregate;
for(I jj = row_start; jj < row_end; jj++)
{
x[Aj[jj]] = next_aggregate;
}
next_aggregate++;
}
}

for(I i = 0; i < n_row; i++)
{
if(x[i])
{
continue;    
}

for(I jj = static_cast<I>(Ap[i]); jj < static_cast<I>(Ap[i+1]); jj++)
{
const I j = Aj[jj];

const I xj = x[j];
if(xj > 0)
{
x[i] = -xj;
break;
}
}
}

next_aggregate--;

for(I i = 0; i < n_row; i++)
{
const I xi = x[i];

if(xi != 0)
{
if(xi > 0)
x[i] = xi - 1;
else if(xi == -n_row)
x[i] = -1;
else
x[i] = -xi - 1;
continue;
}

const I row_start = Ap[i];
const I row_end   = Ap[i+1];

x[i] = next_aggregate;

for(I jj = row_start; jj < row_end; jj++)
{
const I j = Aj[jj];

if(x[j] == 0)  
{
x[j] = next_aggregate;
}
}
next_aggregate++;
}


return next_aggregate; 
}




static void ConstructScalarMatrix(const std::size_t n_row, const std::size_t block_size,
const std::size_t Ap[],
const std::size_t Aj[],
SparseMatrixType& Ascalar
)
{
Ascalar.resize(n_row/block_size,n_row/block_size,0.0);
std::size_t scalar_size = (Ap[n_row]-Ap[0])/(block_size*block_size);
Ascalar.reserve(scalar_size);

for(std::size_t i = 0; i < n_row; i++)
{
if(i%block_size == 0)
{
std::size_t iscalar = i/block_size;
const std::size_t row_start = Ap[i];
const std::size_t row_end   = Ap[i+1];

for(std::size_t jj = row_start; jj < row_end; jj++)
{
const std::size_t j = Aj[jj];
if(j%block_size == 0)
{
std::size_t jscalar = j/block_size;
Ascalar.push_back(iscalar,jscalar,0.0);
}
}
}
}
}

static void ExpandScalarMatrix(const std::size_t n_row, const std::size_t block_size,
const std::size_t Ap[],
const std::size_t Aj[],
SparseMatrixType& Aexpanded
)
{
Aexpanded.resize(n_row*block_size,n_row*block_size,0.0);
std::size_t expanded_size = (Ap[n_row]-Ap[0])*block_size*block_size;
Aexpanded.reserve(expanded_size);

for(std::size_t i = 0; i < n_row; i++)
{
const std::size_t row_start = Ap[i];
const std::size_t row_end   = Ap[i+1];

for(std::size_t isub=0; isub<block_size; isub++)
{
std::size_t iexpanded = i*block_size + isub;
for(std::size_t jj = row_start; jj < row_end; jj++)
{
const std::size_t j = Aj[jj];

for(std::size_t jsub=0; jsub<block_size; jsub++)
{
std::size_t jexpanded = j*block_size+jsub;
Aexpanded.push_back(iexpanded,jexpanded,0.0);
}
}
}
}
}

























}; 









} 

#endif 
