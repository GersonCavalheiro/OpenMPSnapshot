#ifndef MTTKRP_HPP
#define MTTKRP_HPP

#include <omp.h>
#include "master_library.hpp"
#include "omp_lib.hpp"



namespace v1
{
inline void mttkrp( const MatrixXd &X_mat, const MatrixXd &KR_p, const VectorXi Tns_dims, int Mode,  const unsigned int n_thrds, MatrixXd &Mttkrp)
{
#ifndef EIGEN_DONT_PARALLELIZE
Eigen::setNbThreads(1);
#endif

int rows_X_mat = X_mat.rows();
int cols_X_mat = X_mat.cols();

VectorXi rest_dims = Tns_dims;
rest_dims(Mode) = 1;
int max_dim = rest_dims.maxCoeff();
int rounds;
int rounds_sz;
VectorXi offset;

int cols_X_mat_full = rest_dims.prod();                         

Mttkrp.setZero();

if( cols_X_mat < cols_X_mat_full)                               
{   
if( cols_X_mat > n_thrds)
{
rounds = n_thrds;                                   
rounds_sz = cols_X_mat / n_thrds;               

int residual = cols_X_mat % n_thrds;                

if( residual != 0)                                  
{
VectorXi offset_tmp(rounds,1);
offset_tmp.setConstant(rounds_sz);              
offset_tmp(rounds - 1) += residual;             
offset = offset_tmp;

}
else
{
VectorXi offset_tmp(rounds,1);
offset_tmp.setConstant(rounds_sz);              
offset = offset_tmp;    
}
}
else
{
rounds = 1;      
rounds_sz = cols_X_mat;                                    
VectorXi offset_tmp(rounds,1);
offset_tmp.setConstant(rounds_sz);
offset = offset_tmp;
}


}
else 
{   
rounds = max_dim;
rounds_sz = cols_X_mat_full/max_dim;
VectorXi offset_tmp(rounds,1);
offset_tmp.setConstant(rounds_sz);
offset = offset_tmp;         

}

#pragma omp parallel for reduction(sum: Mttkrp) default(shared) num_threads(n_thrds)
for(int block = 0; block < rounds; block++ )
{
Mttkrp.noalias() += X_mat.block(0, block * rounds_sz, Mttkrp.rows(), offset(block)) * KR_p.block(block * rounds_sz, 0, offset(block), Mttkrp.cols());
}
}
} 

namespace v2
{
inline void mttkrp( const MatrixXd &X_mat, const MatrixXd &KR_p, const VectorXi Tns_dims, int Mode,  const unsigned int n_thrds, MatrixXd &Mttkrp)
{
Eigen :: setNbThreads(1);

int rows_X_mat = X_mat.rows();
int cols_X_mat = X_mat.cols();

VectorXi rest_dims = Tns_dims;
rest_dims(Mode) = 1;
int max_dim = rest_dims.maxCoeff();
int rounds;
int offset;

int cols_X_mat_full = rest_dims.prod();          

Mttkrp.setZero();

if( cols_X_mat < cols_X_mat_full)                
{   
offset = cols_X_mat/5;                      
rounds = 5;
}
else
{
offset = cols_X_mat_full/max_dim;
rounds = max_dim;
}

#pragma omp parallel for reduction(sum: Mttkrp) default(shared) num_threads(n_thrds)
for(int block = 0; block < rounds; block++ )
{
Mttkrp.noalias() += X_mat.block(0, block * offset, Mttkrp.rows(), offset) * KR_p.block(block * offset, 0, offset, Mttkrp.cols());
}
}
} 

namespace partial 
{





template <std::size_t  TNS_ORDER>
void mttpartialkrp(const int tensor_order, const Ref<const VectorXi> tensor_dims, const int tensor_rank, const int current_mode,
std::array<MatrixXd, TNS_ORDER> &Init_Factors, const Ref<const MatrixXd> Tensor_X, MatrixXd &MTTKRP,
const unsigned int num_threads)
{
#ifndef EIGEN_DONT_PARALLELIZE
Eigen::setNbThreads(1);
#endif

MTTKRP.setZero();

int mode_N = tensor_order - 1;

int mode_1 = 0;

if (current_mode == mode_N)
{
mode_N--;
}
else if (current_mode == mode_1)
{
mode_1 = 1;
}


int dim = tensor_dims.prod() / tensor_dims(current_mode);

int num_of_blocks = dim / tensor_dims(mode_1);

VectorXi rows_offset(tensor_order - 2);
for (int ii = tensor_order - 3, jj = mode_N; ii >= 0; ii--, jj--)
{
if (jj == current_mode)
{
jj--;
}
if (ii == tensor_order - 3)
{
rows_offset(ii) = num_of_blocks / tensor_dims(jj);
}
else
{
rows_offset(ii) = rows_offset(ii + 1) / tensor_dims(jj);
}
}


#pragma omp parallel for reduction(sum: MTTKRP) schedule(static, 1) num_threads(num_threads) proc_bind(close)
for (int block_idx = 0; block_idx < num_of_blocks; block_idx++)
{
MatrixXd Kr(1, tensor_rank);

Kr = Init_Factors[mode_N].row((block_idx / rows_offset(tensor_order - 3)) % tensor_dims(mode_N));
MatrixXd PartialKR(tensor_dims(mode_1), tensor_rank);

for (int ii = tensor_order - 4, jj = mode_N - 1; ii >= 0; ii--, jj--)
{
if (jj == current_mode)
{
jj--;
}
Kr = (Init_Factors[jj].row((block_idx / rows_offset(ii)) % tensor_dims(jj))).cwiseProduct(Kr);
}

for (int row = 0; row < tensor_dims(mode_1); row++)
{
PartialKR.row(row)  = ((Init_Factors[mode_1].row(row)).cwiseProduct(Kr));
}

MTTKRP.noalias() += Tensor_X.block(0, block_idx * tensor_dims(mode_1), tensor_dims(current_mode), tensor_dims(mode_1)) * PartialKR;
}
#ifndef EIGEN_DONT_PARALLELIZE
Eigen::setNbThreads(num_threads);
#endif
}








} 

#endif 
