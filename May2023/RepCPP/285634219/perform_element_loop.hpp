#ifndef _perform_element_loop_hpp_
#define _perform_element_loop_hpp_


#include <BoxIterator.hpp>
#include <simple_mesh_description.hpp>
#include <SparseMatrix_functions.hpp>
#include <box_utils.hpp>
#include <Hex8_box_utils.hpp>
#include <Hex8_ElemData.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif


namespace miniFE {

template<typename GlobalOrdinal,
typename MatrixType, typename VectorType>
void
perform_element_loop(const simple_mesh_description<GlobalOrdinal>& mesh,
const Box& local_elem_box,
MatrixType& A, VectorType& b,
Parameters& )
{
typedef typename MatrixType::ScalarType Scalar;

int global_elems_x = mesh.global_box[0][1];
int global_elems_y = mesh.global_box[1][1];
int global_elems_z = mesh.global_box[2][1];


GlobalOrdinal num_elems = get_num_ids<GlobalOrdinal>(local_elem_box);
std::vector<GlobalOrdinal> elemIDs(num_elems);

BoxIterator iter = BoxIterator::begin(local_elem_box);
BoxIterator end  = BoxIterator::end(local_elem_box);

for(size_t i=0; iter != end; ++iter, ++i) {
elemIDs[i] = get_id<GlobalOrdinal>(global_elems_x, global_elems_y, global_elems_z,
iter.x, iter.y, iter.z);
}

const MINIFE_GLOBAL_ORDINAL elemID_size = elemIDs.size();

#pragma omp parallel for shared (elemIDs)
for(MINIFE_GLOBAL_ORDINAL i=0; i < elemID_size; ++i) {
ElemData<GlobalOrdinal,Scalar> elem_data;
compute_gradient_values(elem_data.grad_vals);

get_elem_nodes_and_coords(mesh, elemIDs[i], elem_data);
compute_element_matrix_and_vector(elem_data);
sum_into_global_linear_system(elem_data, A, b);
}

}

}

#endif

