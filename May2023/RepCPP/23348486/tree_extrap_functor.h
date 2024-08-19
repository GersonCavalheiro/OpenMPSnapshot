

#ifndef SRC_TREE_EXTRAP_FUNCTOR_H_
#define SRC_TREE_EXTRAP_FUNCTOR_H_

#include <vector>

#include <cheb_node.hpp>
#include <profile.hpp>
#include <pvfmm_common.hpp>

#include "utils/common.h"
#include "utils/cubic.h"

namespace tbslas {

template <typename Real_t, class Tree_t>
class FieldExtrapFunctor {
public:
explicit FieldExtrapFunctor(Tree_t *tp, Tree_t *tc) : tp_(tp), tc_(tc) {
typedef typename Tree_t::Node_t Node_t;
tbslas::SimConfig *sim_config = tbslas::SimConfigSingleton::Instance();

Node_t *n_curr = tp_->PostorderFirst();
while (n_curr != NULL) {
if (!n_curr->IsGhost() && n_curr->IsLeaf()) break;
n_curr = tp_->PostorderNxt(n_curr);
}
data_dof_ = n_curr->DataDOF();
}

virtual ~FieldExtrapFunctor() {}

void operator()(const Real_t *query_points_pos, int num_points, Real_t *out) {
tbslas::NodeFieldFunctor<Real_t, Tree_t> tp_evaluator(tp_);
tbslas::NodeFieldFunctor<Real_t, Tree_t> tc_evaluator(tc_);
std::vector<Real_t> tnc_pnts_val;
tnc_pnts_val.resize(num_points * data_dof_);
tc_evaluator(query_points_pos, num_points, tnc_pnts_val.data());
std::vector<Real_t> tnp_pnts_val;
tnp_pnts_val.resize(num_points * data_dof_);
tp_evaluator(query_points_pos, num_points, tnp_pnts_val.data());
Real_t ccoeff = 3.0 / 2;
Real_t pcoeff = 0.5;

#pragma omp parallel for
for (int i = 0; i < tnc_pnts_val.size(); i++) {
out[i] = ccoeff * tnc_pnts_val[i] - pcoeff * tnp_pnts_val[i];
}
}

void update(Tree_t *new_tree, Real_t time) {
delete tp_;
tp_ = tc_;
tc_ = new_tree;
}

private:
Tree_t *tc_;
Tree_t *tp_;

int data_dof_;
};

}  
#endif  
