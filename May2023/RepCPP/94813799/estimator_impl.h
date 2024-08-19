
#ifndef MAPPEL_ESTIMATOR_IMPL_H
#define MAPPEL_ESTIMATOR_IMPL_H

#include <thread>
#include <cmath>
#include <armadillo>

#include "Mappel/estimator.h"
#include "Mappel/estimator_helpers.h"
#include "Mappel/rng.h"
#include "Mappel/numerical.h"
#include "Mappel/display.h"

#ifdef WIN32
using namespace boost::chrono;
#else
using namespace std::chrono;
#endif

namespace mappel {

namespace estimator {




template<class Model>
const int IterativeMaximizer<Model>::MaximizerData::DefaultMaxSeqLength = 50;
template<class Model>
const int IterativeMaximizer<Model>::DefaultIterations=100;
template<class Model>
const double IterativeMaximizer<Model>::convergence_min_function_change_ratio = 1.0e-9; 
template<class Model>
const double IterativeMaximizer<Model>::convergence_min_step_size_ratio = 1.0e-9; 

template<class Model>
const double IterativeMaximizer<Model>::min_eigenvalue_correction_delta = 1e-3; 


template<class Model>
const double IterativeMaximizer<Model>::backtrack_min_ratio = 0.05; 
template<class Model>
const double IterativeMaximizer<Model>::backtrack_max_ratio = 0.50; 
template<class Model>
const double IterativeMaximizer<Model>::backtrack_min_linear_step_ratio = 1e-3; 
template<class Model>
const int IterativeMaximizer<Model>::max_backtracks = 8; 

template<class Model>
const double IterativeMaximizer<Model>::min_profile_bound_residual = 1e-4; 



template<class Model>
const double TrustRegionMaximizer<Model>::rho_cauchy_min = 0.1;  
template<class Model>
const double TrustRegionMaximizer<Model>::rho_obj_min = 0.25;  
template<class Model>
const double TrustRegionMaximizer<Model>::rho_obj_opt = 0.75;  
template<class Model>
const double TrustRegionMaximizer<Model>::trust_radius_decrease_min = 0.125; 
template<class Model>
const double TrustRegionMaximizer<Model>::trust_radius_decrease = 0.25; 
template<class Model>
const double TrustRegionMaximizer<Model>::trust_radius_increase = 2; 
template<class Model>
const double TrustRegionMaximizer<Model>::convergence_min_trust_radius = 1.0e-8; 



template<class Model>
const int SimulatedAnnealingMaximizer<Model>::DefaultNumIterations = 500; 
template<class Model>
const double SimulatedAnnealingMaximizer<Model>::Default_T_Init = 100.; 
template<class Model>
const double SimulatedAnnealingMaximizer<Model>::DefaultCoolingRate = 1.02; 


template<class Model>
Estimator<Model>::Estimator(const Model &_model)
: model(_model),
exit_counts(NumExitCodes,arma::fill::zeros)
{ }

template<class Model>
const Model& Estimator<Model>::get_model()
{ return model; }

template<class Model>
void Estimator<Model>::estimate_max(const ModelDataT<Model> &data, MLEData &mle)
{
auto theta_init = model.make_param();
theta_init.zeros();
StencilT<Model> stencil;
estimate_max(data, theta_init, mle, stencil);
}

template<class Model>
void Estimator<Model>::estimate_max(const ModelDataT<Model> &data, const ParamT<Model> &theta_init, MLEData &mle)
{
StencilT<Model> mle_stencil;
estimate_max(data, theta_init, mle, mle_stencil);
}

template<class Model>
void Estimator<Model>::estimate_max(const ModelDataT<Model> &data, const ParamT<Model> &theta_init, MLEData &mle, StencilT<Model> &mle_stencil)
{
try {
auto start_walltime = ClockT::now();
compute_estimate(data, theta_init, mle, mle_stencil);
record_walltime(start_walltime, 1);
return;
} catch(std::exception &err) {
#ifdef DEBUG
std::cout<<"std::exception: "<<err.what()<<std::endl;
std::cout<<"Theta init: "<<theta_init.t();
print_text_image(std::cout,data);
#endif
record_exit_code(ExitCode::Error);
} catch(...) {
#ifdef DEBUG
std::cout<<"Unknown exception"<<std::endl;
std::cout<<"Theta init: "<<theta_init.t();
print_text_image(std::cout,data);
#endif
record_exit_code(ExitCode::Error);
}
mle.theta = theta_init;
mle.rllh = -INFINITY;
mle.obsI.zeros();
}

template<class Model>
void Estimator<Model>::estimate_max_debug(const ModelDataT<Model> &data, const ParamT<Model> &theta_init,  MLEDebugData &mle)
{
try {
auto start_walltime = ClockT::now();
StencilT<Model> mle_stencil;
compute_estimate_debug(data,theta_init,mle,mle_stencil);
record_walltime(start_walltime, 1);
return;
} catch(std::exception &err) {
std::cout<<"std::exception: "<<err.what()<<std::endl;
std::cout<<"Theta init: "<<theta_init.t();
print_text_image(std::cout,data);
this->record_exit_code(ExitCode::Error);
} catch(...) {
std::cout<<"Unknown exception"<<std::endl;
std::cout<<"Theta init: "<<theta_init.t();
print_text_image(std::cout,data);
this->record_exit_code(ExitCode::Error);
}
mle.theta = theta_init;
mle.rllh = -INFINITY;
mle.obsI.zeros();
}

template<class Model>
void Estimator<Model>::estimate_max_stack(const ModelDataStackT<Model> &data_stack, MLEDataStack &mle_data_stack)
{
ParamVecT<Model> theta_init(model.get_num_params(), model.get_size_image_stack(data_stack), arma::fill::zeros);
estimate_max_stack(data_stack, theta_init, mle_data_stack);
}

template<class Model>
double Estimator<Model>::estimate_profile_max(const ModelDataT<Model> &data, const IdxVecT &fixed_idxs,
const ParamT<Model> &fixed_theta_init, StencilT<Model> &stencil_max)
{
try {
double rllh;
auto start_walltime = ClockT::now();
rllh = compute_profile_estimate(data,fixed_theta_init,fixed_idxs,stencil_max);
record_walltime(start_walltime, 1);
return rllh;
} catch(std::exception &err) {
#ifdef DEBUG
std::cout<<"std::exception: "<<err.what()<<std::endl;
std::cout<<"Theta init: "<<fixed_theta_init.t();
print_text_image(std::cout,data);
#endif
record_exit_code(ExitCode::Error);
} catch(...) {
#ifdef DEBUG
std::cout<<"Unknown exception"<<std::endl;
std::cout<<"Theta init: "<<fixed_theta_init.t();
print_text_image(std::cout,data);
#endif
record_exit_code(ExitCode::Error);
}
return -INFINITY;
}

template<class Model>
void
Estimator<Model>::estimate_profile_bounds(const ModelDataT<Model> &data, ProfileBoundsData &est)
{
auto start_walltime = ClockT::now();
est.initialize_arrays(model.get_num_params());
try {
for(IdxT n=0; n<est.Nparams_est; n++) {
auto step = subroutine::solve_profile_initial_step(est.mle.obsI, est.estimated_idxs(n), est.target_rllh_delta);
compute_profile_bound(data,est,step,n,0); 
compute_profile_bound(data,est,-step,n,1);
}
record_walltime(start_walltime, 2);
return;
} catch(std::exception &err) {
#ifdef DEBUG
std::cout<<"std::exception: "<<err.what()<<std::endl;
std::cout<<"Theta mle: "<<est.mle.theta.t()<<" estimated_idxsx:"<<est.estimated_idxs<<" llh_delta:"<<est.target_rllh_delta<<std::endl;
print_text_image(std::cout,data);
#endif
record_exit_code(ExitCode::Error);
} catch(...) {
#ifdef DEBUG
std::cout<<"Unknown exception"<<std::endl;
std::cout<<"Theta mle: "<<est.mle.theta.t()<<" estimated_idxs:"<<est.estimated_idxs<<" llh_delta:"<<est.target_rllh_delta<<std::endl;
print_text_image(std::cout,data);
#endif
record_exit_code(ExitCode::Error);
}
est.profile_lb = est.mle.theta;
est.profile_ub = est.mle.theta;
est.profile_points_lb.zeros();
est.profile_points_ub.zeros();
est.profile_points_lb_rllh = -INFINITY;
est.profile_points_lb_rllh = -INFINITY;
}

template<class Model>
void
Estimator<Model>::estimate_profile_bounds_debug(const ModelDataT<Model> &data, ProfileBoundsDebugData &est)
{
try {
auto start_walltime = ClockT::now();
compute_profile_bound_debug(data,est);
record_walltime(start_walltime, 2);
return;
} catch(std::exception &err) {
std::cout<<"std::exception: "<<err.what()<<std::endl;
std::cout<<"Theta mle: "<<est.mle.theta.t()<<" estimated_idx:"<<est.estimated_idx<<" llh_delta:"<<est.target_rllh_delta<<std::endl;
print_text_image(std::cout,data);
record_exit_code(ExitCode::Error);
} catch(...) {
std::cout<<"Unknown exception"<<std::endl;
std::cout<<"Theta mle: "<<est.mle.theta.t()<<" param_idx:"<<est.estimated_idx<<" llh_delta:"<<est.target_rllh_delta<<std::endl;
print_text_image(std::cout,data);
record_exit_code(ExitCode::Error);
}
}



template<class Model>
void Estimator<Model>::compute_estimate_debug(const ModelDataT<Model> &im, const ParamT<Model> &theta_init,
MLEDebugData &mle_debug, StencilT<Model> &mle_stencil)
{
mle_debug.sequence.set_size(model.get_num_params(),1);
mle_debug.sequence_rllh.set_size(1);
MLEData mle;
compute_estimate(im,theta_init,mle,mle_stencil);
mle_debug.theta = mle.theta;
mle_debug.rllh = mle.rllh;
mle_debug.obsI = mle.obsI;
mle_debug.sequence.col(0) = mle_debug.theta;
mle_debug.sequence_rllh(0) = mle_debug.rllh;
}

template<class Model>
double Estimator<Model>::compute_profile_estimate(const ModelDataT<Model> &data, const ParamT<Model> &theta_init,
const IdxVecT &fixed_idxs, StencilT<Model> &max_stencil)
{
std::ostringstream msg;
msg<<"Profile likelihood not implemented for this estimator on model: "<<model.name;
throw NotImplementedError(msg.str());
}

template<class Model>
void Estimator<Model>::compute_profile_bound(const ModelDataT<Model> &data, ProfileBoundsData &est,
const VecT &init_step, IdxT param_idx, IdxT which_bound)
{
std::ostringstream msg;
msg<<"compute_profile_bound not implemented for this estimator on model: "<<model.name;
throw NotImplementedError(msg.str());
}

template<class Model>
void Estimator<Model>::compute_profile_bound_debug(const ModelDataT<Model> &data, ProfileBoundsDebugData &est)
{
std::ostringstream msg;
msg<<"compute_profile_bound_debug not implemented for this estimator on model: "<<model.name;
throw NotImplementedError(msg.str());
}

template<class Model>
StatsT Estimator<Model>::get_stats()
{
StatsT stats;
stats["num_estimations"] = num_estimations;
stats["total_walltime"] = total_walltime;
stats["mean_walltime"] = total_walltime/num_estimations;
stats["num_exit_error"] = exit_counts(static_cast<IdxT>(ExitCode::Error));
stats["num_exit_success"] = exit_counts(static_cast<IdxT>(ExitCode::Success));
stats["num_exit_step_size"] = exit_counts(static_cast<IdxT>(ExitCode::StepSize));
stats["num_exit_grad_ratio"] = exit_counts(static_cast<IdxT>(ExitCode::GradRatio));
stats["num_exit_trust_region_radius"] = exit_counts(static_cast<IdxT>(ExitCode::TrustRegionRadius));
stats["num_exit_max_iter"] = exit_counts(static_cast<IdxT>(ExitCode::MaxIter));
stats["num_exit_max_backtracks"] = exit_counts(static_cast<IdxT>(ExitCode::MaxBacktracks));
return stats;
}

template<class Model>
void Estimator<Model>::clear_stats()
{
num_estimations = 0;
total_walltime = 0.;
exit_counts.zeros();
}

template<class Model>
std::ostream& operator<<(std::ostream &out, Estimator<Model> &estimator)
{
auto stats=estimator.get_stats();
out<<"[Estimator: "<<estimator.name()<<"<"<<estimator.model.name<<">]\n";
for(auto& stat: stats) out<<"\t"<<stat.first<<"="<<stat.second<<"\n";
return out;
}

template<class Model>
void Estimator<Model>::record_walltime(ClockT::time_point start_walltime, int num_estimations_)
{
double walltime = duration_cast<duration<double>>(ClockT::now() - start_walltime).count();
total_walltime += walltime;
num_estimations += num_estimations_;
}



template<class Model>
ThreadedEstimator<Model>::ThreadedEstimator(const Model &model)
: Estimator<Model>(model),
max_threads(std::thread::hardware_concurrency()),
num_threads(1)
{ }

template<class Model>
void ThreadedEstimator<Model>::estimate_max_stack(const ModelDataStackT<Model> &data_stack, const ParamVecT<Model> &theta_init_stack,
MLEDataStack &mle_stack)
{
auto start_walltime=ClockT::now();
mle_stack.Ndata = model.get_size_image_stack(data_stack);
#pragma omp parallel
{
if(omp_get_thread_num()==0) num_threads = omp_get_num_threads();
MLEData mle;
StencilT<Model> mle_stencil;
#pragma omp for
for(IdxT n=0; n<mle_stack.Ndata; n++) {
auto im = model.get_image_from_stack(data_stack,n);
try {
this->compute_estimate(im, theta_init_stack.col(n), mle, mle_stencil);
mle_stack.theta.col(n) = mle.theta;
mle_stack.rllh(n) = mle.rllh;
mle_stack.obsI.slice(n) = mle.obsI;
} catch(std::exception &err) {
#ifdef DEBUG
std::cout<<"std::exception: "<<err.what()<<std::endl;
std::cout<<"Theta init: "<<theta_init_stack.col(n).t();
print_text_image(std::cout,im);
#endif
mle_stack.rllh(n) = -INFINITY;
record_exit_code(ExitCode::Error);
} catch(...) {
#ifdef DEBUG
std::cout<<"Unknown exception"<<std::endl;
std::cout<<"Theta init: "<<theta_init_stack.col(n).t();
print_text_image(std::cout,im);
#endif
mle_stack.rllh(n) = -INFINITY;
record_exit_code(ExitCode::Error);
}
}
}
this->record_walltime(start_walltime, mle_stack.Ndata);
}

template<class Model>
void ThreadedEstimator<Model>::estimate_profile_max(const ModelDataT<Model> &data,
const ParamVecT<Model> &theta_init, ProfileLikelihoodData &prof)
{
auto start_walltime=ClockT::now();
prof.Nfixed = prof.fixed_idxs.n_elem;
prof.Nvalues = prof.fixed_values.n_cols;
auto Np = model.get_num_params();
auto fixed_theta_init = theta_init;
fixed_theta_init.rows(prof.fixed_idxs) = prof.fixed_values;
prof.profile_likelihood.set_size(prof.Nvalues);
prof.profile_parameters.set_size(Np,prof.Nvalues);

#pragma omp parallel
{
if(omp_get_thread_num()==0) num_threads = omp_get_num_threads();
StencilT<Model> max_stencil;
#pragma omp for
for(IdxT n=0; n<prof.Nvalues; n++){
try {
prof.profile_likelihood(n) = this->compute_profile_estimate(data, fixed_theta_init.col(n), prof.fixed_idxs, max_stencil);
prof.profile_parameters.col(n) = max_stencil.theta;
} catch(std::exception &err) {
#ifdef DEBUG
std::cout<<"std::exception: "<<err.what()<<std::endl;
std::cout<<"Fixed Theta init: "<<fixed_theta_init.col(n).t();
print_text_image(std::cout,data);
#endif
prof.profile_likelihood(n)=-INFINITY;
prof.profile_parameters.col(n)=fixed_theta_init.col(n);
record_exit_code(ExitCode::Error);
} catch(...) {
#ifdef DEBUG
std::cout<<"Unknown exception"<<std::endl;
std::cout<<"Fixed Theta init: "<<fixed_theta_init.col(n).t();
print_text_image(std::cout,data);
#endif
prof.profile_likelihood(n)=-INFINITY;
prof.profile_parameters.col(n)=fixed_theta_init.col(n);
record_exit_code(ExitCode::Error);
}
}
}
this->record_walltime(start_walltime, prof.Nvalues);
}

template<class Model>
void ThreadedEstimator<Model>::estimate_profile_bounds_parallel(const ModelDataT<Model> &data, ProfileBoundsData &est)
{
auto start_walltime=ClockT::now();
auto Np = model.get_num_params();
if(est.estimated_idxs.is_empty()) est.estimated_idxs = arma::regspace<IdxVecT>(0,Np-1);
est.initialize_arrays(model.get_num_params());
#pragma omp parallel
{
if(omp_get_thread_num()==0) num_threads = omp_get_num_threads();
#pragma omp for
for(IdxT n=0; n<est.Nparams_est; n++) {
try {
auto step = subroutine::solve_profile_initial_step(est.mle.obsI, est.estimated_idxs(n), est.target_rllh_delta);
this->compute_profile_bound(data,est,step,est.estimated_idxs(n),0); 
this->compute_profile_bound(data,est,-step,est.estimated_idxs(n),1);
} catch(std::exception &err) {
#ifdef DEBUG
std::cout<<"std::exception: "<<err.what()<<std::endl;
std::cout<<"Theta mle: "<<est.mle.theta.t()<<" param_idx:"<<n<<" llh_delta:"<<est.target_rllh_delta<<std::endl;
print_text_image(std::cout,data);
#endif
record_exit_code(ExitCode::Error);
} catch(...) {
#ifdef DEBUG
std::cout<<"Unknown exception"<<std::endl;
std::cout<<"Theta mle: "<<est.mle.theta.t()<<" param_idx:"<<n<<" llh_delta:"<<est.target_rllh_delta<<std::endl;
print_text_image(std::cout,data);
#endif
record_exit_code(ExitCode::Error);
}
}
}
this->record_walltime(start_walltime, est.Nparams_est*2);
}

template<class Model>
void ThreadedEstimator<Model>::estimate_profile_bounds_stack(const ModelDataStackT<Model> &data_stack, ProfileBoundsDataStack &est_stack)
{
auto start_walltime=ClockT::now();
auto Np = model.get_num_params();
est_stack.Ndata = model.get_size_image_stack(data_stack);
if(est_stack.estimated_idxs.is_empty()) est_stack.estimated_idxs = arma::regspace<IdxVecT>(0,Np-1);
est_stack.initialize_arrays(model.get_num_params());
#pragma omp parallel
{
if(omp_get_thread_num()==0) num_threads = omp_get_num_threads();
ProfileBoundsData est;
est.estimated_idxs = est_stack.estimated_idxs;
est.initialize_arrays(Np);
est.target_rllh_delta = est_stack.target_rllh_delta;
#pragma omp for
for(IdxT k=0; k<est_stack.Ndata; k++) {
auto data = model.get_image_from_stack(data_stack,k);
est.mle.theta = est_stack.mle.theta.col(k);
est.mle.rllh = est_stack.mle.rllh(k);
est.mle.obsI = est_stack.mle.obsI.slice(k);
for(IdxT n=0; n<est_stack.Nparams_est; n++) {
try {
auto step = subroutine::solve_profile_initial_step(est.mle.obsI, est.estimated_idxs(n), est.target_rllh_delta);
this->compute_profile_bound(data,est,step,est.estimated_idxs(n),0); 
this->compute_profile_bound(data,est,-step,est.estimated_idxs(n),1);
} catch(std::exception &err) {
#ifdef DEBUG
std::cout<<"std::exception: "<<err.what()<<std::endl;
std::cout<<"Theta mle: "<<est.mle.theta.col(k).t()<<" param_idx:"<<n<<std::endl;
print_text_image(std::cout,data);
#endif
record_exit_code(ExitCode::Error);
} catch(...) {
#ifdef DEBUG
std::cout<<"Unknown exception"<<std::endl;
std::cout<<"Theta mle: "<<est.mle.theta.col(k).t()<<" param_idx:"<<n<<std::endl;
print_text_image(std::cout,data);
#endif
record_exit_code(ExitCode::Error);
}
}
est_stack.profile_lb.col(k) = est.profile_lb;
est_stack.profile_ub.col(k) = est.profile_ub;
est_stack.profile_points_lb.slice(k) = est.profile_points_lb;
est_stack.profile_points_ub.slice(k) = est.profile_points_ub;
est_stack.profile_points_lb_rllh.col(k) = est.profile_points_lb_rllh;
est_stack.profile_points_ub_rllh.col(k) = est.profile_points_ub_rllh;
}
}
this->record_walltime(start_walltime, est_stack.Nparams_est*2*est_stack.Ndata);
}

template<class Model>
StatsT ThreadedEstimator<Model>::get_stats()
{
std::lock_guard<std::mutex> lock(mtx);
auto stats = Estimator<Model>::get_stats();
stats["num_threads"] = num_threads;
stats["total_thread_time"] = num_threads*this->total_walltime;
stats["mean_estimation_time"] =  num_threads*this->total_walltime / this->num_estimations;
return stats;
}

template<class Model>
StatsT ThreadedEstimator<Model>::get_debug_stats()
{
return ThreadedEstimator<Model>::get_stats();
}

template<class Model>
void ThreadedEstimator<Model>::clear_stats()
{
std::lock_guard<std::mutex> lock(mtx);
Estimator<Model>::clear_stats();
num_threads=1;
}

template<class Model>
void ThreadedEstimator<Model>::record_exit_code(ExitCode code)
{
std::lock_guard<std::mutex> lock(mtx);
this->exit_counts(static_cast<IdxT>(code))++;
}


template<class Model>
void HeuristicEstimator<Model>::compute_estimate(const ModelDataT<Model> &im, const ParamT<Model> &theta_init,
MLEData &mle_data, StencilT<Model> &mle_stencil)
{
mle_stencil = model.initial_theta_estimate(im,theta_init);
mle_data.theta = mle_stencil.theta;
mle_data.rllh = methods::objective::rllh(model, im, mle_stencil);
mle_data.obsI = methods::observed_information(model, im, mle_stencil);
this->record_exit_code(ExitCode::Success);
}

template<class Model>
StatsT HeuristicEstimator<Model>::get_stats()
{
auto stats = ThreadedEstimator<Model>::get_stats();
std::lock_guard<std::mutex> lock(this->mtx);
double N = static_cast<double>(this->num_estimations);
stats["total_iterations"] = N;
stats["mean_iterations"] = 1;
stats["total_fun_evals"] = 0;
stats["total_der_evals"] = 0;
stats["mean_fun_evals"] = 0;
stats["mean_der_evals"] = 0;
return stats;
}

template<class Model>
StatsT HeuristicEstimator<Model>::get_debug_stats()
{
return get_stats();
}


template<class Model>
void CGaussHeuristicEstimator<Model>::compute_estimate(const ModelDataT<Model> &data, const ParamT<Model> &theta_init,
MLEData &mle, StencilT<Model> &mle_stencil)
{
mle.theta = cgauss_heuristic_compute_estimate(model,data,theta_init);
if(!model.theta_in_bounds(mle.theta)) {
std::ostringstream msg;
msg<<"CGaussHeuristicEstimator produced out-of-bounds values. Theta_init: "<<theta_init.t()<<" Theta_est:"<<mle.theta.t();
throw NumericalError(msg.str());
}
mle_stencil = model.make_stencil(mle.theta);
mle.rllh = methods::objective::rllh(model, data, mle_stencil);
mle.obsI = methods::observed_information(model, data, mle_stencil);
this->record_exit_code(ExitCode::Success);
}

template<class Model>
StatsT CGaussHeuristicEstimator<Model>::get_stats()
{
auto stats = ThreadedEstimator<Model>::get_stats();
std::lock_guard<std::mutex> lock(this->mtx);
double N = static_cast<double>(this->num_estimations);
stats["total_iterations"] = N;
stats["mean_iterations"] = 1;
stats["total_fun_evals"] = 0;
stats["total_der_evals"] = 0;
stats["mean_fun_evals"] = 0;
stats["mean_der_evals"] = 0;
return stats;
}

template<class Model>
StatsT CGaussHeuristicEstimator<Model>::get_debug_stats()
{
return get_stats();
}


template<class Model>
const int CGaussMLE<Model>::DefaultIterations=50;

template<class Model>
StatsT CGaussMLE<Model>::get_stats()
{
auto stats = ThreadedEstimator<Model>::get_stats();
std::lock_guard<std::mutex> lock(this->mtx);
double N = static_cast<double>(this->num_estimations);
stats["total_iterations"] = N*num_iterations;
stats["mean_iterations"] = num_iterations;
stats["total_fun_evals"] = num_iterations;
stats["total_der_evals"] = N*num_iterations;
stats["mean_fun_evals"] = num_iterations;
stats["mean_der_evals"] = N*num_iterations;
return stats;
}

template<class Model>
StatsT CGaussMLE<Model>::get_debug_stats()
{
return get_stats();
}

template<class Model>
void CGaussMLE<Model>::compute_estimate(const ModelDataT<Model> &im, const ParamT<Model> &theta_init_,
MLEData &mle_data, StencilT<Model> &mle_stencil)
{
auto theta_init = theta_init_;
if(!model.theta_in_bounds(theta_init)) {
auto init_est = cgauss_heuristic_compute_estimate(model,im,theta_init);
ParamT<Model> lb = model.get_lbound()+model.bounds_epsilon;
ParamT<Model> ub = model.get_ubound()-model.bounds_epsilon;
for(IdxT i=0;i<model.get_num_params();i++)
if(theta_init(i)<lb(i) || theta_init(i)>ub(i))
theta_init(i) = init_est(i);
}
mle_data.theta = cgauss_compute_estimate(model,im,theta_init,num_iterations);
if(!model.theta_in_bounds(mle_data.theta)) {
mle_stencil = model.make_stencil(theta_init);
this->record_exit_code(ExitCode::Error);
} else {
mle_stencil = model.make_stencil(mle_data.theta);
this->record_exit_code(ExitCode::Success);
}
mle_data.rllh = methods::objective::rllh(model,im,mle_stencil);
mle_data.obsI = methods::observed_information(model, im, mle_stencil);
}

template<class Model>
void CGaussMLE<Model>::compute_estimate_debug(const ModelDataT<Model> &im, const ParamT<Model> &theta_init_,
MLEDebugData &mle, StencilT<Model> &mle_stencil)
{
auto theta_init = theta_init_;
if(!model.theta_in_bounds(theta_init)) {
auto init_est = cgauss_heuristic_compute_estimate(model,im,theta_init);
ParamT<Model> lb = model.get_lbound()+model.bounds_epsilon;
ParamT<Model> ub = model.get_ubound()-model.bounds_epsilon;
for(IdxT i=0;i<model.get_num_params();i++)
if(theta_init(i)<lb(i) || theta_init(i)>ub(i))
theta_init(i) = init_est(i);
}
mle.theta = cgauss_compute_estimate_debug(model,im,theta_init,num_iterations,mle.sequence);
methods::objective::rllh_stack(model, im, mle.sequence, mle.sequence_rllh);
if(!model.theta_in_bounds(mle.theta)) {
mle_stencil = model.make_stencil(theta_init);
this->record_exit_code(ExitCode::Error);
} else {
mle_stencil = model.make_stencil(mle.theta);
this->record_exit_code(ExitCode::Success);
}
mle.rllh = methods::objective::rllh(model,im,mle_stencil);
mle.obsI = methods::observed_information(model, im, mle_stencil);
}

template<class Model>
IterativeMaximizer<Model>::IterativeMaximizer(const Model &model, int max_iterations)
: ThreadedEstimator<Model>(model), 
max_iterations(max_iterations)
{}

template<class Model>
IterativeMaximizer<Model>::MaximizerData::MaximizerData(const Model &model, const ModelDataT<Model> &im,
const StencilT<Model> &s, bool save_seq)
: MaximizerData(model,im,s,arma::datum::nan,save_seq)
{ }

template<class Model>
IterativeMaximizer<Model>::MaximizerData::MaximizerData(const Model &model, const ModelDataT<Model> &im,
const StencilT<Model> &s, double rllh_, bool save_seq)
: im(im),
grad(model.make_param()),
rllh(rllh_),
num_params(model.get_num_params()),
s0(s),
current_stencil(true)
{
if(!std::isfinite(rllh)) rllh = methods::objective::rllh(model,im,s);
if (save_seq){
max_seq_len = DefaultMaxSeqLength;
theta_seq = model.make_param_stack(max_seq_len);
backtrack_idxs.set_size(max_seq_len);
backtrack_idxs.zeros();
seq_rllh.set_size(max_seq_len);
seq_rllh.zeros();
record_iteration(); 
}
}

template<class Model>
void IterativeMaximizer<Model>::MaximizerData::expand_max_seq_len()
{
max_seq_len*=4;
theta_seq.resize(theta_seq.n_rows,max_seq_len);
seq_rllh.resize(max_seq_len);
backtrack_idxs.resize(max_seq_len);
}

template<class Model>
void IterativeMaximizer<Model>::MaximizerData::record_iteration(const ParamT<Model> &accpeted_theta)
{
nIterations++;
if(max_seq_len>0) {
if(seq_len>=max_seq_len) expand_max_seq_len();
theta_seq.col(seq_len) = accpeted_theta;
seq_rllh(seq_len) = rllh;
seq_len++;
}
}

template<class Model>
void IterativeMaximizer<Model>::MaximizerData::record_backtrack(const ParamT<Model> &rejected_theta, double rejected_rllh)
{
nBacktracks++;
if(max_seq_len>0) {
if(seq_len>=max_seq_len) expand_max_seq_len();
theta_seq.col(seq_len) = rejected_theta;
backtrack_idxs(seq_len) = 1;
seq_rllh(seq_len) = rejected_rllh;
seq_len++;
}
}

template<class Model>
void IterativeMaximizer<Model>::MaximizerData::set_fixed_parameters(const IdxVecT &fixed_idxs_)
{
if(!fixed_idxs_.is_empty()) {
fixed_idxs = fixed_idxs_;
IdxVecT free_parameters_mask(num_params,arma::fill::ones);
free_parameters_mask(fixed_idxs).zeros();
free_idxs = arma::find(free_parameters_mask);
}
}

template<class Model>
StatsT IterativeMaximizer<Model>::get_stats()
{
auto stats = ThreadedEstimator<Model>::get_stats();
std::lock_guard<std::mutex> lock(mtx);
double N = static_cast<double>(this->num_estimations);
stats["total_iterations"] = total_iterations;
stats["total_backtracks"] = total_backtracks;
stats["total_fun_evals"] = total_fun_evals;
stats["total_der_evals"] = total_der_evals;
stats["mean_iterations"] = total_iterations/N;
stats["mean_backtracks"] = total_backtracks/N;
stats["mean_fun_evals"] = total_fun_evals/N;
stats["mean_der_evals"] = total_der_evals/N;
stats["const_min_function_change_ratio"] = convergence_min_function_change_ratio;
stats["const_min_step_size_ratio"] = convergence_min_step_size_ratio;
stats["const_max_iterations"] = max_iterations;
stats["const_max_backtracks"] = max_backtracks;
return stats;
}

template<class Model>
StatsT IterativeMaximizer<Model>::get_debug_stats()
{
auto stats =  IterativeMaximizer<Model>::get_stats();

stats["debugIterative"]=1;
auto backtrack_idxs = last_backtrack_idxs;
for(unsigned n=0; n<backtrack_idxs.n_elem; n++) {
std::ostringstream out;
out<<"backtrack_idxs."<<n+1;
stats[out.str()] = backtrack_idxs(n);
}
return stats;
}


template<class Model>
void IterativeMaximizer<Model>::clear_stats()
{
ThreadedEstimator<Model>::clear_stats();
std::lock_guard<std::mutex> lock(mtx);
total_iterations = 0;
total_backtracks = 0;
total_fun_evals = 0;
total_der_evals = 0;
}

template<class Model>
void IterativeMaximizer<Model>::record_run_statistics(const MaximizerData &data)
{
std::lock_guard<std::mutex> lock(mtx);
total_iterations += data.nIterations;
total_backtracks += data.nBacktracks;
total_fun_evals += data.nIterations + data.nBacktracks;
total_der_evals += data.nIterations;
if(data.has_theta_sequence()) last_backtrack_idxs = data.get_backtrack_idxs(); 
}

template<class Model>
bool IterativeMaximizer<Model>::backtrack(MaximizerData &data)
{
if (!data.step.is_finite()) {
std::ostringstream msg;
msg<<"Step is not finite: "<<data.step.t();
throw NumericalError(msg.str());
}

if(arma::dot(data.grad, data.step)<=0){
std::ostringstream msg;
msg<<"Backtrack with Negative Gradient. grad:"<<data.grad.t()<<" step:"<<data.step.t()<<" <grad, step>: "<<arma::dot(data.grad, data.step);
throw NumericalError(msg.str());
}

double lambda = 1.0;  
double min_lambda = convergence_min_step_size_ratio*(1+arma::norm(data.theta()))/arma::norm(data.step); 
data.save_stencil();  
for(int n=0; n<max_backtracks; n++){
if(lambda < min_lambda) {
data.restore_stencil(); 
this->record_exit_code(ExitCode::StepSize);
return true; 
}
auto new_theta = model.bounded_theta(model.reflected_theta(data.saved_theta() + lambda*data.step));
data.set_stencil(model.make_stencil(new_theta,false)); 
double can_rllh = methods::objective::rllh(model, data.im, data.stencil()); 
if(!std::isfinite(can_rllh)) {
std::ostringstream msg;
msg<<"Candidate theta is inbounds but rllh is non-finite. new_theta:"<<new_theta.t()<<" rllh:"<<can_rllh;
throw NumericalError(msg.str());
}

double linear_step = lambda*arma::dot(data.grad, data.step);
double minimum_increase = backtrack_min_linear_step_ratio*linear_step;
if(can_rllh > data.rllh + minimum_increase) { 
data.rllh = can_rllh; 
data.stencil().compute_derivatives(); 
data.record_iteration();  
return false; 
}
data.record_backtrack(can_rllh); 
lambda *= clamp( -.5*linear_step/(can_rllh - data.rllh - linear_step), this->backtrack_min_ratio,this->backtrack_max_ratio);
}
data.restore_stencil();
this->record_exit_code(ExitCode::MaxBacktracks);
return true; 
}

template<class Model>
bool IterativeMaximizer<Model>::profile_bound_backtrack(MaximizerData &data, IdxT fixed_idx, double target_rllh,
double old_fval, const VecT& fgrad)
{
if (!data.step.is_finite()) {
std::ostringstream msg;
msg<<"Step is not finite: "<<data.step.t();
throw NumericalError(msg.str());
}

if(arma::dot(fgrad, data.step)>=0){
std::ostringstream msg;
msg<<"Profile bounds backtrack: Not a descent direction. Step:"<<data.step.t()<<" <grad, step>: "<<arma::dot(fgrad, data.step);
throw NumericalError(msg.str());
}

double lambda = 1.0; 
double min_step_norm_sq = square(convergence_min_step_size_ratio*(1+arma::norm(data.theta()))); 
data.save_stencil();  
for(int k=0; k<max_backtracks; k++){
if(square(lambda)*norm_sq(data.step) < min_step_norm_sq) {
data.restore_stencil(); 
this->record_exit_code(ExitCode::StepSize);
return true; 
}
auto new_theta = model.bounded_theta(model.reflected_theta(data.saved_theta() + lambda*data.step));
data.set_stencil(model.make_stencil(new_theta)); 
double can_rllh = methods::objective::rllh(model, data.im, data.stencil());
VecT fvec = methods::objective::grad(model, data.im, data.stencil());
fvec(fixed_idx) = can_rllh-target_rllh;
double fval = .5*arma::dot(fvec,fvec);
if(!std::isfinite(fval)) {
std::ostringstream msg;
msg<<"Candidate theta is inbounds but fval is non-finite. new_theta:"<<new_theta.t()<<" fval:"<<fval<<" fvec:"<<fvec.t();
throw NumericalError(msg.str());
}

double linear_step = lambda*arma::dot(fgrad, data.step);
double minimum_fval_decrease = this->backtrack_min_linear_step_ratio*linear_step;
if (fval < old_fval + minimum_fval_decrease) { 
data.rllh = can_rllh; 
data.record_iteration();  
return false; 
}
data.record_backtrack(can_rllh); 
lambda *= clamp(-.5*linear_step/(fval - old_fval - linear_step), this->backtrack_min_ratio,this->backtrack_max_ratio);
}
data.restore_stencil();
this->record_exit_code(ExitCode::MaxBacktracks);
return true; 
}

template<class Model>
bool IterativeMaximizer<Model>::convergence_test_grad_ratio(const VecT &grad, double fval)
{
if(norm_sq(grad) < square(convergence_min_function_change_ratio*(1+fabs(fval)))) {
this->record_exit_code(ExitCode::GradRatio);
return true; 
}
return false;
}

template<class Model>
bool IterativeMaximizer<Model>::convergence_test_step_size(const VecT &new_theta, const VecT &old_theta)
{
if(norm_sq(new_theta-old_theta) < norm_sq(convergence_min_step_size_ratio*old_theta)) {
this->record_exit_code(ExitCode::StepSize);
return true; 
}
return false;
}


template<class Model>
void IterativeMaximizer<Model>::compute_estimate(const ModelDataT<Model> &im, const ParamT<Model> &theta_init, MLEData &mle, StencilT<Model> &mle_stencil)
{
auto theta_init_stencil = this->model.initial_theta_estimate(im, theta_init);
if(!theta_init_stencil.derivatives_computed) throw LogicalError("Stencil has no computed derivatives");
MaximizerData data(model, im, theta_init_stencil);
maximize(data);
mle_stencil = data.stencil();
mle.rllh = data.rllh;
mle.theta = data.theta();
mle.obsI = methods::observed_information(model,im,mle_stencil);
record_run_statistics(data);
}

template<class Model>
void IterativeMaximizer<Model>::compute_estimate_debug(const ModelDataT<Model> &im, const ParamT<Model> &theta_init,
MLEDebugData &mle, StencilT<Model> &mle_stencil)
{
auto theta_init_stencil = this->model.initial_theta_estimate(im, theta_init);
if(!theta_init_stencil.derivatives_computed)  throw LogicalError("Stencil has no computed derivatives");
MaximizerData data(model, im, theta_init_stencil, -INFINITY, true); 
maximize(data);
mle_stencil = data.stencil();
mle.rllh = data.rllh;
mle.theta = data.theta();
mle.obsI = methods::observed_information(model,im,mle_stencil);
mle.sequence = data.get_theta_sequence();
mle.sequence_rllh = data.get_theta_sequence_rllh();
record_run_statistics(data);
}

template<class Model>
double IterativeMaximizer<Model>::compute_profile_estimate(const ModelDataT<Model> &im, const ParamT<Model> &theta_init,
const IdxVecT& fixed_idxs, StencilT<Model> &max_stencil)
{
auto theta_init_stencil = this->model.initial_theta_estimate(im, theta_init);
if(!theta_init_stencil.derivatives_computed)  throw LogicalError("Stencil has no computed derivatives");
MaximizerData data(model, im, theta_init_stencil);
data.set_fixed_parameters(fixed_idxs);
maximize(data);
max_stencil = data.stencil();
record_run_statistics(data);
return data.rllh;
}

template<class Model>
void IterativeMaximizer<Model>::compute_profile_bound(const ModelDataT<Model> &im, ProfileBoundsData &est,
const VecT &init_step, IdxT param_idx, IdxT which_bound)
{
MaximizerData data(model, im, model.make_stencil(est.mle.theta+init_step));
solve_profile_bound(data, est.mle, est.target_rllh_delta, param_idx, which_bound);
IdxT k=0;
bool found=false;
for(;k<est.estimated_idxs.n_elem;k++)
if(est.estimated_idxs(k)==param_idx) {
found=true;
break;
}
if(!found) throw LogicalError("Unknown param_idx.");
if(which_bound) { 
est.profile_points_ub.col(k) = data.theta();
est.profile_ub(k) = est.profile_points_ub(param_idx,k);
est.profile_points_ub_rllh(k) = data.rllh;
} else { 
est.profile_points_lb.col(k) = data.theta();
est.profile_lb(k) = est.profile_points_lb(param_idx,k);
est.profile_points_lb_rllh(k) = data.rllh;
}
record_run_statistics(data);
}

template<class Model>
void IterativeMaximizer<Model>::compute_profile_bound_debug(const ModelDataT<Model> &im, ProfileBoundsDebugData &est)
{
auto step = subroutine::solve_profile_initial_step(est.mle.obsI, est.estimated_idx, est.target_rllh_delta);

MaximizerData data_lb(model, im, model.make_stencil(est.mle.theta+step), -INFINITY, true); 
solve_profile_bound(data_lb, est.mle, est.target_rllh_delta, est.estimated_idx, 0);
est.profile_lb = data_lb.theta()(est.estimated_idx);
est.Nseq_lb = data_lb.get_sequence_len();
est.sequence_lb = data_lb.get_theta_sequence();
est.sequence_lb_rllh = data_lb.get_theta_sequence_rllh();
record_run_statistics(data_lb);

MaximizerData data_ub(model, im, model.make_stencil(est.mle.theta-step), -INFINITY, true); 
solve_profile_bound(data_ub, est.mle, est.target_rllh_delta, est.estimated_idx, 1);
est.profile_ub = data_ub.theta()(est.estimated_idx);
est.Nseq_ub = data_ub.get_sequence_len();
est.sequence_ub = data_ub.get_theta_sequence();
est.sequence_ub_rllh = data_ub.get_theta_sequence_rllh();
record_run_statistics(data_ub);
}

template<class Model>
void IterativeMaximizer<Model>::solve_profile_bound(MaximizerData &data, MLEData &mle, double llh_delta, IdxT fixed_idx, IdxT which_bound)
{
std::ostringstream msg;
msg<<"solve_profile_bound not implemented for maximizer:"<<this->name();
throw NotImplementedError(msg.str());
}

template<class Model>
void IterativeMaximizer<Model>::local_maximize(const ModelDataT<Model> &im, StencilT<Model> &theta_stencil, MLEData &mle)
{
MaximizerData data(model, im, theta_stencil, mle.rllh);
maximize(data);

record_run_statistics(data);
theta_stencil = data.stencil();
mle.theta = data.theta();
mle.rllh = data.rllh;
mle.obsI = methods::observed_information(model, im, theta_stencil);
}

template<class Model>
void IterativeMaximizer<Model>::local_maximize(const ModelDataT<Model> &im, StencilT<Model> &theta_stencil, MLEDebugData &mle)
{
MaximizerData data(model, im, theta_stencil, mle.rllh);
maximize(data);

record_run_statistics(data);
theta_stencil = data.stencil();
mle.theta = data.theta();
mle.rllh = data.rllh;
mle.obsI = methods::observed_information(model, im, theta_stencil);
mle.sequence = data.get_theta_sequence();
mle.sequence_rllh = data.get_theta_sequence_rllh();
}

template<class Model>
void IterativeMaximizer<Model>::local_profile_maximize(const ModelDataT<Model> &im, const IdxVecT &fixed_param_idxs,
StencilT<Model> &theta_stencil, MLEDebugData &mle)
{
MaximizerData data(model, im, theta_stencil, mle.rllh);
data.set_fixed_parameters(fixed_param_idxs);
maximize(data);

record_run_statistics(data);
theta_stencil = data.stencil();
mle.theta = data.theta();
mle.rllh = data.rllh;
mle.obsI = methods::observed_information(model, im, theta_stencil);
mle.sequence = data.get_theta_sequence();
mle.sequence_rllh = data.get_theta_sequence_rllh();
}

template<class Model>
void NewtonDiagonalMaximizer<Model>::maximize(MaximizerData &data)
{
auto grad2 = model.make_param();
for(int n=0; n<this->max_iterations; n++) { 
methods::objective::grad2(model, data.im, data.stencil(), data.grad, grad2); 
if(data.has_fixed_parameters()) data.grad(data.fixed_idxs).zeros(); 
if(this->convergence_test_grad_ratio(data.grad,data.rllh)) return;
data.step = data.grad/arma::abs(grad2);
if(this->backtrack(data)) return; 
}
this->record_exit_code(ExitCode::MaxIter);
}



template<class Model>
void NewtonMaximizer<Model>::solve_profile_bound(MaximizerData &data, MLEData &mle, double llh_delta, IdxT fixed_idx, IdxT which_bound)
{
double target_rllh = mle.rllh+llh_delta;
double fixed_param_lbound, fixed_param_ubound;
if(which_bound) { 
fixed_param_lbound = mle.theta(fixed_idx);
fixed_param_ubound = model.get_ubound()(fixed_idx);
} else { 
fixed_param_lbound = model.get_lbound()(fixed_idx);
fixed_param_ubound = mle.theta(fixed_idx);
}
auto hess = model.make_param_mat();
for(int n=0; n<this->max_iterations; n++) { 
methods::objective::hessian(model, data.im, data.stencil(), data.grad, hess);
MatT J = arma::symmatu(hess);
VecT fvec = data.grad;
fvec(fixed_idx) = data.rllh-target_rllh;
J.row(fixed_idx) = data.grad.t(); 
double fval = .5*arma::dot(fvec,fvec); 
VecT fgrad = J*fvec; 
if(this->convergence_test_grad_ratio(fgrad,fval)) return;

MatT Jinv = arma::inv(J);
VecT newton_step = -Jinv*fvec;

double fixed_param_step = newton_step(fixed_idx);
double fixed_param_val = data.theta()(fixed_idx);
if(fixed_param_step+fixed_param_val<=fixed_param_lbound){
std::cout<<"$$$ Lbound failure! fixed_param_val:"<<fixed_param_val<<" fixed_param_step:"<<fixed_param_step<<" fixed_param_bounds:["<<fixed_param_lbound<<","<<fixed_param_ubound<<"]\n";
std::cout<<"param_idx:"<<fixed_idx<<" which_bound:"<<which_bound<<std::endl;
std::cout<<"newton step:"<<newton_step<<std::endl;
double rho = (fixed_param_lbound-fixed_param_val)/(2*fixed_param_step);
newton_step*=rho;
std::cout<<"rho: "<<rho<<" new newton step:"<<newton_step.t();
}
if(fixed_param_step+fixed_param_val >=fixed_param_ubound){
std::cout<<"$$$ Ubound failure! fixed_param_val:"<<fixed_param_val<<" fixed_param_step:"<<fixed_param_step<<" fixed_param_bounds:["<<fixed_param_lbound<<","<<fixed_param_ubound<<"]\n";
double rho = (fixed_param_ubound-fixed_param_val)/(2*fixed_param_step);
newton_step*=rho;
}

VecT z = -Jinv.col(fixed_idx);
VecT Hz = hess*z;
double a2 = 2*arma::dot(z,Hz); 
double b = 2*(arma::dot(newton_step,Hz)-1);
double c = arma::dot(newton_step,hess*newton_step);
double resid = b*b-2*a2*c;
if(resid >= this->min_profile_bound_residual) {
double width = sqrt(resid)/a2;
double center = -b/a2;
double s0 = center-width;
double s1 = center+width;
VecT step0 = newton_step - s0*z;
VecT step1 = newton_step - s1*z;
double mu0 = arma::dot(step0,mle.obsI*step0);
double mu1 = arma::dot(step1,mle.obsI*step1);
data.step = (mu0<mu1) ? step0 : step1;
} else {
data.step = newton_step;
}
if(arma::dot(fgrad, data.step)>=0){
for(IdxT i=0;i<data.grad.n_elem;i++) data.step(i)=std::copysign(data.step(i),-fgrad(i));
}

if(this->profile_bound_backtrack(data,fixed_idx,target_rllh,fval,fgrad)) return;
}
this->record_exit_code(ExitCode::MaxIter);
}


template<class Model>
void NewtonMaximizer<Model>::maximize(MaximizerData &data)
{
auto hess = model.make_param_mat();
data.step = model.make_param();
data.step.zeros();
MatT Hhat;
VecT ghat;
VecT lb, ub;
if(data.has_fixed_parameters()) {
lb = model.get_lbound()(data.free_idxs);
ub = model.get_ubound()(data.free_idxs);
} else {
lb = model.get_lbound();
ub = model.get_ubound();
}
for(int n=0; n < this->max_iterations; n++) { 
methods::objective::hessian(model, data.im, data.stencil(), data.grad, hess);
if(data.has_fixed_parameters()) data.grad(data.fixed_idxs).zeros(); 
if(this->convergence_test_grad_ratio(data.grad,data.rllh)) return;

VecT v,Jv;
subroutine::compute_bound_scaling_vec(data.theta(), -data.grad, model.get_lbound(), model.get_ubound(), v, Jv);
VecT Dinv = arma::sqrt(arma::abs(v));
subroutine::compute_scaled_problem(hess, data.grad, Dinv, Jv, Hhat, ghat);

if(data.has_fixed_parameters()) {
data.step(data.free_idxs) = Dinv(data.free_idxs)%arma::solve(arma::symmatu(hess(data.free_idxs,data.free_idxs)), -data.grad(data.free_idxs));
} else {
data.step = Dinv%arma::solve(arma::symmatu(Hhat), -ghat);
}
if(arma::dot(data.step,data.grad)<=0) {
for(IdxT n=0; n<data.step.n_elem;n++) data.step(n) = std::copysign(data.step(n), data.grad(n));
}
data.step = subroutine::bound_step(data.step, data.theta(), lb, ub);
if(this->backtrack(data)) return;
}
this->record_exit_code(ExitCode::MaxIter);
}

template<class Model>
void QuasiNewtonMaximizer<Model>::maximize(MaximizerData &data)
{
auto grad_old=model.make_param();
data.step = model.make_param();
data.step.zeros();
auto G=model.make_param_mat(); 
VecT last_step, delta_grad, q;
for(int n=0; n < this->max_iterations; n++) { 
if(n==0) {
auto hess=model.make_param_mat();
methods::objective::hessian(model, data.im, data.stencil(), data.grad, hess);
if(data.has_fixed_parameters()) data.grad(data.fixed_idxs).zeros(); 
if(this->convergence_test_grad_ratio(data.grad,data.rllh)) return;

if(data.has_fixed_parameters()) {
G = -arma::symmatu(hess(data.free_idxs,data.free_idxs));
} else {
G = -arma::symmatu(hess);
}

if(!is_positive_definite(G)) {
G = arma::diagmat(arma::clamp(arma::abs(G.diag()),1e-2,1e2));
if(!is_positive_definite(G)){
throw NumericalError("Could not correct indefinite matrix.");
}
}
G = arma::inv(arma::symmatu(G));
} else {
grad_old = data.grad;
data.grad = methods::objective::grad(model, data.im, data.stencil());
if(data.has_fixed_parameters()) data.grad(data.fixed_idxs).zeros(); 
if(this->convergence_test_grad_ratio(data.grad,data.rllh)) return;
last_step = data.theta() - data.saved_theta(); 
if(data.has_fixed_parameters()) {
last_step = last_step(data.free_idxs);
delta_grad = grad_old(data.free_idxs) - data.grad(data.free_idxs); 
} else {
delta_grad = grad_old - data.grad; 
}
VecT q = G*delta_grad;
double rho = fabs(1./arma::dot(delta_grad, last_step));
double gamma = 1./arma::dot(delta_grad,q);
VecT u = rho*last_step - gamma*q;
G += rho*last_step*last_step.t() - gamma*q*q.t() + (1/gamma)*u*u.t();
}

if(data.has_fixed_parameters()) {
data.step(data.free_idxs) = G*data.grad(data.free_idxs);
} else {
data.step = G*data.grad;
}

if(arma::dot(data.step,data.grad)<=0) {
for(IdxT n=0; n<data.step.n_elem;n++) data.step(n) = std::copysign(data.step(n), data.grad(n));
}
if(this->backtrack(data)) return;
}
this->record_exit_code(ExitCode::MaxIter);
}

template<class Model>
void TrustRegionMaximizer<Model>::maximize(MaximizerData &data)
{
auto hess = model.make_param_mat();
double delta = 1.0; 

VecT lb, ub;
if(data.has_fixed_parameters()) {
lb = model.get_lbound()(data.free_idxs);
ub = model.get_ubound()(data.free_idxs);
} else {
lb = model.get_lbound();
ub = model.get_ubound();
}
MatT Hhat;
VecT ghat;
for(int n=0; n<this->max_iterations; n++) { 
methods::objective::hessian(model, data.im, data.stencil(), data.grad, hess);
if(data.has_fixed_parameters()) {
if(this->convergence_test_grad_ratio(data.grad(data.free_idxs),data.rllh)) return;
} else {
if(this->convergence_test_grad_ratio(data.grad,data.rllh)) return;
}
VecT v,Jv;
subroutine::compute_bound_scaling_vec(data.theta(), -data.grad, model.get_lbound(), model.get_ubound(), v, Jv);
VecT D = 1./arma::sqrt(arma::abs(v));

VecT Dinv = 1./D;

subroutine::compute_scaled_problem(hess, data.grad, Dinv, Jv, Hhat, ghat);

double rho_cauchy, quad_model_opt_bnd, quad_model_cauchy_bnd;
VecT s_hat_cauchy_bnd;
VecT s_hat_opt_bnd;
if(data.has_fixed_parameters()) {
data.grad(data.fixed_idxs).zeros(); 
MatT free_Hhat = Hhat(data.free_idxs,data.free_idxs);
VecT free_ghat = ghat(data.free_idxs);
VecT free_D = D(data.free_idxs);
VecT free_Dinv = Dinv(data.free_idxs);

VecT free_theta = data.theta()(data.free_idxs);
if(n==0) delta = subroutine::compute_initial_trust_radius(free_ghat); 

VecT s_hat_opt = subroutine::solve_TR_subproblem(free_ghat, free_Hhat, delta);
s_hat_opt_bnd = free_D%subroutine::bound_step(s_hat_opt%free_Dinv, free_theta, lb, ub);
VecT s_hat_cauchy = subroutine::compute_cauchy_point(free_ghat, free_Hhat, delta);
s_hat_cauchy_bnd = free_D%subroutine::bound_step(s_hat_cauchy%free_Dinv, free_theta, lb, ub);

quad_model_opt_bnd = subroutine::compute_quadratic_model_value(s_hat_opt_bnd, free_ghat, free_Hhat);
quad_model_cauchy_bnd = subroutine::compute_quadratic_model_value(s_hat_cauchy_bnd, free_ghat, free_Hhat);
rho_cauchy = quad_model_opt_bnd / quad_model_cauchy_bnd;
} else {
if(n==0) delta = subroutine::compute_initial_trust_radius(ghat); 

VecT s_hat_opt = subroutine::solve_TR_subproblem(ghat, Hhat, delta);
s_hat_opt_bnd = D%subroutine::bound_step(s_hat_opt%Dinv, data.theta(), lb, ub);
VecT s_hat_cauchy = subroutine::compute_cauchy_point(ghat, Hhat, delta);
s_hat_cauchy_bnd = D%subroutine::bound_step(s_hat_cauchy%Dinv, data.theta(), lb, ub);

quad_model_opt_bnd = subroutine::compute_quadratic_model_value(s_hat_opt_bnd, ghat, Hhat);
quad_model_cauchy_bnd = subroutine::compute_quadratic_model_value(s_hat_cauchy_bnd, ghat, Hhat);
rho_cauchy = quad_model_opt_bnd / quad_model_cauchy_bnd;
}

data.save_stencil();
double model_improvement;
VecT s_hat;
if(rho_cauchy < rho_cauchy_min) { 
s_hat = s_hat_cauchy_bnd;
model_improvement = -quad_model_cauchy_bnd;
} else { 
s_hat = s_hat_opt_bnd;
model_improvement = -quad_model_opt_bnd;
}
if(data.has_fixed_parameters()) {
for(IdxT k=0; k<data.num_fixed_parameters(); k++) s_hat.insert_rows(data.fixed_idxs(k),1);
}
data.set_stencil(model.make_stencil(data.saved_theta() + Dinv % s_hat));
double old_rllh = data.rllh;
double can_rllh = methods::objective::rllh(model, data.im, data.stencil());
double obj_improvement = can_rllh - old_rllh; 
double rho_obj =  obj_improvement / model_improvement;
if(rho_obj < rho_obj_min) { 
if(rho_obj_min < 0) {
delta = trust_radius_decrease_min*delta;
} else {
delta = std::max(trust_radius_decrease_min*delta, trust_radius_decrease*arma::norm(s_hat));
}
data.record_backtrack(can_rllh);
data.restore_stencil(); 
} else { 
data.rllh = can_rllh;
if(rho_obj > rho_obj_opt) { 
if(rho_cauchy > rho_obj_opt) { 
delta = std::max(delta, trust_radius_increase*arma::norm(s_hat));
} else if(delta>1 && rho_cauchy <= rho_obj_min) {
delta = std::max(trust_radius_decrease*delta, arma::norm(s_hat));
}
}
data.record_iteration();
}

if(delta < convergence_min_trust_radius) {
this->record_exit_code(ExitCode::TrustRegionRadius);
return;
}
if(this->convergence_test_step_size(data.theta(),data.saved_theta())) return;
}
this->record_exit_code(ExitCode::MaxIter);
}




template<class Model>
void SimulatedAnnealingMaximizer<Model>::compute_estimate(const ModelDataT<Model> &im, const ParamT<Model> &theta_init,
MLEData &mle, StencilT<Model> &mle_stencil)
{
MatT sequence;
VecT sequence_rllh;
IdxVecT fixed_params_idx;
mle_stencil = anneal(im, model.initial_theta_estimate(im,theta_init), fixed_params_idx, mle.rllh, sequence, sequence_rllh);
mle.theta = mle_stencil.theta;
mle.obsI = methods::observed_information(model, im, mle_stencil);
}

template<class Model>
void SimulatedAnnealingMaximizer<Model>::compute_estimate_debug(const ModelDataT<Model> &im, const ParamT<Model> &theta_init,
MLEDebugData &mle, StencilT<Model> &mle_stencil)
{
IdxVecT fixed_params_mask;
mle_stencil = anneal(im, model.initial_theta_estimate(im,theta_init), fixed_params_mask, mle.rllh, mle.sequence, mle.sequence_rllh);
mle.theta = mle_stencil.theta;
mle.obsI = methods::observed_information(model, im, mle_stencil);
}

template<class Model>
double SimulatedAnnealingMaximizer<Model>::compute_profile_estimate(const ModelDataT<Model> &im, const ParamT<Model> &theta_init,
const IdxVecT &fixed_idxs, StencilT<Model> &max_stencil)
{
MatT sequence;
VecT sequence_rllh;
double rllh;
max_stencil = anneal(im, model.initial_theta_estimate(im,theta_init), fixed_idxs, rllh, sequence, sequence_rllh);
return rllh;
}


template<class Model>
StencilT<Model>
SimulatedAnnealingMaximizer<Model>::anneal(const ModelDataT<Model> &im, const StencilT<Model> &theta_init, const IdxVecT &fixed_params_idxs,
double &max_rllh, MatT &sequence, VecT &sequence_rllh)
{
bool has_fixed_parameters = !fixed_params_idxs.is_empty();
IdxVecT fixed_params_mask;
if(has_fixed_parameters) {
fixed_params_mask.zeros();
fixed_params_mask(fixed_params_idxs).ones();
}
auto &rng = model.get_rng_generator();
UniformDistT uniform;
TrustRegionMaximizer<Model> newton_max(model);
IdxT num_seq_allocate = num_iterations+16; 
sequence = model.make_param_stack(num_seq_allocate);
sequence_rllh.set_size(num_seq_allocate);
sequence.col(0)=theta_init.theta;
sequence_rllh(0)=methods::objective::rllh(model, im, theta_init);
max_rllh = sequence_rllh(0);
double T = T_init; 
int naccepted=1;
StencilT<Model> can_stencil;
StencilT<Model> max_stencil=theta_init;
double old_rllh = max_rllh;
for(int n=1; n<num_iterations; n++){
ParamT<Model> can_theta = sequence.col(naccepted-1);
if(has_fixed_parameters) {
model.sample_mcmc_candidate(n, can_theta, fixed_params_mask);
} else {
model.sample_mcmc_candidate(n, can_theta);
}

if(!model.theta_in_bounds(can_theta)) { 
n--;
continue;
}
can_stencil = model.make_stencil(can_theta);
double can_rllh = methods::objective::rllh(model, im, can_stencil);
if(can_rllh < old_rllh && uniform(rng) > exp((can_rllh-old_rllh)/T)) continue; 
T /= cooling_rate;
sequence.col(naccepted) = can_theta;
sequence_rllh(naccepted) = can_rllh;
old_rllh=can_rllh;
if(can_rllh > max_rllh) {
max_rllh = can_rllh;
max_stencil = std::move(can_stencil);
}
naccepted++;
}

MatT newton_seq;
VecT newton_seq_rllh;
MLEDebugData mle;
mle.rllh = max_rllh;
if(has_fixed_parameters) {
newton_max.local_profile_maximize(im, fixed_params_mask, max_stencil, mle);
} else {
newton_max.local_maximize(im, max_stencil, mle);
}
this->exit_counts += newton_max.get_exit_counts();

sequence = arma::join_horiz(sequence.head_cols(naccepted),mle.sequence);
sequence_rllh = arma::join_vert(sequence_rllh.head(naccepted),mle.sequence_rllh);
record_run_statistics(num_iterations+newton_max.get_total_iterations(),
num_iterations+newton_max.get_total_fun_evals(),
newton_max.get_total_der_evals());
return max_stencil;
}

template<class Model>
void SimulatedAnnealingMaximizer<Model>::record_run_statistics(int num_iters, int num_fun_evals, int num_der_evals)
{
std::lock_guard<std::mutex> lock(mtx);
total_iterations += num_iters;
total_fun_evals += num_fun_evals;
total_der_evals += num_der_evals;
}

template<class Model>
StatsT SimulatedAnnealingMaximizer<Model>::get_stats()
{
auto stats = ThreadedEstimator<Model>::get_stats();
std::lock_guard<std::mutex> lock(mtx);
double N = static_cast<double>(this->num_estimations);
stats["num_iterations"] = num_iterations;
stats["total_iterations"] = total_iterations;
stats["total_fun_evals"] = total_fun_evals;
stats["total_der_evals"] = total_der_evals;
stats["mean_iterations"] = total_iterations/N;
stats["mean_fun_evals"] = total_fun_evals/N;
stats["mean_der_evals"] = total_der_evals/N;
stats["const_num_iterations"] = num_iterations;
stats["const_T_init"] = T_init;
stats["const_cooling_rate"] = cooling_rate;

return stats;
}

template<class Model>
StatsT SimulatedAnnealingMaximizer<Model>::get_debug_stats()
{
return get_stats();
}

} 

} 

#endif 
