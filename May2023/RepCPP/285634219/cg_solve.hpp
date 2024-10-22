#ifndef _cg_solve_hpp_
#define _cg_solve_hpp_


#include <cmath>
#include <limits>

#include <Vector_functions.hpp>
#include <mytimer.hpp>

#include <outstream.hpp>


namespace miniFE {

template<typename Scalar>
void print_vec(const std::vector<Scalar>& vec, const std::string& name)
{
for(size_t i=0; i<vec.size(); ++i) {
std::cout << name << "["<<i<<"]: " << vec[i] << std::endl;
}
}

template<typename VectorType>
bool breakdown(typename VectorType::ScalarType inner,
const VectorType& v,
const VectorType& w)
{
typedef typename VectorType::ScalarType Scalar;
typedef typename TypeTraits<Scalar>::magnitude_type magnitude;


magnitude vnorm = std::sqrt(dot(v,v));
magnitude wnorm = std::sqrt(dot(w,w));
return std::abs(inner) <= 100*vnorm*wnorm*std::numeric_limits<magnitude>::epsilon();
}

template<typename OperatorType,
typename VectorType,
typename Matvec>
void
cg_solve(OperatorType& A,
const VectorType& b,
VectorType& x,
Matvec matvec,
typename OperatorType::LocalOrdinalType max_iter,
typename TypeTraits<typename OperatorType::ScalarType>::magnitude_type& tolerance,
typename OperatorType::LocalOrdinalType& num_iters,
typename TypeTraits<typename OperatorType::ScalarType>::magnitude_type& normr,
timer_type* my_cg_times)
{
typedef typename OperatorType::ScalarType ScalarType;
typedef typename OperatorType::GlobalOrdinalType GlobalOrdinalType;
typedef typename OperatorType::LocalOrdinalType LocalOrdinalType;
typedef typename TypeTraits<ScalarType>::magnitude_type magnitude_type;

timer_type t0 = 0, tWAXPY = 0, tDOT = 0, tMATVEC = 0, tMATVECDOT = 0;
timer_type total_time = mytimer();

int myproc = 0;
#ifdef HAVE_MPI
MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
#endif

if (!A.has_local_indices) {
std::cerr << "miniFE::cg_solve ERROR, A.has_local_indices is false, needs to be true. This probably means "
<< "miniFE::make_local_matrix(A) was not called prior to calling miniFE::cg_solve."
<< std::endl;
return;
}

size_t nrows = A.rows.size();
LocalOrdinalType ncols = A.num_cols;

VectorType r(b.startIndex, nrows);
VectorType p(0, ncols);
VectorType Ap(b.startIndex, nrows);

normr = 0;
magnitude_type rtrans = 0;
magnitude_type oldrtrans = 0;

LocalOrdinalType print_freq = max_iter/10;
if (print_freq>50) print_freq = 50;
if (print_freq<1)  print_freq = 1;

ScalarType one = 1.0;
ScalarType zero = 0.0;

MINIFE_SCALAR* MINIFE_RESTRICT r_ptr = &r.coefs[0];
MINIFE_SCALAR* MINIFE_RESTRICT p_ptr = &p.coefs[0];
MINIFE_SCALAR* MINIFE_RESTRICT Ap_ptr = &Ap.coefs[0];
MINIFE_SCALAR* MINIFE_RESTRICT x_ptr = &x.coefs[0];
const MINIFE_SCALAR* MINIFE_RESTRICT b_ptr = &b.coefs[0];

const MINIFE_LOCAL_ORDINAL* MINIFE_RESTRICT const Arowoffsets = &A.row_offsets[0];
const MINIFE_GLOBAL_ORDINAL* MINIFE_RESTRICT const Acols    = &A.packed_cols[0];
const MINIFE_SCALAR* MINIFE_RESTRICT const Acoefs            = &A.packed_coefs[0];

#pragma omp target data map(to: r_ptr[0:r.coefs.size()],  \
p_ptr[0:p.coefs.size()],  \
Ap_ptr[0:Ap.coefs.size()], \
b_ptr[0:b.coefs.size()],  \
Arowoffsets[0:A.row_offsets.size()], \
Acols[0:A.packed_cols.size()], \
Acoefs[0:A.packed_coefs.size()]) \
map(tofrom: x_ptr[0:x.coefs.size()])  
{

TICK(); waxpby(one, x, zero, x, p); TOCK(tWAXPY);

#ifdef MINIFE_DEBUG
print_vec(p.coefs, "p");
#endif

TICK();
matvec(A, p, Ap);
TOCK(tMATVEC);

TICK(); waxpby(one, b, -one, Ap, r); TOCK(tWAXPY);

TICK(); rtrans = dot_r2(r); TOCK(tDOT);

#ifdef MINIFE_DEBUG
std::cout << "rtrans="<<rtrans<<std::endl;
#endif

normr = std::sqrt(rtrans);

if (myproc == 0) {
std::cout << "Initial Residual = "<< normr << std::endl;
}

magnitude_type brkdown_tol = std::numeric_limits<magnitude_type>::epsilon();

#ifdef MINIFE_DEBUG
std::ostream& os = outstream();
os << "brkdown_tol = " << brkdown_tol << std::endl;
#endif

#ifdef MINIFE_DEBUG_OPENMP
std::cout << "Starting CG Solve Phase..." << std::endl;
#endif

for(LocalOrdinalType k=1; k <= max_iter && normr > tolerance; ++k) {
if (k == 1) {
TICK(); waxpby(one, r, zero, r, p); TOCK(tWAXPY);
}
else {
oldrtrans = rtrans;
TICK(); rtrans = dot_r2(r); TOCK(tDOT);
magnitude_type beta = rtrans/oldrtrans;
TICK(); daxpby(one, r, beta, p); TOCK(tWAXPY);
}

normr = sqrt(rtrans);

if (myproc == 0 && (k%print_freq==0 || k==max_iter)) {
std::cout << "Iteration = "<<k<<"   Residual = "<<normr<<std::endl;
}

magnitude_type alpha = 0;
magnitude_type p_ap_dot = 0;

TICK(); matvec(A, p, Ap); TOCK(tMATVEC);
TICK(); p_ap_dot = dot(Ap, p); TOCK(tDOT);

#ifdef MINIFE_DEBUG
os << "iter " << k << ", p_ap_dot = " << p_ap_dot;
os.flush();
#endif
if (p_ap_dot < brkdown_tol) {
if (p_ap_dot < 0 || breakdown(p_ap_dot, Ap, p)) {
std::cerr << "miniFE::cg_solve ERROR, numerical breakdown!"<<std::endl;
#ifdef MINIFE_DEBUG
os << "ERROR, numerical breakdown!"<<std::endl;
#endif
my_cg_times[WAXPY] = tWAXPY;
my_cg_times[DOT] = tDOT;
my_cg_times[MATVEC] = tMATVEC;
my_cg_times[TOTAL] = mytimer() - total_time;
break; 
}
else brkdown_tol = 0.1 * p_ap_dot;
}
alpha = rtrans/p_ap_dot;
#ifdef MINIFE_DEBUG
os << ", rtrans = " << rtrans << ", alpha = " << alpha << std::endl;
#endif

TICK(); daxpby(alpha, p, one, x);
daxpby(-alpha, Ap, one, r); TOCK(tWAXPY);

num_iters = k;
}

}

my_cg_times[WAXPY] = tWAXPY;
my_cg_times[DOT] = tDOT;
my_cg_times[MATVEC] = tMATVEC;
my_cg_times[MATVECDOT] = tMATVECDOT;
my_cg_times[TOTAL] = mytimer() - total_time;
}

}

#endif

