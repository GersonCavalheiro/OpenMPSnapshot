#pragma once

#include <cstdio> 
#include <cstdint> 
#include <cassert> 
#include <vector> 

#include "status.hxx" 

#include "constants.hxx" 
#include "green_memory.hxx" 

#ifdef HAS_TFQMRGPU

#ifdef HAS_NO_CUDA
#include "tfQMRgpu/include/tfqmrgpu_cudaStubs.hxx" 
#define devPtr const __restrict__
#else  
#include <cuda.h>
#endif 
#include "tfQMRgpu/include/tfqmrgpu_memWindow.h" 

#else  

#include <utility> 
typedef std::pair<size_t,size_t> memWindow_t;
#ifdef HAS_NO_CUDA
typedef size_t cudaStream_t;
#endif 

#endif 

#include "green_sparse.hxx"    
#include "green_kinetic.hxx"   
#include "green_potential.hxx" 
#include "green_dyadic.hxx"    


#ifdef    debug_printf
#undef  debug_printf
#endif 

#ifdef    DEBUG
#define debug_printf(...) { std::printf(__VA_ARGS__); std::fflush(stdout); }
#else  
#define debug_printf(...)
#endif 

namespace green_action {

struct atom_t {
double pos[3]; 
double sigma; 
int32_t gid; 
int32_t ia; 
int16_t shifts[3]; 
uint8_t nc; 
int8_t numax; 
}; 

struct atom_image_t {
double pos[3]; 
float  oneoversqrtsigma; 
int8_t shifts[3]; 
int8_t lmax; 
}; 

struct plan_t {


std::vector<uint16_t> colindx; 
memWindow_t colindxwin; 

std::vector<uint32_t> subset; 
memWindow_t subsetwin; 
memWindow_t matBwin; 

memWindow_t matXwin; 
memWindow_t vec3win; 

uint32_t nRows = 0; 
uint32_t nCols = 0; 
std::vector<uint32_t> rowstart; 

size_t gpu_mem = 0; 

float residuum_reached    = 3e38;
float flops_performed     = 0.f;
float flops_performed_all = 0.f;
int   iterations_needed   = -99;


int echo = 9;

std::vector<int64_t> global_target_indices; 
std::vector<int64_t> global_source_indices; 
double r_truncation   = 9e18; 
float r_confinement   = 9e18; 
float V_confinement   = 1; 
std::complex<double> E_param; 


green_kinetic::kinetic_plan_t kinetic[3]; 

uint32_t* RowStart = nullptr; 
uint32_t* rowindx  = nullptr; 
int16_t (*source_coords)[3+1] = nullptr; 
int16_t (*target_coords)[3+1] = nullptr; 
float   (*rowCubePos)[3+1]    = nullptr; 
float   (*colCubePos)[3+1]    = nullptr; 
int16_t (*target_minus_source)[3+1] = nullptr; 
double  (**Veff)[64]          = nullptr; 
int32_t*  veff_index          = nullptr; 

double *grid_spacing_trunc = nullptr; 
double (*phase)[2][2]      = nullptr; 

bool noncollinear_spin = false;

green_dyadic::dyadic_plan_t dyadic_plan;

plan_t() {
debug_printf("# default constructor for %s\n", __func__);
} 

~plan_t() {
debug_printf("# destruct %s\n", __func__);
free_memory(RowStart);
free_memory(rowindx);
free_memory(source_coords);
free_memory(target_coords);
free_memory(target_minus_source);
for (int mag = 0; mag < 4*(nullptr != Veff); ++mag) {
free_memory(Veff[mag]);
} 
free_memory(Veff);
free_memory(veff_index);
free_memory(colCubePos);
free_memory(rowCubePos);
free_memory(grid_spacing_trunc);
free_memory(phase);
} 

}; 





template <typename floating_point_t=float, int R1C2=2, int Noco=1, int n64=64>
class action_t { 
public:
typedef floating_point_t real_t;
static int constexpr LM = Noco*n64, 
LN = LM;    
action_t(plan_t *plan)
: p(plan), apc(nullptr), aac(nullptr)
{
assert((1 == Noco && (1 == R1C2 || 2 == R1C2)) || (2 == Noco && 2 == R1C2));
debug_printf("# construct %s\n", __func__);
char* buffer{nullptr};
take_memory(buffer);
assert(nullptr != plan);
} 

~action_t() {
debug_printf("# destruct %s\n", __func__);
free_memory(apc);
} 

void take_memory(char* &buffer) {
auto const & dp = p->dyadic_plan;
auto const natomcoeffs = dp.AtomImageStarts ? dp.AtomImageStarts[dp.nAtomImages] : 0;
auto const n = size_t(natomcoeffs) * p->nCols;
apc = get_memory<real_t[R1C2][Noco][LM]>(n, p->echo, "apc");
} 

void transfer(char* const buffer, cudaStream_t const streamId=0) {
} 

bool has_preconditioner() const { return false; }


double multiply( 
real_t         (*const __restrict y)[R1C2][LM][LM] 
, real_t   const (*const __restrict x)[R1C2][LM][LM] 
, uint16_t const (*const __restrict colIndex) 
, uint32_t const nnzb 
, uint32_t const nCols=1 
, unsigned const l2nX=0  
, cudaStream_t const streamId=0 
, bool const precondition=false
)
{
assert(p);
if (2 == Noco) assert(p->noncollinear_spin && "Also the plan needs to be created with Noco=2");
double nops{0};

if (p->echo > 3) std::printf("\n");
if (p->echo > 2) std::printf("# green_action::multiply\n");

nops += green_potential::multiply<real_t,R1C2,Noco>(y, x, p->Veff, p->veff_index,
p->target_minus_source, p->grid_spacing_trunc, nnzb, p->E_param,
p->V_confinement, pow2(p->r_confinement), p->echo);

for (int dd = 0; dd < 3; ++dd) { 
nops += p->kinetic[dd].multiply<real_t,R1C2,Noco>(y, x, p->phase[dd], p->echo);
} 

nops += green_dyadic::multiply<real_t,R1C2,Noco>(y, apc, x, p->dyadic_plan,
p->rowindx, colIndex, p->rowCubePos, nnzb, p->echo);

if (p->echo > 4) std::printf("# green_action::multiply %g Gflop\n", nops*1e-9);

return nops;
} 

plan_t * get_plan() { return p; }

private: 

plan_t *p; 

real_t (*apc)[R1C2][Noco][LM]; 
real_t (*aac)[R1C2][Noco][LM]; 

}; 


#ifdef  NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

inline status_t test_construction_and_destruction(int const echo=0) {
{   plan_t plan;
if (echo > 4) std::printf("# %s for action_t\n", __func__);
{ action_t<float ,1,1> action(&plan); }
{ action_t<float ,2,1> action(&plan); }
{ action_t<float ,2,2> action(&plan); }
{ action_t<double,1,1> action(&plan); }
{ action_t<double,2,1> action(&plan); }
{ action_t<double,2,2> action(&plan); }
if (echo > 5) std::printf("# Hint: to test action_t::multiply, please envoke --test green_function\n");
} 
if (echo > 6) {
std::printf("# %s sizeof(atom_t) = %ld Byte\n", __func__, sizeof(atom_t));
std::printf("# %s sizeof(atom_image_t) = %ld Byte\n", __func__, sizeof(atom_image_t));
std::printf("# %s sizeof(plan_t) = %ld Byte\n", __func__, sizeof(plan_t));
} 
return 0;
} 

inline status_t all_tests(int const echo=0) {
status_t stat(0);
stat += test_construction_and_destruction(echo);
return stat;
} 

#endif 

} 
