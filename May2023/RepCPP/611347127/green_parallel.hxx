#pragma once

#include "status.hxx" 
#include "simple_stats.hxx" 

namespace green_parallel {

int init(int argc=0, char **argv=nullptr); 

unsigned size(void); 

int rank(void); 

int finalize(void); 

int max(uint16_t data[], size_t const n); 

int allreduce(simple_stats::Stats<double> & stats); 

typedef uint16_t rank_int_t;

status_t dyadic_exchange(                                    
double       *const mat_out 
, std::vector<int32_t> const & requests  
, double const *const mat_inp 
, std::vector<int32_t> const & offerings 
, rank_int_t const owner_rank[] 
, uint32_t const nall 
, int const count
, bool const debug=true
, int const echo=0 
); 

status_t potential_exchange(
double    (*const Veff[4])[64]  
, std::vector<int64_t> const & requests  
, double const (*const Vinp)[64]  
, std::vector<int64_t> const & offerings 
, rank_int_t const owner_rank[] 
, uint32_t const nb[3] 
, int const Noco=1 
, bool const debug=true
, int const echo=0 
); 

status_t all_tests(int echo=0); 

} 

