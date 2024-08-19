#ifndef TL_OMP_SIMD_HPP
#define TL_OMP_SIMD_HPP
#include "tl-pragmasupport.hpp"
namespace TL
{
namespace OpenMP
{
class Simd : public TL::PragmaCustomCompilerPhase
{
public:
Simd();
virtual void run(TL::DTO& dto);
virtual void pre_run(TL::DTO& dto);
virtual void phase_cleanup(TL::DTO& dto);
virtual ~Simd() { }
private:
std::string _simd_enabled_str;
std::string _svml_enabled_str;
std::string _fast_math_enabled_str;
std::string _avx2_enabled_str;
std::string _neon_enabled_str;
std::string _romol_enabled_str;
std::string _knc_enabled_str;
std::string _knl_enabled_str;
std::string _only_adjacent_accesses_str;
std::string _only_aligned_accesses_str;
std::string _overlap_in_place_str;
bool _simd_enabled;
bool _svml_enabled;
bool _fast_math_enabled;
bool _avx2_enabled;
bool _neon_enabled;
bool _romol_enabled;
bool _knc_enabled;
bool _knl_enabled;
bool _only_adjacent_accesses_enabled;
bool _only_aligned_accesses_enabled;
bool _overlap_in_place;
void set_simd(const std::string simd_enabled_str);
void set_svml(const std::string svml_enabled_str);
void set_fast_math(const std::string fast_math_enabled_str);
void set_avx2(const std::string avx2_enabled_str);
void set_neon(const std::string neon_enabled_str);
void set_romol(const std::string romol_enabled_str);
void set_knc(const std::string knc_enabled_str);
void set_knl(const std::string knl_enabled_str);
void set_only_adjcent_accesses(const std::string only_adjacent_accesses_str);
void set_only_aligned_accesses(const std::string only_aligned_accesses_str);
void set_overlap_in_place(const std::string overlap_in_place_str);
};
}
}
#endif 
