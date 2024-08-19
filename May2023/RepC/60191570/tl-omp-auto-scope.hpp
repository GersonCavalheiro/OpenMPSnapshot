#ifndef TL_OMP_AUTO_SCOPE_HPP
#define TL_OMP_AUTO_SCOPE_HPP
#include "tl-analysis-interface.hpp"
#include "tl-pragmasupport.hpp"
namespace TL {
namespace OpenMP {
class AutoScopePhase : public TL::PragmaCustomCompilerPhase
{
private:
std::string _auto_scope_enabled_str;
bool _auto_scope_enabled;
void set_auto_scope(const std::string auto_scope_enabled_str);
std::string _ompss_mode_str;
bool _ompss_mode_enabled;
void set_ompss_mode( const std::string& ompss_mode_str);
public:
AutoScopePhase( );
virtual ~AutoScopePhase( ) {}
virtual void pre_run(TL::DTO& dto);
virtual void run( TL::DTO& dto );
};
}
}
#endif 
