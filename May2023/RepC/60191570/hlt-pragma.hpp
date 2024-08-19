#ifndef HLT_PRAGMA_HPP
#define HLT_PRAGMA_HPP
#include "tl-pragmasupport.hpp"
namespace TL
{
namespace HLT
{
class HLTPragmaPhase : public PragmaCustomCompilerPhase
{
public:
HLTPragmaPhase();
virtual void run(TL::DTO& dto);
private:
void do_loop_unroll(TL::PragmaCustomStatement construct);
void do_loop_normalize(TL::PragmaCustomStatement construct);
void do_loop_collapse(TL::PragmaCustomStatement construct);
};
}
}
#endif 
