#ifndef TL_CHECKPOINT_BASE_HPP
#define TL_CHECKPOINT_BASE_HPP
#include "tl-pragmasupport.hpp"
namespace TL
{
namespace Checkpoint
{
class Base : public TL::PragmaCustomCompilerPhase
{
public:
Base();
void store_directive_handler_post(TL::PragmaCustomDirective);
void load_directive_handler_post(TL::PragmaCustomDirective);
void init_directive_handler_post(TL::PragmaCustomDirective);
void shutdown_directive_handler_post(TL::PragmaCustomDirective);
};
}
}
#endif 
