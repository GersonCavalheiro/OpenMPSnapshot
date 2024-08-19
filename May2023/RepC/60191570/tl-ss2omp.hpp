#ifndef TL_SS2OMP_HPP
#define TL_SS2OMP_HPP
#include "tl-pragmasupport.hpp"
namespace TL
{
class SS2OpenMP : public PragmaCustomCompilerPhase
{
private:
public:
SS2OpenMP()
: PragmaCustomCompilerPhase("css")
{
set_phase_name("Superscalar to OpenMP");
set_phase_description("This phase converts a subset of Superscalar into OpenMP plus dependences");
on_directive_post["task"].connect(functor(&SS2OpenMP::on_post_task, *this));
on_directive_post["target"].connect(functor(&SS2OpenMP::construct_not_implemented, *this));
register_directive("start");
register_directive("finish");
register_directive("barrier");
register_directive("wait");
register_directive("restart");
register_directive("mutex");
on_directive_post["start"].connect(functor(&SS2OpenMP::remove_directive, *this));
on_directive_post["finish"].connect(functor(&SS2OpenMP::on_post_finish, *this));
on_directive_post["barrier"].connect(functor(&SS2OpenMP::on_post_barrier, *this));
on_directive_post["wait"].connect(functor(&SS2OpenMP::on_post_wait, *this));
on_directive_post["mutex"].connect(functor(&SS2OpenMP::directive_not_implemented, *this));
on_directive_post["restart"].connect(functor(&SS2OpenMP::directive_not_implemented, *this));
}
void on_post_task(PragmaCustomConstruct construct);
void on_post_wait(PragmaCustomConstruct construct);
void on_post_finish(PragmaCustomConstruct construct);
void on_post_barrier(PragmaCustomConstruct construct);
void remove_directive(PragmaCustomConstruct);
void directive_not_implemented(PragmaCustomConstruct);
void construct_not_implemented(PragmaCustomConstruct);
virtual void run(DTO& dto);
};
}
#endif 
