#ifndef HLT_COLLAPSE_LOOP_HPP
#define HLT_COLLAPSE_LOOP_HPP
#include "hlt-transform.hpp"
#include "tl-scope.hpp"
namespace TL { namespace HLT {
class LIBHLT_CLASS LoopCollapse : public Transform
{
private:
Nodecl::NodeclBase _loop;
Nodecl::NodeclBase _transformation;
TL::Scope _pragma_context;
int _collapse_factor;
Nodecl::List _post_transformation_stmts;
TL::ObjectList<TL::Symbol> _omp_capture_symbols;
public:
LoopCollapse();
LoopCollapse& set_loop(Nodecl::NodeclBase loop);
LoopCollapse& set_collapse_factor(int collapse_factor);
LoopCollapse& set_pragma_context(const TL::Scope& context);
void collapse();
Nodecl::NodeclBase get_whole_transformation() const { return _transformation; }
Nodecl::NodeclBase get_post_transformation_stmts() const;
TL::ObjectList<TL::Symbol> get_omp_capture_symbols() const;
};
}}
#endif 
