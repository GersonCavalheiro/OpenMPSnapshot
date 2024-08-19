#ifdef HAVE_CONFIG_H
#include <config.h>
#endif
#include "tl-omp-base.hpp"
namespace TL { namespace OpenMP {
void Base::ompss_target_handler_pre(TL::PragmaCustomStatement stmt) { }
void Base::ompss_target_handler_post(TL::PragmaCustomStatement stmt)
{
TL::PragmaCustomLine pragma_line = stmt.get_pragma_line();
pragma_line.diagnostic_unused_clauses();
stmt.replace(stmt.get_statements());
}
void Base::ompss_target_handler_pre(TL::PragmaCustomDeclaration decl) { }
void Base::ompss_target_handler_post(TL::PragmaCustomDeclaration decl)
{
TL::PragmaCustomLine pragma_line = decl.get_pragma_line();
if (decl.get_nested_pragma().is_null())
{
Nodecl::NodeclBase result;
ObjectList<Nodecl::NodeclBase> devices;
ObjectList<Nodecl::NodeclBase> symbols;
const locus_t* locus = decl.get_locus();
PragmaCustomClause device_clause = pragma_line.get_clause("device");
if (device_clause.is_defined())
{
ObjectList<std::string> device_names = device_clause.get_tokenized_arguments();
for (ObjectList<std::string>::iterator it = device_names.begin();
it != device_names.end();
++it)
{
devices.append(Nodecl::Text::make(*it, locus));
}
}
ERROR_CONDITION(!decl.has_symbol(),
"%s: expecting a function declaration or definition", decl.get_locus_str().c_str());
Symbol sym = decl.get_symbol();
symbols.append(Nodecl::Symbol::make(sym, locus));
result = Nodecl::OmpSs::TargetDeclaration::make(
Nodecl::List::make(devices),
Nodecl::List::make(symbols),
locus);
pragma_line.diagnostic_unused_clauses();
decl.replace(result);
}
else
{
pragma_line.diagnostic_unused_clauses();
Nodecl::Utils::remove_from_enclosing_list(decl);
}
}
}  }
