#include "tl-source.hpp"
#include "tl-lowering-visitor.hpp"
#include "tl-nodecl-utils.hpp"
#include "tl-counters.hpp"
#include "cxx-cexpr.h"
#include "tl-predicateutils.hpp"
#include "tl-devices.hpp"
#include "tl-nanos.hpp"
namespace TL { namespace Nanox {
void LoweringVisitor::visit(const Nodecl::OpenMP::ForAppendix& construct)
{
Nodecl::List distribute_environment = construct.get_environment().as<Nodecl::List>();
Nodecl::OpenMP::Schedule schedule = distribute_environment.find_first<Nodecl::OpenMP::Schedule>();
ERROR_CONDITION(schedule.is_null(), "Schedule tree is missing", 0);
std::string schedule_name = schedule.get_text();
std::string ompss_prefix = "ompss_";
bool is_ompss_schedule = (schedule_name.substr(0, ompss_prefix.size()) == ompss_prefix);
std::string openmp_prefix0 = "omp_";
std::string openmp_prefix1 = "openmp_";
bool is_explicit_openmp_schedule = (schedule_name.substr(0, openmp_prefix0.size()) == openmp_prefix0)
|| (schedule_name.substr(0, openmp_prefix1.size()) == openmp_prefix1);
if (!is_ompss_schedule
&& !is_explicit_openmp_schedule)
{
if (_lowering->in_ompss_mode())
{
std::string fixed_schedule = schedule_name;
fixed_schedule = "ompss_" + schedule_name;
is_ompss_schedule = true;
schedule.set_text(fixed_schedule);
}
}
if (is_explicit_openmp_schedule)
{
std::string fixed_schedule;
if (schedule_name.substr(0, openmp_prefix0.size()) == openmp_prefix0)
fixed_schedule = schedule_name.substr(openmp_prefix0.size());
else if (schedule_name.substr(0, openmp_prefix1.size()) == openmp_prefix1)
fixed_schedule = schedule_name.substr(openmp_prefix1.size());
schedule.set_text(fixed_schedule);
}
if (is_ompss_schedule)
{
Nodecl::NodeclBase new_construct = construct;
#if 0
bool generate_final_stmts =
Nanos::Version::interface_is_at_least("master", 5024) &&
!_lowering->final_clause_transformation_disabled();
if (_lowering->in_ompss_mode() && generate_final_stmts)
{
new_construct =
Nodecl::OpenMP::ForAppendix::make(distribute_environment,
construct.get_loop(),
construct.get_prependix(),
construct.get_appendix());
Nodecl::NodeclBase copied_statements_placeholder;
TL::Source code;
code
<< "{"
<<      as_type(TL::Type::get_bool_type()) << "mcc_is_in_final;"
<<      "nanos_err_t mcc_err_in_final = nanos_in_final(&mcc_is_in_final);"
<<      "if (mcc_err_in_final != NANOS_OK) nanos_handle_error(mcc_err_in_final);"
<<      "if (mcc_is_in_final)"
<<      "{"
<<          statement_placeholder(copied_statements_placeholder)
<<      "}"
<<      "else"
<<      "{"
<<          as_statement(new_construct)
<<      "}"
<< "}"
;
if (IS_FORTRAN_LANGUAGE)
Source::source_language = SourceLanguage::C;
Nodecl::NodeclBase if_else_tree = code.parse_statement(construct);
if (IS_FORTRAN_LANGUAGE)
Source::source_language = SourceLanguage::Current;
construct.replace(if_else_tree);
Nodecl::NodeclBase final_stmt_list = copied_statements_placeholder.get_parent();
std::map<Nodecl::NodeclBase, Nodecl::NodeclBase>::iterator it = _final_stmts_map.find(construct);
ERROR_CONDITION(it == _final_stmts_map.end(), "Unreachable code", 0);
copied_statements_placeholder.replace(it->second);
ERROR_CONDITION(!copied_statements_placeholder.is_in_list(), "Unreachable code\n", 0);
walk(final_stmt_list);
}
#endif
lower_for_slicer(new_construct.as<Nodecl::OpenMP::For>(),
construct.get_prependix(),
construct.get_appendix());
}
else
{
lower_for_worksharing(construct.as<Nodecl::OpenMP::For>(),
construct.get_prependix(),
construct.get_appendix());
}
}
} }
