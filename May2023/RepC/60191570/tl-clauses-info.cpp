#include "tl-clauses-info.hpp"
namespace TL
{
ClausesInfo::ClausesInfo()
{
}
ClausesInfo::DirectiveClauses& ClausesInfo::lookup_map(AST_t a)
{
return _directive_clauses_map[a];
}
void ClausesInfo::set_all_clauses(AST_t directive, ObjectList<std::string> all_clauses)
{
_directive_clauses_map[directive].all_clauses = all_clauses;
}
void ClausesInfo::add_referenced_clause(AST_t directive, std::string clause_name)
{
_directive_clauses_map[directive].referenced_clauses.append(clause_name);
}
void ClausesInfo::add_referenced_clause(AST_t directive, const ObjectList<std::string> & clause_names)
{
for(ObjectList<std::string>::const_iterator it = clause_names.begin();
it != clause_names.end();
++it)
{
_directive_clauses_map[directive].referenced_clauses.append(*it);
}
}
void ClausesInfo::set_locus_info(AST_t directive)
{
_directive_clauses_map[directive].file = directive.get_file();
std::stringstream line;
line << directive.get_line();
_directive_clauses_map[directive].line = line.str();
}
void ClausesInfo::set_pragma(const PragmaCustomConstruct& directive)
{
_directive_clauses_map[directive.get_ast()].pragma = directive.get_pragma() + " " + directive.get_directive();
}
bool ClausesInfo::directive_already_defined(AST_t directive)
{
bool defined = false;
if (_directive_clauses_map.find(directive) != _directive_clauses_map.end())
{
defined = true;
}
return defined;
}
ObjectList<std::string> ClausesInfo::get_unreferenced_clauses(AST_t directive)
{
DirectiveClauses directive_entry = _directive_clauses_map[directive];
ObjectList<std::string> unref = directive_entry.all_clauses.filter(not_in_set(directive_entry.referenced_clauses));
return unref;
}
std::string ClausesInfo::get_locus_info(AST_t directive)
{
return _directive_clauses_map[directive].file + ":" + _directive_clauses_map[directive].line;
}
std::string ClausesInfo::get_pragma(AST_t directive)
{
return _directive_clauses_map[directive].pragma;
}
}
