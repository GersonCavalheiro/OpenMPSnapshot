#ifndef TL_CLAUSESINFO_HPP
#define TL_CLAUSESINFO_HPP
#include <string>
#include "tl-object.hpp"
#include "tl-objectlist.hpp"
#include "tl-type-fwd.hpp"
#include "tl-nodecl-fwd.hpp"
#include <map>
namespace TL
{
class PragmaCustomConstruct;
class LIBTL_CLASS ClausesInfo : public Object
{
private:
struct DirectiveClauses
{
ObjectList<std::string> referenced_clauses;
ObjectList<std::string> all_clauses;
std::string file;
std::string line;
std::string pragma;
};
std::map<Nodecl::NodeclBase, DirectiveClauses> _directive_clauses_map;
DirectiveClauses& lookup_map(Nodecl::NodeclBase a);
public:
ClausesInfo();
void set_all_clauses(Nodecl::NodeclBase directive, ObjectList<std::string> all_clauses);
void add_referenced_clause(Nodecl::NodeclBase directive, std::string clause_name);
void add_referenced_clause(Nodecl::NodeclBase directive, const ObjectList<std::string> & clause_names);
void set_locus_info(Nodecl::NodeclBase directive);
void set_pragma(const PragmaCustomConstruct& directive);
bool directive_already_defined(Nodecl::NodeclBase directive);
ObjectList<std::string> get_unreferenced_clauses(Nodecl::NodeclBase directive);
std::string get_locus_info(Nodecl::NodeclBase directive);
std::string get_pragma(Nodecl::NodeclBase directive);
};
}
#endif 
