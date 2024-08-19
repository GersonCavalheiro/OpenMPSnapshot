#include "tl-node.hpp"
#include "tl-pcfg-utils.hpp"
namespace TL {
namespace Analysis {
PCFGLoopControl::PCFGLoopControl()
: _init(NULL), _cond(NULL), _next(NULL)
{}
PCFGLoopControl::~PCFGLoopControl()
{
delete _init;
delete _cond;
delete _next;
}
PCFGTryBlock::PCFGTryBlock()
: _handler_parents(), _handler_exits(), _nhandlers(-1)
{}
PCFGTryBlock::~PCFGTryBlock()
{}
PCFGSwitch::PCFGSwitch(Node* condition, Node* exit )
: _condition(condition), _exit(exit)
{}
PCFGSwitch::~PCFGSwitch()
{
delete _condition;
delete _exit;
}
void PCFGSwitch::set_condition(Node* condition)
{
_condition = condition;
}
PCFGPragmaInfo::PCFGPragmaInfo()
: _clauses()
{}
PCFGPragmaInfo::PCFGPragmaInfo(const PCFGPragmaInfo& p)
{
_clauses = p._clauses;
}
PCFGPragmaInfo::PCFGPragmaInfo(const NBase& clause)
: _clauses(ObjectList<NBase>(1, clause))
{}
bool PCFGPragmaInfo::has_clause(node_t kind) const
{
for (ObjectList<NBase>::const_iterator it = _clauses.begin(); it != _clauses.end(); ++it)
{
if (it->get_kind() == kind)
return true;
}
return false;
}
NBase PCFGPragmaInfo::get_clause(node_t kind) const
{
for (ObjectList<NBase>::const_iterator it = _clauses.begin(); it != _clauses.end(); ++it)
if (it->get_kind() == kind)
return *it;
internal_error("No clause with kind %d found in pragma info.\n", kind);
}
void PCFGPragmaInfo::add_clause(const NBase& clause)
{
_clauses.append(clause);
}
ObjectList<NBase> PCFGPragmaInfo::get_clauses() const
{
return _clauses;
}
PCFGVisitUtils::PCFGVisitUtils()
: _last_nodes(), _return_nodes(), _outer_nodes(),
_continue_nodes(), _break_nodes(), _labeled_nodes(), _goto_nodes(),
_switch_nodes(), _nested_loop_nodes(), _tryblock_nodes(),
_pragma_nodes(), _context_nodecl(), _section_nodes(), _assert_nodes(),
_environ_entry_exit(), _is_vector(false), _nid(0)
{}
std::string print_node_list(const ObjectList<Node*>& list)
{
std::string result;
for(ObjectList<Node*>::const_iterator it = list.begin(); it != list.end(); )
{
std::stringstream ss; ss << (*it)->get_id();
result +=  ss.str();
++it;
if(it != list.end())
result += ", ";
}
return result;
}
}
}
