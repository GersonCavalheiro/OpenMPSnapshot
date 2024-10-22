

#include "template_stack.hpp"
#include <cassert>
#include "files.hpp"

#ifdef BOOST_MSVC
#pragma warning(disable : 4355)
#endif

namespace quickbook
{
template_symbol::template_symbol(
std::string const& identifier_,
std::vector<std::string> const& params_,
value const& content_,
template_scope const* lexical_parent_)
: identifier(identifier_)
, params(params_)
, content(content_)
, lexical_parent(lexical_parent_)
{
assert(
content.get_tag() == template_tags::block ||
content.get_tag() == template_tags::phrase ||
content.get_tag() == template_tags::snippet);
}

template_stack::template_stack()
: scope(template_stack::parser(*this)), scopes(), parent_1_4(0)
{
scopes.push_front(template_scope());
parent_1_4 = &scopes.front();
}

template_symbol* template_stack::find(std::string const& symbol) const
{
for (template_scope const* i = &*scopes.begin(); i;
i = i->parent_scope) {
if (template_symbol* ts =
boost::spirit::classic::find(i->symbols, symbol.c_str()))
return ts;
}
return 0;
}

template_symbol* template_stack::find_top_scope(
std::string const& symbol) const
{
return boost::spirit::classic::find(
scopes.front().symbols, symbol.c_str());
}

template_symbols const& template_stack::top() const
{
BOOST_ASSERT(!scopes.empty());
return scopes.front().symbols;
}

template_scope const& template_stack::top_scope() const
{
BOOST_ASSERT(!scopes.empty());
return scopes.front();
}

bool template_stack::add(template_symbol const& ts)
{
BOOST_ASSERT(!scopes.empty());
BOOST_ASSERT(ts.lexical_parent);

if (this->find_top_scope(ts.identifier)) {
return false;
}

boost::spirit::classic::add(
scopes.front().symbols, ts.identifier.c_str(), ts);

return true;
}

void template_stack::push()
{
template_scope const& old_front = scopes.front();
scopes.push_front(template_scope());
scopes.front().parent_1_4 = parent_1_4;
scopes.front().parent_scope = &old_front;
parent_1_4 = &scopes.front();
}

void template_stack::pop()
{
parent_1_4 = scopes.front().parent_1_4;
scopes.pop_front();
}

void template_stack::start_template(template_symbol const* symbol)
{
if (symbol->content.get_file()->version() >= 105u) {
parent_1_4 = scopes.front().parent_1_4;
scopes.front().parent_scope = symbol->lexical_parent;
}
else {
scopes.front().parent_scope = scopes.front().parent_1_4;
}
}
}
