#ifndef TL_PRAGMASUPPORT_HPP
#define TL_PRAGMASUPPORT_HPP
#include "tl-common.hpp"
#include <string>
#include <stack>
#include <algorithm>
#include "tl-clauses-info.hpp"
#include "tl-compilerphase.hpp"
#include "tl-handler.hpp"
#include "tl-source.hpp"
#include "cxx-attrnames.h"
namespace TL
{
class LIBTL_CLASS ClauseTokenizer
{
public:
virtual ObjectList<std::string> tokenize(const std::string& str) const = 0;
virtual ~ClauseTokenizer() { }
};
class LIBTL_CLASS NullClauseTokenizer : public ClauseTokenizer
{
public:
virtual ObjectList<std::string> tokenize(const std::string& str) const
{
ObjectList<std::string> result;
result.append(str);
return result;
}
};
class LIBTL_CLASS ExpressionTokenizer : public ClauseTokenizer
{
public:
virtual ObjectList<std::string> tokenize(const std::string& str) const
{
int bracket_nesting = 0;
ObjectList<std::string> result;
std::string temporary("");
for (std::string::const_iterator it = str.begin();
it != str.end();
it++)
{
const char & c(*it);
if (c == ',' 
&& bracket_nesting == 0
&& temporary != "")
{
result.append(temporary);
temporary = "";
}
else
{
if (c == '('
|| c == '{'
|| c == '[')
{
bracket_nesting++;
}
else if (c == ')'
|| c == '}'
|| c == ']')
{
bracket_nesting--;
}
temporary += c;
}
}
if (temporary != "")
{
result.append(temporary);
}
return result;
}
};
class LIBTL_CLASS ExpressionTokenizerTrim : public ExpressionTokenizer
{
public:
virtual ObjectList<std::string> tokenize(const std::string& str) const
{
ObjectList<std::string> result;
result = ExpressionTokenizer::tokenize(str);
std::transform(result.begin(), result.end(), result.begin(), trimExp);
return result;
}
private:
static std::string trimExp (const std::string &str) {
ssize_t first = str.find_first_not_of(" \t");
ssize_t last = str.find_last_not_of(" \t");
return str.substr(first, last - first + 1);
}
};
class LIBTL_CLASS PragmaCustomClause : public LangConstruct
{
private:
ObjectList<std::string> _clause_names;
ObjectList<AST_t> filter_pragma_clause();
public:
PragmaCustomClause(const std::string& src, AST_t ref, ScopeLink scope_link)
: LangConstruct(ref, scope_link) 
{
_clause_names.push_back(src);
}
PragmaCustomClause(const ObjectList<std::string> & src, AST_t ref, ScopeLink scope_link)
: LangConstruct(ref, scope_link), _clause_names(src) 
{
}
std::string get_clause_name() { return _clause_names[0]; }
bool is_defined();
ObjectList<Expression> get_expression_list();
ObjectList<IdExpression> get_id_expressions(IdExpressionCriteria criteria = VALID_SYMBOLS);
ObjectList<IdExpression> id_expressions(IdExpressionCriteria criteria = VALID_SYMBOLS);
ObjectList<std::string> get_arguments();
ObjectList<std::string> get_arguments(const ClauseTokenizer&);
ObjectList<ObjectList<std::string> > get_arguments_unflattened();
ObjectList<AST_t> get_arguments_tree();
};
class LIBTL_CLASS PragmaCustomConstruct : public LangConstruct, public LinkData
{
private:
DTO* _dto;
public:
PragmaCustomConstruct(AST_t ref, ScopeLink scope_link)
: LangConstruct(ref, scope_link),
_dto(NULL)
{
}
std::string get_pragma() const;
std::string get_directive() const;
bool is_directive() const;
bool is_construct() const;
Statement get_statement() const;
AST_t get_declaration() const;
AST_t get_pragma_line() const;
void init_clause_info() const;
bool is_function_definition() const;
bool is_parameterized() const;
ObjectList<IdExpression> get_parameter_id_expressions(IdExpressionCriteria criteria = VALID_SYMBOLS) const;
ObjectList<Expression> get_parameter_expressions() const;
ObjectList<std::string> get_parameter_arguments() const;
ObjectList<std::string> get_parameter_arguments(const ClauseTokenizer& tokenizer) const;
ObjectList<std::string> get_clause_names() const;
PragmaCustomClause get_clause(const std::string& name) const;
PragmaCustomClause get_clause(const ObjectList<std::string>& names) const;
void set_dto(DTO* dto);
bool get_show_warnings();
};
LIBTL_EXTERN bool is_pragma_custom(const std::string& pragma_preffix, 
AST_t ast,
ScopeLink scope_link);
LIBTL_EXTERN bool is_pragma_custom_directive(const std::string& pragma_preffix, 
const std::string& pragma_directive, 
AST_t ast,
ScopeLink scope_link);
LIBTL_EXTERN bool is_pragma_custom_construct(const std::string& pragma_preffix, 
const std::string& pragma_directive, 
AST_t ast,
ScopeLink scope_link);
typedef std::map<std::string, Signal1<PragmaCustomConstruct> > CustomFunctorMap;
class LIBTL_CLASS PragmaCustomDispatcher : public TraverseFunctor
{
private:
std::string _pragma_handled;
CustomFunctorMap& _pre_map;
CustomFunctorMap& _post_map;
DTO* _dto;
bool _warning_clauses;
std::stack<PragmaCustomConstruct*> _construct_stack;
void dispatch_pragma_construct(CustomFunctorMap& search_map, PragmaCustomConstruct& pragma_custom_construct);
public:
PragmaCustomDispatcher(const std::string& pragma_handled, 
CustomFunctorMap& pre_map,
CustomFunctorMap& post_map,
bool warning_clauses);
virtual void preorder(Context ctx, AST_t node);
virtual void postorder(Context ctx, AST_t node);
void set_dto(DTO* dto);
void set_warning_clauses(bool warning);
};
class LIBTL_CLASS PragmaCustomCompilerPhase : public CompilerPhase
{
private:
std::string _pragma_handled;
PragmaCustomDispatcher _pragma_dispatcher;
public:
PragmaCustomCompilerPhase(const std::string& pragma_handled);
virtual void pre_run(DTO& data_flow);
virtual void run(DTO& data_flow);
CustomFunctorMap on_directive_pre;
CustomFunctorMap on_directive_post;
void register_directive(const std::string& name);
void register_construct(const std::string& name, bool bound_to_statement = false);
void warning_pragma_unused_clauses(bool warning);
};
}
#endif 
