#ifndef TL_STATEMENT_HPP
#define TL_STATEMENT_HPP
#include "tl-common.hpp"
#include "tl-langconstruct.hpp"
namespace TL
{
class LIBTL_CLASS Condition : public LangConstruct
{
public:
Condition(AST_t ref, ScopeLink sl)
: LangConstruct(ref, sl)
{
}
bool is_expression() const; 
Expression get_expression() const;
bool is_declaration() const;
Declaration get_declaration() const;
};
class LIBTL_CLASS Statement : public LangConstruct
{
private:
public:
Statement(AST_t ref, ScopeLink scope_link)
: LangConstruct(ref, scope_link)
{
}
bool is_compound_statement() const;
ObjectList<Statement> get_inner_statements() const;
bool is_in_compound_statement() const;
Statement next() const;
Statement previous() const;
bool is_first() const;
bool is_last() const;
const static PredicateAttr predicate;
void prepend(Statement st);
void append(Statement st);
bool is_declaration() const;
bool is_simple_declaration() const;
Declaration get_simple_declaration() const;
bool is_expression() const;
Expression get_expression() const;
bool breaks_flow();
Statement get_pragma_line() const;
bool is_pragma_construct() const;
Statement get_pragma_construct_statement() const;
bool is_pragma_directive() const;
};
class LIBTL_CLASS ForStatement : public Statement
{
private:
AST_t _induction_variable;
AST_t _lower_bound;
AST_t _upper_bound;
AST_t _step;
Source _operator_bound;
void gather_for_information();
bool check_statement();
public:
ForStatement(AST_t ref, ScopeLink scope_link)
: Statement(ref, scope_link)
{
if (check_statement())
{
gather_for_information();
}
}
ForStatement(const Statement& st)
: Statement(st)
{
if (check_statement())
{
gather_for_information();
}
}
IdExpression get_induction_variable();
Expression get_lower_bound() const;
Expression get_upper_bound() const;
Expression get_step() const;
Source get_bound_operator() const;
Statement get_loop_body() const;
bool regular_loop() const;
bool is_regular_loop() const;
AST_t get_iterating_init() const;
Expression get_iterating_condition() const;
Expression get_iterating_expression() const;
const static PredicateAttr predicate;
};
class LIBTL_CLASS WhileStatement : public Statement
{
public:
WhileStatement(AST_t ref, ScopeLink sl)
: Statement(ref, sl)
{
}
Condition get_condition() const;
Statement get_body() const;
const static PredicateAttr predicate;
};
class LIBTL_CLASS IfStatement : public Statement
{
public:
IfStatement(AST_t ref, ScopeLink sl)
: Statement(ref, sl)
{
}
Condition get_condition() const;
Statement get_then_body() const;
bool has_else() const;
Statement get_else_body() const;
const static PredicateAttr predicate;
};
class LIBTL_CLASS DoWhileStatement : public Statement
{
public:
DoWhileStatement(AST_t ref, ScopeLink sl)
: Statement(ref, sl)
{
}
Statement get_body() const;
Expression get_expression() const;
const static PredicateAttr predicate;
};
class LIBTL_CLASS CaseStatement : public Statement
{
public:
CaseStatement(AST_t ast, ScopeLink sl)
: Statement(ast, sl)
{
}
Expression get_case_expression() const;
Statement get_statement() const;
const static PredicateAttr predicate;
};
class LIBTL_CLASS DefaultStatement : public Statement
{
public:
DefaultStatement(AST_t ast, ScopeLink sl)
: Statement(ast, sl)
{
}
Statement get_statement() const;
const static PredicateAttr predicate;
};
class LIBTL_CLASS BreakStatement : public Statement
{
public:
BreakStatement(AST_t ast, ScopeLink sl)
: Statement(ast, sl)
{
}
const static PredicateAttr predicate;
};
class LIBTL_CLASS ContinueStatement : public Statement
{
public:
ContinueStatement(AST_t ast, ScopeLink sl)
: Statement(ast, sl)
{
}
const static PredicateAttr predicate;
};
class LIBTL_CLASS SwitchStatement : public Statement
{
public:
SwitchStatement(AST_t ref, ScopeLink sl)
: Statement(ref, sl)
{
}
Condition get_condition() const;
Statement get_switch_body() const;
ObjectList<CaseStatement> get_cases() const;
ObjectList<DefaultStatement> get_defaults() const;
const static PredicateAttr predicate;
};
class LIBTL_CLASS TryStatement : public Statement
{
public:
TryStatement(AST_t ref, ScopeLink sl)
: Statement(ref, sl)
{
}
Statement get_try_protected_block() const;
ObjectList<Declaration> get_try_handler_declarations() const;
ObjectList<Statement> get_try_handler_blocks() const;
const static PredicateAttr predicate;
};
class LIBTL_CLASS ReturnStatement : public Statement
{
public:
ReturnStatement(AST_t ref, ScopeLink sl)
: Statement(ref, sl)
{
}
bool has_return_expression() const;
Expression get_return_expression() const;
const static PredicateAttr predicate;
};
class LIBTL_CLASS GotoStatement : public Statement
{
public:
GotoStatement(AST_t ref, ScopeLink sl)
: Statement(ref, sl)
{
}
std::string get_label() const;
const static PredicateAttr predicate;
};
class LIBTL_CLASS LabeledStatement : public Statement
{
public:
LabeledStatement(AST_t ref, ScopeLink sl)
: Statement(ref, sl)
{
}
Statement get_labeled_statement() const;
std::string get_label() const;
const static PredicateAttr predicate;
};
class LIBTL_CLASS EmptyStatement : public Statement
{
public:
EmptyStatement(AST_t ref, ScopeLink sl)
: Statement(ref, sl)
{}
const static PredicateAttr predicate;
};
}
#endif 
