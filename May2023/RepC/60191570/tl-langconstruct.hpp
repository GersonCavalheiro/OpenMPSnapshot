#ifndef TL_LANGCONSTRUCT_HPP
#define TL_LANGCONSTRUCT_HPP
#include "tl-common.hpp"
#include "tl-object.hpp"
#include "tl-ast.hpp"
#include "tl-symbol.hpp"
#include "tl-scopelink.hpp"
#include "tl-builtin.hpp"
#include "tl-source.hpp"
#include "tl-type.hpp"
#include "cxx-attrnames.h"
#include "cxx-macros.h"
#include <iostream>
#include <set>
#include <string>
#include <utility>
namespace TL
{
class FunctionDefinition;
class Statement;
class IdExpression;
class LIBTL_CLASS LangConstruct : public TL::Object
{
protected:
AST_t _ref;
ScopeLink _scope_link;
public:
LangConstruct(AST_t ref, ScopeLink scope_link)
: _ref(ref), _scope_link(scope_link)
{
}
enum SymbolsWanted
{
ALL_SYMBOLS = 0,
ONLY_OBJECTS,
ONLY_VARIABLES = ONLY_OBJECTS, 
ONLY_FUNCTIONS
};
virtual std::string prettyprint() const;
operator std::string() const;
AST_t get_ast() const
{
return _ref;
}
ScopeLink get_scope_link() const
{
return _scope_link;
}
Scope get_scope() const
{
return _scope_link.get_scope(_ref);
}
FunctionDefinition get_enclosing_function() const;
Statement get_enclosing_statement() const;
ObjectList<IdExpression> all_symbol_occurrences(SymbolsWanted symbols = ALL_SYMBOLS) const;
ObjectList<IdExpression> non_local_symbol_occurrences(SymbolsWanted symbols = ALL_SYMBOLS) const;
ObjectList<Symbol> non_local_symbols(SymbolsWanted symbols = ALL_SYMBOLS) const;
ObjectList<IdExpression> local_symbol_occurrences() const;
const static AlwaysFalse<AST_t> predicate;
virtual ~LangConstruct()
{
}
};
LIBTL_EXTERN std::ostream& operator<< (std::ostream& o, const LangConstruct& lang_construct);
class Declaration;
enum IdExpressionCriteria
{
VALID_SYMBOLS = 0,
INVALID_SYMBOLS,
ALL_FOUND_SYMBOLS
};
class Expression;
class Statement;
class LIBTL_CLASS IdExpression : public LangConstruct
{
private:
public:
IdExpression(AST_t ref, ScopeLink scope_link)
: LangConstruct(ref, scope_link)
{
}
std::string mangle_id_expression() const;
std::string get_qualified_part() const;
std::string get_unqualified_part(bool with_template_id = false) const;
bool is_qualified() const;
bool is_unqualified() const;
bool is_template_id() const;
std::string get_template_name() const;
std::string get_template_arguments() const;
Symbol get_symbol() const;
Symbol get_computed_symbol() const;
Declaration get_declaration() const;
static const PredicateAttr predicate;
Expression get_expression() const;
};
class TemplateHeader;
class LinkageSpecifier;
class TemplateParameterConstruct;
class DeclaredEntity;
class LIBTL_CLASS FunctionDefinition : public LangConstruct
{
private:
public:
FunctionDefinition(AST_t ref, ScopeLink scope_link)
: LangConstruct(ref, scope_link)
{
if (ref.is_list())
{
ASTIterator iter = ref.get_list_iterator();
AST_t item = iter.item();
if (FunctionDefinition::predicate(item))
{
_ref = item;
}
}
}
void prepend_sibling(AST_t);
IdExpression get_function_name() const;
Statement get_function_body() const;
bool is_templated() const;
ObjectList<TemplateHeader> get_template_header() const;
bool has_linkage_specifier() const;
ObjectList<LinkageSpecifier> get_linkage_specifier() const;
DeclaredEntity get_declared_entity() const;
Symbol get_function_symbol() const;
AST_t get_point_of_declaration() const;
static const PredicateAttr predicate;
};
class LIBTL_CLASS Expression : public LangConstruct
{
private:
AST_t _orig;
static AST_t advance_over_nests(AST_t);
public :
enum OperationKind
{
UNKNOWN = 0,
DERREFERENCE,
REFERENCE,
PLUS,
MINUS,
ADDITION,
SUBSTRACTION,
MULTIPLICATION,
DIVISION,
MODULUS,
SHIFT_LEFT,
SHIFT_RIGHT,
LOGICAL_OR,
LOGICAL_AND,
LOGICAL_NOT,
BITWISE_OR,
BITWISE_AND,
BITWISE_XOR,
BITWISE_NOT,
LOWER_THAN,
GREATER_THAN,
LOWER_EQUAL_THAN,
GREATER_EQUAL_THAN,
COMPARISON,
DIFFERENT,
PREINCREMENT,
POSTINCREMENT,
PREDECREMENT,
POSTDECREMENT,
CONDITIONAL,
ASSIGNMENT
};
Type get_type() const;
Type get_type(bool &is_lvalue) const;
Expression(AST_t ref, ScopeLink scope_link)
: LangConstruct(ref, scope_link)
{
this->_orig = this->_ref;
this->_ref = advance_over_nests(this->_ref);
}
bool is_id_expression();
bool is_accessed_member();
IdExpression get_id_expression();
bool is_binary_operation();
Expression get_first_operand();
Expression get_second_operand();
bool is_unary_operation();
Expression get_unary_operand();
bool is_casting();
AST_t get_cast_type();
Expression get_casted_expression();
bool is_literal();
bool is_function_call();
Expression get_called_expression();
ObjectList<Expression> get_argument_list();
bool is_named_function_call();
Symbol get_called_entity();
bool is_assignment();
bool is_operation_assignment();
bool is_array_subscript();
Expression get_subscript_expression();
Expression get_subscripted_expression();
bool is_this_variable();
Symbol get_this_symbol();
bool is_this_access();
bool is_member_access();
bool is_pointer_member_access();
Expression get_accessed_entity();
IdExpression get_accessed_member();
bool is_conditional();
Expression get_condition_expression();
Expression get_true_expression();
Expression get_false_expression();
OperationKind get_operation_kind();
std::string get_operator_str();
DEPRECATED bool is_array_section();
bool is_array_section_range();
bool is_array_section_size();
Expression array_section_item();
Expression array_section_lower();
Expression array_section_upper();
bool is_shaping_expression();
Expression shaped_expression();
ObjectList<Expression> shape_list();
bool is_throw_expression();
Expression get_throw_expression();
static const PredicateAttr predicate;
Expression get_enclosing_expression();
Expression get_top_enclosing_expression();
bool is_constant();
int evaluate_constant_int_expression(bool &valid);
AST_t original_tree()
{
return _orig;
}
bool has_symbol();
Symbol get_symbol();
bool is_top_level_expression();
bool is_sizeof();
bool is_sizeof_typeid();
};
class LIBTL_CLASS ParameterDeclaration : public LangConstruct
{
private:
Type _type;
public:
ParameterDeclaration(AST_t tree, ScopeLink sl, Type parameter_type)
: LangConstruct(tree, sl), _type(parameter_type)
{
}
bool is_named() const;
IdExpression get_name() const;
Type get_type() const
{
return _type;
}
static const PredicateAttr predicate;
};
class LIBTL_CLASS DeclaredEntity : public LangConstruct
{
public :
DeclaredEntity(AST_t ast, ScopeLink scope_link)
: LangConstruct(ast, scope_link)
{
}
IdExpression get_declared_entity() const DEPRECATED;
Symbol get_declared_symbol() const;
AST_t get_declared_tree() const;
bool has_initializer() const;
Expression get_initializer() const;
AST_t get_declarator_tree() const;
bool is_functional_declaration() const;
ObjectList<ParameterDeclaration> get_parameter_declarations() const;
ObjectList<ParameterDeclaration> get_parameter_declarations(bool &has_ellipsis) const;
bool functional_declaration_lacks_prototype() const;
static const PredicateAttr predicate;
};
class TypeSpec : public LangConstruct
{
public:
TypeSpec(AST_t ast, ScopeLink scope_link)
: LangConstruct(ast, scope_link)
{
}
bool is_class_specifier() const;
Symbol get_class_symbol() const;
bool is_enum_specifier() const;
Symbol get_enum_symbol() const;
Type get_type() const;
static const PredicateAttr predicate;
};
class LIBTL_CLASS DeclarationSpec : public LangConstruct
{
public:
DeclarationSpec(AST_t ast, ScopeLink scope_link)
: LangConstruct(ast, scope_link)
{
}
TypeSpec get_type_spec() const;
};
class LIBTL_CLASS Declaration : public LangConstruct
{
public:
Declaration(AST_t ast, ScopeLink scope_link)
: LangConstruct(ast, scope_link)
{
PredicateAttr decl_stmt(LANG_IS_DECLARATION_STATEMENT);
if (decl_stmt(ast))
{
ast = ast.get_link_to_child(LANG_DECLARATION_STATEMENT_DECLARATION);
}
}
ObjectList<DeclaredEntity> get_declared_entities() const;
DeclarationSpec get_declaration_specifiers() const;
bool has_linkage_specifier() const;
ObjectList<LinkageSpecifier> get_linkage_specifier() const;
bool is_templated() const;
ObjectList<TemplateHeader> get_template_header() const;
AST_t get_point_of_declaration() const;
bool is_empty_declaration() const;
static const PredicateAttr predicate;
};
class LIBTL_CLASS TemplateParameterConstruct : public LangConstruct
{
public:
TemplateParameterConstruct(AST_t ast, ScopeLink scope_link)
: LangConstruct(ast, scope_link)
{
}
bool is_named() const;
std::string get_name() const;
bool is_type() const ;
bool is_nontype() const;
bool is_template() const;
Symbol get_symbol() const;
};
class LIBTL_CLASS TemplateHeader : public LangConstruct
{
public:
TemplateHeader(AST_t ast, ScopeLink scope_link)
: LangConstruct(ast, scope_link)
{
}
ObjectList<TemplateParameterConstruct> get_parameters() const;
virtual std::string prettyprint() const;
};
class LIBTL_CLASS LinkageSpecifier : public LangConstruct
{
public:
LinkageSpecifier(AST_t ast, ScopeLink scope_link)
: LangConstruct(ast, scope_link)
{
}
virtual std::string prettyprint() const;
};
class LIBTL_CLASS GCCAttribute : public LangConstruct
{
private:
public:
GCCAttribute(AST_t ast, ScopeLink scope_link)
: LangConstruct(ast, scope_link)
{
}
std::string get_name() const;
bool has_argument_list() const;
ObjectList<Expression> get_argument_list() const;
};
class LIBTL_CLASS GCCAttributeSpecifier : public LangConstruct
{
private:
public:
GCCAttributeSpecifier(AST_t ast, ScopeLink scope_link)
: LangConstruct(ast, scope_link)
{
}
ObjectList<GCCAttribute> get_gcc_attribute_list() const;
static const PredicateAttr predicate;
};
class LIBTL_CLASS ReplaceIdExpression
{
protected:
std::map<Symbol, std::string> _repl_map;
std::string _repl_this;
public:
ReplaceIdExpression()
{
}
void add_replacement(Symbol sym, AST_t ast);
void add_replacement(Symbol sym, const std::string& str);
void add_replacement(Symbol sym, Source src);
void add_replacement(Symbol sym, const std::string& str, AST_t ref_tree, ScopeLink scope_link) DEPRECATED;
void add_replacement(Symbol sym, Source src, AST_t ref_tree, ScopeLink scope_link) DEPRECATED;
void add_this_replacement(const std::string& str);
void add_this_replacement(Source src);
void add_this_replacement(AST_t ast);
bool has_replacement(Symbol sym) const;
template <class T>
T replace(T orig_stmt) const
{
std::pair<AST_t, ScopeLink> modified_statement = 
orig_stmt.get_ast().duplicate_with_scope(orig_stmt.get_scope_link());
T result(modified_statement.first, modified_statement.second);
ObjectList<IdExpression> id_expressions = result.non_local_symbol_occurrences();
for (ObjectList<IdExpression>::iterator it = id_expressions.begin();
it != id_expressions.end();
it++)
{
Symbol sym = it->get_symbol();
if (_repl_map.find(sym) != _repl_map.end())
{
AST_t orig_ast = it->get_ast();
Source src;
src << _repl_map.find(sym)->second
;
AST_t repl_ast = src.parse_expression(orig_ast, 
orig_stmt.get_scope_link(),
Source::DO_NOT_CHECK_EXPRESSION);
orig_ast.replace_with(repl_ast);
}
}
if (_repl_this != "")
{
ObjectList<AST_t> this_references = result.get_ast().depth_subtrees(PredicateAttr(LANG_IS_THIS_VARIABLE));
for (ObjectList<AST_t>::iterator it = this_references.begin();
it != this_references.end();
it++)
{
AST_t &orig_ast(*it);
Source src;
src << _repl_this
;
AST_t repl_ast = src.parse_expression(orig_ast, 
orig_stmt.get_scope_link(),
Source::DO_NOT_CHECK_EXPRESSION);
orig_ast.replace(repl_ast);
}
}
return result;
}
};
class LIBTL_CLASS ReplaceSrcIdExpression
{
protected:
static const char* prettyprint_callback(AST a, void* data);
std::map<Symbol, std::string> _repl_map;
std::string _repl_this;
ScopeLink _sl;
bool _do_not_replace_declarators;
bool _ignore_pragmas;
public:
ReplaceSrcIdExpression(ScopeLink sl)
: _sl(sl), 
_do_not_replace_declarators(false),
_ignore_pragmas(false) { }
void add_replacement(Symbol sym, const std::string& str);
bool has_replacement(Symbol sym) const;
std::string get_replacement(Symbol sym) const;
void add_this_replacement(const std::string& str);
Source replace(AST_t a) const;
Source replace(LangConstruct a) const;
void set_replace_declarators(bool b);
void set_ignore_pragma(bool b);
ScopeLink get_scope_link() const;
virtual ~ReplaceSrcIdExpression() { }
};
class LIBTL_CLASS GetSymbolFromAST : public Functor<Symbol, AST_t>
{
private:
ScopeLink scope_link;
public:
virtual Symbol do_(AST_t& ast) const 
{
Scope sc = scope_link.get_scope(ast);
Symbol result = sc.get_symbol_from_id_expr(ast);
return result;
}
GetSymbolFromAST(ScopeLink _scope_link)
: scope_link(_scope_link)
{
}
~GetSymbolFromAST()
{
}
};
}
#endif 
#include "tl-statement.hpp"
