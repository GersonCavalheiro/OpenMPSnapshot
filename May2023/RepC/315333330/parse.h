#ifndef GO_PARSE_H
#define GO_PARSE_H
class Lex;
class Gogo;
class Named_object;
class Type;
class Typed_identifier;
class Typed_identifier_list;
class Channel_type;
class Function_type;
class Block;
class Expression;
class Expression_list;
class Struct_field_list;
class Case_clauses;
class Type_case_clauses;
class Select_clauses;
class Statement;
class Label;
class Parse
{
public:
Parse(Lex*, Gogo*);
void
program();
private:
enum Precedence
{
PRECEDENCE_INVALID = -1,
PRECEDENCE_NORMAL = 0,
PRECEDENCE_OROR,
PRECEDENCE_ANDAND,
PRECEDENCE_RELOP,
PRECEDENCE_ADDOP,
PRECEDENCE_MULOP
};
struct Range_clause
{
bool found;
Expression* index;
Expression* value;
Expression* range;
Range_clause()
: found(false), index(NULL), value(NULL), range(NULL)
{ }
};
struct Type_switch
{
bool found;
std::string name;
Location location;
Expression* expr;
Type_switch()
: found(false), name(), location(Linemap::unknown_location()),
expr(NULL)
{ }
};
class Enclosing_var
{
public:
Enclosing_var(Named_object* var, Named_object* in_function,
unsigned int index)
: var_(var), in_function_(in_function), index_(index)
{ }
Enclosing_var()
: var_(NULL), in_function_(NULL), index_(-1U)
{ }
Named_object*
var() const
{ return this->var_; }
Named_object*
in_function() const
{ return this->in_function_; }
unsigned int
index() const
{ return this->index_; }
private:
Named_object* var_;
Named_object* in_function_;
unsigned int index_;
};
struct Enclosing_var_comparison
{
bool
operator()(const Enclosing_var&, const Enclosing_var&) const;
};
typedef std::set<Enclosing_var, Enclosing_var_comparison> Enclosing_vars;
typedef std::map<std::string, const Typed_identifier*> Names;
const Token*
peek_token();
const Token*
advance_token();
void
unget_token(const Token&);
Location
location();
typedef std::vector<std::pair<Statement*, Label*> > Bc_stack;
void identifier_list(Typed_identifier_list*);
Expression_list* expression_list(Expression*, bool may_be_sink,
bool may_be_composite_lit);
bool qualified_ident(std::string*, Named_object**);
Type* type();
bool type_may_start_here();
Type* type_name(bool issue_error);
Type* array_type(bool may_use_ellipsis);
Type* map_type();
Type* struct_type();
void field_decl(Struct_field_list*);
Type* pointer_type();
Type* channel_type();
void check_signature_names(const Typed_identifier_list*, Names*);
Function_type* signature(Typed_identifier*, Location);
bool parameters(Typed_identifier_list**, bool* is_varargs);
Typed_identifier_list* parameter_list(bool* is_varargs);
void parameter_decl(bool, Typed_identifier_list*, bool*, bool*, bool*);
bool result(Typed_identifier_list**);
Location block();
Type* interface_type(bool record);
void method_spec(Typed_identifier_list*);
void declaration();
bool declaration_may_start_here();
void decl(void (Parse::*)(void*, unsigned int), void*, unsigned int pragmas);
void list(void (Parse::*)(void*, unsigned int), void*, bool);
void const_decl();
void const_spec(int, Type**, Expression_list**);
void type_decl(unsigned int pragmas);
void type_spec(void*, unsigned int pragmas);
void var_decl();
void var_spec(void*, unsigned int pragmas);
void init_vars(const Typed_identifier_list*, Type*, Expression_list*,
bool is_coloneq, Location);
bool init_vars_from_call(const Typed_identifier_list*, Type*, Expression*,
bool is_coloneq, Location);
bool init_vars_from_map(const Typed_identifier_list*, Type*, Expression*,
bool is_coloneq, Location);
bool init_vars_from_receive(const Typed_identifier_list*, Type*,
Expression*, bool is_coloneq, Location);
bool init_vars_from_type_guard(const Typed_identifier_list*, Type*,
Expression*, bool is_coloneq,
Location);
Named_object* init_var(const Typed_identifier&, Type*, Expression*,
bool is_coloneq, bool type_from_init, bool* is_new,
Expression_list* vars, Expression_list* vals);
Named_object* create_dummy_global(Type*, Expression*, Location);
void finish_init_vars(Expression_list* vars, Expression_list* vals,
Location);
void simple_var_decl_or_assignment(const std::string&, Location,
bool may_be_composite_lit,
Range_clause*, Type_switch*);
void function_decl(unsigned int pragmas);
Typed_identifier* receiver();
Expression* operand(bool may_be_sink, bool *is_parenthesized);
Expression* enclosing_var_reference(Named_object*, Named_object*,
bool may_be_sink, Location);
Expression* composite_lit(Type*, int depth, Location);
Expression* function_lit();
Expression* create_closure(Named_object* function, Enclosing_vars*,
Location);
Expression* primary_expr(bool may_be_sink, bool may_be_composite_lit,
bool* is_type_switch, bool* is_parenthesized);
Expression* selector(Expression*, bool* is_type_switch);
Expression* index(Expression*);
Expression* call(Expression*);
Expression* expression(Precedence, bool may_be_sink,
bool may_be_composite_lit, bool* is_type_switch,
bool *is_parenthesized);
bool expression_may_start_here();
Expression* unary_expr(bool may_be_sink, bool may_be_composite_lit,
bool* is_type_switch, bool* is_parenthesized);
Type* reassociate_chan_direction(Channel_type*, Location);
Expression* qualified_expr(Expression*, Location);
Expression* id_to_expression(const std::string&, Location, bool);
void statement(Label*);
bool statement_may_start_here();
void labeled_stmt(const std::string&, Location);
Expression* simple_stat(bool, bool*, Range_clause*, Type_switch*);
bool simple_stat_may_start_here();
void statement_list();
bool statement_list_may_start_here();
void expression_stat(Expression*);
void send_stmt(Expression*, bool may_be_composite_lit);
void inc_dec_stat(Expression*);
void assignment(Expression*, bool may_be_composite_lit, Range_clause*);
void tuple_assignment(Expression_list*, bool may_be_composite_lit,
Range_clause*);
void send();
void go_or_defer_stat();
void return_stat();
void if_stat();
void switch_stat(Label*);
Statement* expr_switch_body(Label*, Expression*, Location);
void expr_case_clause(Case_clauses*, bool* saw_default);
Expression_list* expr_switch_case(bool*);
Statement* type_switch_body(Label*, const Type_switch&, Location);
void type_case_clause(const std::string&, Expression*, Type_case_clauses*,
bool* saw_default, std::vector<Named_object*>*);
void type_switch_case(std::vector<Type*>*, bool*);
void select_stat(Label*);
void comm_clause(Select_clauses*, bool* saw_default);
bool comm_case(bool*, Expression**, Expression**, Expression**,
std::string*, std::string*, bool*);
bool send_or_recv_stmt(bool*, Expression**, Expression**, Expression**,
std::string*, std::string*);
void for_stat(Label*);
void for_clause(Expression**, Block**);
void range_clause_decl(const Typed_identifier_list*, Range_clause*);
void range_clause_expr(const Expression_list*, Range_clause*);
void push_break_statement(Statement*, Label*);
void push_continue_statement(Statement*, Label*);
void pop_break_statement();
void pop_continue_statement();
Statement* find_bc_statement(const Bc_stack*, const std::string&);
void break_stat();
void continue_stat();
void goto_stat();
void package_clause();
void import_decl();
void import_spec(void*, unsigned int pragmas);
bool
skip_past_error(Operator op);
Expression*
verify_not_sink(Expression*);
Statement*
find_bc_statement(const Bc_stack*, const std::string&) const;
void
mark_var_used(Named_object*);
Lex* lex_;
Token token_;
Token unget_token_;
bool unget_token_valid_;
bool is_erroneous_function_;
Gogo* gogo_;
Bc_stack* break_stack_;
Bc_stack* continue_stack_;
Enclosing_vars enclosing_vars_;
};
#endif 
