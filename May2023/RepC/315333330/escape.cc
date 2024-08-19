#include "go-system.h"
#include <limits>
#include <stack>
#include <sstream>
#include "gogo.h"
#include "types.h"
#include "expressions.h"
#include "statements.h"
#include "escape.h"
#include "lex.h"
#include "ast-dump.h"
#include "go-optimize.h"
#include "go-diagnostics.h"
#include "go-sha1.h"
Type*
Node::type() const
{
if (this->object() != NULL
&& this->object()->is_variable())
return this->object()->var_value()->type();
else if (this->object() != NULL
&& this->object()->is_function())
return this->object()->func_value()->type();
else if (this->expr() != NULL)
return this->expr()->type();
else if (this->is_indirect())
{
if (this->child()->type()->deref()->is_void_type())
return this->child()->type();
else
return this->child()->type()->deref();
}
else if (this->statement() != NULL
&& this->statement()->temporary_statement() != NULL)
return this->statement()->temporary_statement()->type();
else
return NULL;
}
Location
Node::location() const
{
if (this->object() != NULL && !this->object()->is_sink())
return this->object()->location();
else if (this->expr() != NULL)
return this->expr()->location();
else if (this->statement() != NULL)
return this->statement()->location();
else if (this->is_indirect())
return this->child()->location();
else
return Linemap::unknown_location();
}
Location
Node::definition_location() const
{
if (this->object() != NULL && !this->object()->is_sink())
{
Named_object* no = this->object();
if (no->is_variable() || no->is_result_variable())
return no->location();
}
else if (this->expr() != NULL)
{
Var_expression* ve = this->expr()->var_expression();
if (ve != NULL)
{
Named_object* no = ve->named_object();
if (no->is_variable() || no->is_result_variable())
return no->location();
}
Enclosed_var_expression* eve = this->expr()->enclosed_var_expression();
if (eve != NULL)
{
Named_object* no = eve->variable();
if (no->is_variable() || no->is_result_variable())
return no->location();
}
}
return this->location();
}
std::string
strip_packed_prefix(Gogo* gogo, const std::string& s)
{
std::string packed_prefix = "." + gogo->pkgpath() + ".";
std::string fmt = s;
for (size_t pos = fmt.find(packed_prefix);
pos != std::string::npos;
pos = fmt.find(packed_prefix))
fmt.erase(pos, packed_prefix.length());
return fmt;
}
std::string
Node::ast_format(Gogo* gogo) const
{
std::ostringstream ss;
if (this->is_sink())
ss << ".sink";
else if (this->object() != NULL)
{
Named_object* no = this->object();
if (no->is_function() && no->func_value()->enclosing() != NULL)
return "func literal";
ss << no->message_name();
}
else if (this->expr() != NULL)
{
Expression* e = this->expr();
bool is_call = e->call_expression() != NULL;
if (is_call)
e->call_expression()->fn();
Func_expression* fe = e->func_expression();;
bool is_closure = fe != NULL && fe->closure() != NULL;
if (is_closure)
{
if (is_call)
return "(func literal)()";
return "func literal";
}
Ast_dump_context::dump_to_stream(this->expr(), &ss);
}
else if (this->statement() != NULL)
{
Statement* s = this->statement();
Goto_unnamed_statement* unnamed = s->goto_unnamed_statement();
if (unnamed != NULL)
{
Statement* derived = unnamed->unnamed_label()->derived_from();
if (derived != NULL)
{
switch (derived->classification())
{
case Statement::STATEMENT_FOR:
case Statement::STATEMENT_FOR_RANGE:
return "for loop";
break;
case Statement::STATEMENT_SWITCH:
return "switch";
break;
case Statement::STATEMENT_TYPE_SWITCH:
return "type switch";
break;
default:
break;
}
}
}
Temporary_statement* tmp = s->temporary_statement();
if (tmp != NULL)
{
ss << "tmp." << (uintptr_t) tmp;
if (tmp->init() != NULL)
{
ss << " [ = ";
Ast_dump_context::dump_to_stream(tmp->init(), &ss);
ss << " ]";
}
}
else
Ast_dump_context::dump_to_stream(s, &ss);
}
else if (this->is_indirect())
return "*(" + this->child()->ast_format(gogo) + ")";
std::string s = strip_packed_prefix(gogo, ss.str());
return s.substr(0, s.find_last_not_of(' ') + 1);
}
std::string
Node::details()
{
std::stringstream details;
if (!this->is_sink())
details << " l(" << Linemap::location_to_line(this->location()) << ")";
bool is_varargs = false;
bool is_address_taken = false;
bool is_in_heap = false;
bool is_assigned = false;
std::string class_name;
Expression* e = this->expr();
Named_object* node_object = NULL;
if (this->object() != NULL)
node_object = this->object();
else if (e != NULL && e->var_expression() != NULL)
node_object = e->var_expression()->named_object();
if (node_object)
{
if (node_object->is_variable())
{
Variable* var = node_object->var_value();
is_varargs = var->is_varargs_parameter();
is_address_taken = (var->is_address_taken()
|| var->is_non_escaping_address_taken());
is_in_heap = var->is_in_heap();
is_assigned = var->init() != NULL;
if (var->is_global())
class_name = "PEXTERN";
else if (var->is_parameter())
class_name = "PPARAM";
else if (var->is_closure())
class_name = "PPARAMREF";
else
class_name = "PAUTO";
}
else if (node_object->is_result_variable())
class_name = "PPARAMOUT";
else if (node_object->is_function()
|| node_object->is_function_declaration())
class_name = "PFUNC";
}
else if (e != NULL && e->enclosed_var_expression() != NULL)
{
Named_object* enclosed = e->enclosed_var_expression()->variable();
if (enclosed->is_variable())
{
Variable* var = enclosed->var_value();
is_address_taken = (var->is_address_taken()
|| var->is_non_escaping_address_taken());
}
else
{
Result_variable* var = enclosed->result_var_value();
is_address_taken = (var->is_address_taken()
|| var->is_non_escaping_address_taken());
}
class_name = "PPARAMREF";
}
if (!class_name.empty())
{
details << " class(" << class_name;
if (is_in_heap)
details << ",heap";
details << ")";
}
switch ((this->encoding() & ESCAPE_MASK))
{
case Node::ESCAPE_UNKNOWN:
break;
case Node::ESCAPE_HEAP:
details << " esc(h)";
break;
case Node::ESCAPE_NONE:
details << " esc(no)";
break;
case Node::ESCAPE_NEVER:
details << " esc(N)";
break;
default:
details << " esc(" << this->encoding() << ")";
break;
}
if (this->state_ != NULL && this->state_->loop_depth != 0)
details << " ld(" << this->state_->loop_depth << ")";
if (is_varargs)
details << " isddd(1)";
if (is_address_taken)
details << " addrtaken";
if (is_assigned)
details << " assigned";
return details.str();
}
std::string
Node::op_format() const
{
std::stringstream op;
Ast_dump_context adc(&op, false);
if (this->expr() != NULL)
{
Expression* e = this->expr();
switch (e->classification())
{
case Expression::EXPRESSION_UNARY:
adc.dump_operator(e->unary_expression()->op());
break;
case Expression::EXPRESSION_BINARY:
adc.dump_operator(e->binary_expression()->op());
break;
case Expression::EXPRESSION_CALL:
op << "function call";
break;
case Expression::EXPRESSION_FUNC_REFERENCE:
if (e->func_expression()->is_runtime_function())
{
switch (e->func_expression()->runtime_code())
{
case Runtime::GOPANIC:
op << "panic";
break;
case Runtime::GROWSLICE:
op << "append";
break;
case Runtime::SLICECOPY:
case Runtime::SLICESTRINGCOPY:
case Runtime::TYPEDSLICECOPY:
op << "copy";
break;
case Runtime::MAKECHAN:
case Runtime::MAKECHAN64:
case Runtime::MAKEMAP:
case Runtime::MAKESLICE:
case Runtime::MAKESLICE64:
op << "make";
break;
case Runtime::DEFERPROC:
op << "defer";
break;
case Runtime::GORECOVER:
op << "recover";
break;
case Runtime::CLOSE:
op << "close";
break;
default:
break;
}
}
break;
case Expression::EXPRESSION_ALLOCATION:
op << "new";
break;
case Expression::EXPRESSION_RECEIVE:
op << "<-";
break;
default:
break;
}
}
if (this->statement() != NULL)
{
switch (this->statement()->classification())
{
case Statement::STATEMENT_DEFER:
op << "defer";
break;
case Statement::STATEMENT_RETURN:
op << "return";
break;
default:
break;
}
}
if (this->is_indirect())
op << "*";
return op.str();
}
Node::Escape_state*
Node::state(Escape_context* context, Named_object* fn)
{
if (this->state_ == NULL)
{
if (this->expr() != NULL && this->expr()->var_expression() != NULL)
{
Named_object* var_no = this->expr()->var_expression()->named_object();
Node* var_node = Node::make_node(var_no);
this->state_ = var_node->state(context, fn);
}
else
{
this->state_ = new Node::Escape_state;
if (fn == NULL)
fn = context->current_function();
this->state_->fn = fn;
}
}
go_assert(this->state_ != NULL);
return this->state_;
}
Node::~Node()
{
if (this->state_ != NULL)
{
if (this->expr() == NULL || this->expr()->var_expression() == NULL)
delete this->state_;
}
}
int
Node::encoding()
{
if (this->expr() != NULL
&& this->expr()->var_expression() != NULL)
{
Named_object* no = this->expr()->var_expression()->named_object();
int enc = Node::make_node(no)->encoding();
this->encoding_ = enc;
}
return this->encoding_;
}
void
Node::set_encoding(int enc)
{
this->encoding_ = enc;
if (this->expr() != NULL)
{
if (this->expr()->var_expression() != NULL)
{
Named_object* no = this->expr()->var_expression()->named_object();
Node::make_node(no)->set_encoding(enc);
}
else if (this->expr()->func_expression() != NULL)
{
Expression* closure = this->expr()->func_expression()->closure();
if (closure != NULL)
Node::make_node(closure)->set_encoding(enc);
}
}
}
bool
Node::is_big(Escape_context* context) const
{
Type* t = this->type();
if (t == NULL
|| t->is_call_multiple_result_type()
|| t->is_sink_type()
|| t->is_void_type()
|| t->is_abstract())
return false;
int64_t size;
bool ok = t->backend_type_size(context->gogo(), &size);
bool big = ok && (size < 0 || size > 10 * 1024 * 1024);
if (this->expr() != NULL)
{
if (this->expr()->allocation_expression() != NULL)
{
ok = t->deref()->backend_type_size(context->gogo(), &size);
big = big || size <= 0 || size >= (1 << 16);
}
else if (this->expr()->call_expression() != NULL)
{
Call_expression* call = this->expr()->call_expression();
Func_expression* fn = call->fn()->func_expression();
if (fn != NULL
&& fn->is_runtime_function()
&& (fn->runtime_code() == Runtime::MAKESLICE
|| fn->runtime_code() == Runtime::MAKESLICE64))
{
Expression_list::iterator p = call->args()->begin();
++p;
Expression* e = *p;
if (e->temporary_reference_expression() != NULL)
{
Temporary_reference_expression* tre = e->temporary_reference_expression();
if (tre->statement() != NULL && tre->statement()->init() != NULL)
e = tre->statement()->init();
}
Numeric_constant nc;
unsigned long v;
if (e->numeric_constant_value(&nc)
&& nc.to_unsigned_long(&v) == Numeric_constant::NC_UL_VALID)
big = big || v >= (1 << 16);
}
}
}
return big;
}
bool
Node::is_sink() const
{
if (this->object() != NULL
&& this->object()->is_sink())
return true;
else if (this->expr() != NULL
&& this->expr()->is_sink_expression())
return true;
return false;
}
std::map<Named_object*, Node*> Node::objects;
std::map<Expression*, Node*> Node::expressions;
std::map<Statement*, Node*> Node::statements;
std::vector<Node*> Node::indirects;
Node*
Node::make_node(Named_object* no)
{
if (Node::objects.find(no) != Node::objects.end())
return Node::objects[no];
Node* n = new Node(no);
std::pair<Named_object*, Node*> val(no, n);
Node::objects.insert(val);
return n;
}
Node*
Node::make_node(Expression* e)
{
if (Node::expressions.find(e) != Node::expressions.end())
return Node::expressions[e];
Node* n = new Node(e);
std::pair<Expression*, Node*> val(e, n);
Node::expressions.insert(val);
return n;
}
Node*
Node::make_node(Statement* s)
{
if (Node::statements.find(s) != Node::statements.end())
return Node::statements[s];
Node* n = new Node(s);
std::pair<Statement*, Node*> val(s, n);
Node::statements.insert(val);
return n;
}
Node*
Node::make_indirect_node(Node* child)
{
Node* n = new Node(child);
Node::indirects.push_back(n);
return n;
}
int
Node::max_encoding(int e, int etype)
{
if ((e & ESCAPE_MASK) > etype)
return e;
if (etype == Node::ESCAPE_NONE || etype == Node::ESCAPE_RETURN)
return (e & ~ESCAPE_MASK) | etype;
return etype;
}
int
Node::note_inout_flows(int e, int index, Level level)
{
if (level.value() <= 0 && level.suffix_value() > 0)
return Node::max_encoding(e|ESCAPE_CONTENT_ESCAPES, Node::ESCAPE_NONE);
if (level.value() < 0)
return Node::ESCAPE_HEAP;
if (level.value() >  ESCAPE_MAX_ENCODED_LEVEL)
level = Level::From(ESCAPE_MAX_ENCODED_LEVEL);
int encoded = level.value() + 1;
int shift = ESCAPE_BITS_PER_OUTPUT_IN_TAG * index + ESCAPE_RETURN_BITS;
int old = (e >> shift) & ESCAPE_BITS_MASK_FOR_TAG;
if (old == 0
|| (encoded != 0 && encoded < old))
old = encoded;
int encoded_flow = old << shift;
if (((encoded_flow >> shift) & ESCAPE_BITS_MASK_FOR_TAG) != old)
{
return Node::ESCAPE_HEAP;
}
return (e & ~(ESCAPE_BITS_MASK_FOR_TAG << shift)) | encoded_flow;
}
Escape_context::Escape_context(Gogo* gogo, bool recursive)
: gogo_(gogo), current_function_(NULL), recursive_(recursive),
sink_(Node::make_node(Named_object::make_sink())), loop_depth_(0),
flood_id_(0), pdepth_(0)
{
Node::Escape_state* state = this->sink_->state(this, NULL);
state->loop_depth = -1;
}
std::string
debug_function_name(Named_object* fn)
{
if (fn == NULL)
return "<S>";
if (!fn->is_function())
return Gogo::unpack_hidden_name(fn->name());
std::string fnname = Gogo::unpack_hidden_name(fn->name());
if (fn->func_value()->is_method())
{
Type* rt = fn->func_value()->type()->receiver()->type();
switch (rt->classification())
{
case Type::TYPE_NAMED:
fnname = rt->named_type()->name() + "." + fnname;
break;
case Type::TYPE_POINTER:
{
Named_type* nt = rt->points_to()->named_type();
if (nt != NULL)
fnname = "(*" + nt->name() + ")." + fnname;
break;
}
default:
break;
}
}
return fnname;
}
std::string
Escape_context::current_function_name() const
{
return debug_function_name(this->current_function_);
}
void
Escape_context::init_retvals(Node* n, Function_type* fntype)
{
if (fntype == NULL || fntype->results() == NULL)
return;
Node::Escape_state* state = n->state(this, NULL);
state->retvals.clear();
Location loc = n->location();
int i = 0;
char buf[50];
for (Typed_identifier_list::const_iterator p = fntype->results()->begin();
p != fntype->results()->end();
++p, ++i)
{
snprintf(buf, sizeof buf, ".out%d", i);
Variable* dummy_var = new Variable(p->type(), NULL, false, false,
false, loc);
dummy_var->set_is_used();
Named_object* dummy_no =
Named_object::make_variable(buf, NULL, dummy_var);
Node* dummy_node = Node::make_node(dummy_no);
Node::Escape_state* dummy_node_state = dummy_node->state(this, NULL);
dummy_node_state->loop_depth = this->loop_depth_;
state->retvals.push_back(dummy_node);
}
}
Node*
Escape_context::add_dereference(Node* n)
{
Expression* e = n->expr();
Location loc = n->location();
Node* ind;
if (e != NULL
&& e->type()->points_to() != NULL
&& !e->type()->points_to()->is_void_type())
{
Expression* deref_expr = Expression::make_unary(OPERATOR_MULT, e, loc);
ind = Node::make_node(deref_expr);
}
else
ind = Node::make_indirect_node(n);
Node::Escape_state* state = ind->state(this, NULL);
state->loop_depth = n->state(this, NULL)->loop_depth;
return ind;
}
void
Escape_context::track(Node* n)
{
n->set_encoding(Node::ESCAPE_NONE);
Node::Escape_state* state = n->state(this, NULL);
state->loop_depth = this->loop_depth_;
this->noesc_.push_back(n);
}
std::string
Escape_note::make_tag(int encoding)
{
char buf[50];
snprintf(buf, sizeof buf, "esc:0x%x", encoding);
return buf;
}
int
Escape_note::parse_tag(std::string* tag)
{
if (tag == NULL || tag->substr(0, 4) != "esc:")
return Node::ESCAPE_UNKNOWN;
int encoding = (int)strtol(tag->substr(4).c_str(), NULL, 0);
if (encoding == 0)
return Node::ESCAPE_UNKNOWN;
return encoding;
}
Go_optimize optimize_allocation_flag("allocs", true);
static bool
escape_hash_match(std::string suffix, std::string name)
{
if (suffix.empty())
return true;
if (suffix.at(0) == '-')
return !escape_hash_match(suffix.substr(1), name);
const char* p = name.c_str();
Go_sha1_helper* sha1_helper = go_create_sha1_helper();
sha1_helper->process_bytes(p, strlen(p));
std::string s = sha1_helper->finish();
delete sha1_helper;
int j = suffix.size() - 1;
for (int i = s.size() - 1; i >= 0; i--)
{
char c = s.at(i);
for (int k = 0; k < 8; k++, j--, c>>=1)
{
if (j < 0)
return true;
char bit = suffix.at(j) - '0';
if ((c&1) != bit)
return false;
}
}
return false;
}
void
Gogo::analyze_escape()
{
if (saw_errors())
return;
if (!optimize_allocation_flag.is_enabled()
&& !this->compiling_runtime())
return;
this->discover_analysis_sets();
if (!this->debug_escape_hash().empty())
std::cerr << "debug-escape-hash " << this->debug_escape_hash() << "\n";
for (std::vector<Analysis_set>::iterator p = this->analysis_sets_.begin();
p != this->analysis_sets_.end();
++p)
{
std::vector<Named_object*> stack = p->first;
if (!this->debug_escape_hash().empty())
{
bool match = false;
for (std::vector<Named_object*>::const_iterator fn = stack.begin();
fn != stack.end();
++fn)
match = match || escape_hash_match(this->debug_escape_hash(), (*fn)->message_name());
if (!match)
{
for (std::vector<Named_object*>::iterator fn = stack.begin();
fn != stack.end();
++fn)
if ((*fn)->is_function())
{
Function_type* fntype = (*fn)->func_value()->type();
fntype->set_is_tagged();
std::cerr << "debug-escape-hash disables " << debug_function_name(*fn) << "\n";
}
continue;
}
for (std::vector<Named_object*>::const_iterator fn = stack.begin();
fn != stack.end();
++fn)
if ((*fn)->is_function())
std::cerr << "debug-escape-hash triggers " << debug_function_name(*fn) << "\n";
}
Escape_context* context = new Escape_context(this, p->second);
for (std::vector<Named_object*>::reverse_iterator fn = stack.rbegin();
fn != stack.rend();
++fn)
{
context->set_current_function(*fn);
this->assign_connectivity(context, *fn);
}
std::set<Node*> dsts = context->dsts();
Unordered_map(Node*, int) escapes;
for (std::set<Node*>::iterator n = dsts.begin();
n != dsts.end();
++n)
{
escapes[*n] = (*n)->encoding();
this->propagate_escape(context, *n);
}
for (;;)
{
bool done = true;
for (std::set<Node*>::iterator n = dsts.begin();
n != dsts.end();
++n)
{
if ((*n)->object() == NULL
&& ((*n)->expr() == NULL
|| ((*n)->expr()->var_expression() == NULL
&& (*n)->expr()->enclosed_var_expression() == NULL
&& (*n)->expr()->func_expression() == NULL)))
continue;
if (escapes[*n] != (*n)->encoding())
{
done = false;
if (this->debug_escape_level() > 2)
go_inform((*n)->location(), "Reflooding %s %s",
debug_function_name((*n)->state(context, NULL)->fn).c_str(),
(*n)->ast_format(this).c_str());
escapes[*n] = (*n)->encoding();
this->propagate_escape(context, *n);
}
}
if (done)
break;
}
for (std::vector<Named_object*>::iterator fn = stack.begin();
fn != stack.end();
++fn)
this->tag_function(context, *fn);
if (this->debug_escape_level() != 0)
{
std::vector<Node*> noesc = context->non_escaping_nodes();
for (std::vector<Node*>::const_iterator n = noesc.begin();
n != noesc.end();
++n)
{
Node::Escape_state* state = (*n)->state(context, NULL);
if ((*n)->encoding() == Node::ESCAPE_NONE)
go_inform((*n)->location(), "%s %s does not escape",
strip_packed_prefix(this, debug_function_name(state->fn)).c_str(),
(*n)->ast_format(this).c_str());
}
}
delete context;
}
}
class Escape_analysis_discover : public Traverse
{
public:
Escape_analysis_discover(Gogo* gogo)
: Traverse(traverse_functions | traverse_func_declarations),
gogo_(gogo), component_ids_()
{ }
int
function(Named_object*);
int
function_declaration(Named_object*);
int
visit(Named_object*);
int
visit_code(Named_object*, int);
private:
static int id;
typedef Unordered_map(Named_object*, int) Component_ids;
Gogo* gogo_;
Component_ids component_ids_;
std::stack<Named_object*> stack_;
};
int Escape_analysis_discover::id = 0;
int
Escape_analysis_discover::function(Named_object* fn)
{
this->visit(fn);
return TRAVERSE_CONTINUE;
}
int
Escape_analysis_discover::function_declaration(Named_object* fn)
{
this->visit(fn);
return TRAVERSE_CONTINUE;
}
int
Escape_analysis_discover::visit(Named_object* fn)
{
Component_ids::const_iterator p = this->component_ids_.find(fn);
if (p != this->component_ids_.end())
return p->second;
this->id++;
int id = this->id;
this->component_ids_[fn] = id;
this->id++;
int min = this->id;
this->stack_.push(fn);
min = this->visit_code(fn, min);
if ((min == id || min == id + 1)
&& ((fn->is_function() && fn->func_value()->enclosing() == NULL)
|| fn->is_function_declaration()))
{
bool recursive = min == id;
std::vector<Named_object*> group;
for (; !this->stack_.empty(); this->stack_.pop())
{
Named_object* n = this->stack_.top();
if (n == fn)
{
this->stack_.pop();
break;
}
group.push_back(n);
this->component_ids_[n] = std::numeric_limits<int>::max();
}
group.push_back(fn);
this->component_ids_[fn] = std::numeric_limits<int>::max();
std::reverse(group.begin(), group.end());
this->gogo_->add_analysis_set(group, recursive);
}
return min;
}
class Escape_discover_expr : public Traverse
{
public:
Escape_discover_expr(Escape_analysis_discover* ead, int min)
: Traverse(traverse_expressions),
ead_(ead), min_(min)
{ }
int
min()
{ return this->min_; }
int
expression(Expression** pexpr);
private:
Escape_analysis_discover* ead_;
int min_;
};
int
Escape_discover_expr::expression(Expression** pexpr)
{
Expression* e = *pexpr;
Named_object* fn = NULL;
if (e->call_expression() != NULL
&& e->call_expression()->fn()->func_expression() != NULL)
{
fn = e->call_expression()->fn()->func_expression()->named_object();
}
else if (e->func_expression() != NULL
&& e->func_expression()->closure() != NULL)
{
fn = e->func_expression()->named_object();
}
if (fn != NULL)
this->min_ = std::min(this->min_, this->ead_->visit(fn));
return TRAVERSE_CONTINUE;
}
int
Escape_analysis_discover::visit_code(Named_object* fn, int min)
{
if (!fn->is_function())
return min;
Escape_discover_expr ede(this, min);
fn->func_value()->traverse(&ede);
return ede.min();
}
void
Gogo::discover_analysis_sets()
{
Escape_analysis_discover ead(this);
this->traverse(&ead);
}
class Escape_analysis_loop : public Traverse
{
public:
Escape_analysis_loop()
: Traverse(traverse_statements)
{ }
int
statement(Block*, size_t*, Statement*);
};
int
Escape_analysis_loop::statement(Block*, size_t*, Statement* s)
{
if (s->label_statement() != NULL)
s->label_statement()->label()->set_nonlooping();
else if (s->goto_statement() != NULL)
{
if (s->goto_statement()->label()->nonlooping())
s->goto_statement()->label()->set_looping();
}
return TRAVERSE_CONTINUE;
}
class Escape_analysis_assign : public Traverse
{
public:
Escape_analysis_assign(Escape_context* context, Named_object* fn)
: Traverse(traverse_statements
| traverse_expressions),
context_(context), fn_(fn)
{ }
int
statement(Block*, size_t*, Statement*);
int
expression(Expression**);
void
call(Call_expression* call);
void
assign(Node* dst, Node* src);
void
assign_deref(Node* dst, Node* src);
int
assign_from_note(std::string* note, const std::vector<Node*>& dsts,
Node* src);
void
flows(Node* dst, Node* src);
private:
Escape_context* context_;
Named_object* fn_;
};
static bool
is_self_assignment(Expression* lhs, Expression* rhs)
{
Unary_expression* lue =
(lhs->field_reference_expression() != NULL
? lhs->field_reference_expression()->expr()->unary_expression()
: lhs->unary_expression());
Var_expression* lve =
(lue != NULL && lue->op() == OPERATOR_MULT ? lue->operand()->var_expression() : NULL);
Array_index_expression* raie = rhs->array_index_expression();
String_index_expression* rsie = rhs->string_index_expression();
Expression* rarray =
(raie != NULL && raie->end() != NULL && raie->array()->type()->is_slice_type()
? raie->array()
: (rsie != NULL && rsie->type()->is_string_type() ? rsie->string() : NULL));
Unary_expression* rue =
(rarray != NULL && rarray->field_reference_expression() != NULL
? rarray->field_reference_expression()->expr()->unary_expression()
: (rarray != NULL ? rarray->unary_expression() : NULL));
Var_expression* rve =
(rue != NULL && rue->op() == OPERATOR_MULT ? rue->operand()->var_expression() : NULL);
return lve != NULL && rve != NULL
&& lve->named_object() == rve->named_object();
}
int
Escape_analysis_assign::statement(Block*, size_t*, Statement* s)
{
bool is_for_statement = (s->is_block_statement()
&& s->block_statement()->is_lowered_for_statement());
if (is_for_statement)
this->context_->increase_loop_depth();
s->traverse_contents(this);
if (is_for_statement)
this->context_->decrease_loop_depth();
Gogo* gogo = this->context_->gogo();
int debug_level = gogo->debug_escape_level();
if (debug_level > 1
&& s->unnamed_label_statement() == NULL
&& s->expression_statement() == NULL
&& !s->is_block_statement())
{
Node* n = Node::make_node(s);
std::string fn_name = this->context_->current_function_name();
go_inform(s->location(), "[%d] %s esc: %s",
this->context_->loop_depth(), fn_name.c_str(),
n->ast_format(gogo).c_str());
}
switch (s->classification())
{
case Statement::STATEMENT_VARIABLE_DECLARATION:
{
Named_object* var = s->variable_declaration_statement()->var();
Node* var_node = Node::make_node(var);
Node::Escape_state* state = var_node->state(this->context_, NULL);
state->loop_depth = this->context_->loop_depth();
if (var->is_variable()
&& var->var_value()->init() != NULL)
{
Node* init_node = Node::make_node(var->var_value()->init());
this->assign(var_node, init_node);
}
}
break;
case Statement::STATEMENT_TEMPORARY:
{
Expression* init = s->temporary_statement()->init();
if (init != NULL)
this->assign(Node::make_node(s), Node::make_node(init));
}
break;
case Statement::STATEMENT_LABEL:
{
Label_statement* label_stmt = s->label_statement();
if (label_stmt->label()->looping())
this->context_->increase_loop_depth();
if (debug_level > 1)
{
std::string label_type = (label_stmt->label()->looping()
? "looping"
: "nonlooping");
go_inform(s->location(), "%s %s label",
label_stmt->label()->name().c_str(),
label_type.c_str());
}
}
break;
case Statement::STATEMENT_SWITCH:
case Statement::STATEMENT_TYPE_SWITCH:
break;
case Statement::STATEMENT_ASSIGNMENT:
{
Assignment_statement* assn = s->assignment_statement();
Expression* lhs = assn->lhs();
Expression* rhs = assn->rhs();
Node* lhs_node = Node::make_node(lhs);
Node* rhs_node = Node::make_node(rhs);
if (is_self_assignment(lhs, rhs))
{
if (debug_level != 0)
go_inform(s->location(), "%s ignoring self-assignment to %s",
strip_packed_prefix(gogo, this->context_->current_function_name()).c_str(),
lhs_node->ast_format(gogo).c_str());
break;
}
this->assign(lhs_node, rhs_node);
}
break;
case Statement::STATEMENT_SEND:
{
Node* sent_node = Node::make_node(s->send_statement()->val());
this->assign(this->context_->sink(), sent_node);
}
break;
case Statement::STATEMENT_DEFER:
if (this->context_->loop_depth() == 1)
{
Node* n = Node::make_node(s);
n->set_encoding(Node::ESCAPE_NONE);
break;
}
case Statement::STATEMENT_GO:
{
Thunk_statement* thunk = s->thunk_statement();
if (thunk->call()->call_expression() == NULL)
break;
Call_expression* call = thunk->call()->call_expression();
Node* func_node = Node::make_node(call->fn());
this->assign(this->context_->sink(), func_node);
if (call->args() != NULL)
{
for (Expression_list::const_iterator p = call->args()->begin();
p != call->args()->end();
++p)
{
Node* arg_node = Node::make_node(*p);
this->assign(this->context_->sink(), arg_node);
}
}
}
break;
default:
break;
}
return TRAVERSE_SKIP_COMPONENTS;
}
static void
move_to_heap(Gogo* gogo, Expression *expr)
{
Named_object* no;
if (expr->var_expression() != NULL)
no = expr->var_expression()->named_object();
else if (expr->enclosed_var_expression() != NULL)
no = expr->enclosed_var_expression()->variable();
else
return;
if ((no->is_variable()
&& !no->var_value()->is_global())
|| no->is_result_variable())
{
Node* n = Node::make_node(expr);
if (gogo->debug_escape_level() != 0)
go_inform(n->definition_location(),
"moved to heap: %s",
n->ast_format(gogo).c_str());
if (gogo->compiling_runtime() && gogo->package_name() == "runtime")
go_error_at(expr->location(),
"%s escapes to heap, not allowed in runtime",
n->ast_format(gogo).c_str());
}
}
int
Escape_analysis_assign::expression(Expression** pexpr)
{
Gogo* gogo = this->context_->gogo();
int debug_level = gogo->debug_escape_level();
Node* n = Node::make_node(*pexpr);
if ((n->encoding() & ESCAPE_MASK) != int(Node::ESCAPE_HEAP)
&& n->is_big(this->context_))
{
if (debug_level > 1)
go_inform((*pexpr)->location(), "%s too large for stack",
n->ast_format(gogo).c_str());
move_to_heap(gogo, *pexpr);
n->set_encoding(Node::ESCAPE_HEAP);
(*pexpr)->address_taken(true);
this->assign(this->context_->sink(), n);
}
if ((*pexpr)->func_expression() == NULL)
(*pexpr)->traverse_subexpressions(this);
if (debug_level > 1)
{
Node* n = Node::make_node(*pexpr);
std::string fn_name = this->context_->current_function_name();
go_inform((*pexpr)->location(), "[%d] %s esc: %s",
this->context_->loop_depth(), fn_name.c_str(),
n->ast_format(gogo).c_str());
}
switch ((*pexpr)->classification())
{
case Expression::EXPRESSION_CALL:
{
Call_expression* call = (*pexpr)->call_expression();
if (call->is_builtin())
{
Builtin_call_expression* bce = call->builtin_call_expression();
switch (bce->code())
{
case Builtin_call_expression::BUILTIN_PANIC:
{
Node* panic_arg = Node::make_node(call->args()->front());
this->assign(this->context_->sink(), panic_arg);
}
break;
case Builtin_call_expression::BUILTIN_APPEND:
{
if (call->is_varargs())
{
Node* appended = Node::make_node(call->args()->back());
this->assign_deref(this->context_->sink(), appended);
if (debug_level > 2)
go_inform((*pexpr)->location(),
"special treatment of append(slice1, slice2...)");
}
else
{
for (Expression_list::const_iterator pa =
call->args()->begin() + 1;
pa != call->args()->end();
++pa)
{
Node* arg = Node::make_node(*pa);
this->assign(this->context_->sink(), arg);
}
}
Node* appendee = Node::make_node(call->args()->front());
this->assign_deref(this->context_->sink(), appendee);
}
break;
case Builtin_call_expression::BUILTIN_COPY:
{
Node* copied = Node::make_node(call->args()->back());
this->assign_deref(this->context_->sink(), copied);
}
break;
default:
break;
}
break;
}
Func_expression* fe = call->fn()->func_expression();
if (fe != NULL && fe->is_runtime_function())
{
switch (fe->runtime_code())
{
case Runtime::MAKECHAN:
case Runtime::MAKECHAN64:
case Runtime::MAKEMAP:
case Runtime::MAKESLICE:
case Runtime::MAKESLICE64:
this->context_->track(n);
break;
case Runtime::MAPASSIGN:
{
Node* key_node = Node::make_node(call->args()->back());
this->assign_deref(this->context_->sink(), key_node);
}
break;
case Runtime::SELECTSEND:
{
Node* arg_node = Node::make_node(call->args()->back());
this->assign_deref(this->context_->sink(), arg_node);
}
break;
case Runtime::IFACEE2T2:
case Runtime::IFACEI2T2:
{
Node* src_node = Node::make_node(call->args()->at(1));
Node* dst_node;
Expression* arg2 = call->args()->at(2);
Unary_expression* ue =
(arg2->conversion_expression() != NULL
? arg2->conversion_expression()->expr()->unary_expression()
: arg2->unary_expression());
if (ue != NULL && ue->op() == OPERATOR_AND)
{
if (!ue->operand()->type()->has_pointer())
break;
dst_node = Node::make_node(ue->operand());
}
else
dst_node = this->context_->add_dereference(Node::make_node(arg2));
this->assign(dst_node, src_node);
}
break;
default:
break;
}
}
else
this->call(call);
}
break;
case Expression::EXPRESSION_ALLOCATION:
this->context_->track(n);
break;
case Expression::EXPRESSION_STRING_CONCAT:
this->context_->track(n);
break;
case Expression::EXPRESSION_CONVERSION:
{
Type_conversion_expression* tce = (*pexpr)->conversion_expression();
Type* ft = tce->expr()->type();
Type* tt = tce->type();
if ((ft->is_string_type() && tt->is_slice_type())
|| (ft->is_slice_type() && tt->is_string_type())
|| (ft->integer_type() != NULL && tt->is_string_type()))
{
this->context_->track(n);
break;
}
Node* tce_node = Node::make_node(tce);
Node* converted = Node::make_node(tce->expr());
this->context_->track(tce_node);
this->assign(tce_node, converted);
}
break;
case Expression::EXPRESSION_FIXED_ARRAY_CONSTRUCTION:
case Expression::EXPRESSION_SLICE_CONSTRUCTION:
{
Node* array_node = Node::make_node(*pexpr);
if ((*pexpr)->slice_literal() != NULL)
this->context_->track(array_node);
Expression_list* vals = ((*pexpr)->slice_literal() != NULL
? (*pexpr)->slice_literal()->vals()
: (*pexpr)->array_literal()->vals());
if (vals != NULL)
{
for (Expression_list::const_iterator p = vals->begin();
p != vals->end();
++p)
if ((*p) != NULL)
this->assign(array_node, Node::make_node(*p));
}
}
break;
case Expression::EXPRESSION_STRUCT_CONSTRUCTION:
{
Node* struct_node = Node::make_node(*pexpr);
Expression_list* vals = (*pexpr)->struct_literal()->vals();
if (vals != NULL)
{
for (Expression_list::const_iterator p = vals->begin();
p != vals->end();
++p)
{
if ((*p) != NULL)
this->assign(struct_node, Node::make_node(*p));
}
}
}
break;
case Expression::EXPRESSION_HEAP:
{
Node* pointer_node = Node::make_node(*pexpr);
Node* lit_node = Node::make_node((*pexpr)->heap_expression()->expr());
this->context_->track(pointer_node);
this->assign(pointer_node, lit_node);
}
break;
case Expression::EXPRESSION_BOUND_METHOD:
{
Node* bound_node = Node::make_node(*pexpr);
this->context_->track(bound_node);
Expression* obj = (*pexpr)->bound_method_expression()->first_argument();
Node* obj_node = Node::make_node(obj);
this->assign(this->context_->sink(), obj_node);
}
break;
case Expression::EXPRESSION_MAP_CONSTRUCTION:
{
Map_construction_expression* mce = (*pexpr)->map_literal();
Node* map_node = Node::make_node(mce);
this->context_->track(map_node);
if (mce->vals() != NULL)
{
for (Expression_list::const_iterator p = mce->vals()->begin();
p != mce->vals()->end();
++p)
{
if ((*p) != NULL)
this->assign(this->context_->sink(), Node::make_node(*p));
}
}
}
break;
case Expression::EXPRESSION_FUNC_REFERENCE:
{
Func_expression* fe = (*pexpr)->func_expression();
if (fe->closure() != NULL)
{
Node* closure_node = Node::make_node(fe);
this->context_->track(closure_node);
Heap_expression* he = fe->closure()->heap_expression();
Struct_construction_expression* sce = he->expr()->struct_literal();
Expression_list::const_iterator p = sce->vals()->begin();
++p;
for (; p != sce->vals()->end(); ++p)
{
Node* enclosed_node = Node::make_node(*p);
this->context_->track(enclosed_node);
this->assign(closure_node, enclosed_node);
}
}
}
break;
case Expression::EXPRESSION_UNARY:
{
if ((*pexpr)->unary_expression()->op() != OPERATOR_AND)
break;
Node* addr_node = Node::make_node(*pexpr);
this->context_->track(addr_node);
Expression* operand = (*pexpr)->unary_expression()->operand();
Named_object* var = NULL;
if (operand->var_expression() != NULL)
var = operand->var_expression()->named_object();
else if (operand->enclosed_var_expression() != NULL)
var = operand->enclosed_var_expression()->variable();
if (var == NULL)
break;
if (var->is_variable()
&& !var->var_value()->is_parameter())
{
Node::Escape_state* addr_state =
addr_node->state(this->context_, NULL);
Node* operand_node = Node::make_node(operand);
Node::Escape_state* operand_state =
operand_node->state(this->context_, NULL);
if (operand_state->loop_depth != 0)
addr_state->loop_depth = operand_state->loop_depth;
}
else if ((var->is_variable()
&& var->var_value()->is_parameter())
|| var->is_result_variable())
{
Node::Escape_state* addr_state =
addr_node->state(this->context_, NULL);
addr_state->loop_depth = 1;
}
}
break;
case Expression::EXPRESSION_ARRAY_INDEX:
{
Array_index_expression* aie = (*pexpr)->array_index_expression();
if (aie->end() != NULL && !aie->array()->type()->is_slice_type())
{
Expression* addr = Expression::make_unary(OPERATOR_AND, aie->array(),
aie->location());
Node* addr_node = Node::make_node(addr);
n->set_child(addr_node);
this->context_->track(addr_node);
}
}
break;
default:
break;
}
return TRAVERSE_SKIP_COMPONENTS;
}
void
Escape_analysis_assign::call(Call_expression* call)
{
Gogo* gogo = this->context_->gogo();
int debug_level = gogo->debug_escape_level();
Func_expression* fn = call->fn()->func_expression();
Function_type* fntype = call->get_function_type();
bool indirect = false;
if (fntype == NULL
|| (fntype->is_method()
&& fntype->receiver()->type()->interface_type() != NULL)
|| fn == NULL
|| (fn->named_object()->is_function()
&& fn->named_object()->func_value()->enclosing() != NULL))
indirect = true;
Node* call_node = Node::make_node(call);
std::vector<Node*> arg_nodes;
if (call->fn()->interface_field_reference_expression() != NULL)
{
Interface_field_reference_expression* ifre =
call->fn()->interface_field_reference_expression();
Node* field_node = Node::make_node(ifre->expr());
arg_nodes.push_back(field_node);
}
if (call->args() != NULL)
{
for (Expression_list::const_iterator p = call->args()->begin();
p != call->args()->end();
++p)
arg_nodes.push_back(Node::make_node(*p));
}
if (indirect)
{
for (std::vector<Node*>::iterator p = arg_nodes.begin();
p != arg_nodes.end();
++p)
{
if (debug_level > 2)
go_inform(call->location(),
"esccall:: indirect call <- %s, untracked",
(*p)->ast_format(gogo).c_str());
this->assign(this->context_->sink(), *p);
}
this->context_->init_retvals(call_node, fntype);
Node* fn_node = Node::make_node(call->fn());
std::vector<Node*> retvals = call_node->state(this->context_, NULL)->retvals;
for (std::vector<Node*>::const_iterator p = retvals.begin();
p != retvals.end();
++p)
this->assign_deref(*p, fn_node);
return;
}
if (fn != NULL
&& fn->named_object()->is_function()
&& !fntype->is_tagged())
{
if (debug_level > 2)
go_inform(call->location(), "esccall:: %s in recursive group",
call_node->ast_format(gogo).c_str());
Function* f = fn->named_object()->func_value();
const Bindings* callee_bindings = f->block()->bindings();
Function::Results* results = f->result_variables();
if (results != NULL)
{
Node::Escape_state* state = call_node->state(this->context_, NULL);
for (Function::Results::const_iterator p1 = results->begin();
p1 != results->end();
++p1)
{
Node* result_node = Node::make_node(*p1);
state->retvals.push_back(result_node);
}
}
std::vector<Node*>::iterator p = arg_nodes.begin();
if (fntype->is_method())
{
std::string rcvr_name = fntype->receiver()->name();
if (rcvr_name.empty() || Gogo::is_sink_name(rcvr_name)
|| !fntype->receiver()->type()->has_pointer())
;
else
{
Named_object* rcvr_no =
callee_bindings->lookup_local(fntype->receiver()->name());
go_assert(rcvr_no != NULL);
Node* rcvr_node = Node::make_node(rcvr_no);
if (fntype->receiver()->type()->points_to() == NULL
&& (*p)->expr()->type()->points_to() != NULL)
this->assign_deref(rcvr_node, *p);
else
this->assign(rcvr_node, *p);
}
++p;
}
const Typed_identifier_list* til = fntype->parameters();
if (til != NULL)
{
for (Typed_identifier_list::const_iterator p1 = til->begin();
p1 != til->end();
++p1, ++p)
{
if (p1->name().empty() || Gogo::is_sink_name(p1->name()))
continue;
Named_object* param_no =
callee_bindings->lookup_local(p1->name());
go_assert(param_no != NULL);
Expression* arg = (*p)->expr();
if (arg->var_expression() != NULL
&& arg->var_expression()->named_object() == param_no)
continue;
Node* param_node = Node::make_node(param_no);
this->assign(param_node, *p);
}
for (; p != arg_nodes.end(); ++p)
{
if (debug_level > 2)
go_inform(call->location(), "esccall:: ... <- %s, untracked",
(*p)->ast_format(gogo).c_str());
this->assign(this->context_->sink(), *p);
}
}
return;
}
if (debug_level > 2)
go_inform(call->location(), "esccall:: %s not recursive",
call_node->ast_format(gogo).c_str());
Node::Escape_state* call_state = call_node->state(this->context_, NULL);
if (!call_state->retvals.empty())
go_error_at(Linemap::unknown_location(),
"esc already decorated call %s",
call_node->ast_format(gogo).c_str());
this->context_->init_retvals(call_node, fntype);
std::vector<Node*>::iterator p = arg_nodes.begin();
if (fntype->is_method()
&& p != arg_nodes.end())
{
std::string* note = fntype->receiver()->note();
if (fntype->receiver()->type()->points_to() == NULL
&& (*p)->expr()->type()->points_to() != NULL)
this->assign_from_note(note, call_state->retvals,
this->context_->add_dereference(*p));
else
{
if (!Type::are_identical(fntype->receiver()->type(),
(*p)->expr()->type(), true, NULL))
{
this->context_->track(*p);
}
this->assign_from_note(note, call_state->retvals, *p);
}
p++;
}
const Typed_identifier_list* til = fntype->parameters();
if (til != NULL)
{
for (Typed_identifier_list::const_iterator pn = til->begin();
pn != til->end() && p != arg_nodes.end();
++pn, ++p)
{
if (!Type::are_identical(pn->type(), (*p)->expr()->type(),
true, NULL))
{
this->context_->track(*p);
}
Type* t = pn->type();
if (t != NULL
&& t->has_pointer())
{
std::string* note = pn->note();
int enc = this->assign_from_note(note, call_state->retvals, *p);
if (enc == Node::ESCAPE_NONE
&& !call->is_deferred()
&& !call->is_concurrent())
{
}
}
}
for (; p != arg_nodes.end(); ++p)
{
if (debug_level > 2)
go_inform(call->location(), "esccall:: ... <- %s, untracked",
(*p)->ast_format(gogo).c_str());
this->assign(this->context_->sink(), *p);
}
}
}
void
Escape_analysis_assign::assign(Node* dst, Node* src)
{
Gogo* gogo = this->context_->gogo();
int debug_level = gogo->debug_escape_level();
if (debug_level > 1)
go_inform(dst->location(), "[%d] %s escassign: %s(%s)[%s] = %s(%s)[%s]",
this->context_->loop_depth(),
strip_packed_prefix(gogo, this->context_->current_function_name()).c_str(),
dst->ast_format(gogo).c_str(), dst->details().c_str(),
dst->op_format().c_str(),
src->ast_format(gogo).c_str(), src->details().c_str(),
src->op_format().c_str());
if (dst->is_indirect())
dst = this->context_->sink();
else if (dst->expr() != NULL)
{
Expression* e = dst->expr();
switch (e->classification())
{
case Expression::EXPRESSION_VAR_REFERENCE:
{
Named_object* var = e->var_expression()->named_object();
if (var->is_variable() && var->var_value()->is_global())
dst = this->context_->sink();
}
break;
case Expression::EXPRESSION_FIELD_REFERENCE:
{
Expression* strct = e->field_reference_expression()->expr();
if (strct->heap_expression() != NULL)
{
dst = this->context_->sink();
break;
}
Node* struct_node = Node::make_node(strct);
this->assign(struct_node, src);
return;
}
case Expression::EXPRESSION_ARRAY_INDEX:
{
Array_index_expression* are = e->array_index_expression();
if (!are->array()->type()->is_slice_type())
{
Node* array_node = Node::make_node(are->array());
this->assign(array_node, src);
return;
}
dst = this->context_->sink();
}
break;
case Expression::EXPRESSION_UNARY:
if (e->unary_expression()->op() == OPERATOR_MULT)
dst = this->context_->sink();
break;
case Expression::EXPRESSION_MAP_INDEX:
{
Expression* index = e->map_index_expression()->index();
Node* index_node = Node::make_node(index);
this->assign(this->context_->sink(), index_node);
dst = this->context_->sink();
}
break;
case Expression::EXPRESSION_TEMPORARY_REFERENCE:
{
Statement* t = dst->expr()->temporary_reference_expression()->statement();
dst = Node::make_node(t);
}
break;
default:
break;
}
}
if (src->object() != NULL)
this->flows(dst, src);
else if (src->is_indirect())
this->flows(dst, src);
else if (src->expr() != NULL)
{
Expression* e = src->expr();
switch (e->classification())
{
case Expression::EXPRESSION_VAR_REFERENCE:
case Expression::EXPRESSION_ENCLOSED_VAR_REFERENCE:
case Expression::EXPRESSION_HEAP:
case Expression::EXPRESSION_FIXED_ARRAY_CONSTRUCTION:
case Expression::EXPRESSION_SLICE_CONSTRUCTION:
case Expression::EXPRESSION_MAP_CONSTRUCTION:
case Expression::EXPRESSION_STRUCT_CONSTRUCTION:
case Expression::EXPRESSION_ALLOCATION:
case Expression::EXPRESSION_BOUND_METHOD:
case Expression::EXPRESSION_STRING_CONCAT:
this->flows(dst, src);
break;
case Expression::EXPRESSION_UNSAFE_CONVERSION:
{
Expression* underlying = e->unsafe_conversion_expression()->expr();
Node* underlying_node = Node::make_node(underlying);
this->assign(dst, underlying_node);
}
break;
case Expression::EXPRESSION_CALL:
{
Call_expression* call = e->call_expression();
if (call->is_builtin())
{
Builtin_call_expression* bce = call->builtin_call_expression();
if (bce->code() == Builtin_call_expression::BUILTIN_APPEND)
{
Node* appendee = Node::make_node(call->args()->front());
this->assign(dst, appendee);
}
break;
}
Func_expression* fe = call->fn()->func_expression();
if (fe != NULL && fe->is_runtime_function())
{
switch (fe->runtime_code())
{
case Runtime::MAKECHAN:
case Runtime::MAKECHAN64:
case Runtime::MAKEMAP:
case Runtime::MAKESLICE:
case Runtime::MAKESLICE64:
this->flows(dst, src);
break;
default:
break;
}
break;
}
else if (fe != NULL
&& fe->named_object()->is_function()
&& fe->named_object()->func_value()->is_method()
&& (call->is_deferred()
|| call->is_concurrent()))
{
Node* rcvr_node = Node::make_node(call->args()->front());
this->assign(dst, rcvr_node);
break;
}
Node* call_node = Node::make_node(e);
Node::Escape_state* call_state = call_node->state(this->context_, NULL);
std::vector<Node*> retvals = call_state->retvals;
for (std::vector<Node*>::const_iterator p = retvals.begin();
p != retvals.end();
++p)
this->flows(dst, *p);
}
break;
case Expression::EXPRESSION_CALL_RESULT:
{
Call_result_expression* cre = e->call_result_expression();
Call_expression* call = cre->call()->call_expression();
if (call->is_builtin())
break;
if (call->fn()->func_expression() != NULL
&& call->fn()->func_expression()->is_runtime_function())
{
switch (call->fn()->func_expression()->runtime_code())
{
case Runtime::IFACEE2E2:
case Runtime::IFACEI2E2:
case Runtime::IFACEE2I2:
case Runtime::IFACEI2I2:
case Runtime::IFACEE2T2P:
case Runtime::IFACEI2T2P:
{
if (cre->index() != 0)
break;
Node* arg_node = Node::make_node(call->args()->back());
this->assign(dst, arg_node);
}
break;
default:
break;
}
break;
}
Node* call_node = Node::make_node(call);
Node* ret_node = call_node->state(context_, NULL)->retvals[cre->index()];
this->assign(dst, ret_node);
}
break;
case Expression::EXPRESSION_FUNC_REFERENCE:
if (e->func_expression()->closure() != NULL)
this->flows(dst, src);
break;
case Expression::EXPRESSION_CONVERSION:
{
Type_conversion_expression* tce = e->conversion_expression();
Type* ft = tce->expr()->type();
Type* tt = tce->type();
if ((ft->is_string_type() && tt->is_slice_type())
|| (ft->is_slice_type() && tt->is_string_type())
|| (ft->integer_type() != NULL && tt->is_string_type()))
{
this->flows(dst, src);
break;
}
Expression* underlying = tce->expr();
this->assign(dst, Node::make_node(underlying));
}
break;
case Expression::EXPRESSION_FIELD_REFERENCE:
{
if (!e->type()->has_pointer())
break;
}
case Expression::EXPRESSION_TYPE_GUARD:
case Expression::EXPRESSION_ARRAY_INDEX:
case Expression::EXPRESSION_STRING_INDEX:
{
Expression* left = NULL;
if (e->field_reference_expression() != NULL)
{
left = e->field_reference_expression()->expr();
if (left->unary_expression() != NULL
&& left->unary_expression()->op() == OPERATOR_MULT)
{
this->flows(dst, src);
break;
}
}
else if (e->type_guard_expression() != NULL)
left = e->type_guard_expression()->expr();
else if (e->array_index_expression() != NULL)
{
Array_index_expression* aie = e->array_index_expression();
if (aie->end() != NULL)
if (aie->array()->type()->is_slice_type())
left = aie->array();
else
{
go_assert(src->child() != NULL);
this->assign(dst, src->child());
break;
}
else if (!aie->array()->type()->is_slice_type())
{
Node* array_node = Node::make_node(aie->array());
this->assign(dst, array_node);
break;
}
else
{
this->flows(dst, src);
break;
}
}
else if (e->string_index_expression() != NULL)
{
String_index_expression* sie = e->string_index_expression();
if (e->type()->is_string_type())
left = sie->string();
else
{
this->flows(dst, src);
break;
}
}
go_assert(left != NULL);
Node* left_node = Node::make_node(left);
this->assign(dst, left_node);
}
break;
case Expression::EXPRESSION_BINARY:
{
switch (e->binary_expression()->op())
{
case OPERATOR_PLUS:
case OPERATOR_MINUS:
case OPERATOR_XOR:
case OPERATOR_OR:
case OPERATOR_MULT:
case OPERATOR_DIV:
case OPERATOR_MOD:
case OPERATOR_LSHIFT:
case OPERATOR_RSHIFT:
case OPERATOR_AND:
case OPERATOR_BITCLEAR:
{
Node* left = Node::make_node(e->binary_expression()->left());
this->assign(dst, left);
Node* right = Node::make_node(e->binary_expression()->right());
this->assign(dst, right);
}
break;
default:
break;
}
}
break;
case Expression::EXPRESSION_UNARY:
{
switch (e->unary_expression()->op())
{
case OPERATOR_PLUS:
case OPERATOR_MINUS:
case OPERATOR_XOR:
{
Node* op_node =
Node::make_node(e->unary_expression()->operand());
this->assign(dst, op_node);
}
break;
case OPERATOR_MULT:
case OPERATOR_AND:
this->flows(dst, src);
break;
default:
break;
}
}
break;
case Expression::EXPRESSION_TEMPORARY_REFERENCE:
{
Statement* temp = e->temporary_reference_expression()->statement();
this->assign(dst, Node::make_node(temp));
}
break;
default:
break;
}
}
else if (src->statement() != NULL && src->statement()->temporary_statement() != NULL)
this->flows(dst, src);
}
void
Escape_analysis_assign::assign_deref(Node* dst, Node* src)
{
if (src->expr() != NULL)
{
switch (src->expr()->classification())
{
case Expression::EXPRESSION_BOOLEAN:
case Expression::EXPRESSION_STRING:
case Expression::EXPRESSION_INTEGER:
case Expression::EXPRESSION_FLOAT:
case Expression::EXPRESSION_COMPLEX:
case Expression::EXPRESSION_NIL:
case Expression::EXPRESSION_IOTA:
return;
default:
break;
}
}
this->assign(dst, this->context_->add_dereference(src));
}
int
Escape_analysis_assign::assign_from_note(std::string* note,
const std::vector<Node*>& dsts,
Node* src)
{
int enc = Escape_note::parse_tag(note);
if (src->expr() != NULL)
{
switch (src->expr()->classification())
{
case Expression::EXPRESSION_BOOLEAN:
case Expression::EXPRESSION_STRING:
case Expression::EXPRESSION_INTEGER:
case Expression::EXPRESSION_FLOAT:
case Expression::EXPRESSION_COMPLEX:
case Expression::EXPRESSION_NIL:
case Expression::EXPRESSION_IOTA:
return enc;
default:
break;
}
}
if (this->context_->gogo()->debug_escape_level() > 2)
go_inform(src->location(), "assignfromtag:: src=%s em=%s",
src->ast_format(context_->gogo()).c_str(),
Escape_note::make_tag(enc).c_str());
if (enc == Node::ESCAPE_UNKNOWN)
{
this->assign(this->context_->sink(), src);
return enc;
}
else if (enc == Node::ESCAPE_NONE)
return enc;
if ((enc & ESCAPE_CONTENT_ESCAPES) != 0)
this->assign(this->context_->sink(), this->context_->add_dereference(src));
int save_enc = enc;
enc >>= ESCAPE_RETURN_BITS;
for (std::vector<Node*>::const_iterator p = dsts.begin();
enc != 0 && p != dsts.end();
++p)
{
int bits = enc & ESCAPE_BITS_MASK_FOR_TAG;
if (bits > 0)
{
Node* n = src;
for (int i = 0; i < bits - 1; ++i)
{
n = this->context_->add_dereference(n);
}
this->assign(*p, n);
}
enc >>= ESCAPE_BITS_PER_OUTPUT_IN_TAG;
}
return save_enc;
}
void
Escape_analysis_assign::flows(Node* dst, Node* src)
{
if (src->type() != NULL && !src->type()->has_pointer())
return;
if (dst->is_sink() && dst != this->context_->sink())
return;
Node::Escape_state* dst_state = dst->state(this->context_, NULL);
Node::Escape_state* src_state = src->state(this->context_, NULL);
if (dst == src
|| dst_state == src_state
|| dst_state->flows.find(src) != dst_state->flows.end())
return;
Gogo* gogo = this->context_->gogo();
if (gogo->debug_escape_level() > 2)
go_inform(Linemap::unknown_location(), "flows:: %s <- %s",
dst->ast_format(gogo).c_str(), src->ast_format(gogo).c_str());
if (dst_state->flows.empty())
this->context_->add_dst(dst);
dst_state->flows.insert(src);
}
void
Gogo::assign_connectivity(Escape_context* context, Named_object* fn)
{
if (!fn->is_function())
return;
int save_depth = context->loop_depth();
context->set_loop_depth(1);
Escape_analysis_assign ea(context, fn);
Function::Results* res = fn->func_value()->result_variables();
if (res != NULL)
{
for (Function::Results::const_iterator p = res->begin();
p != res->end();
++p)
{
Node* res_node = Node::make_node(*p);
Node::Escape_state* res_state = res_node->state(context, fn);
res_state->fn = fn;
res_state->loop_depth = 0;
if (context->recursive())
ea.flows(context->sink(), res_node);
}
}
const Bindings* callee_bindings = fn->func_value()->block()->bindings();
Function_type* fntype = fn->func_value()->type();
Typed_identifier_list* params = (fntype->parameters() != NULL
? fntype->parameters()->copy()
: new Typed_identifier_list);
if (fntype->receiver() != NULL)
params->push_back(*fntype->receiver());
for (Typed_identifier_list::const_iterator p = params->begin();
p != params->end();
++p)
{
if (p->name().empty() || Gogo::is_sink_name(p->name()))
continue;
Named_object* param_no = callee_bindings->lookup_local(p->name());
go_assert(param_no != NULL);
Node* param_node = Node::make_node(param_no);
Node::Escape_state* param_state = param_node->state(context, fn);
param_state->fn = fn;
param_state->loop_depth = 1;
if (!p->type()->has_pointer())
continue;
if (fn->package() != NULL)
param_node->set_encoding(Node::ESCAPE_HEAP);
else
{
param_node->set_encoding(Node::ESCAPE_NONE);
context->track(param_node);
}
}
Escape_analysis_loop el;
fn->func_value()->traverse(&el);
fn->func_value()->traverse(&ea);
context->set_loop_depth(save_depth);
}
class Escape_analysis_flood
{
public:
Escape_analysis_flood(Escape_context* context)
: context_(context)
{ }
void
flood(Level, Node* dst, Node* src, int);
private:
Escape_context* context_;
};
void
Escape_analysis_flood::flood(Level level, Node* dst, Node* src,
int extra_loop_depth)
{
if (src->expr() != NULL)
{
switch (src->expr()->classification())
{
case Expression::EXPRESSION_BOOLEAN:
case Expression::EXPRESSION_STRING:
case Expression::EXPRESSION_INTEGER:
case Expression::EXPRESSION_FLOAT:
case Expression::EXPRESSION_COMPLEX:
case Expression::EXPRESSION_NIL:
case Expression::EXPRESSION_IOTA:
return;
default:
break;
}
}
Node::Escape_state* src_state = src->state(this->context_, NULL);
if (src_state->flood_id == this->context_->flood_id())
{
level = level.min(src_state->level);
if (level == src_state->level)
{
if (src_state->max_extra_loop_depth >= extra_loop_depth
|| src_state->loop_depth >= extra_loop_depth)
return;
src_state->max_extra_loop_depth = extra_loop_depth;
}
}
else
src_state->max_extra_loop_depth = -1;
src_state->flood_id = this->context_->flood_id();
src_state->level = level;
int mod_loop_depth = std::max(extra_loop_depth, src_state->loop_depth);
Gogo* gogo = this->context_->gogo();
int debug_level = gogo->debug_escape_level();
if (debug_level > 1)
go_inform(Linemap::unknown_location(),
"escwalk: level:{%d %d} depth:%d "
"op=%s %s(%s) "
"scope:%s[%d] "
"extraloopdepth=%d",
level.value(), level.suffix_value(), this->context_->pdepth(),
src->op_format().c_str(),
src->ast_format(gogo).c_str(),
src->details().c_str(),
debug_function_name(src_state->fn).c_str(),
src_state->loop_depth,
extra_loop_depth);
this->context_->increase_pdepth();
Named_object* src_no = NULL;
if (src->expr() != NULL && src->expr()->var_expression() != NULL)
src_no = src->expr()->var_expression()->named_object();
else
src_no = src->object();
bool src_is_param = (src_no != NULL
&& src_no->is_variable()
&& src_no->var_value()->is_parameter());
Named_object* dst_no = NULL;
if (dst->expr() != NULL && dst->expr()->var_expression() != NULL)
dst_no = dst->expr()->var_expression()->named_object();
else
dst_no = dst->object();
bool dst_is_result = dst_no != NULL && dst_no->is_result_variable();
Node::Escape_state* dst_state = dst->state(this->context_, NULL);
if (src_is_param
&& dst_is_result
&& src_state->fn == dst_state->fn
&& (src->encoding() & ESCAPE_MASK) < int(Node::ESCAPE_HEAP)
&& dst->encoding() != Node::ESCAPE_HEAP)
{
if (debug_level != 0)
{
if (debug_level == 1)
go_inform(src->definition_location(),
"leaking param: %s to result %s level=%d",
src->ast_format(gogo).c_str(),
dst->ast_format(gogo).c_str(),
level.value());
else
go_inform(src->definition_location(),
"leaking param: %s to result %s level={%d %d}",
src->ast_format(gogo).c_str(),
dst->ast_format(gogo).c_str(),
level.value(), level.suffix_value());
}
if ((src->encoding() & ESCAPE_MASK) != Node::ESCAPE_RETURN)
{
int enc =
Node::ESCAPE_RETURN | (src->encoding() & ESCAPE_CONTENT_ESCAPES);
src->set_encoding(enc);
}
int enc = Node::note_inout_flows(src->encoding(),
dst_no->result_var_value()->index(),
level);
src->set_encoding(enc);
level = level.copy();
for (std::set<Node*>::const_iterator p = src_state->flows.begin();
p != src_state->flows.end();
++p)
this->flood(level, dst, *p, extra_loop_depth);
return;
}
if (src_is_param
&& dst->encoding() == Node::ESCAPE_HEAP
&& (src->encoding() & ESCAPE_MASK) < int(Node::ESCAPE_HEAP)
&& level.value() > 0)
{
int enc =
Node::max_encoding((src->encoding() | ESCAPE_CONTENT_ESCAPES),
Node::ESCAPE_NONE);
src->set_encoding(enc);
if (debug_level != 0)
go_inform(src->definition_location(), "mark escaped content: %s",
src->ast_format(gogo).c_str());
}
bool src_leaks = (level.value() <= 0
&& level.suffix_value() <= 0
&& dst_state->loop_depth < mod_loop_depth);
src_leaks = src_leaks || (level.value() <= 0
&& (dst->encoding() & ESCAPE_MASK) == Node::ESCAPE_HEAP);
int osrcesc = src->encoding();
if (src_is_param
&& (src_leaks || dst_state->loop_depth < 0)
&& (src->encoding() & ESCAPE_MASK) < int(Node::ESCAPE_HEAP))
{
if (level.suffix_value() > 0)
{
int enc =
Node::max_encoding((src->encoding() | ESCAPE_CONTENT_ESCAPES),
Node::ESCAPE_NONE);
src->set_encoding(enc);
if (debug_level != 0 && osrcesc != src->encoding())
go_inform(src->definition_location(), "leaking param content: %s",
src->ast_format(gogo).c_str());
}
else
{
if (debug_level != 0)
go_inform(src->definition_location(), "leaking param: %s",
src->ast_format(gogo).c_str());
src->set_encoding(Node::ESCAPE_HEAP);
}
}
else if (src->expr() != NULL)
{
Expression* e = src->expr();
if (e->enclosed_var_expression() != NULL)
{
if (src_leaks && debug_level != 0)
go_inform(src->location(), "leaking closure reference %s",
src->ast_format(gogo).c_str());
Node* enclosed_node =
Node::make_node(e->enclosed_var_expression()->variable());
this->flood(level, dst, enclosed_node, -1);
}
else if (e->heap_expression() != NULL
|| (e->unary_expression() != NULL
&& e->unary_expression()->op() == OPERATOR_AND))
{
Expression* underlying;
if (e->heap_expression())
underlying = e->heap_expression()->expr();
else
underlying = e->unary_expression()->operand();
Node* underlying_node = Node::make_node(underlying);
underlying->address_taken(src_leaks);
if (src_leaks)
{
src->set_encoding(Node::ESCAPE_HEAP);
if (osrcesc != src->encoding())
{
move_to_heap(gogo, underlying);
if (debug_level > 1)
go_inform(src->location(),
"%s escapes to heap, level={%d %d}, "
"dst.eld=%d, src.eld=%d",
src->ast_format(gogo).c_str(), level.value(),
level.suffix_value(), dst_state->loop_depth,
mod_loop_depth);
else if (debug_level > 0)
go_inform(src->location(), "%s escapes to heap",
src->ast_format(gogo).c_str());
}
this->flood(level.decrease(), dst,
underlying_node, mod_loop_depth);
extra_loop_depth = mod_loop_depth;
}
else
{
this->flood(level.decrease(), dst, underlying_node, -1);
}
}
else if (e->slice_literal() != NULL)
{
Slice_construction_expression* slice = e->slice_literal();
if (slice->vals() != NULL)
{
for (Expression_list::const_iterator p = slice->vals()->begin();
p != slice->vals()->end();
++p)
{
if ((*p) != NULL)
this->flood(level.decrease(), dst, Node::make_node(*p), -1);
}
}
if (src_leaks)
{
src->set_encoding(Node::ESCAPE_HEAP);
if (debug_level != 0 && osrcesc != src->encoding())
go_inform(src->location(), "%s escapes to heap",
src->ast_format(gogo).c_str());
extra_loop_depth = mod_loop_depth;
}
}
else if (e->call_expression() != NULL)
{
Call_expression* call = e->call_expression();
if (call->is_builtin())
{
Builtin_call_expression* bce = call->builtin_call_expression();
if (bce->code() == Builtin_call_expression::BUILTIN_APPEND)
{
Expression* appendee = call->args()->front();
this->flood(level, dst, Node::make_node(appendee), -1);
}
}
else if (call->fn()->func_expression() != NULL
&& call->fn()->func_expression()->is_runtime_function())
{
switch (call->fn()->func_expression()->runtime_code())
{
case Runtime::MAKECHAN:
case Runtime::MAKECHAN64:
case Runtime::MAKEMAP:
case Runtime::MAKESLICE:
case Runtime::MAKESLICE64:
if (src_leaks)
{
src->set_encoding(Node::ESCAPE_HEAP);
if (debug_level != 0 && osrcesc != src->encoding())
go_inform(src->location(), "%s escapes to heap",
src->ast_format(gogo).c_str());
extra_loop_depth = mod_loop_depth;
}
break;
default:
break;
}
}
else if (src_state->retvals.size() > 0)
{
go_assert(src_state->retvals.size() == 1);
if (debug_level > 2)
go_inform(src->location(), "[%d] dst %s escwalk replace src: %s with %s",
this->context_->loop_depth(),
dst->ast_format(gogo).c_str(),
src->ast_format(gogo).c_str(),
src_state->retvals[0]->ast_format(gogo).c_str());
src = src_state->retvals[0];
src_state = src->state(this->context_, NULL);
}
}
else if (e->allocation_expression() != NULL && src_leaks)
{
src->set_encoding(Node::ESCAPE_HEAP);
if (debug_level != 0 && osrcesc != src->encoding())
go_inform(src->location(), "%s escapes to heap",
src->ast_format(gogo).c_str());
extra_loop_depth = mod_loop_depth;
}
else if ((e->map_literal() != NULL
|| e->string_concat_expression() != NULL
|| (e->func_expression() != NULL && e->func_expression()->closure() != NULL)
|| e->bound_method_expression() != NULL)
&& src_leaks)
{
src->set_encoding(Node::ESCAPE_HEAP);
if (debug_level != 0 && osrcesc != src->encoding())
go_inform(src->location(), "%s escapes to heap",
src->ast_format(gogo).c_str());
extra_loop_depth = mod_loop_depth;
}
else if (e->conversion_expression() != NULL && src_leaks)
{
Type_conversion_expression* tce = e->conversion_expression();
Type* ft = tce->expr()->type();
Type* tt = tce->type();
if ((ft->is_string_type() && tt->is_slice_type())
|| (ft->is_slice_type() && tt->is_string_type())
|| (ft->integer_type() != NULL && tt->is_string_type()))
{
src->set_encoding(Node::ESCAPE_HEAP);
if (debug_level != 0 && osrcesc != src->encoding())
go_inform(src->location(), "%s escapes to heap",
src->ast_format(gogo).c_str());
extra_loop_depth = mod_loop_depth;
}
}
else if (e->array_index_expression() != NULL
&& !e->array_index_expression()->array()->type()->is_slice_type())
{
Array_index_expression* aie = e->array_index_expression();
if (aie->end() != NULL)
{
this->flood(level, dst, src->child(), -1);
}
else
{
Expression* underlying = e->array_index_expression()->array();
Node* underlying_node = Node::make_node(underlying);
this->flood(level, dst, underlying_node, -1);
}
}
else if ((e->field_reference_expression() != NULL
&& e->field_reference_expression()->expr()->unary_expression() == NULL)
|| e->type_guard_expression() != NULL
|| (e->array_index_expression() != NULL
&& e->array_index_expression()->end() != NULL)
|| (e->string_index_expression() != NULL
&& e->type()->is_string_type()))
{
Expression* underlying;
if (e->field_reference_expression() != NULL)
underlying = e->field_reference_expression()->expr();
else if (e->type_guard_expression() != NULL)
underlying = e->type_guard_expression()->expr();
else if (e->array_index_expression() != NULL)
underlying = e->array_index_expression()->array();
else
underlying = e->string_index_expression()->string();
Node* underlying_node = Node::make_node(underlying);
this->flood(level, dst, underlying_node, -1);
}
else if ((e->field_reference_expression() != NULL
&& e->field_reference_expression()->expr()->unary_expression() != NULL)
|| e->array_index_expression() != NULL
|| e->map_index_expression() != NULL
|| (e->unary_expression() != NULL
&& e->unary_expression()->op() == OPERATOR_MULT))
{
Expression* underlying;
if (e->field_reference_expression() != NULL)
{
underlying = e->field_reference_expression()->expr();
underlying = underlying->unary_expression()->operand();
}
else if (e->array_index_expression() != NULL)
underlying = e->array_index_expression()->array();
else if (e->map_index_expression() != NULL)
underlying = e->map_index_expression()->map();
else
underlying = e->unary_expression()->operand();
Node* underlying_node = Node::make_node(underlying);
this->flood(level.increase(), dst, underlying_node, -1);
}
else if (e->temporary_reference_expression() != NULL)
{
Statement* t = e->temporary_reference_expression()->statement();
this->flood(level, dst, Node::make_node(t), -1);
}
}
else if (src->is_indirect())
this->flood(level.increase(), dst, src->child(), -1);
level = level.copy();
for (std::set<Node*>::const_iterator p = src_state->flows.begin();
p != src_state->flows.end();
++p)
this->flood(level, dst, *p, extra_loop_depth);
this->context_->decrease_pdepth();
}
void
Gogo::propagate_escape(Escape_context* context, Node* dst)
{
if (dst->object() == NULL
&& (dst->expr() == NULL
|| (dst->expr()->var_expression() == NULL
&& dst->expr()->enclosed_var_expression() == NULL
&& dst->expr()->func_expression() == NULL)))
return;
Node::Escape_state* state = dst->state(context, NULL);
Gogo* gogo = context->gogo();
if (gogo->debug_escape_level() > 1)
go_inform(Linemap::unknown_location(), "escflood:%d: dst %s scope:%s[%d]",
context->flood_id(), dst->ast_format(gogo).c_str(),
debug_function_name(state->fn).c_str(),
state->loop_depth);
Escape_analysis_flood eaf(context);
for (std::set<Node*>::const_iterator p = state->flows.begin();
p != state->flows.end();
++p)
{
context->increase_flood_id();
eaf.flood(Level::From(0), dst, *p, -1);
}
}
class Escape_analysis_tag
{
public:
Escape_analysis_tag(Escape_context* context)
: context_(context)
{ }
void
tag(Named_object* fn);
private:
Escape_context* context_;
};
void
Escape_analysis_tag::tag(Named_object* fn)
{
if (fn->package() != NULL)
return;
if (fn->is_function_declaration())
{
Function_declaration* fdcl = fn->func_declaration_value();
if ((fdcl->pragmas() & GOPRAGMA_NOESCAPE) != 0)
{
Function_type* fntype = fdcl->type();
if (fntype->parameters() != NULL)
{
const Typed_identifier_list* til = fntype->parameters();
int i = 0;
for (Typed_identifier_list::const_iterator p = til->begin();
p != til->end();
++p, ++i)
if (p->type()->has_pointer())
fntype->add_parameter_note(i, Node::ESCAPE_NONE);
}
}
}
if (!fn->is_function())
return;
Function_type* fntype = fn->func_value()->type();
Bindings* bindings = fn->func_value()->block()->bindings();
if (fntype->is_method()
&& !fntype->receiver()->name().empty()
&& !Gogo::is_sink_name(fntype->receiver()->name()))
{
Named_object* rcvr_no = bindings->lookup(fntype->receiver()->name());
go_assert(rcvr_no != NULL);
Node* rcvr_node = Node::make_node(rcvr_no);
switch ((rcvr_node->encoding() & ESCAPE_MASK))
{
case Node::ESCAPE_NONE: 
case Node::ESCAPE_RETURN:
if (fntype->receiver()->type()->has_pointer())
fntype->add_receiver_note(rcvr_node->encoding());
break;
case Node::ESCAPE_HEAP: 
break;
default:
break;
}
}
int i = 0;
if (fntype->parameters() != NULL)
{
const Typed_identifier_list* til = fntype->parameters();
for (Typed_identifier_list::const_iterator p = til->begin();
p != til->end();
++p, ++i)
{
if (p->name().empty() || Gogo::is_sink_name(p->name()))
{
if (p->type()->has_pointer())
fntype->add_parameter_note(i, Node::ESCAPE_NONE);
continue;
}
Named_object* param_no = bindings->lookup(p->name());
go_assert(param_no != NULL);
Node* param_node = Node::make_node(param_no);
switch ((param_node->encoding() & ESCAPE_MASK))
{
case Node::ESCAPE_NONE: 
case Node::ESCAPE_RETURN:
if (p->type()->has_pointer())
fntype->add_parameter_note(i, param_node->encoding());
break;
case Node::ESCAPE_HEAP: 
break;
default:
break;
}
}
}
fntype->set_is_tagged();
}
void
Gogo::tag_function(Escape_context* context, Named_object* fn)
{
Escape_analysis_tag eat(context);
eat.tag(fn);
}
void
Gogo::reclaim_escape_nodes()
{
Node::reclaim_nodes();
}
void
Node::reclaim_nodes()
{
for (std::map<Named_object*, Node*>::iterator p = Node::objects.begin();
p != Node::objects.end();
++p)
delete p->second;
Node::objects.clear();
for (std::map<Expression*, Node*>::iterator p = Node::expressions.begin();
p != Node::expressions.end();
++p)
delete p->second;
Node::expressions.clear();
for (std::map<Statement*, Node*>::iterator p = Node::statements.begin();
p != Node::statements.end();
++p)
delete p->second;
Node::statements.clear();
for (std::vector<Node*>::iterator p = Node::indirects.begin();
p != Node::indirects.end();
++p)
delete *p;
Node::indirects.clear();
}
