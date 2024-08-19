#include "go-system.h"
#include "go-c.h"
#include "go-diagnostics.h"
#include "operator.h"
#include "lex.h"
#include "types.h"
#include "expressions.h"
#include "statements.h"
#include "runtime.h"
#include "gogo.h"
class Mark_address_taken : public Traverse
{
public:
Mark_address_taken(Gogo* gogo)
: Traverse(traverse_expressions),
gogo_(gogo)
{ }
int
expression(Expression**);
private:
Gogo* gogo_;
};
int
Mark_address_taken::expression(Expression** pexpr)
{
Expression* expr = *pexpr;
Unary_expression* ue = expr->unary_expression();
if (ue != NULL)
ue->check_operand_address_taken(this->gogo_);
Array_index_expression* aie = expr->array_index_expression();
if (aie != NULL
&& aie->end() != NULL
&& !aie->array()->type()->is_slice_type())
{
bool escapes = false;
Node* n = Node::make_node(expr);
if (n->child() == NULL
|| (n->child()->encoding() & ESCAPE_MASK) != Node::ESCAPE_NONE)
escapes = true;
aie->array()->address_taken(escapes);
}
if (expr->allocation_expression() != NULL)
{
Node* n = Node::make_node(expr);
if ((n->encoding() & ESCAPE_MASK) == Node::ESCAPE_NONE)
expr->allocation_expression()->set_allocate_on_stack();
}
if (expr->heap_expression() != NULL)
{
Node* n = Node::make_node(expr);
if ((n->encoding() & ESCAPE_MASK) == Node::ESCAPE_NONE)
expr->heap_expression()->set_allocate_on_stack();
}
if (expr->slice_literal() != NULL)
{
Node* n = Node::make_node(expr);
if ((n->encoding() & ESCAPE_MASK) == Node::ESCAPE_NONE)
expr->slice_literal()->set_storage_does_not_escape();
}
Unsafe_type_conversion_expression* uce =
expr->unsafe_conversion_expression();
if (uce != NULL
&& uce->type()->is_slice_type()
&& Node::make_node(uce->expr())->encoding() == Node::ESCAPE_NONE
&& uce->expr()->call_expression() != NULL)
{
Call_expression* call = uce->expr()->call_expression();
if (call->fn()->func_expression() != NULL
&& call->fn()->func_expression()->runtime_code() == Runtime::MAKESLICE)
{
Expression* len_arg = call->args()->at(1);
Expression* cap_arg = call->args()->at(2);
Numeric_constant nclen;
Numeric_constant nccap;
unsigned long vlen;
unsigned long vcap;
if (len_arg->numeric_constant_value(&nclen)
&& cap_arg->numeric_constant_value(&nccap)
&& nclen.to_unsigned_long(&vlen) == Numeric_constant::NC_UL_VALID
&& nccap.to_unsigned_long(&vcap) == Numeric_constant::NC_UL_VALID)
{
Location loc = expr->location();
Type* elmt_type = expr->type()->array_type()->element_type();
Expression* len_expr =
Expression::make_integer_ul(vcap, cap_arg->type(), loc);
Type* array_type = Type::make_array_type(elmt_type, len_expr);
Expression* alloc = Expression::make_allocation(array_type, loc);
alloc->allocation_expression()->set_allocate_on_stack();
Expression* array = Expression::make_unary(OPERATOR_MULT, alloc, loc);
Expression* zero = Expression::make_integer_ul(0, len_arg->type(), loc);
Expression* slice =
Expression::make_array_index(array, zero, len_arg, cap_arg, loc);
*pexpr = slice;
}
}
}
return TRAVERSE_CONTINUE;
}
class Check_escape : public Traverse
{
public:
Check_escape(Gogo* gogo)
: Traverse(traverse_expressions | traverse_variables),
gogo_(gogo)
{ }
int
expression(Expression**);
int
variable(Named_object*);
private:
Gogo* gogo_;
};
int
Check_escape::variable(Named_object* no)
{
if ((no->is_variable() && no->var_value()->is_in_heap())
|| (no->is_result_variable()
&& no->result_var_value()->is_in_heap()))
go_error_at(no->location(),
"%s escapes to heap, not allowed in runtime",
no->message_name().c_str());
return TRAVERSE_CONTINUE;
}
int
Check_escape::expression(Expression** pexpr)
{
Expression* expr = *pexpr;
Func_expression* fe = expr->func_expression();
if (fe != NULL && fe->closure() != NULL)
{
Node* n = Node::make_node(expr);
if (n->encoding() == Node::ESCAPE_HEAP)
go_error_at(expr->location(),
"heap-allocated closure, not allowed in runtime");
}
return TRAVERSE_CONTINUE;
}
class Write_barriers : public Traverse
{
public:
Write_barriers(Gogo* gogo)
: Traverse(traverse_functions | traverse_variables | traverse_statements),
gogo_(gogo), function_(NULL)
{ }
int
function(Named_object*);
int
variable(Named_object*);
int
statement(Block*, size_t* pindex, Statement*);
private:
Gogo* gogo_;
Function* function_;
};
int
Write_barriers::function(Named_object* no)
{
go_assert(this->function_ == NULL);
this->function_ = no->func_value();
int t = this->function_->traverse(this);
this->function_ = NULL;
if (t == TRAVERSE_EXIT)
return t;
return TRAVERSE_SKIP_COMPONENTS;
}
int
Write_barriers::variable(Named_object* no)
{
if (!no->is_variable())
return TRAVERSE_CONTINUE;
Variable* var = no->var_value();
if (!var->is_global())
return TRAVERSE_CONTINUE;
Expression* init = var->init();
if (init == NULL)
return TRAVERSE_CONTINUE;
if (!var->type()->has_pointer())
return TRAVERSE_CONTINUE;
init = Expression::make_cast(var->type(), init, var->location());
if (!var->has_pre_init() && init->is_static_initializer())
return TRAVERSE_CONTINUE;
if (!var->type()->in_heap())
return TRAVERSE_CONTINUE;
if (var->type()->points_to() != NULL && !var->type()->points_to()->in_heap())
return TRAVERSE_CONTINUE;
this->gogo_->check_self_dep(no);
Location loc = init->location();
Expression* ref = Expression::make_var_reference(no, loc);
Statement_inserter inserter(this->gogo_, var);
Statement* s = this->gogo_->assign_with_write_barrier(NULL, NULL, &inserter,
ref, init, loc);
var->add_preinit_statement(this->gogo_, s);
var->clear_init();
return TRAVERSE_CONTINUE;
}
int
Write_barriers::statement(Block* block, size_t* pindex, Statement* s)
{
switch (s->classification())
{
default:
break;
case Statement::STATEMENT_VARIABLE_DECLARATION:
{
Variable_declaration_statement* vds =
s->variable_declaration_statement();
Named_object* no = vds->var();
Variable* var = no->var_value();
Expression* init = var->init();
if (init == NULL)
break;
if (!var->is_in_heap())
break;
if (!var->type()->has_pointer())
break;
if (!var->type()->in_heap())
break;
if (var->type()->points_to() != NULL
&& !var->type()->points_to()->in_heap())
break;
Function* function = this->function_;
Location loc = init->location();
Statement_inserter inserter(block, pindex);
var->clear_init();
inserter.insert(s);
Expression* ref = Expression::make_var_reference(no, loc);
Statement* assign = this->gogo_->assign_with_write_barrier(function,
block,
&inserter,
ref, init,
loc);
block->replace_statement(*pindex, assign);
}
break;
case Statement::STATEMENT_ASSIGNMENT:
{
Assignment_statement* as = s->assignment_statement();
Expression* lhs = as->lhs();
Expression* rhs = as->rhs();
if (!this->gogo_->assign_needs_write_barrier(lhs))
break;
Function* function = this->function_;
Location loc = as->location();
Statement_inserter inserter = Statement_inserter(block, pindex);
Statement* assign = this->gogo_->assign_with_write_barrier(function,
block,
&inserter,
lhs, rhs,
loc);
block->replace_statement(*pindex, assign);
}
break;
}
return TRAVERSE_CONTINUE;
}
void
Gogo::add_write_barriers()
{
if (saw_errors())
return;
Mark_address_taken mat(this);
this->traverse(&mat);
if (this->compiling_runtime() && this->package_name() == "runtime")
{
Check_escape chk(this);
this->traverse(&chk);
}
Write_barriers wb(this);
this->traverse(&wb);
}
Named_object*
Gogo::write_barrier_variable()
{
static Named_object* write_barrier_var;
if (write_barrier_var == NULL)
{
Location bloc = Linemap::predeclared_location();
Type* uint32_type = Type::lookup_integer_type("uint32");
Variable* var = new Variable(uint32_type, NULL, true, false, false,
bloc);
bool add_to_globals;
Package* package = this->add_imported_package("runtime", "_", false,
"runtime", "runtime",
bloc, &add_to_globals);
write_barrier_var = Named_object::make_variable("writeBarrier",
package, var);
}
return write_barrier_var;
}
bool
Gogo::assign_needs_write_barrier(Expression* lhs)
{
if (!lhs->type()->has_pointer())
return false;
while (true)
{
if (!lhs->type()->in_heap())
return false;
if (lhs->type()->points_to() != NULL
&& !lhs->type()->points_to()->in_heap())
return false;
Field_reference_expression* fre = lhs->field_reference_expression();
if (fre == NULL)
break;
lhs = fre->expr();
}
if (lhs->temporary_reference_expression() != NULL)
return false;
if (lhs->is_sink_expression())
return false;
Var_expression* ve = lhs->var_expression();
if (ve != NULL)
{
Named_object* no = ve->named_object();
if (no->is_variable())
{
Variable* var = no->var_value();
if (!var->is_global() && !var->is_in_heap())
return false;
}
else if (no->is_result_variable())
{
Result_variable* rvar = no->result_var_value();
if (!rvar->is_in_heap())
return false;
}
}
Struct_type* st = lhs->type()->struct_type();
if (st != NULL)
{
bool in_heap = false;
const Struct_field_list* fields = st->fields();
for (Struct_field_list::const_iterator p = fields->begin();
p != fields->end();
p++)
{
Type* ft = p->type();
if (!ft->has_pointer())
continue;
if (!ft->in_heap())
continue;
if (ft->points_to() != NULL && !ft->points_to()->in_heap())
continue;
in_heap = true;
break;
}
if (!in_heap)
return false;
}
return true;
}
Statement*
Gogo::assign_with_write_barrier(Function* function, Block* enclosing,
Statement_inserter* inserter, Expression* lhs,
Expression* rhs, Location loc)
{
if (function != NULL
&& ((function->pragmas() & GOPRAGMA_NOWRITEBARRIER) != 0
|| (function->pragmas() & GOPRAGMA_NOWRITEBARRIERREC) != 0))
go_error_at(loc, "write barrier prohibited");
Type* type = lhs->type();
go_assert(type->has_pointer());
Expression* addr;
if (lhs->unary_expression() != NULL
&& lhs->unary_expression()->op() == OPERATOR_MULT)
addr = lhs->unary_expression()->operand();
else
{
addr = Expression::make_unary(OPERATOR_AND, lhs, loc);
addr->unary_expression()->set_does_not_escape();
}
Temporary_statement* lhs_temp = Statement::make_temporary(NULL, addr, loc);
inserter->insert(lhs_temp);
lhs = Expression::make_temporary_reference(lhs_temp, loc);
if (!Type::are_identical(type, rhs->type(), false, NULL)
&& rhs->type()->interface_type() != NULL
&& !rhs->is_variable())
{
Temporary_statement* temp = Statement::make_temporary(NULL, rhs, loc);
inserter->insert(temp);
rhs = Expression::make_temporary_reference(temp, loc);
}
rhs = Expression::convert_for_assignment(this, type, rhs, loc);
Temporary_statement* rhs_temp = NULL;
if (!rhs->is_variable() && !rhs->is_constant())
{
rhs_temp = Statement::make_temporary(NULL, rhs, loc);
inserter->insert(rhs_temp);
rhs = Expression::make_temporary_reference(rhs_temp, loc);
}
Expression* indir =
Expression::make_dereference(lhs, Expression::NIL_CHECK_DEFAULT, loc);
Statement* assign = Statement::make_assignment(indir, rhs, loc);
lhs = Expression::make_temporary_reference(lhs_temp, loc);
if (rhs_temp != NULL)
rhs = Expression::make_temporary_reference(rhs_temp, loc);
Type* unsafe_ptr_type = Type::make_pointer_type(Type::make_void_type());
lhs = Expression::make_unsafe_cast(unsafe_ptr_type, lhs, loc);
Expression* call;
switch (type->base()->classification())
{
default:
go_unreachable();
case Type::TYPE_ERROR:
return assign;
case Type::TYPE_POINTER:
case Type::TYPE_FUNCTION:
case Type::TYPE_MAP:
case Type::TYPE_CHANNEL:
call = Runtime::make_call(Runtime::WRITEBARRIERPTR, loc, 2, lhs, rhs);
break;
case Type::TYPE_STRING:
case Type::TYPE_STRUCT:
case Type::TYPE_ARRAY:
case Type::TYPE_INTERFACE:
{
rhs = Expression::make_unary(OPERATOR_AND, rhs, loc);
rhs->unary_expression()->set_does_not_escape();
call = Runtime::make_call(Runtime::TYPEDMEMMOVE, loc, 3,
Expression::make_type_descriptor(type, loc),
lhs, rhs);
}
break;
}
return this->check_write_barrier(enclosing, assign,
Statement::make_statement(call, false));
}
Statement*
Gogo::check_write_barrier(Block* enclosing, Statement* without,
Statement* with)
{
Location loc = without->location();
Named_object* wb = this->write_barrier_variable();
Expression* ref = Expression::make_var_reference(wb, loc);
Expression* zero = Expression::make_integer_ul(0, ref->type(), loc);
Expression* cond = Expression::make_binary(OPERATOR_EQEQ, ref, zero, loc);
Block* then_block = new Block(enclosing, loc);
then_block->add_statement(without);
Block* else_block = new Block(enclosing, loc);
else_block->add_statement(with);
return Statement::make_if_statement(cond, then_block, else_block, loc);
}
