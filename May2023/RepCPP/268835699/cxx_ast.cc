#include "cxx_ast.hh"

namespace __preprocessor__ {
namespace __cxx__ {
std::vector<for_statement_t> for_statements(const clang_ast_t &ast,clang_ast_node_t *root) {
std::vector<for_statement_t> statements;
auto visitor=[&statements](clang_ast_node_t& node,size_t depth) {
if(node.token==CXCursor_ForStmt)
statements.push_back(for_statement_t(&node));
};
ast.traverse(visitor,root);
return statements;
}
std::vector<for_statement_t> for_statements(const std::vector<std::string>& src,const clang_ast_t& ast,clang_ast_node_t* root) {
std::vector<for_statement_t> statements;
auto visitor=[&statements,&src](clang_ast_node_t& node,size_t depth) {
if(node.token==CXCursor_ForStmt)
statements.push_back(for_statement_t(node,src));
};
ast.traverse(visitor,root);
return statements;
}
void remove_unexposed(clang_ast_t &ast,clang_ast_node_t *root) {
auto visitor=[&ast](clang_ast_node_t& node,size_t depth) {
if((node.token==CXCursor_UnexposedExpr)||(node.token==CXCursor_UnexposedStmt)||(node.token==CXCursor_UnexposedDecl)||(node.token==CXCursor_UnexposedAttr))
ast.erase(ast(&node));
};
ast.template traverse<astTraversal::DFSPreorderPost>(visitor,root);
}
variable_t variable_type(const clang_ast_node_t &node) {
if(node.token==CXCursor_VarDecl) {
if(node.parent!=nullptr) {
if(node.parent->token==CXCursor_DeclStmt)
if(node.parent->parent!=nullptr)
if(node.parent->parent->token==CXCursor_ForStmt)
return FOR_VAR;
}
if(node.children.size()>0)
if(node.children.front()->token==CXCursor_LambdaExpr)
return LAMBDA_FUNCTION;
return REGULAR_VAR;
}
else if((node.token==CXCursor_ParmDecl)&&(node.parent!=nullptr)) {
if(node.parent->token==CXCursor_FunctionDecl)
return FUNCTION_VAR;
else if(node.parent->token==CXCursor_LambdaExpr)
return LAMBDA_VAR;
}
else if(node.token==CXCursor_DeclRefExpr) {
if(node.parent!=nullptr)
if(node.parent->token!=CXCursor_CallExpr)
return REFERENCE_VAR;
}
return NOT_VAR;
}
std::pair<std::map<clang_ast_node_t*,size_t>,std::vector<clang_ast_node_t*>> pragma_code_blocks(const std::vector<pragma_ast_node_t*>& pragma_translation,const clang_ast_t &ast,clang_ast_node_t *root) {
std::pair<std::map<clang_ast_node_t*,size_t>,std::vector<clang_ast_node_t*>> blocks=std::pair<std::map<clang_ast_node_t*,size_t>,std::vector<clang_ast_node_t*>>(std::map<clang_ast_node_t*,size_t>(),
std::vector<clang_ast_node_t*>(pragma_translation.size()));
std::map<clang_ast_node_t*,size_t>& forward_table=blocks.first;
std::vector<clang_ast_node_t*>& list=blocks.second;
auto visitor=[&pragma_translation,&forward_table,&list](const clang_ast_node_t& node,size_t depth) {
if(node.token==CXCursor_DeclStmt)
for(clang_ast_node_t* n : node.children )
if(n!=nullptr)
if(n->token==CXCursor_TypedefDecl)
if(__pragma__::is_pragma_name(std::get<std::string>(n->value))) {
auto pragma_id=__pragma__::pragma_id(std::get<std::string>(n->value));
auto type=__pragma__::pragma_type(*pragma_translation[pragma_id]);
if(((type&ptok_t::TASK)==ptok_t::TASK)||(!((type&ptok_t::DEPEND)==ptok_t::DEPEND))) {
auto it=std::find(node.parent->children.cbegin(),node.parent->children.cend(),&node);
++it;
list[pragma_id]=*it;
forward_table[*it]=pragma_id;
}
else {
list[pragma_id]=const_cast<clang_ast_node_t*>(&node);
forward_table[const_cast<clang_ast_node_t*>(&node)]=pragma_id;
}
}
};
ast.traverse(visitor,root);
return blocks;
}
std::vector<std::string> pragma_identifier_list(const clang_ast_t &ast,clang_ast_node_t *root) {
std::vector<std::string> list;
auto visitor=[&list](const clang_ast_node_t& node,size_t depth) {
if(node.token==CXCursor_TypedefDecl)
if(__pragma__::is_pragma_name(std::get<std::string>(node.value)))
list.push_back(std::get<std::string>(node.value));
};
ast.traverse(visitor,root);
return list;
}
std::string remove_tasks(const std::vector<std::string>& src,const std::vector<pragma_ast_node_t*> &pragma_translation,const std::vector<clang_ast_node_t*>& cxx_translation,const clang_ast_t &ast,clang_ast_node_t *root,bool leave) {
std::vector<size_t> tasks_to_remove;
auto visitor=[&tasks_to_remove](const clang_ast_node_t& node,size_t depth) {
if(node.token==CXCursor_TypedefDecl)
if(__pragma__::is_pragma_name(std::get<std::string>(node.value)))
tasks_to_remove.push_back(__pragma__::pragma_id(std::get<std::string>(node.value)));
};
ast.traverse(visitor,root);
int line_begin=0,line_end=src.size()>0?src.size()-1:0;
std::string base_identation="";
if(root!=nullptr) {
line_begin=root->extent.begin.line-1;
line_end=root->extent.end.line-1;
base_identation=identation(src[line_begin]);
}
std::vector<std::pair<int,int>> lines_to_remove;
for(size_t i: tasks_to_remove) {
if(!leave) {
if((__pragma__::pragma_type(*pragma_translation[i])&(ptok_t::TASK|ptok_t::DEPEND))==(ptok_t::TASK|ptok_t::DEPEND))
lines_to_remove.push_back(std::make_pair(cxx_translation[i]->extent.begin.line-1,cxx_translation[i]->extent.end.line-1));
else if((__pragma__::pragma_type(*pragma_translation[i])&ptok_t::TASK)==ptok_t::TASK)
lines_to_remove.push_back(std::make_pair(cxx_translation[i]->extent.begin.line-2,cxx_translation[i]->extent.end.line-1));
}
else
if((__pragma__::pragma_type(*pragma_translation[i])&ptok_t::TASK)==ptok_t::TASK)
lines_to_remove.push_back(std::make_pair(cxx_translation[i]->extent.begin.line-1,cxx_translation[i]->extent.end.line-1));
}
std::string result="";
size_t pos=0;
for(int i=line_begin;i<=line_end;){
if(pos<lines_to_remove.size()) {
if(i<lines_to_remove[pos].first) {
result+=remove_single_identation(src[i],base_identation)+"\n";
++i;
}
else
i=lines_to_remove[pos++].second+1;
}
else {
result+=remove_single_identation(src[i],base_identation)+"\n";
++i;
}
}
auto last=result.back();
if(last=='\n')
return result.erase(result.size()-1);
return result;
}
}
}
