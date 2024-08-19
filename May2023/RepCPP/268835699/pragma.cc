#include "pragma.hh"

namespace __preprocessor__ {
namespace __pragma__ {
std::string pragma_name(size_t id) {
return pragma_name_identifier+std::to_string(id);
}
size_t pragma_id(const std::string &name) {
return std::stoull(name.substr(pragma_name_identifier.size()));
}
bool is_pragma_name(const std::string& name) {
if(name.size()>pragma_name_identifier.size()) {
bool result=(name.compare(0,pragma_name_identifier.size(),pragma_name_identifier)==0);
if(result)
for(size_t i=pragma_name_identifier.size();i<name.size();++i)
result=result&&(std::isdigit(name[i]));
return result;
}
return false;
}
ptok_t pragma_type(const pragma_ast_node_t &pragma) {
if((pragma.token==ptok_t::PRAGMA)||(pragma.token==ptok_t::DEPENDENCY))
return std::get<ptok_t>(pragma.value);
else
return pragma.token;
}

std::vector<pragma_ast_node_t*> environment_pragmas(const std::vector<pragma_ast_node_t*>& pragmas){
std::vector<pragma_ast_node_t*> env_pragmas;
for(pragma_ast_node_t* pragma : pragmas)
if(pragma->token==ptok_t::PRAGMA)
if(((pragma_type(*pragma)&ptok_t::TASK)!=ptok_t::TASK)&&((pragma_type(*pragma)&ptok_t::DEPEND)!=ptok_t::DEPEND))
env_pragmas.push_back(pragma);
return env_pragmas;
}
std::pair<std::map<pragma_ast_node_t*,size_t>,std::vector<pragma_ast_node_t*>>
transform_pragmas_identifiers(std::vector<std::string>& src,const pragma_ast_t& ast) {
std::pair<std::map<pragma_ast_node_t*,size_t>,std::vector<pragma_ast_node_t*>> translation_table=std::pair<std::map<pragma_ast_node_t*,size_t>,std::vector<pragma_ast_node_t*>>(std::map<pragma_ast_node_t*,size_t>(),std::vector<pragma_ast_node_t*>());
std::map<pragma_ast_node_t*,size_t>& forward_table=translation_table.first;
std::vector<pragma_ast_node_t*>& pragma_list=translation_table.second;
std::vector<size_t> pragma_lines;
size_t i=0;
auto line_collector=[&forward_table,&pragma_list,&i,&pragma_lines](const pragma_ast_node_t& node,size_t depth) {
if(node.token==ptok_t::PRAGMA) {
pragma_lines.push_back(node.extent.begin.line);
forward_table[const_cast<pragma_ast_node_t*>(&node)]=i++;
pragma_list.push_back(const_cast<pragma_ast_node_t*>(&node));
}
};
ast.traverse(line_collector,nullptr);
i=0;
for(size_t line : pragma_lines )
src[line-1]=identation(src,line-1)+"typedef int "+pragma_name(i++)+";";
return translation_table;
}
std::pair<ptok_t,std::vector<std::string>> coarsening_options(const pragma_ast_t &ast,pragma_ast_node_t* root) {
std::pair<ptok_t,std::vector<std::string>> options(ptok_t::ROOT,std::vector<std::string>());
ptok_t& method=options.first;
std::vector<std::string>& opts=options.second;
auto visitor=[&opts,&method](const pragma_ast_node_t& node,size_t depth) {
if(node.token==ptok_t::SYMBOL)
if(node.parent->token==ptok_t::COARSENING_OPTS) {
opts.push_back(std::get<std::string>(node.value));
method=node.parent->parent->token;
}
};
ast.traverse(visitor,root);
return options;
}

std::string pragma_thread_num(const pragma_ast_t &ast, pragma_ast_node_t *root) {
std::string thread_num="";
auto visitor=[&thread_num](const pragma_ast_node_t& node,size_t depth) {
if(node.token==ptok_t::NUM_THREADS)
thread_num=std::get<std::string>(node.value);
};
ast.traverse(visitor,root);
return thread_num;
}

std::vector<dependency_t> dependency_list(const pragma_ast_t &ast,pragma_ast_node_t* root) {
using namespace std;
std::vector<dependency_t> dependencies;
auto visitor=[&dependencies](pragma_ast_node_t& node,size_t depth) {
switch (pragma_type(node)){
case ptok_t::DEPENDENCY:
dependencies.push_back(dependency_t(std::get<string>((*node[0]).value),std::get<string>((*node[1]).value),"",ptok_t::DEPENDENCY));
break;
case ptok_t::SIMPLE_DEPENDENCY:
dependencies.push_back(dependency_t("",std::get<string>((*node[0]).value),"",ptok_t::SIMPLE_DEPENDENCY));
break;
case ptok_t::CONDITIONAL_DEPENDENCY:
dependencies.push_back(dependency_t(std::get<string>((*node[0]).value),std::get<string>((*node[1]).value),std::get<string>((*node[2]).value),ptok_t::CONDITIONAL_DEPENDENCY));
break;
case ptok_t::SIMPLE_CONDITIONAL_DEPENDENCY:
dependencies.push_back(dependency_t("",std::get<string>((*node[0]).value),std::get<string>((*node[1]).value),ptok_t::SIMPLE_CONDITIONAL_DEPENDENCY));
break;
default:
break;
}
};
ast.traverse(visitor,root);
return dependencies;
}
}
}
