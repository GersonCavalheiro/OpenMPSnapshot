#include "preprocessor.hh"

namespace __preprocessor__ {
const std::string Preprocessor::clang_options="-std=c++17 -I/usr/include -I/usr/local/include";
Preprocessor::Preprocessor() {
}
Preprocessor::Preprocessor(Preprocessor&& preprocessor): input_file_path(std::move(preprocessor.input_file_path)), output_file_path(std::move(preprocessor.output_file_path)),
tmp_file_path(std::move(preprocessor.tmp_file_path)),input_file(std::move(preprocessor.input_file)),output_file(std::move(preprocessor.output_file)),
pragmaAST(std::move(preprocessor.pragmaAST)),clangAST(std::move(preprocessor.clangAST)),
pragma_translation_tables(std::move(preprocessor.pragma_translation_tables)),clang_translation_tables(std::move(preprocessor.clang_translation_tables)),
environment_pragmas(std::move(preprocessor.environment_pragmas)){
}
Preprocessor::~Preprocessor() {
input_file_path.clear();
output_file_path.clear();
tmp_file_path.clear();
input_file.clear();
output_file.clear();
pragmaAST.clear();
clangAST.clear();
pragma_translation_tables.first.clear();
pragma_translation_tables.second.clear();
clang_translation_tables.first.clear();
clang_translation_tables.second.clear();
environment_pragmas.clear();
}

Preprocessor& Preprocessor::operator=(Preprocessor&& preprocessor) {
input_file_path=std::move(preprocessor.input_file_path);
output_file_path=std::move(preprocessor.output_file_path);
tmp_file_path=std::move(preprocessor.tmp_file_path);
input_file=std::move(preprocessor.input_file);
output_file=std::move(preprocessor.output_file);
pragmaAST=std::move(preprocessor.pragmaAST);
clangAST=std::move(preprocessor.clangAST);
input_file=std::move(preprocessor.input_file);
output_file=std::move(preprocessor.output_file);
pragma_translation_tables=std::move(preprocessor.pragma_translation_tables);
clang_translation_tables=std::move(preprocessor.clang_translation_tables);
environment_pragmas=std::move(preprocessor.environment_pragmas);
return *this;
}

void Preprocessor::pragma_parse() {
using namespace __preprocessor__::__pragma__;
if(input_file_path.empty())
return;
pragmaAST=pragma_parser(input_file_path.string());
pragma_translation_tables=transform_pragmas_identifiers(input_file,pragmaAST);
this->environment_pragmas=__pragma__::environment_pragmas(pragma_translation_tables.second);
output_file=input_file;
}
void Preprocessor::clang_parse(std::string options,CXTranslationUnit_Flags flags) {
using namespace __io__;
using namespace __preprocessor__::__pragma__;
using namespace __preprocessor__::__cxx__;
if(input_file_path.empty())
return;
tmp_file_path=input_file_path.parent_path();
tmp_file_path/="tmp_pragma_dag_cxx_"+input_file_path.filename().string();
write(input_file,tmp_file_path.string());
clangAST=__clang__::parse(tmp_file_path.string(),options,flags);
remove_unexposed(clangAST);
clang_translation_tables=pragma_code_blocks(pragma_translation_tables.second,clangAST);
remove_file(tmp_file_path.string());
}
void Preprocessor::parse(std::string filename,std::string options,CXTranslationUnit_Flags flags) {
using namespace __io__;
input_file_path=std::filesystem::path(filename);
input_file=read(input_file_path.string());
pragma_parse();
clang_parse(options,flags);
}
std::vector<std::string> Preprocessor::block_preprocess(pragma_ast_node_t &pragma,clang_ast_node_t &block,const std::vector<std::string>& coarsening_opts,std::string base_identation,bool timing,bool print) {
using namespace __io__;
using namespace __util__;
using namespace __pragma__;
using namespace __cxx__;
using namespace __graph__;
using __io__::operator <<;
auto idenlvl=[&base_identation] (int i){  return i>0?base_identation+std::string(i,'\t'):base_identation; };
int idlvl=0;
std::string block_size_name="ompDAGdimx_";
std::string block_num_name="ompDAGnumx_";
auto code_block=remove_tasks(input_file,pragma_translation_tables.second,clang_translation_tables.second,clangAST,&block);
size_t for_count=count(code_block,"for");
std::vector<std::string> result({idenlvl(idlvl++)+"{"});
if(timing) {
result.push_back(idenlvl(idlvl)+"__ompDAG__::cpu_timer ompDAGTimer;");
result.push_back(idenlvl(idlvl)+"__ompDAG__::ompDAGTiming_t ompDAGTiming;");
}
auto pragma_identifiers=pragma_identifier_list(clangAST,&block);
auto for_list=for_statements(input_file,clangAST,&block);
std::vector<std::vector<std::string>> for_limits;
std::map<std::string,std::vector<dependency_t>> dependencies;
if(print&&timing)
insert(result,vertex_type(idenlvl(idlvl),"__ompDAG__::int_"+std::to_string(for_count)+" "+"begin;","__ompDAG__::int_"+std::to_string(for_count)+" "+"end;",
"int tid;","double etime;"));
else
insert(result,vertex_type(idenlvl(idlvl),"__ompDAG__::int_"+std::to_string(for_count)+" "+"begin;","__ompDAG__::int_"+std::to_string(for_count)+" "+"end;"));
std::string task_num="(", depend_num="";
if(for_list.size()>for_count)
for_list.erase(for_list.begin()+for_count,for_list.end());
for(auto pragma_identifier : pragma_identifiers) {
auto pragma=pragma_translation_tables.second[pragma_id(pragma_identifier)];
if((pragma_type(*pragma)&ptok_t::DEPEND)==ptok_t::DEPEND)
dependencies[pragma_identifier]=dependency_list(pragmaAST,pragma);
}
size_t i=0;
for(auto &[k,v] : dependencies )
i+=v.size();
depend_num=std::to_string(i);
i=0;
for(auto statement : for_list) {
for_limits.push_back(block_iteration(input_file,statement));
std::string tmp=source(input_file,*statement.declaration);
code_block=replace_pattern(code_block,exchange_blank_regex(tmp),"int "+index_name+std::to_string(i)+"=0;");
code_block=replace_pattern(code_block,exact(for_limits.back()[0]),index_name+std::to_string(i));
code_block=replace_pattern(code_block,exact(exchange_escape_regex(for_limits.back()[2])),block_num_name+std::to_string(i));
result.push_back(idenlvl(idlvl)+"int "+block_size_name+std::to_string(i)+" = " +coarsening_opts[i]+";");
result.push_back(idenlvl(idlvl)+"int "+block_num_name+std::to_string(i)+" = ("+for_limits.back()[3]+"+"+block_size_name+std::to_string(i)+"-1)/"+block_size_name+std::to_string(i)+";");
if(task_num.size()==1)
task_num+=block_num_name+std::to_string(i);
else
task_num+="*"+block_num_name+std::to_string(i);
i++;
}
task_num+=")";
depend_num+="*"+task_num;
if(timing)
result.push_back(idenlvl(idlvl)+"ompDAGTimer.start();");
result.push_back(idenlvl(idlvl)+"__ompDAG__::GraphCXS<"+vertex_type_name+",float,__ompDAG__::cpuAllocator> "+graph_name+"("+task_num+","+depend_num+",-1);");
if(timing) {
result.push_back(idenlvl(idlvl)+"ompDAGTimer.stop();");
result.push_back(idenlvl(idlvl)+"ompDAGTiming.allocation=ompDAGTimer.elapsed_time();");
}
result.push_back(idenlvl(idlvl)+"auto "+graph_creation_function_name+" = [&]( ) {");
result.push_back(idenlvl(++idlvl)+"size_t __pos__ = 0, __task__ = 0;");
insert(result,toLines(line_identation(code_block,idenlvl(idlvl))));
auto pragma_positions=patterns<regex_mode::POSITIONS>(result,pragma_name_identifier+int_regex);
std::string tmp=identation(result[pragma_positions.front().first.line-1]);
result.insert(result.begin()+pragma_positions.front().first.line-1,tmp+vertex_type_name+" v;");
int off=1,line=0;
for(size_t i=0;i<for_count;++i) {
result.insert(result.begin()+pragma_positions.front().first.line-1+(off++),
tmp+"v.begin["+std::to_string(i)+"] = "+index_name+std::to_string(i)+"*"+block_size_name+std::to_string(i)+"+"+for_limits[i][1]+";");
result.insert(result.begin()+pragma_positions.front().first.line-1+(off++),
tmp+"v.end["+std::to_string(i)+"] = __ompDAG__::__min__(v.begin["+std::to_string(i)+"]+"+block_size_name+std::to_string(i)+","+for_limits[i][2]+");");
}
for(auto pos : pragma_positions ) {
std::string dp="";
std::string pragma=result[pos.first.line-1+off].substr(pos.first.column-1,pos.second);
for( auto dependency : dependencies.at(pragma) ) {
std::string din=dependency.in;
i=0;
for( auto vid : for_limits) {
din=replace_pattern(din,exact(for_limits[i][0]),index_name+std::to_string(i));
din=replace_pattern(din,exact(exchange_escape_regex(for_limits[i][2])),block_num_name+std::to_string(i));
++i;
}
switch(dependency.type) {
case ptok_t::SIMPLE_DEPENDENCY :
dp+=graph_name+".indxs(__pos__++)="+din+";\n";
break;
case ptok_t::SIMPLE_CONDITIONAL_DEPENDENCY :
dp+="if( "+dependency.condition+" )\n\t"+graph_name+".indxs(__pos__++)="+din+";\n";
break;
default:
break;
}
}
i=0;
for( auto vid : for_limits) {
dp=replace_pattern(dp,exact(for_limits[i][0]),"(v.end["+std::to_string(i)+"]-1)");
++i;
}
auto l=line_identation(toLines(dp),tmp);
result.insert(result.begin()+pos.first.line-1+off,l.begin(),l.end());
off+=l.size();
result.erase(result.begin()+pos.first.line-1+off);
--off;
line=pos.first.line-1+off;
}
result.insert(result.begin()+line+1,tmp+graph_name+".ptr(__task__+1)=__pos__;");
result.insert(result.begin()+line+2,tmp+graph_name+".vertices(__task__++)=v;");
result.push_back(idenlvl(idlvl)+graph_name+".setNZV(__pos__);");
result.push_back(idenlvl(--idlvl)+"};");
result.push_back(idenlvl(idlvl++)+"auto "+task_name+"=[&](int vid, int tid) {");
result.push_back(idenlvl(idlvl)+vertex_type_name+"& vertex = "+graph_name+".vertices(vid);");
std::string task_str=remove_tasks(input_file,pragma_translation_tables.second,clang_translation_tables.second,clangAST,&block,true);
std::map<std::string,std::string> tasks;
for(auto pragma_identifier : pragma_identifiers) {
auto pragma=pragma_translation_tables.second[pragma_id(pragma_identifier)];
auto clang_pragma=clang_translation_tables.second[pragma_id(pragma_identifier)];
if((pragma_type(*pragma)&ptok_t::TASK)==ptok_t::TASK)
tasks[pragma_identifier]=remove_identation(source(input_file,*clang_pragma),identation(input_file[clang_pragma->extent.begin.line-1]))+"\n";
}
i=0;
for(auto statement : for_list) {
std::string tmp=source(input_file,*statement.declaration);
task_str=replace_pattern(task_str,exchange_blank_regex(tmp),"int "+index_name+std::to_string(i)+"=vertex.begin["+std::to_string(i)+"];");
task_str=replace_pattern(task_str,exact(for_limits[i][0]),index_name+std::to_string(i));
task_str=replace_pattern(task_str,exact(exchange_escape_regex(for_limits[i][2])),"vertex.end["+std::to_string(i)+"]");
for (auto& [k,v] : tasks )
v=replace_pattern(v,exact(for_limits[i][0]),index_name+std::to_string(i));
i++;
}
std::vector<std::string> task=toLines(task_str);
pragma_positions=patterns<regex_mode::POSITIONS>(task,pragma_name_identifier+int_regex);
off=0;
for(auto pos : pragma_positions ) {
std::string pragma=task[pos.first.line-1+off].substr(pos.first.column-1,pos.second);
if(tasks.find(pragma)!=tasks.end()) {
auto l=line_identation(toLines(tasks[pragma]),identation(task[pos.first.line-1+off]));
task.insert(task.begin()+pos.first.line-1+off,l.begin(),l.end());
off+=l.size();
}
task.erase(task.begin()+pos.first.line-1+off);
--off;
}
insert(result,line_identation(task,idenlvl(idlvl)));
result.push_back(idenlvl(--idlvl)+"};");
if(timing)
result.push_back(idenlvl(idlvl)+"ompDAGTimer.start();");
result.push_back(idenlvl(idlvl)+graph_creation_function_name+"();");
if(timing) {
result.push_back(idenlvl(idlvl)+"ompDAGTimer.stop();");
result.push_back(idenlvl(idlvl)+"ompDAGTiming.creation=ompDAGTimer.elapsed_time();");
}
result.push_back(idenlvl(idlvl)+"std::vector<int> ompDAGGraphOrder,ompDAGGraphIndegree;");
if(timing)
result.push_back(idenlvl(idlvl)+"ompDAGTimer.start();");
result.push_back(idenlvl(idlvl)+"ompDAGGraphOrder.reserve("+graph_name+".v());");
if(timing) {
result.push_back(idenlvl(idlvl)+"ompDAGTimer.stop();");
result.push_back(idenlvl(idlvl)+"ompDAGTiming.order_allocation=ompDAGTimer.elapsed_time();");
}
result.push_back(idenlvl(idlvl)+"auto ompDAGGraphOrderPush=[&ompDAGGraphOrder](int v,decltype("+graph_name+")& g){ ompDAGGraphOrder.push_back(v); };");
if(timing)
result.push_back(idenlvl(idlvl)+"ompDAGTimer.start();");
result.push_back(idenlvl(idlvl)+"__ompDAG__::topologicalSort("+graph_name+",ompDAGGraphOrderPush,ompDAGGraphIndegree);");
if(timing) {
result.push_back(idenlvl(idlvl)+"ompDAGTimer.stop();");
result.push_back(idenlvl(idlvl)+"ompDAGTiming.topological_sort=ompDAGTimer.elapsed_time();");
}
auto thread_num=pragma_thread_num(pragmaAST,&pragma);
if(timing)
result.push_back(idenlvl(idlvl)+"ompDAGTimer.start();");
if(print&&timing)
tmp="__ompDAG__::OMPUserOrderDebug";
else
tmp="__ompDAG__::OMPUserOrder";
if(!thread_num.empty())
result.push_back(idenlvl(idlvl)+"__ompDAG__::ompGraph<"+tmp+">("+graph_name+",ompDAGTask,ompDAGGraphOrder,"+thread_num+");");
else
result.push_back(idenlvl(idlvl)+"__ompDAG__::ompGraph<"+tmp+">("+graph_name+",ompDAGTask,ompDAGGraphOrder);");
if(timing) {
result.push_back(idenlvl(idlvl)+"ompDAGTimer.stop();");
result.push_back(idenlvl(idlvl)+"ompDAGTiming.execution=ompDAGTimer.elapsed_time();");
}
if(timing&&print)
result.push_back(idenlvl(idlvl)+"ompDAGTimer.start();");
if(print) {
result.push_back(idenlvl(idlvl)+"std::ofstream ompDAGGraphFile=std::move(__ompDAG__::open_file<1>(\"omp_dag_graph"+std::to_string(graph_counter++)+".graphml\"));");
if(timing&&print)
result.push_back(idenlvl(idlvl)+"__ompDAG__::ompDAGGraphWriter<true>("+graph_name+",ompDAGGraphFile);");
else
result.push_back(idenlvl(idlvl)+"__ompDAG__::ompDAGGraphWriter("+graph_name+",ompDAGGraphFile);");
result.push_back(idenlvl(idlvl)+"__ompDAG__::close_file(ompDAGGraphFile);");
}
if(timing&&print) {
result.push_back(idenlvl(idlvl)+"ompDAGTimer.stop();");
result.push_back(idenlvl(idlvl)+"ompDAGTiming.printing=ompDAGTimer.elapsed_time();");
}
if(timing)
result.push_back(idenlvl(idlvl)+"ompDAGTiming.report(std::cerr);");
result.push_back(idenlvl(--idlvl)+"}");
return result;
}
int Preprocessor::preprocess_environment_pragmas(int offset,pragma_ast_node_t& pragma,clang_ast_node_t& block,bool timing,bool print) {
using namespace __pragma__;
int position=block.extent.begin.line-2+offset,lines=block.extent.end.line-block.extent.begin.line+2;
output_file.erase(output_file.begin()+position,output_file.begin()+position+lines);
std::string base_identation=identation(input_file[block.extent.begin.line-1]);
auto coarsening_opts=coarsening_options(pragmaAST,&pragma);
std::vector<std::string> result;
if((pragma_type(pragma)&ptok_t::COARSENING)==ptok_t::COARSENING)
switch(coarsening_opts.first) {
case ptok_t::BLOCK:
result=block_preprocess(pragma,block,coarsening_opts.second,base_identation,timing,print);
break;
default:
break;
}
output_file.insert(output_file.begin()+position,result.begin(),result.end());
return (static_cast<int>(result.size())-lines);
}
void Preprocessor::serial() {
std::string serial=__io__::toString(input_file);
serial=__util__::replace_pattern(serial,".*"+__pragma__::pragma_name_identifier+".*\\n","");
output_file=__io__::toLines(serial);
}
void Preprocessor::preprocess(std::string filename,bool timing,bool print,bool s) {
using namespace __io__;
if(!s) {
int offset=0;
for( auto pragma : environment_pragmas ) {
clang_ast_node_t* block=clang_translation_tables.second.at(pragma_translation_tables.first.at(pragma));
offset+=preprocess_environment_pragmas(offset,*pragma,*block,timing,print);
}
output_file.insert(output_file.begin(),"#include <omp_dag.hh>");
}
else
serial();
output_file_path=std::filesystem::path(filename);
if(std::filesystem::exists(output_file_path)) {
std::filesystem::path move_path(output_file_path.parent_path());
move_path/=output_file_path.filename().string()+".original";
std::filesystem::rename(output_file_path,move_path);
}
write(output_file,output_file_path.string());
}
}
