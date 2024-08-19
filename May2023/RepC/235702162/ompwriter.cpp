#include "clang/Driver/Options.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include <stack>
#include <map>
#include <vector>
#include <fstream>
#include "jsonParser.h"
using namespace std;
using namespace clang;
using namespace llvm;
Rewriter rewriter;
typedef struct relative_loop_inst_id {
string filename;
string functionName;
long long int functionLoopID;
long long int loopInstructionID;
} relative_loop_inst_id;
struct InputFile {
string filename;
map<Stmt*, bool> visited;
map<Stmt*, relative_loop_inst_id> mapStmtInstID;
map<string, Stmt*> mapInstIDStmt;
};
stack <struct InputFile> FileStack;
map <std::string, std::string> fileDir;
class PragmaVisitor : public RecursiveASTVisitor<PragmaVisitor> {
private:
ASTContext *astContext; 
MangleContext *mangleContext;
std::string json;
JSONParser *jsonFile;
public:
explicit PragmaVisitor(CompilerInstance *CI, std::string json) 
: astContext(&(CI->getASTContext())) { 
rewriter.setSourceMgr(astContext->getSourceManager(),
astContext->getLangOpts());
this->json = json;
}
std::string getSourceSnippet(SourceRange sourceRange, bool allTokens, bool jsonForm) {
if (!sourceRange.isValid())
return std::string();
SourceLocation bLoc(sourceRange.getBegin());
SourceLocation eLoc(sourceRange.getEnd());
const SourceManager& mng = astContext->getSourceManager();
std::pair<FileID, unsigned> bLocInfo = mng.getDecomposedLoc(bLoc);
std::pair<FileID, unsigned> eLocInfo = mng.getDecomposedLoc(eLoc);
FileID FID = bLocInfo.first;
unsigned bFileOffset = bLocInfo.second;
unsigned eFileOffset = eLocInfo.second;
int length = eFileOffset - bFileOffset;
if (length <= 0)
return std::string();
bool Invalid = false;
const char *BufStart = mng.getBufferData(FID, &Invalid).data();
if (Invalid)
return std::string();
if (allTokens == true) {
while (true) {
if (BufStart[(bFileOffset + length)] == ';')
break;
if (BufStart[(bFileOffset + length)] == '}')
break;
if (length == eFileOffset)
break;
length++;
}
}
if (length != eFileOffset)
length++;
std::string snippet = StringRef(BufStart + bFileOffset, length).trim().str();
snippet = replace_all(snippet, "\\", "\\\\");
snippet = replace_all(snippet, "\"", "\\\"");
if (jsonForm == true)
snippet = "\"" + replace_all(snippet, "\n", "\",\n\"") + "\"";
return snippet;
}
std::string replace_all(std::string str, std::string from, std::string to) {
int pos = 0;
while((pos = str.find(from, pos)) != std::string::npos) {
str.replace(pos, from.length(), to);
pos = pos + to.length();
}
return str;
}
void visitNodes(Stmt *st, vector<Stmt*> & nodes_list) {
if (!st)
return;
nodes_list.push_back(st);
if (CapturedStmt *CPTSt = dyn_cast<CapturedStmt>(st)) {
visitNodes(CPTSt->getCapturedStmt(), nodes_list);
return;
}
for (auto I = st->child_begin(), IE = st->child_end(); I != IE; I++) {
visitNodes((*I)->IgnoreContainers(true), nodes_list);
}
}
void recoverCodeSnippetsID(Stmt *st, std::string functionName, long long int loopID) {
if (!st)
return;
map<Stmt*, long long int> mapped_statments;
string snippet = getSourceSnippet(st->getSourceRange(), true, false);
vector<pair<int, int> > separator_ref;
int line = 0;
int column = 0;
for (int i = 0, ie = snippet.size(); i < ie; i++) {
if (snippet[i] == ';')
separator_ref.push_back(make_pair(line, column));
column++;
if (snippet[i] == '\n') {
line++;
column = 0;
}
}
if (snippet[(snippet.size() - 1)] != ';')
separator_ref.push_back(make_pair(line, column));
vector<Stmt*> nodes_list;
visitNodes(st, nodes_list);
for (int i = 0, ie = nodes_list.size(); i != ie; i++) {
FullSourceLoc StartLocation = astContext->getFullLoc(nodes_list[i]->getBeginLoc());
FullSourceLoc EndLocation = astContext->getFullLoc(nodes_list[i]->getEndLoc());
if (!StartLocation.isValid() || !EndLocation.isValid()) 
continue;
int line  = EndLocation.getSpellingLineNumber();
int column = EndLocation.getSpellingColumnNumber();
mapped_statments[nodes_list[i]] = -1;
for (int j = 0, je = separator_ref.size(); j != je; j++) {
if (separator_ref[i].first == line) {
if (separator_ref[i].second >= column) {
mapped_statments[nodes_list[i]] = j + 1;
break;
}
}
else if (separator_ref[i].first > line) {
mapped_statments[nodes_list[i]] = j + 1;
break;
}
}
if (mapped_statments[nodes_list[i]] == -1)
mapped_statments[nodes_list[i]] = separator_ref.size();
struct InputFile& currFile = FileStack.top();
currFile.mapStmtInstID[nodes_list[i]].filename = currFile.filename;
currFile.mapStmtInstID[nodes_list[i]].functionName = functionName;
currFile.mapStmtInstID[nodes_list[i]].functionLoopID = loopID;
currFile.mapStmtInstID[nodes_list[i]].loopInstructionID = mapped_statments[nodes_list[i]];
std::string index = std::string();
index += currFile.filename + "," + functionName + ",";
index += std::to_string(loopID) + "," + std::to_string(mapped_statments[nodes_list[i]]);
if (currFile.mapInstIDStmt.count(index) == 0) {
currFile.mapInstIDStmt[index] = nodes_list[i];
if (jsonFile->directives.count(index) > 0) {
rewriter.InsertText(st->getBeginLoc(),jsonFile->directives[index], true, true);
rewriter.overwriteChangedFiles();
}
}
currFile.mapInstIDStmt[index] = nodes_list[i];
}
}
void NewInputFile(string filename) {
struct InputFile newfile;
filename.erase(filename.begin(), filename.begin() + filename.rfind("/") + 1);
newfile.filename = filename;
FileStack.push(newfile);
}
virtual bool VisitDecl(Decl *D) {
if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
if (FD->doesThisDeclarationHaveABody()) {
const SourceManager& mng = astContext->getSourceManager();
if (astContext->getSourceManager().isInSystemHeader(D->getLocation())) {
return true;
}
string dirname = mng.getFilename(D->getBeginLoc());
if (dirname.rfind("/") == std::string::npos)
return true;
dirname.erase(dirname.begin() + dirname.rfind("/"), dirname.end());
if ((this->json != std::string()) && (dirname != std::string())) {
string jsonName = dirname + "/" + this->json;
jsonFile = new JSONParser(jsonName);
}
string filename = mng.getFilename(D->getBeginLoc());
if (FileStack.empty() || FileStack.top().filename != filename) {
NewInputFile(filename);
}
struct InputFile& currFile = FileStack.top();
vector<Stmt*> nodes_list;
visitNodes(FD->getBody(), nodes_list);
string functionName = FD->getNameInfo().getName().getAsString();
currFile.mapStmtInstID[FD->getBody()].filename = currFile.filename;
currFile.mapStmtInstID[FD->getBody()].functionName = functionName;
currFile.mapStmtInstID[FD->getBody()].functionLoopID = -1;
currFile.mapStmtInstID[FD->getBody()].loopInstructionID = -1;
std::string index = std::string();
index += currFile.filename + "," + functionName;
currFile.mapInstIDStmt[index] = FD->getBody();
map<int, Stmt*> loops;
for (int i = 0, ie = nodes_list.size(); i != ie; i++) {
if (isa<DoStmt>(nodes_list[i]) || isa<ForStmt>(nodes_list[i]) || isa<WhileStmt>(nodes_list[i])) {
FullSourceLoc StartLocation = astContext->getFullLoc(nodes_list[i]->getBeginLoc());
FullSourceLoc EndLocation = astContext->getFullLoc(nodes_list[i]->getEndLoc());
if (!StartLocation.isValid() || !EndLocation.isValid()) 
continue;
int line  = StartLocation.getSpellingLineNumber();
loops[line] = nodes_list[i];
}
}
int id = 0;
for (map<int, Stmt*>::iterator I = loops.begin(), IE = loops.end(); I != IE; I++) {
id++;
Stmt *st = nullptr;
if (ForStmt *forst = dyn_cast<ForStmt>(I->second))
st = forst;
if (DoStmt *dost = dyn_cast<DoStmt>(I->second))
st = dost;
if (WhileStmt *whst = dyn_cast<WhileStmt>(I->second))
st = whst;
currFile.mapStmtInstID[st].filename = currFile.filename;
currFile.mapStmtInstID[st].functionName = functionName;
currFile.mapStmtInstID[st].functionLoopID = id;
currFile.mapStmtInstID[st].loopInstructionID = -1;
std::string index = std::string();
index += currFile.filename + "," + functionName + "," + std::to_string(id);
if (currFile.mapInstIDStmt.count(index) == 0) {
currFile.mapInstIDStmt[index] = I->second;
if (jsonFile->directives.count(index) > 0) {
rewriter.InsertText(st->getBeginLoc(),jsonFile->directives[index], true, true);
rewriter.overwriteChangedFiles();
}
}
if (ForStmt *forst = dyn_cast<ForStmt>(I->second))
st = forst->getBody();
if (DoStmt *dost = dyn_cast<DoStmt>(I->second))
st = dost->getBody();
if (WhileStmt *whst = dyn_cast<WhileStmt>(I->second))
st = whst->getBody();
if (CompoundStmt *cmps = dyn_cast<CompoundStmt>(st))
st = cmps->body_front();
recoverCodeSnippetsID(st, functionName, id);
}
}
}
return true;
}
int getLineForAbstractHandle(std::string index) {
struct InputFile& currFile = FileStack.top();
FullSourceLoc StartLocation = astContext->getFullLoc(currFile.mapInstIDStmt[index]->getBeginLoc());
FullSourceLoc EndLocation = astContext->getFullLoc(currFile.mapInstIDStmt[index]->getEndLoc());
return StartLocation.getSpellingLineNumber();
}
};
class PragmaASTConsumer : public ASTConsumer {
private:
PragmaVisitor *visitor; 
public:
explicit PragmaASTConsumer(CompilerInstance *CI, std::string json)
: visitor(new PragmaVisitor(CI, json)) 
{ }
virtual void HandleTranslationUnit(ASTContext &Context) {
visitor->TraverseDecl(Context.getTranslationUnitDecl());
}
};
class PragmaPluginAction : public PluginASTAction {
protected:
std::string json = std::string();
unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, 
StringRef file) {
return make_unique<PragmaASTConsumer>(&CI, this->json);
}
bool ParseArgs(const CompilerInstance &CI, const vector<string> &args) {
for (unsigned i = 0, e = args.size(); i != e; ++i) {
if (args[i].find("-write-json=") == 0) {
this->json = args[i];
this->json.erase(this->json.begin(), this->json.begin() + 12);
}
}
return true;
}
};
static FrontendPluginRegistry::Add<PragmaPluginAction> X
("-write-omp", "OMP Writer");
