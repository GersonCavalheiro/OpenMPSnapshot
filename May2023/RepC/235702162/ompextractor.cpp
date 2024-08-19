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
using namespace std;
using namespace clang;
using namespace llvm;
Rewriter rewriter;
struct Node {
string name;
unsigned int id;
unsigned int sline, scol;
unsigned int eline, ecol;
};
typedef struct relative_loop_inst_id {
string filename;
string functionName;
long long int functionLoopID;
long long int loopInstructionID;
} relative_loop_inst_id;
struct InputFile {
string filename;
string labels;
map<Stmt*, bool> visited;
map<Stmt*, bool> isInsideTargetRegion;
map<Stmt*, string> mapFunctionName;
map<string, map<Stmt*, long long int> > functionLoopID;
map<Stmt*, map<Stmt*, long long int> > loopInstructionID;
map<Stmt*, relative_loop_inst_id> loopInstID;
stack <struct Node> NodeStack;
};
stack <struct InputFile> FileStack;
long long int opCount = 0;
class PragmaVisitor : public RecursiveASTVisitor<PragmaVisitor> {
private:
ASTContext *astContext; 
MangleContext *mangleContext;
bool ClDCSnippet;
public:
explicit PragmaVisitor(CompilerInstance *CI, bool ClDCSnippet) 
: astContext(&(CI->getASTContext())) { 
rewriter.setSourceMgr(astContext->getSourceManager(),
astContext->getLangOpts());
this->ClDCSnippet = ClDCSnippet;
}
void CreateLoopNode(Stmt *st) {
struct Node N;
struct InputFile& currFile = FileStack.top();
FullSourceLoc StartLocation = astContext->getFullLoc(st->getBeginLoc());
FullSourceLoc EndLocation = astContext->getFullLoc(st->getEndLoc());
if (!StartLocation.isValid() || !EndLocation.isValid() || (currFile.visited.count(st) != 0)) {
N.sline = -1;
return;
}
currFile.visited[st] = true;
Stmt *body = nullptr;
if (ForStmt *forst = dyn_cast<ForStmt>(st))
body = forst->getBody();
if (DoStmt *dost = dyn_cast<DoStmt>(st))
body = dost->getBody();
if (WhileStmt *whst = dyn_cast<WhileStmt>(st))
body = whst->getBody();
std:string snippet = getSourceSnippet(body->getSourceRange(), true, true);
N.id = currFile.functionLoopID[currFile.mapFunctionName[st]][st];
N.sline = StartLocation.getSpellingLineNumber();
N.scol = StartLocation.getSpellingColumnNumber();
N.eline = EndLocation.getSpellingLineNumber();
N.ecol = EndLocation.getSpellingColumnNumber();
N.name = st->getStmtClassName() + to_string(N.id);
currFile.labels += "\"loop - object id : " + to_string(opCount++) + "\":{\n";
currFile.labels += "\"file\":\"" + currFile.filename + "\",\n";
currFile.labels += "\"function\":\"" + currFile.mapFunctionName[st] + "\",\n";
currFile.labels += "\"loop id\":\"" + to_string(N.id) + "\",\n";
currFile.labels += "\"loop line\":\"" + to_string(N.sline) + "\",\n";
currFile.labels += "\"loop column\":\"" + to_string(N.scol) + "\",\n";
currFile.labels += "\"pragma type\":\"NULL\",\n";
currFile.labels += "\"ordered\":\"false\",\n";
currFile.labels += "\"offload\":\"false\",\n";
currFile.labels += "\"multiversioned\":\"false\"";
if (ClDCSnippet == true)
currFile.labels += ",\n\"code snippet\":[" + snippet + "]";
currFile.labels += "\n},\n";
}
string classifyPragma(OMPExecutableDirective *OMPLD, bool insideParallelRegion) {
if (isa<OMPDistributeDirective>(OMPLD)) {
return "distribute";
}
else if (isa<OMPDistributeParallelForDirective>(OMPLD)) {
return "distribute parallel for";
}
else if (isa<OMPDistributeParallelForSimdDirective>(OMPLD)) {
return "distribute parallel for smid";
}
else if (isa<OMPDistributeSimdDirective>(OMPLD)) {
return "distribute simd";
}
else if (isa<OMPForDirective>(OMPLD)) {
if (insideParallelRegion)
return "parallel for";
return "for";
}
else if (isa<OMPForSimdDirective>(OMPLD)) {
if (insideParallelRegion)
return "parallel for simd";
return "for simd";
}
else if (isa<OMPParallelForDirective>(OMPLD)) {
return "parallel for";
}
else if (isa<OMPParallelForSimdDirective>(OMPLD)) {
return "parallel for simd";
}
else if (isa<OMPSimdDirective>(OMPLD)) {
return "simd";
}
else if (isa<OMPTargetParallelForDirective>(OMPLD)) {
return "target parallel for";
}
else if (isa<OMPTargetParallelForSimdDirective>(OMPLD)) {
return "target parallel for simd";
}
else if (isa<OMPTargetSimdDirective>(OMPLD)) {
return "target simd";
}
else if (isa<OMPTargetTeamsDistributeDirective>(OMPLD)) {
return "target teams ditribute";
}
else if (isa<OMPTargetTeamsDistributeParallelForDirective>(OMPLD)) {
return "target teams distribute parallel for";
}
else if (isa<OMPTargetTeamsDistributeParallelForSimdDirective>(OMPLD)) {
return "target teams ditribute parallel for simd";
}
else if (isa<OMPTargetTeamsDistributeSimdDirective>(OMPLD)) {
return "target teams ditribute simd";
}
else if (isa<OMPTaskLoopDirective>(OMPLD)) {
return "taskloop";
}
else if (isa<OMPTaskLoopSimdDirective>(OMPLD)) {
return "taskloop simd";
}
else if (isa<OMPTeamsDistributeDirective>(OMPLD)) {
return "teams ditribute";
}
else if (isa<OMPTeamsDistributeParallelForDirective>(OMPLD)) {
return "teams ditribute parallel for";
}
else if (isa<OMPTeamsDistributeParallelForSimdDirective>(OMPLD)) {
return "teams ditribute parallel for simd";
}
else if (isa<OMPTeamsDistributeSimdDirective>(OMPLD)) {
return "teams ditribute simd";
}
else if (isa<OMPTargetDataDirective>(OMPLD)) {
return "target data";
}
return string();
}
bool isTargetDirective(OMPExecutableDirective *OMPLD) {
if (isa<OMPTargetParallelForDirective>(OMPLD) ||
isa<OMPTargetParallelForSimdDirective>(OMPLD) ||
isa<OMPTargetTeamsDistributeDirective>(OMPLD) ||
isa<OMPTargetTeamsDistributeParallelForDirective>(OMPLD) ||
isa<OMPTargetTeamsDistributeParallelForSimdDirective>(OMPLD) ||
isa<OMPTargetTeamsDistributeSimdDirective>(OMPLD) ||
isa<OMPTargetParallelDirective>(OMPLD) ||
isa<OMPTargetTeamsDirective>(OMPLD) ||
isa<OMPTargetUpdateDirective>(OMPLD) ||
isa<OMPTargetDirective>(OMPLD))
return true;
return false;
}
string getStrForStmt(Stmt *st) {
if (!st) {
return string();
}
if (DeclRefExpr *DRex = dyn_cast<DeclRefExpr>(st)) {
return DRex->getFoundDecl()->getNameAsString();
}
if (IntegerLiteral *IL = dyn_cast<IntegerLiteral>(st)) {
return to_string((int) IL->getValue().roundToDouble());
}
if (OMPArraySectionExpr *OMPcl = dyn_cast<OMPArraySectionExpr>(st)) {
std::string offsets = getStrForStmt(OMPcl->getBase()->IgnoreCasts());
offsets += "[" + getStrForStmt(OMPcl->getLowerBound()->IgnoreImpCasts()) + ":";
offsets += getStrForStmt(OMPcl->getLength()->IgnoreImpCasts()) + "]";
return offsets;
}
if (ArraySubscriptExpr *ASExp = dyn_cast<ArraySubscriptExpr>(st)) {
std::string offsets = getStrForStmt(ASExp->getBase()->IgnoreImpCasts());
offsets += "[" + getStrForStmt(ASExp->getIdx()->IgnoreImpCasts()) + "]";
return offsets;
}
if (ConstantExpr *ConstExp = dyn_cast<ConstantExpr>(st)) {
return getStrForStmt(ConstExp->getSubExpr());
}
if (UnaryOperator *Uop = dyn_cast<UnaryOperator>(st)) {
return getStrForStmt(Uop->getSubExpr());
}
return string();
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
std::string recoverOperandsForClause(OMPClause *clause) {
if (OMPReductionClause *OMPcl = dyn_cast<OMPReductionClause>(clause)) {
std::string op = OMPcl->getNameInfo().getName().getAsString();
if (op.size() > 8)
op.erase(op.begin(), op.begin() + 8);
return (op + ":");
}
return std::string();
}
void recoverClause(OMPClause *clause, std::string clause_type, map<string, string> & clauses,
MutableArrayRef<Expr *>::iterator list_begin,  MutableArrayRef<Expr *>::iterator list_end) {
clauses[clause_type] = std::string();
std::string operands = recoverOperandsForClause(clause);
for (MutableArrayRef<Expr *>::iterator I = list_begin, IE = list_end; I != IE; I++) {
if (Stmt *Nmdc = dyn_cast<Stmt>(*I)) {
clauses[clause_type] += "\"" + operands + getStrForStmt(Nmdc) + "\",";
}
}
if (clauses[clause_type].size() > 0) {
clauses[clause_type].erase(clauses[clause_type].end()-1, clauses[clause_type].end());
}
}
void ClassifyClause(OMPClause *clause, map<string, string> & clauseType) {
if (clause->isImplicit())
return;
if (isa<OMPIfClause>(clause) ||
isa<OMPFinalClause>(clause)) {
clauseType["multiversioned"] = "true";
return;
}
if (OMPCollapseClause *OMPCc = dyn_cast<OMPCollapseClause>(clause)) {
clauseType["collapse"] = getStrForStmt(OMPCc->getNumForLoops());
}
if (OMPOrderedClause *OMPcl = dyn_cast<OMPOrderedClause>(clause)) {
clauseType["ordered"] = "true";
}
if (OMPPrivateClause *OMPcl = dyn_cast<OMPPrivateClause>(clause))
recoverClause(clause, "private", clauseType, OMPcl->varlist_begin(), OMPcl->varlist_end());
if (OMPSharedClause *OMPcl = dyn_cast<OMPSharedClause>(clause))
recoverClause(clause, "shared", clauseType, OMPcl->varlist_begin(), OMPcl->varlist_end());
if (OMPFirstprivateClause *OMPcl = dyn_cast<OMPFirstprivateClause>(clause))
recoverClause(clause, "firstprivate", clauseType, OMPcl->varlist_begin(), OMPcl->varlist_end());
if (OMPLastprivateClause *OMPcl = dyn_cast<OMPLastprivateClause>(clause))
recoverClause(clause, "lastprivate", clauseType, OMPcl->varlist_begin(), OMPcl->varlist_end());
if (OMPLinearClause *OMPcl = dyn_cast<OMPLinearClause>(clause))
recoverClause(clause, "linear", clauseType, OMPcl->varlist_begin(), OMPcl->varlist_end());
if (OMPReductionClause *OMPcl = dyn_cast<OMPReductionClause>(clause)) 
recoverClause(clause, "reduction", clauseType, OMPcl->varlist_begin(), OMPcl->varlist_end());
if (OMPMapClause *OMPcl = dyn_cast<OMPMapClause>(clause)) {
std::string index = "map" + std::to_string(OMPcl->getMapType());
recoverClause(clause, index, clauseType, OMPcl->varlist_begin(), OMPcl->varlist_end());
}
}
void CreateLoopDirectiveNode(Stmt *stmt, map<string, string> clauseType) {
struct Node N;
struct InputFile& currFile = FileStack.top();
Stmt *st = stmt;
std::string inductionVar = std::string();
if (OMPExecutableDirective *OMPLD = dyn_cast<OMPExecutableDirective>(stmt))
st = OMPLD->getInnermostCapturedStmt()->getCapturedStmt();
if (isa<DoStmt>(st) || isa<ForStmt>(st) || isa<WhileStmt>(st)) {
if (ForStmt *fstmt = dyn_cast<ForStmt>(st)) {
if (UnaryOperator *unop = dyn_cast<UnaryOperator>(fstmt->getInc())) {
inductionVar = getStrForStmt(unop);
}
if (BinaryOperator *biop = dyn_cast<BinaryOperator>(fstmt->getInc())) {
inductionVar = getStrForStmt(biop->getLHS());
}
errs() << "IND var =>> " << inductionVar << "\n";
}
}
else
return;
FullSourceLoc StartLocation = astContext->getFullLoc(st->getBeginLoc());
FullSourceLoc EndLocation = astContext->getFullLoc(st->getEndLoc());
if (!StartLocation.isValid() || !EndLocation.isValid() || (currFile.visited.count(st) != 0)) {
return;
}
currFile.visited[st] = true;
Stmt *body = nullptr;
if (ForStmt *forst = dyn_cast<ForStmt>(st))
body = forst->getBody();
if (DoStmt *dost = dyn_cast<DoStmt>(st))
body = dost->getBody();
if (WhileStmt *whst = dyn_cast<WhileStmt>(st))
body = whst->getBody();
std:string snippet = getSourceSnippet(body->getSourceRange(), true, true);
N.id = currFile.functionLoopID[currFile.mapFunctionName[st]][st];
N.sline = StartLocation.getSpellingLineNumber();
N.scol = StartLocation.getSpellingColumnNumber();
N.eline = EndLocation.getSpellingLineNumber();
N.ecol = EndLocation.getSpellingColumnNumber();
N.name = st->getStmtClassName() + to_string(N.id);
if (OMPExecutableDirective *OMPED = dyn_cast<OMPExecutableDirective>(stmt))
clauseType["pragma type"] = classifyPragma(OMPED, (clauseType.count("parallel") > 0) == true);
currFile.labels += "\"loop - object id : " + to_string(opCount++) + "\":{\n";
currFile.labels += "\"file\":\"" + currFile.filename + "\",\n";
currFile.labels += "\"function\":\"" + currFile.mapFunctionName[st] + "\",\n";
currFile.labels += "\"loop id\":\"" + to_string(N.id) + "\",\n";
currFile.labels += "\"loop line\":\"" + to_string(N.sline) + "\",\n";
currFile.labels += "\"loop column\":\"" + to_string(N.scol) + "\",\n";
currFile.labels += "\"pragma type\":\"" + clauseType["pragma type"] + "\",\n";
currFile.labels += "\"ordered\":\"" + ((clauseType.count("ordered") > 0) ? (clauseType["ordered"]) : "false") + "\",\n";
currFile.labels += "\"offload\":\"" + ((clauseType.count("offload") > 0) ? (clauseType["offload"]) : "false") + "\",\n";
currFile.labels += "\"multiversioned\":\""+ ((clauseType.count("multiversioned") > 0) ? (clauseType["multiversioned"]) : "false") + "\"";
if (inductionVar != std::string())
currFile.labels += ",\n\"induction variable\":\"" + inductionVar + "\"";
if (clauseType.count("shared") > 0)
currFile.labels += ",\n\"shared\":[" + ((clauseType.count("shared") > 0) ? (clauseType["shared"]) : "") + "]";
if (clauseType.count("private") > 0)
currFile.labels += ",\n\"private\":[" + ((clauseType.count("private") > 0) ? (clauseType["private"]) : "") + "]";
if (clauseType.count("firstprivate") > 0)
currFile.labels += ",\n\"firstprivate\":[" + ((clauseType.count("firstprivate") > 0) ? (clauseType["firstprivate"]) : "") + "]";
if (clauseType.count("lastprivate") > 0)
currFile.labels += ",\n\"lastprivate\":[" + ((clauseType.count("lastprivate") > 0) ? (clauseType["lastprivate"]) : "") + "]";
if (clauseType.count("linear") > 0)
currFile.labels += ",\n\"linear\":[" + ((clauseType.count("linear") > 0) ? (clauseType["linear"]) : "") + "]";
if (clauseType.count("reduction") > 0)
currFile.labels += ",\n\"reduction\":[" + ((clauseType.count("reduction") > 0) ? (clauseType["reduction"]) : "") + "]";
if (clauseType.count("map1") > 0)
currFile.labels += ",\n\"map to\":[" + ((clauseType.count("map1") > 0) ? (clauseType["map1"]) : "") + "]";
if (clauseType.count("map2") > 0)
currFile.labels += ",\n\"map from\":[" + ((clauseType.count("map2") > 0) ? (clauseType["map2"]) : "") + "]";
if (clauseType.count("map3") > 0)
currFile.labels += ",\n\"map tofrom\":[" + ((clauseType.count("map3") > 0) ? (clauseType["map3"]) : "") + "]";
if (clauseType.count("dependence list") > 0)
currFile.labels += ",\n\"dependence list\":[" + ((clauseType.count("dependence list") > 0) ? (clauseType["dependence list"]) : "") + "]";
if (ClDCSnippet == true)
currFile.labels += ",\n\"code snippet\":[" + snippet + "]";
currFile.labels += "\n},\n";
}
void NewInputFile(string filename) {
struct InputFile newfile;
struct Node root;
newfile.filename = filename;
FileStack.push(newfile);
root.id = ++opCount;
root.name = filename;
root.sline = 0;
root.scol = 0;
root.eline = ~0;
root.ecol = ~0;
FileStack.top().NodeStack.push(root);
}      
std::string replace_all(std::string str, std::string from, std::string to) {
int pos = 0;
while((pos = str.find(from, pos)) != std::string::npos) {
str.replace(pos, from.length(), to);
pos = pos + to.length();
}
return str;
}
std::string getSourceSnippet(SourceRange sourceRange, bool allTokens, bool jsonForm) {
SourceLocation bLoc(sourceRange.getBegin());
SourceLocation eLoc(sourceRange.getEnd());
const SourceManager& mng = astContext->getSourceManager();
std::pair<FileID, unsigned> bLocInfo = mng.getDecomposedLoc(bLoc);
std::pair<FileID, unsigned> eLocInfo = mng.getDecomposedLoc(eLoc);
FileID FID = bLocInfo.first;
unsigned bFileOffset = bLocInfo.second;
unsigned eFileOffset = eLocInfo.second;
unsigned length = eFileOffset - bFileOffset;
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
length++;
}
}
length++;
if (ClDCSnippet == false)
return std::string(); 
std::string snippet = StringRef(BufStart + bFileOffset, length).trim().str();
snippet = replace_all(snippet, "\\", "\\\\");
snippet = replace_all(snippet, "\"", "\\\"");
if (jsonForm == true)
snippet = "\"" + replace_all(snippet, "\n", "\",\n\"") + "\"";
return snippet;
}
void insertStmtDirectives(Stmt *st, std::string directive, std::string snippet, map<string, string> & clauses) {
struct InputFile& currFile = FileStack.top();
FullSourceLoc StartLocation = astContext->getFullLoc(st->getBeginLoc());
FullSourceLoc EndLocation = astContext->getFullLoc(st->getEndLoc());
if (!StartLocation.isValid() || !EndLocation.isValid()) {
return;
}
currFile.labels += "\"" + directive + " - object id : " + std::to_string(opCount++) + "\":{\n";
currFile.labels += "\"pragma type\":\"" + directive + "\",\n";
currFile.labels += "\"file\":\"" + currFile.loopInstID[st].filename + "\",\n";
currFile.labels += "\"function\":\"" + currFile.loopInstID[st].functionName  + "\",\n";
currFile.labels += "\"loop id\":\"" + to_string(currFile.loopInstID[st].functionLoopID) + "\",\n";
currFile.labels += "\"statement id\":\"" + to_string(currFile.loopInstID[st].loopInstructionID) + "\",\n";
currFile.labels += "\"snippet line\":\"" + to_string(StartLocation.getSpellingLineNumber()) + "\",\n";
currFile.labels += "\"snippet column\":\"" + to_string(StartLocation.getSpellingColumnNumber()) + "\"";
if (ClDCSnippet == true)
currFile.labels += ",\n\"code snippet\":[" + snippet + "]";
currFile.labels += "\n},\n";
if (clauses.count("dependence list") == 0)
clauses["dependence list"] = "\"" + (directive + " - object id : " + std::to_string((opCount - 1))) + "\"";
else
clauses["dependence list"] += ",\"" + (directive + " - object id : " + std::to_string((opCount - 1))) + "\"";
}
void associateEachLoopInside(OMPExecutableDirective *OMPED, map<string, string> & clauses) {
struct InputFile& currFile = FileStack.top();
vector<Stmt*> nodes_list;
visitNodes(OMPED, nodes_list);
if (currFile.visited.count(OMPED) != 0)
return;
currFile.visited[OMPED] = true;
if (isTargetDirective(OMPED))
clauses["offload"] = "true";
if (isa<OMPParallelDirective>(OMPED)) 
clauses["parallel"] = "true";
if (isa<OMPOrderedDirective>(OMPED)) {
const SourceManager& mng = astContext->getSourceManager();
std::string snippet = std::string();
if (ClDCSnippet == true)
snippet = getSourceSnippet(OMPED->getInnermostCapturedStmt()->getSourceRange(), true, true);
insertStmtDirectives(OMPED, "ordered", snippet, clauses);
}
if (OMPAtomicDirective * OMPAD = dyn_cast<OMPAtomicDirective>(OMPED)) {
std::string snippet = std::string();
if (ClDCSnippet == true)
snippet = getSourceSnippet(OMPAD->getInnermostCapturedStmt()->getSourceRange(), true, true);
if (OMPAD->getNumClauses() > 0) {
if (isa<OMPCaptureClause>(OMPED->getClause(0)))
insertStmtDirectives(OMPAD, "atomic capture", snippet, clauses);
else if (isa<OMPWriteClause>(OMPED->getClause(0)))
insertStmtDirectives(OMPAD, "atomic write", snippet, clauses);
else if (isa<OMPReadClause>(OMPED->getClause(0)))
insertStmtDirectives(OMPAD, "atomic read", snippet, clauses);
else if (isa<OMPUpdateClause>(OMPED->getClause(0)))
insertStmtDirectives(OMPAD, "atomic update", snippet, clauses);
}
else
insertStmtDirectives(OMPAD, "atomic", snippet, clauses);
}
clauses["pragma type"] = classifyPragma(OMPED, (clauses.count("parallel") > 0) == true);
if (isa<OMPTargetDataDirective>(OMPED) ||
isa<OMPTargetEnterDataDirective>(OMPED) ||
isa<OMPTargetExitDataDirective>(OMPED))
clauses["offload"] = "false";
for (int i = 0, ie = OMPED->getNumClauses(); i != ie; i++)
ClassifyClause(OMPED->getClause(i), clauses);
for (int i = 0, ie = nodes_list.size(); i != ie; i++) {
if (OMPOrderedDirective *OMPOD = dyn_cast<OMPOrderedDirective>(nodes_list[i]))
associateEachLoopInside(OMPOD, clauses);
if (OMPAtomicDirective *OMPAD = dyn_cast<OMPAtomicDirective>(nodes_list[i]))
associateEachLoopInside(OMPAD, clauses);
}
if ((clauses.count("collapse") != 0) ||
(clauses.count("offload") != 0) ||
(clauses.count("parallel") != 0) ||
(isa<OMPTargetDataDirective>(OMPED) ||
isa<OMPTargetEnterDataDirective>(OMPED) ||
isa<OMPTargetExitDataDirective>(OMPED))) {
if (clauses.count("collapse") != 0)
CreateLoopDirectiveNode(OMPED, clauses);
for (int i = 0, ie = nodes_list.size(); i != ie; i++) {
if (currFile.visited.count(nodes_list[i]) != 0) 
continue;
if (clauses.count("collapse") != 0) {
if (isa<DoStmt>(nodes_list[i]) || isa<ForStmt>(nodes_list[i]) || isa<WhileStmt>(nodes_list[i])) {
clauses["collapse"] = std::to_string(std::stoi(clauses["collapse"]) - 1);
CreateLoopDirectiveNode(nodes_list[i], clauses);
if (clauses["collapse"] == "1")
break;      
}
}
if (OMPExecutableDirective *OMPEN = dyn_cast<OMPExecutableDirective>(nodes_list[i])) {
if (OMPLoopDirective *OMPLD = dyn_cast<OMPLoopDirective>(OMPEN)) {
associateEachLoopInside(OMPLD, clauses);
}
else if (OMPTargetDataDirective *OMPTD = dyn_cast<OMPTargetDataDirective>(OMPEN)) {
associateEachLoopInside(OMPED, clauses); 
}
else if (OMPParallelDirective *OMPPD = dyn_cast<OMPParallelDirective>(OMPEN)) {
associateEachLoopInside(OMPPD, clauses);
}
else if (OMPTargetDirective *OMPTD = dyn_cast<OMPTargetDirective>(OMPEN)) {
associateEachLoopInside(OMPTD, clauses);
}
}
}
}
if (OMPLoopDirective *OMPLD = dyn_cast<OMPLoopDirective>(OMPED)) {
CreateLoopDirectiveNode(OMPLD, clauses);
}
}
void recoverCodeSnippetsID(Stmt *st, map<Stmt*, long long int> & mapped_statments, long long int loopID) {
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
currFile.loopInstID[nodes_list[i]].filename = currFile.filename;
string functionName = currFile.mapFunctionName[nodes_list[i]];
currFile.loopInstID[nodes_list[i]].functionName = functionName;
currFile.loopInstID[nodes_list[i]].functionLoopID = loopID;
currFile.loopInstID[nodes_list[i]].loopInstructionID = mapped_statments[nodes_list[i]];
}
}
virtual bool VisitDecl(Decl *D) {
struct InputFile& currFile = FileStack.top();
if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
if (FD->doesThisDeclarationHaveABody()) {
const SourceManager& mng = astContext->getSourceManager();
if (astContext->getSourceManager().isInSystemHeader(D->getLocation())) {
return true;
}
string filename = mng.getFilename(D->getBeginLoc());
if (FileStack.empty() || FileStack.top().filename != filename) {
NewInputFile(filename);
}
struct InputFile& currFile = FileStack.top();
vector<Stmt*> nodes_list;
visitNodes(FD->getBody(), nodes_list);
string funcName = FD->getNameInfo().getName().getAsString();
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
currFile.mapFunctionName[nodes_list[i]] = funcName;
}
int id = 1;
for (map<int, Stmt*>::iterator I = loops.begin(), IE = loops.end(); I != IE; I++) {
currFile.functionLoopID[funcName][I->second] = id++;
int idInst = 1;
Stmt *st = nullptr;
if (ForStmt *forst = dyn_cast<ForStmt>(I->second))
st = forst->getBody();
if (DoStmt *dost = dyn_cast<DoStmt>(I->second))
st = dost->getBody();
if (WhileStmt *whst = dyn_cast<WhileStmt>(I->second))
st = whst->getBody();
recoverCodeSnippetsID(st, currFile.loopInstructionID[st], currFile.functionLoopID[funcName][I->second]);
}
}
}
return true;
}
virtual bool VisitStmt(Stmt *st) {
const SourceManager& mng = astContext->getSourceManager();
if ((st->getBeginLoc().isInvalid()) || 
(mng.isInSystemHeader(st->getBeginLoc()))) {
return true;
} 
if (!isa<OMPExecutableDirective>(st) && !isa<DoStmt>(st) 
&& !isa<ForStmt>(st) && !isa<WhileStmt>(st)) {
return true;
}
if (OMPExecutableDirective *OMPED = dyn_cast<OMPExecutableDirective>(st)) {
map<string, string> clauses;
associateEachLoopInside(OMPED, clauses);
}
if (isa<DoStmt>(st) || isa<ForStmt>(st) || isa<WhileStmt>(st)) {
CreateLoopNode(st);
}
return true;
}
};
class PragmaASTConsumer : public ASTConsumer {
private:
PragmaVisitor *visitor; 
public:
explicit PragmaASTConsumer(CompilerInstance *CI, bool ClDCSnippet)
: visitor(new PragmaVisitor(CI, ClDCSnippet)) 
{ }
void EmptyStack() {
while (!FileStack.empty()) {
FileStack.pop();
}
}
bool writeJsonToFile() {
struct InputFile& currFile = FileStack.top(); 
ofstream outfile;
if (currFile.filename.empty()) {
return false;
}
outfile.open(currFile.filename + ".json");
if (!outfile.is_open()) {
return false;
}
if (currFile.labels.size() >= 2)
currFile.labels.erase(currFile.labels.end() - 2, currFile.labels.end());
outfile << "{\n";
outfile << currFile.labels << "\n}";
return true;
}
virtual void HandleTranslationUnit(ASTContext &Context) {
visitor->TraverseDecl(Context.getTranslationUnitDecl());
while (!FileStack.empty()) {
if (writeJsonToFile()) {
errs() << "Pragma info for file " << FileStack.top().filename;
errs() << " written successfully!\n";
}
else {
errs() << "Failed to write dot file for input file: ";
errs() << FileStack.top().filename << "\n";
}
FileStack.pop();
} 
}
};
class PragmaPluginAction : public PluginASTAction {
protected:
bool ClDCSnippet = false;
unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, 
StringRef file) {
return make_unique<PragmaASTConsumer>(&CI, this->ClDCSnippet);
}
bool ParseArgs(const CompilerInstance &CI, const vector<string> &args) {
for (unsigned i = 0, e = args.size(); i != e; ++i) {
if (args[i] == "-code-snippet-gen") {
ClDCSnippet = true;
}
}
return true;
}
};
static FrontendPluginRegistry::Add<PragmaPluginAction> X
("-extract-omp", "OMP Extractor");
