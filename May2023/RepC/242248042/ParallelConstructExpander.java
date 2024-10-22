package imop.lib.transform.simplify;
import imop.ast.info.OmpClauseInfo;
import imop.ast.info.cfgNodeInfo.ParameterDeclarationInfo;
import imop.ast.node.external.*;
import imop.ast.node.internal.*;
import imop.lib.analysis.flowanalysis.Cell;
import imop.lib.analysis.flowanalysis.FieldCell;
import imop.lib.analysis.flowanalysis.Symbol;
import imop.lib.analysis.typesystem.ArrayType;
import imop.lib.analysis.typesystem.VoidType;
import imop.lib.cfg.info.CompoundStatementCFGInfo;
import imop.lib.cfg.link.autoupdater.AutomatedUpdater;
import imop.lib.cg.CallStack;
import imop.lib.cg.NodeWithStack;
import imop.lib.transform.BasicTransform;
import imop.lib.transform.updater.InsertImmediatePredecessor;
import imop.lib.transform.updater.InsertImmediateSuccessor;
import imop.lib.transform.updater.NodeRemover;
import imop.lib.transform.updater.NodeReplacer;
import imop.lib.transform.updater.sideeffect.SyntacticConstraint;
import imop.lib.util.CellList;
import imop.lib.util.CellSet;
import imop.lib.util.Misc;
import imop.lib.util.ProfileSS;
import imop.parser.FrontEnd;
import imop.parser.Program;
import java.util.*;
public class ParallelConstructExpander {
private static int counter = 0;
public static void mergeParallelRegions(Node node) {
if (!node.getInfo().isConnectedToProgram()) {
Misc.warnDueToLackOfFeature("Cannot invoke merging of parallel regions within a disconnected node.", node);
return;
}
ParallelConstructExpander.counter = 0;
long timeTaken = 0;
long initTime = System.nanoTime();
outer: do {
timeTaken += System.nanoTime() - initTime;
if (Program.useTimerForIncEPARuns && timeTaken/1e9 > Program.secondsForIncEPARuns) {
break;
} else {
initTime = System.nanoTime();
}
List<ParallelConstruct> parConsList = Misc.getExactEncloseeList(node, ParallelConstruct.class);
boolean changed = false;
for (ParallelConstruct parCons : parConsList) {
if (!parCons.getInfo().getCFGNestingNonLeafNodes().stream()
.anyMatch(e -> (e instanceof IterationStatement))) {
continue;
}
changed = expandParRegion(parCons);
ProfileSS.insertCP(); 
AutomatedUpdater.stabilizePTAInCPModes();
counter++;
if (!parCons.getInfo().isConnectedToProgram()) {
continue outer;
}
if (changed) {
parCons.getInfo().removeExtraScopes();
ProfileSS.insertCP();
}
changed |= interchangeUp(parCons);
if (changed) {
ProfileSS.insertCP(); 
AutomatedUpdater.stabilizePTAInCPModes();
continue outer;
}
}
break;
} while (true && counter < Program.numExpansionAllowed);
}
public static boolean expandParRegion(ParallelConstruct parCons) {
Node enclosingNonLeaf = Misc.getEnclosingCFGNonLeafNode(parCons);
assert (enclosingNonLeaf instanceof CompoundStatement)
: "The input is not in normalized form! Found a parallel-construct as a body of a "
+ enclosingNonLeaf.getClass().getSimpleName();
CompoundStatement scope = (CompoundStatement) enclosingNonLeaf;
int indexOfPar = scope.getInfo().getCFGInfo().getElementList().indexOf(parCons);
int lineNum = Misc.getLineNum(parCons);
System.err.println("\t -- Processing the parallel construct at line #" + lineNum + " of size "
+ parCons.getInfo().getCFGInfo().getIntraTaskCFGLeafContents().size());
boolean changed = false;
changed |= expandDownward(parCons, scope, indexOfPar);
ProfileSS.insertCP(); 
AutomatedUpdater.stabilizePTAInCPModes();
indexOfPar = scope.getInfo().getCFGInfo().getElementList().indexOf(parCons);
changed |= expandUpward(parCons, scope, indexOfPar);
ProfileSS.insertCP();
return changed;
}
public static boolean expandDownward(ParallelConstruct parCons, CompoundStatement scope, int index) {
int nextIndex = index + 1;
boolean changed = false;
do {
List<Node> elemList = scope.getInfo().getCFGInfo().getElementList();
int size = elemList.size();
if (nextIndex == size) {
return changed;
}
Node nextNode = elemList.get(nextIndex);
List<OmpClause> clauseList = parCons.getInfo().getOmpClauseList();
CellSet accessedCells = nextNode.getInfo().getSymbolAccesses();
for (OmpClause clause : clauseList) {
Set<String> listedNames = clause.getInfo().getListedNames();
if (clause instanceof OmpReductionClause) {
for (Cell c : accessedCells) {
if (c instanceof Symbol) {
Symbol sym = (Symbol) c;
if (listedNames.contains(sym.getName())) {
return changed;
}
}
}
}
}
if (nextNode instanceof Declaration) {
Declaration decl = (Declaration) nextNode;
if (decl.getInfo().hasInitializer()) {
if (Misc.haveDataDependences(decl.getInfo().getInitializer(), parCons)) {
return changed;
}
}
if (decl.getInfo().clashesSyntacticallyWith(parCons)) {
return changed;
}
scope.getInfo().getCFGInfo().removeElement(decl);
ProfileSS.insertCP();
scope.getInfo().getCFGInfo().addElement(index, decl);
ProfileSS.insertCP(); 
AutomatedUpdater.stabilizePTAInCPModes();
index++;
nextIndex++;
changed = true;
continue;
} else if (nextNode instanceof UnknownCpp) {
return changed;
} else if (nextNode instanceof UnknownPragma) {
return changed;
} else if (nextNode instanceof OmpDirective) {
return changed;
} else if (nextNode instanceof JumpStatement) {
return changed;
} else if (nextNode instanceof ParallelConstruct) {
ParallelConstruct nextPar = (ParallelConstruct) nextNode;
changed = ParallelConstructExpander.merge(parCons, nextPar);
ProfileSS.insertCP(); 
AutomatedUpdater.stabilizePTAInCPModes();
if (!changed) {
return changed;
}
continue;
} else if (nextNode instanceof OmpConstruct) {
return changed;
} else {
if (!nextNode.getInfo().isControlConfined()) {
return changed;
}
Set<NodeWithStack> contentsWithStack = nextNode.getInfo().getCFGInfo()
.getIntraTaskCFGLeafContentsOfSameParLevel(new CallStack());
for (NodeWithStack nS : contentsWithStack) {
Node node = nS.getNode();
if (node instanceof BeginNode) {
BeginNode bn = (BeginNode) node;
if (bn.getParent() instanceof OmpConstruct) {
return changed;
}
}
if (node instanceof OmpDirective) {
return changed;
}
if (node instanceof UnknownCpp || node instanceof UnknownPragma) {
return changed;
}
}
MasterConstruct masterCons = setAndGetLastMaster(parCons);
ProfileSS.insertCP();
if (masterCons == null) {
return changed;
}
NodeRemover.removeNode(nextNode);
ProfileSS.insertCP();
InsertImmediatePredecessor.insert(masterCons.getInfo().getCFGInfo().getNestedCFG().getEnd(), nextNode);
ProfileSS.insertCP(); 
AutomatedUpdater.stabilizePTAInCPModes();
assert (size == scope.getInfo().getCFGInfo().getElementList().size() + 1);
changed = true;
}
} while (true);
}
public static MasterConstruct setAndGetLastMaster(ParallelConstruct parCons) {
CompoundStatementCFGInfo bodyCFGInfo = ((CompoundStatement) parCons.getInfo().getCFGInfo().getBody()).getInfo()
.getCFGInfo();
int size = bodyCFGInfo.getElementList().size();
if (size == 0) {
MasterConstruct newMasterCons = FrontEnd.parseAndNormalize("#pragma omp master \n {}",
MasterConstruct.class);
bodyCFGInfo.addElement(0, newMasterCons);
return newMasterCons;
} else {
Node lastElem = bodyCFGInfo.getElementList().get(size - 1);
if (lastElem instanceof MasterConstruct) {
if (bodyCFGInfo.getElementList().size() == 1) {
return (MasterConstruct) lastElem;
} else {
Node secondLast = bodyCFGInfo.getElementList().get(size - 2);
if (secondLast instanceof BarrierDirective) {
} else {
BarrierDirective barr = FrontEnd.parseAndNormalize("#pragma omp barrier \n",
BarrierDirective.class);
bodyCFGInfo.addElement(size - 1, barr);
ProfileSS.insertCP();
}
return (MasterConstruct) lastElem;
}
} else {
BarrierDirective barr = FrontEnd.parseAndNormalize("#pragma omp barrier \n", BarrierDirective.class);
bodyCFGInfo.addElement(size, barr);
ProfileSS.insertCP();
MasterConstruct newMasterCons = FrontEnd.parseAndNormalize("#pragma omp master \n {}",
MasterConstruct.class);
bodyCFGInfo.addElement(size + 2, newMasterCons);
ProfileSS.insertCP();
return newMasterCons;
}
}
}
public static boolean expandUpward(ParallelConstruct parCons, CompoundStatement scope, int index) {
int prevIndex = index;
boolean changed = false;
List<OmpClause> clauseList = parCons.getInfo().getOmpClauseList();
CellList clauseReadCells = new CellList();
for (OmpClause clause : clauseList) {
if (Misc.isCFGNode(clause)) {
CellList clauseReads = clause.getInfo().getReads();
clauseReadCells.addAll(clauseReads);
}
}
Set<Declaration> testedDecls = new HashSet<>();
outer: do {
prevIndex--;
List<Node> elemList = scope.getInfo().getCFGInfo().getElementList();
int size = elemList.size();
if (prevIndex == -1) {
return changed;
}
Node prevNode = elemList.get(prevIndex);
CellSet prevWriteCellList = prevNode.getInfo().getSymbolWrites();
if (Misc.doIntersect(clauseReadCells, new CellList(prevWriteCellList))) {
if (prevNode instanceof Declaration) {
if (!BasicTransform.pushDeclarationUp((Declaration) prevNode).stream()
.anyMatch(s -> s instanceof SyntacticConstraint)) {
if (testedDecls.contains(prevNode)) {
return changed;
} else {
testedDecls.add((Declaration) prevNode);
prevIndex++; 
continue outer;
}
}
}
return changed;
}
Set<String> writtenIds = null;
if (!prevWriteCellList.isUniversal()) {
writtenIds = new HashSet<>();
for (Cell w : prevWriteCellList) {
if (w instanceof Symbol) {
String idName = ((Symbol) w).getName().trim();
writtenIds.add(idName);
}
}
}
for (OmpClause clause : clauseList) {
if (clause instanceof OmpFirstPrivateClause || clause instanceof OmpReductionClause
|| clause instanceof OmpCopyinClause) {
if (prevWriteCellList.isUniversal()) {
return changed;
} else {
if (Misc.doIntersect(writtenIds, clause.getInfo().getListedNames())) {
if (prevNode instanceof Declaration) {
if (!BasicTransform.pushDeclarationUp((Declaration) prevNode).stream()
.anyMatch(s -> s instanceof SyntacticConstraint)) {
if (testedDecls.contains(prevNode)) {
return changed;
} else {
testedDecls.add((Declaration) prevNode);
prevIndex++; 
continue outer;
}
}
}
return changed;
}
}
}
}
if (prevNode instanceof Declaration) {
Declaration decl = (Declaration) prevNode;
List<String> idList = decl.getInfo().getIDNameList();
assert (idList.size() == 1)
: "Encountered code where a declaration declares more than one symbol! \n " + decl;
String declaredIdName = idList.get(0);
Symbol declaredSymbol = Misc.getSymbolEntry(declaredIdName, parCons);
assert (declaredSymbol != null);
CellSet writtenCells = new CellSet(prevWriteCellList);
CellSet accessedLater = scope.getInfo().getSymbolAccessesStarting(prevIndex + 2);
if (Misc.doIntersect(writtenCells, accessedLater)) {
if (!BasicTransform.pushDeclarationUp(decl).stream()
.anyMatch(s -> s instanceof SyntacticConstraint)) {
ProfileSS.insertCP(); 
AutomatedUpdater.stabilizePTAInCPModes();
if (testedDecls.contains(prevNode)) {
return changed;
} else {
testedDecls.add((Declaration) prevNode);
prevIndex++; 
continue outer;
}
}
return changed;
}
List<OmpClause> clauseListCopy = new ArrayList<>(clauseList);
for (OmpClause clause : clauseListCopy) {
if (clause instanceof OmpSharedClause || clause instanceof OmpPrivateClause) {
Set<String> listedIds = clause.getInfo().getListedNames();
if (!Misc.doIntersect(writtenIds, listedIds)) {
continue;
}
if (clause instanceof OmpSharedClause) {
if (!usedOnlyByMaster(declaredSymbol, parCons) && writtenWithin(declaredSymbol, parCons)) {
if (!BasicTransform.pushDeclarationUp(decl).stream()
.anyMatch(s -> s instanceof SyntacticConstraint)) {
if (testedDecls.contains(prevNode)) {
return changed;
} else {
testedDecls.add((Declaration) prevNode);
prevIndex++; 
continue outer;
}
}
return changed;
}
}
if (BasicTransform.removeSideEffectsFromInitializer(decl)) {
changed = true;
return changed;
}
boolean clauseNeedsToBeRemoved = clause.getInfo().removeListItem(declaredIdName);
if (clauseNeedsToBeRemoved) {
boolean clauseRemoved = parCons.getInfo().removeClause(clause);
if (!clauseRemoved) {
Misc.exitDueToLackOfFeature("Could not remove the clause: " + clause + " from "
+ parCons.getInfo().getOmpClauseList());
}
}
scope.getInfo().getCFGInfo().removeDeclaration(decl);
ProfileSS.insertCP();
CompoundStatement parBody = (CompoundStatement) parCons.getInfo().getCFGInfo().getBody();
parBody.getInfo().getCFGInfo().addDeclaration(0, decl);
ProfileSS.insertCP(); 
AutomatedUpdater.stabilizePTAInCPModes();
assert (size == scope.getInfo().getCFGInfo().getElementList().size() + 1);
changed = true;
continue outer;
}
}
boolean writtenWithin = writtenWithin(declaredSymbol, parCons);
boolean usedApartFromMaster = !usedOnlyByMaster(declaredSymbol, parCons);
if (usedApartFromMaster && writtenWithin) {
if (!BasicTransform.pushDeclarationUp(decl).stream()
.anyMatch(s -> s instanceof SyntacticConstraint)) {
ProfileSS.insertCP(); 
AutomatedUpdater.stabilizePTAInCPModes();
if (testedDecls.contains(prevNode)) {
return changed;
} else {
testedDecls.add((Declaration) prevNode);
prevIndex++; 
continue outer;
}
}
return changed;
}
scope.getInfo().getCFGInfo().removeDeclaration(decl);
ProfileSS.insertCP();
CompoundStatement parBody = (CompoundStatement) parCons.getInfo().getCFGInfo().getBody();
parBody.getInfo().getCFGInfo().addDeclaration(0, decl);
ProfileSS.insertCP();
ProfileSS.insertCP(); 
AutomatedUpdater.stabilizePTAInCPModes();
assert (size == scope.getInfo().getCFGInfo().getElementList().size() + 1);
changed = true;
continue outer;
} else if (prevNode instanceof UnknownCpp) {
return changed;
} else if (prevNode instanceof UnknownPragma) {
return changed;
} else if (prevNode instanceof OmpDirective) {
return changed;
} else if (prevNode instanceof JumpStatement) {
return changed;
} else if (prevNode instanceof ParallelConstruct) {
ParallelConstruct prevPar = (ParallelConstruct) prevNode;
changed = ParallelConstructExpander.merge(prevPar, parCons);
return changed;
} else if (prevNode instanceof OmpConstruct) {
return changed;
} else {
if (!prevNode.getInfo().isControlConfined()) {
return changed;
}
Set<NodeWithStack> contentsWithStack = prevNode.getInfo().getCFGInfo()
.getIntraTaskCFGLeafContentsOfSameParLevel(new CallStack());
for (NodeWithStack nS : contentsWithStack) {
Node node = nS.getNode();
if (node instanceof BeginNode) {
BeginNode bn = (BeginNode) node;
if (bn.getParent() instanceof OmpConstruct) {
return changed;
}
}
if (node instanceof OmpDirective) {
return changed;
}
if (node instanceof UnknownCpp || node instanceof UnknownPragma) {
return changed;
}
}
MasterConstruct masterCons = setAndGetFirstMaster(parCons);
ProfileSS.insertCP();
if (masterCons == null) {
return changed;
}
NodeRemover.removeNode(prevNode);
ProfileSS.insertCP();
InsertImmediateSuccessor.insert(masterCons.getInfo().getCFGInfo().getNestedCFG().getBegin(), prevNode);
ProfileSS.insertCP(); 
AutomatedUpdater.stabilizePTAInCPModes();
assert (size == scope.getInfo().getCFGInfo().getElementList().size() + 1);
changed = true;
continue outer;
}
} while (true);
}
private static boolean writtenWithin(Symbol sym, ParallelConstruct parCons) {
Set<Node> parConsLeafContents = parCons.getInfo().getCFGInfo().getIntraTaskCFGLeafContents();
for (Node leaf : parConsLeafContents) {
if (!leaf.getInfo().mayWrite()) {
continue;
}
CellList writes = leaf.getInfo().getWrites();
if (writes.isUniversal()) {
return true;
}
for (Cell cell : writes) {
if (cell == sym) {
return true;
}
if (cell instanceof FieldCell && sym.getType() instanceof ArrayType && cell == sym.getFieldCell()) {
return true;
}
}
}
return false;
}
private static boolean usedOnlyByMaster(Symbol sym, ParallelConstruct parCons) {
Set<MasterConstruct> allMasterConstructs = new HashSet<>();
CompoundStatement body = (CompoundStatement) parCons.getInfo().getCFGInfo().getBody();
Set<NodeWithStack> contentsWS = body.getInfo().getCFGInfo().getIntraTaskCFGLeafContentsOfSameParLevel();
for (NodeWithStack cfgLeafWS : contentsWS) {
Node cfgLeaf = cfgLeafWS.getNode();
if (cfgLeaf instanceof BeginNode) {
BeginNode beginNode = (BeginNode) cfgLeaf;
if (beginNode.getParent() instanceof MasterConstruct) {
allMasterConstructs.add((MasterConstruct) beginNode.getParent());
}
}
}
Set<Node> masterLeafContents = new HashSet<>();
for (MasterConstruct masterCons : allMasterConstructs) {
masterLeafContents.addAll(masterCons.getInfo().getCFGInfo().getIntraTaskCFGLeafContents());
}
Set<Node> parConsLeafContents = parCons.getInfo().getCFGInfo().getIntraTaskCFGLeafContents();
parConsLeafContents.removeAll(masterLeafContents);
for (Node leaf : parConsLeafContents) {
CellSet accesses = leaf.getInfo().getAccesses();
if (accesses.isUniversal()) {
return false;
}
for (Cell cell : accesses) {
if (cell == sym) {
return false;
}
}
}
return true;
}
public static MasterConstruct setAndGetFirstMaster(ParallelConstruct parCons) {
CompoundStatementCFGInfo bodyCFGInfo = ((CompoundStatement) parCons.getInfo().getCFGInfo().getBody()).getInfo()
.getCFGInfo();
int size = bodyCFGInfo.getElementList().size();
if (size == 0) {
MasterConstruct newMasterCons = FrontEnd.parseAndNormalize("#pragma omp master \n {}",
MasterConstruct.class);
bodyCFGInfo.addElement(0, newMasterCons);
return newMasterCons;
} else {
Node firstElem = bodyCFGInfo.getElementList().get(0);
if (firstElem instanceof MasterConstruct) {
if (bodyCFGInfo.getElementList().size() == 1) {
return (MasterConstruct) firstElem;
} else {
Node second = bodyCFGInfo.getElementList().get(1);
if (second instanceof DummyFlushDirective
&& ((DummyFlushDirective) second).getDummyFlushType() == DummyFlushType.BARRIER_START) {
} else {
BarrierDirective barr = FrontEnd.parseAndNormalize("#pragma omp barrier \n",
BarrierDirective.class);
bodyCFGInfo.addElement(1, barr);
}
return (MasterConstruct) firstElem;
}
} else {
MasterConstruct newMasterCons = FrontEnd.parseAndNormalize("#pragma omp master \n {}",
MasterConstruct.class);
bodyCFGInfo.addElement(0, newMasterCons);
ProfileSS.insertCP();
BarrierDirective barr = FrontEnd.parseAndNormalize("#pragma omp barrier \n", BarrierDirective.class);
ProfileSS.insertCP();
bodyCFGInfo.addElement(1, barr);
ProfileSS.insertCP();
return newMasterCons;
}
}
}
public static boolean interchangeUp(ParallelConstruct parCons) {
CompoundStatement scope = (CompoundStatement) Misc.getEnclosingCFGNonLeafNode(parCons);
if (scope == null) {
return false;
}
Node encloser = Misc.getEnclosingCFGNonLeafNode(scope);
if (scope.getInfo().getCFGInfo().getElementList().size() == 1) {
assert !(encloser instanceof CompoundStatement) : "The parCons was normalized already!"; 
if (encloser instanceof OmpConstruct) {
return false;
} else if (encloser instanceof FunctionDefinition) {
} else if (encloser instanceof SelectionStatement) {
if (encloser instanceof IfStatement) {
IfStatement ifStmt = (IfStatement) encloser;
Set<String> privateIdNames = parCons.getInfo().getListedPrivateNames();
Expression predicate = ifStmt.getInfo().getCFGInfo().getPredicate();
CellSet readsInPred = predicate.getInfo().getSymbolReads();
if (predicate.getInfo().mayWrite()) {
return false;
}
if (readsInPred.isUniversal()) {
if (!privateIdNames.isEmpty()) {
return false;
}
}
for (Cell cell : readsInPred) {
if (cell instanceof Symbol) {
Symbol sym = (Symbol) cell;
String idName = sym.getName().trim();
if (privateIdNames.contains(idName)) {
return false;
}
}
}
if (scope == ifStmt.getInfo().getCFGInfo().getThenBody()) {
if (!ifStmt.getInfo().getCFGInfo().hasElseBody()) {
CompoundStatement enclosingCS = (CompoundStatement) Misc.getEnclosingCFGNonLeafNode(ifStmt);
int indexOfIf = enclosingCS.getInfo().getCFGInfo().getElementList().indexOf(ifStmt);
enclosingCS.getInfo().getCFGInfo().removeElement(indexOfIf);
CompoundStatement parBody = (CompoundStatement) parCons.getInfo().getCFGInfo().getBody();
BarrierDirective barr = FrontEnd.parseAndNormalize("#pragma omp barrier\n",
BarrierDirective.class);
parBody.getInfo().getCFGInfo().addAtLast(barr);
CompoundStatement newParBody = FrontEnd.parseAndNormalize("{}", CompoundStatement.class);
parCons.getInfo().getCFGInfo().setBody(newParBody);
IfStatement newIfStmt = FrontEnd.parseAndNormalize("if (1) {}", IfStatement.class);
newParBody.getInfo().getCFGInfo().addAtLast(newIfStmt);
enclosingCS.getInfo().getCFGInfo().addElement(indexOfIf, parCons);
newIfStmt.getInfo().getCFGInfo().setThenBody(parBody);
newIfStmt.getInfo().getCFGInfo().setPredicate(ifStmt.getInfo().getCFGInfo().getPredicate());
return true;
} else {
CompoundStatement elseBody = (CompoundStatement) ifStmt.getInfo().getCFGInfo()
.getElseBody();
List<Node> elseElements = elseBody.getInfo().getCFGInfo().getElementList();
if (elseElements.size() == 1) {
Node elseNode = elseElements.get(0);
if (elseNode instanceof ParallelConstruct) {
ParallelConstruct elsePar = (ParallelConstruct) elseNode;
if (!ParallelConstructExpander.mergeDataAttributeLists(parCons, elsePar)) {
return false;
}
CompoundStatement enclosingCS = (CompoundStatement) Misc
.getEnclosingCFGNonLeafNode(ifStmt);
int indexOfIf = enclosingCS.getInfo().getCFGInfo().getElementList().indexOf(ifStmt);
enclosingCS.getInfo().getCFGInfo().removeElement(indexOfIf);
CompoundStatement parBody = (CompoundStatement) parCons.getInfo().getCFGInfo()
.getBody();
BarrierDirective barr = FrontEnd.parseAndNormalize("#pragma omp barrier\n",
BarrierDirective.class);
parBody.getInfo().getCFGInfo().addAtLast(barr);
CompoundStatement newParBody = FrontEnd.parseAndNormalize("{}",
CompoundStatement.class);
parCons.getInfo().getCFGInfo().setBody(newParBody);
IfStatement newIfStmt = FrontEnd.parseAndNormalize("if (1) {}", IfStatement.class);
newParBody.getInfo().getCFGInfo().addAtLast(newIfStmt);
enclosingCS.getInfo().getCFGInfo().addElement(indexOfIf, parCons);
newIfStmt.getInfo().getCFGInfo().setThenBody(parBody);
newIfStmt.getInfo().getCFGInfo()
.setPredicate(ifStmt.getInfo().getCFGInfo().getPredicate());
newIfStmt.getInfo().getCFGInfo()
.setElseBody(elsePar.getInfo().getCFGInfo().getBody());
return true;
}
}
if (!bodyFusable(parCons, elseBody)) {
return false;
}
CompoundStatement enclosingCS = (CompoundStatement) Misc.getEnclosingCFGNonLeafNode(ifStmt);
int indexOfIf = enclosingCS.getInfo().getCFGInfo().getElementList().indexOf(ifStmt);
enclosingCS.getInfo().getCFGInfo().removeElement(indexOfIf);
ProfileSS.insertCP();
CompoundStatement parBody = (CompoundStatement) parCons.getInfo().getCFGInfo().getBody();
BarrierDirective barr = FrontEnd.parseAndNormalize("#pragma omp barrier\n",
BarrierDirective.class);
parBody.getInfo().getCFGInfo().addAtLast(barr);
ProfileSS.insertCP();
CompoundStatement newParBody = FrontEnd.parseAndNormalize("{}", CompoundStatement.class);
ProfileSS.insertCP();
parCons.getInfo().getCFGInfo().setBody(newParBody);
ProfileSS.insertCP();
IfStatement newIfStmt = FrontEnd.parseAndNormalize("if (1) {}", IfStatement.class);
newParBody.getInfo().getCFGInfo().addAtLast(newIfStmt);
ProfileSS.insertCP();
enclosingCS.getInfo().getCFGInfo().addElement(indexOfIf, parCons);
ProfileSS.insertCP();
newIfStmt.getInfo().getCFGInfo().setThenBody(parBody);
ProfileSS.insertCP();
newIfStmt.getInfo().getCFGInfo().setPredicate(ifStmt.getInfo().getCFGInfo().getPredicate());
ProfileSS.insertCP();
CompoundStatement newElseBody = FrontEnd.parseAndNormalize("{\n#pragma omp master\n{}}",
CompoundStatement.class);
newIfStmt.getInfo().getCFGInfo().setElseBody(newElseBody);
ProfileSS.insertCP();
MasterConstruct master = (MasterConstruct) newElseBody.getInfo().getCFGInfo()
.getElementList().get(0);
master.getInfo().getCFGInfo().setBody(elseBody);
ProfileSS.insertCP();
return true;
}
} else if (scope == ifStmt.getInfo().getCFGInfo().getElseBody()) {
CompoundStatement thenBody = (CompoundStatement) ifStmt.getInfo().getCFGInfo().getThenBody();
List<Node> thenElements = thenBody.getInfo().getCFGInfo().getElementList();
if (thenElements.size() == 1) {
Node thenNode = thenElements.get(0);
if (thenNode instanceof ParallelConstruct) {
ParallelConstruct thenPar = (ParallelConstruct) thenNode;
if (!ParallelConstructExpander.mergeDataAttributeListsBelow(thenPar, parCons)) {
return false;
}
CompoundStatement enclosingCS = (CompoundStatement) Misc
.getEnclosingCFGNonLeafNode(ifStmt);
int indexOfIf = enclosingCS.getInfo().getCFGInfo().getElementList().indexOf(ifStmt);
enclosingCS.getInfo().getCFGInfo().removeElement(indexOfIf);
ProfileSS.insertCP();
CompoundStatement parBody = (CompoundStatement) parCons.getInfo().getCFGInfo()
.getBody();
BarrierDirective barr = FrontEnd.parseAndNormalize("#pragma omp barrier\n",
BarrierDirective.class);
parBody.getInfo().getCFGInfo().addAtLast(barr);
ProfileSS.insertCP();
CompoundStatement newParBody = FrontEnd.parseAndNormalize("{}",
CompoundStatement.class);
parCons.getInfo().getCFGInfo().setBody(newParBody);
ProfileSS.insertCP();
IfStatement newIfStmt = FrontEnd.parseAndNormalize("if (1) {}", IfStatement.class);
newParBody.getInfo().getCFGInfo().addAtLast(newIfStmt);
ProfileSS.insertCP();
enclosingCS.getInfo().getCFGInfo().addElement(indexOfIf, parCons);
ProfileSS.insertCP();
newIfStmt.getInfo().getCFGInfo().setElseBody(parBody);
ProfileSS.insertCP();
newIfStmt.getInfo().getCFGInfo()
.setPredicate(ifStmt.getInfo().getCFGInfo().getPredicate());
ProfileSS.insertCP();
newIfStmt.getInfo().getCFGInfo().setThenBody(thenPar.getInfo().getCFGInfo().getBody());
ProfileSS.insertCP();
return true;
}
}
if (!bodyFusable(parCons, thenBody)) {
return false;
}
CompoundStatement enclosingCS = (CompoundStatement) Misc.getEnclosingCFGNonLeafNode(ifStmt);
int indexOfIf = enclosingCS.getInfo().getCFGInfo().getElementList().indexOf(ifStmt);
enclosingCS.getInfo().getCFGInfo().removeElement(indexOfIf);
ProfileSS.insertCP();
CompoundStatement parBody = (CompoundStatement) parCons.getInfo().getCFGInfo().getBody();
BarrierDirective barr = FrontEnd.parseAndNormalize("#pragma omp barrier\n",
BarrierDirective.class);
parBody.getInfo().getCFGInfo().addAtLast(barr);
ProfileSS.insertCP();
CompoundStatement newParBody = FrontEnd.parseAndNormalize("{}", CompoundStatement.class);
ProfileSS.insertCP();
parCons.getInfo().getCFGInfo().setBody(newParBody);
ProfileSS.insertCP();
IfStatement newIfStmt = FrontEnd.parseAndNormalize("if (1) {}", IfStatement.class);
newParBody.getInfo().getCFGInfo().addAtLast(newIfStmt);
ProfileSS.insertCP();
enclosingCS.getInfo().getCFGInfo().addElement(indexOfIf, parCons);
ProfileSS.insertCP();
newIfStmt.getInfo().getCFGInfo().setPredicate(ifStmt.getInfo().getCFGInfo().getPredicate());
ProfileSS.insertCP();
newIfStmt.getInfo().getCFGInfo().setElseBody(parBody);
ProfileSS.insertCP();
CompoundStatement newThenBody = FrontEnd.parseAndNormalize("{\n#pragma omp master\n{}}",
CompoundStatement.class);
newIfStmt.getInfo().getCFGInfo().setThenBody(newThenBody);
ProfileSS.insertCP();
MasterConstruct master = (MasterConstruct) newThenBody.getInfo().getCFGInfo().getElementList()
.get(0);
ProfileSS.insertCP();
master.getInfo().getCFGInfo().setBody(thenBody);
ProfileSS.insertCP();
return true;
} else {
assert (false);
}
} else if (encloser instanceof SwitchStatement) {
}
return false;
} else {
Set<String> privateIdNames = parCons.getInfo().getListedPrivateNames();
if (encloser instanceof DoStatement) {
DoStatement doStmt = (DoStatement) encloser;
if (doStmt.getInfo().hasLabelAnnotations()) {
return false;
}
Expression predicate = doStmt.getInfo().getCFGInfo().getPredicate();
CellSet readsInPred = predicate.getInfo().getSymbolReads();
if (predicate.getInfo().mayWrite()) {
return false;
}
if (readsInPred.isUniversal()) {
if (!privateIdNames.isEmpty()) {
return false;
}
}
for (Cell cell : readsInPred) {
assert (cell instanceof Symbol);
if (cell instanceof Symbol) {
Symbol sym = (Symbol) cell;
String idName = sym.getName().trim();
if (privateIdNames.contains(idName)) {
return false;
}
}
}
CompoundStatement enclosingCS = (CompoundStatement) Misc.getEnclosingCFGNonLeafNode(doStmt);
int indexOfDo = enclosingCS.getInfo().getCFGInfo().getElementList().indexOf(doStmt);
enclosingCS.getInfo().getCFGInfo().removeElement(indexOfDo);
CompoundStatement parBody = (CompoundStatement) parCons.getInfo().getCFGInfo().getBody();
BarrierDirective barr = FrontEnd.parseAndNormalize("#pragma omp barrier\n", BarrierDirective.class);
parBody.getInfo().getCFGInfo().addAtLast(barr);
CompoundStatement newParBody = FrontEnd.parseAndNormalize("{}", CompoundStatement.class);
parCons.getInfo().getCFGInfo().setBody(newParBody);
DoStatement newDoStmt = FrontEnd.parseAndNormalize("do {} while(1);", DoStatement.class);
newParBody.getInfo().getCFGInfo().addAtLast(newDoStmt);
enclosingCS.getInfo().getCFGInfo().addElement(indexOfDo, parCons);
newDoStmt.getInfo().getCFGInfo().setBody(parBody);
newDoStmt.getInfo().getCFGInfo().setPredicate(doStmt.getInfo().getCFGInfo().getPredicate());
return true;
} else if (encloser instanceof WhileStatement) {
} else if (encloser instanceof ForStatement) {
ForStatement forStmt = (ForStatement) encloser;
if (forStmt.getInfo().hasLabelAnnotations()) {
return false;
}
Expression initExp = forStmt.getInfo().getCFGInfo().getInitExpression();
Expression stepExp = forStmt.getInfo().getCFGInfo().getStepExpression();
Expression predicate = forStmt.getInfo().getCFGInfo().getTerminationExpression();
if (initExp != null) {
CellSet readsInInit = initExp.getInfo().getSymbolReads();
if (initExp.getInfo().mayWrite()) {
return false;
}
if (readsInInit.isUniversal()) {
if (!privateIdNames.isEmpty()) {
return false;
}
}
for (Cell cell : readsInInit) {
if (cell instanceof Symbol) {
Symbol sym = (Symbol) cell;
String idName = sym.getName().trim();
if (privateIdNames.contains(idName)) {
return false;
}
}
}
}
if (stepExp != null) {
CellSet readsInStep = stepExp.getInfo().getSymbolReads();
if (stepExp.getInfo().mayWrite()) {
return false;
}
if (readsInStep.isUniversal()) {
if (!privateIdNames.isEmpty()) {
return false;
}
}
for (Cell cell : readsInStep) {
if (cell instanceof Symbol) {
Symbol sym = (Symbol) cell;
String idName = sym.getName().trim();
if (privateIdNames.contains(idName)) {
return false;
}
}
}
}
if (predicate != null) {
CellSet readsInPred = predicate.getInfo().getSymbolReads();
if (predicate.getInfo().mayWrite()) {
return false;
}
if (readsInPred.isUniversal()) {
if (!privateIdNames.isEmpty()) {
return false;
}
}
for (Cell cell : readsInPred) {
if (cell instanceof Symbol) {
Symbol sym = (Symbol) cell;
String idName = sym.getName().trim();
if (privateIdNames.contains(idName)) {
return false;
}
}
}
}
CompoundStatement enclosingCS = (CompoundStatement) Misc.getEnclosingCFGNonLeafNode(forStmt);
int indexOfFor = enclosingCS.getInfo().getCFGInfo().getElementList().indexOf(forStmt);
enclosingCS.getInfo().getCFGInfo().removeElement(indexOfFor);
CompoundStatement parBody = (CompoundStatement) parCons.getInfo().getCFGInfo().getBody();
BarrierDirective barr = FrontEnd.parseAndNormalize("#pragma omp barrier\n", BarrierDirective.class);
parBody.getInfo().getCFGInfo().addAtLast(barr);
CompoundStatement newParBody = FrontEnd.parseAndNormalize("{}", CompoundStatement.class);
parCons.getInfo().getCFGInfo().setBody(newParBody);
ForStatement newForStmt = FrontEnd.parseAndNormalize("for(;;){}", ForStatement.class);
newParBody.getInfo().getCFGInfo().addAtLast(newForStmt);
enclosingCS.getInfo().getCFGInfo().addElement(indexOfFor, parCons);
newForStmt.getInfo().getCFGInfo().setBody(parBody);
newForStmt.getInfo().getCFGInfo()
.setInitExpression(forStmt.getInfo().getCFGInfo().getInitExpression());
newForStmt.getInfo().getCFGInfo()
.setTerminationExpression(forStmt.getInfo().getCFGInfo().getTerminationExpression());
newForStmt.getInfo().getCFGInfo()
.setStepExpression(forStmt.getInfo().getCFGInfo().getStepExpression());
return true;
} else {
return false;
}
}
}
if (encloser instanceof FunctionDefinition) {
FunctionDefinition func = (FunctionDefinition) encloser;
if (func.getInfo().isRecursive()) {
return false;
}
ProfileSS.insertCP(); 
AutomatedUpdater.stabilizePTAInCPModes();
return functionSwappableAggressive(func, parCons);
} else if (encloser instanceof WhileStatement) {
Set<String> privateIdNames = parCons.getInfo().getListedPrivateNames();
WhileStatement whileStmt = (WhileStatement) encloser;
if (scope.getInfo().getCFGInfo().getElementList().size() == 1) {
if (whileStmt.getInfo().hasLabelAnnotations()) {
return false;
}
Expression predicate = whileStmt.getInfo().getCFGInfo().getPredicate();
CellSet readsInPred = predicate.getInfo().getSymbolReads();
if (predicate.getInfo().mayWrite()) {
return false;
}
if (readsInPred.isUniversal()) {
if (!privateIdNames.isEmpty()) {
return false;
}
}
for (Cell cell : readsInPred) {
if (cell instanceof Symbol) {
Symbol sym = (Symbol) cell;
String idName = sym.getName().trim();
if (privateIdNames.contains(idName)) {
return false;
}
}
}
CompoundStatement enclosingCS = (CompoundStatement) Misc.getEnclosingCFGNonLeafNode(whileStmt);
int indexOfWhile = enclosingCS.getInfo().getCFGInfo().getElementList().indexOf(whileStmt);
enclosingCS.getInfo().getCFGInfo().removeElement(indexOfWhile);
CompoundStatement parBody = (CompoundStatement) parCons.getInfo().getCFGInfo().getBody();
BarrierDirective barr = FrontEnd.parseAndNormalize("#pragma omp barrier\n", BarrierDirective.class);
parBody.getInfo().getCFGInfo().addAtLast(barr);
CompoundStatement newParBody = FrontEnd.parseAndNormalize("{}", CompoundStatement.class);
parCons.getInfo().getCFGInfo().setBody(newParBody);
WhileStatement newWhileStmt = FrontEnd.parseAndNormalize("while(1){}", WhileStatement.class);
newParBody.getInfo().getCFGInfo().addAtLast(newWhileStmt);
enclosingCS.getInfo().getCFGInfo().addElement(indexOfWhile, parCons);
newWhileStmt.getInfo().getCFGInfo().setBody(parBody);
newWhileStmt.getInfo().getCFGInfo().setPredicate(whileStmt.getInfo().getCFGInfo().getPredicate());
return true;
} else {
return whileSwappableAggressive((WhileStatement) encloser, parCons);
}
} else if (encloser instanceof DoStatement) {
} else if (encloser instanceof ForStatement) {
}
return false;
}
private static boolean bodyFusable(ParallelConstruct parCons, CompoundStatement body) {
List<Node> elementList = body.getInfo().getCFGInfo().getElementList();
if (elementList.isEmpty()) {
return true;
}
if (elementList.size() == 1) {
if (elementList.get(0) instanceof ParallelConstruct) {
return true;
}
}
if (!body.getInfo().isControlConfined()) {
return false;
}
Set<NodeWithStack> contentsWithStack = body.getInfo().getCFGInfo()
.getIntraTaskCFGLeafContentsOfSameParLevel(new CallStack());
for (NodeWithStack nS : contentsWithStack) {
Node node = nS.getNode();
if (node instanceof BeginNode) {
BeginNode bn = (BeginNode) node;
if (bn.getParent() instanceof OmpConstruct) {
return false;
}
}
if (node instanceof OmpDirective) {
return false;
}
if (node instanceof UnknownCpp || node instanceof UnknownPragma) {
return false;
}
}
return true;
}
private static boolean whileSwappableAggressive(WhileStatement whileStmt, ParallelConstruct parCons) {
if (!ParallelConstructExpander.remainingElementsFusableAcrossIteration(whileStmt, parCons)) {
return false;
}
return false;
}
private static boolean functionSwappableAggressive(FunctionDefinition func, ParallelConstruct parCons) {
if (func.getInfo().getFunctionName().equals("main")) {
return false;
}
if (func.getInfo().getReturnType() instanceof VoidType && !func.getInfo().hasAnyRealParams()) {
return ParallelConstructExpander.functionSwappableBasic(func, parCons);
}
for (CallStatement callStmt : func.getInfo().getCallersOfThis()) {
if (callStmt.getInfo().getCalledDefinitions().size() != 1) {
return false;
}
}
if (!ParallelConstructExpander.remainingElementsFusableAcrossFunction(func, parCons)) {
return false;
}
List<OmpClause> clauseList = parCons.getInfo().getOmpClauseList();
outer: for (ParameterDeclaration paramDecl : func.getInfo().getCFGInfo().getParameterDeclarationList()) {
Symbol paramSym = paramDecl.getInfo().getDeclaredSymbol();
if (paramSym.getType() instanceof VoidType) {
continue;
}
String paramName = ParameterDeclarationInfo.getRootParamName(paramDecl);
for (OmpClause clause : clauseList) {
if (clause instanceof OmpReductionClause) {
if (clause.getInfo().getListedNames().contains(paramName)) {
return false;
}
}
}
for (OmpClause clause : clauseList) {
if (clause instanceof OmpPrivateClause) {
if (clause.getInfo().getListedNames().contains(paramName)) {
continue outer;
}
}
}
if (!usedOnlyByMaster(paramSym, parCons) && writtenWithin(paramSym, parCons)) {
return false;
}
}
System.err
.println("\t -- Swapping up the parallel construct above " + func.getInfo().getFunctionName() + "().");
CompoundStatement scope = func.getInfo().getCFGInfo().getBody();
List<Node> elementList = scope.getInfo().getCFGInfo().getElementList();
int parIndex = elementList.indexOf(parCons);
CompoundStatement compStmt = FrontEnd.parseAndNormalize("{}", CompoundStatement.class);
CompoundStatement parBody = (CompoundStatement) parCons.getInfo().getCFGInfo().getBody();
parCons.getInfo().getCFGInfo().setBody(compStmt);
ProfileSS.insertCP();
if (elementList.size() == 1) {
func.getInfo().getCFGInfo().setBody(parBody);
ProfileSS.insertCP();
} else {
NodeReplacer.replaceNodes(parCons, parBody);
}
if (elementList.size() > 1) {
BarrierDirective barrier = FrontEnd.parseAndNormalize("#pragma omp barrier \n", BarrierDirective.class);
scope.getInfo().getCFGInfo().addElement(parIndex + 1, barrier);
}
List<String> paramNames = new ArrayList<>();
for (ParameterDeclaration paramDecl : func.getInfo().getCFGInfo().getParameterDeclarationList()) {
paramNames.add(ParameterDeclarationInfo.getRootParamName(paramDecl));
}
for (CallStatement callStmt : func.getInfo().getCallersOfThis()) {
if (callStmt.getPostCallNode().hasReturnReceiver()) {
}
HashMap<String, String> renamingMap = new HashMap<>();
List<SimplePrimaryExpression> speList = callStmt.getInfo().getArgumentList();
int index = -1;
for (String paramName : paramNames) {
index++;
SimplePrimaryExpression argSPE;
try {
argSPE = speList.get(index);
} catch (IndexOutOfBoundsException e) {
continue;
}
if (argSPE.isAConstant()) {
continue;
} else {
renamingMap.put(paramName, argSPE.getIdentifier().getTokenImage());
}
}
List<OmpClause> newClauses = new ArrayList<>();
for (OmpClause oldClause : parCons.getInfo().getOmpClauseList()) {
if (oldClause instanceof IfClause || oldClause instanceof NumThreadsClause
|| oldClause instanceof OmpFirstPrivateClause || oldClause instanceof OmpCopyinClause
|| oldClause instanceof OmpReductionClause) {
OmpClause newClause = (OmpClause) BasicTransform.obtainRenamedNode(oldClause, renamingMap);
newClauses.add(newClause);
}
}
assert (callStmt.getInfo().getCalledDefinitions().size() == 1);
ParallelConstruct parConsNew = FrontEnd.parseAndNormalize("#pragma omp parallel \n {}",
ParallelConstruct.class);
InsertImmediatePredecessor.insert(callStmt, parConsNew);
ProfileSS.insertCP();
NodeRemover.removeNode(callStmt);
ProfileSS.insertCP();
for (OmpClause newClause : newClauses) {
OmpClause newClauseCopy = OmpClauseInfo.getCopy(newClause);
parConsNew.getInfo().addClause(newClauseCopy);
}
CompoundStatement newParBody = (CompoundStatement) parConsNew.getInfo().getCFGInfo().getBody();
newParBody.getInfo().getCFGInfo().addElement(callStmt);
ProfileSS.insertCP();
}
return true;
}
private static boolean functionSwappableBasic(FunctionDefinition func, ParallelConstruct parCons) {
if (func.getInfo().getFunctionName().equals("main")) {
return false;
}
if (!(func.getInfo().getReturnType() instanceof VoidType)) {
return false;
}
if (func.getInfo().hasAnyRealParams()) {
return false;
}
for (CallStatement callStmt : func.getInfo().getCallersOfThis()) {
if (callStmt.getInfo().getCalledDefinitions().size() != 1) {
return false;
}
}
if (!ParallelConstructExpander.remainingElementsFusableAcrossFunction(func, parCons)) {
return false;
}
System.err
.println("\t -- Swapping up the parallel construct above " + func.getInfo().getFunctionName() + "().");
CompoundStatement scope = func.getInfo().getCFGInfo().getBody();
List<Node> elementList = scope.getInfo().getCFGInfo().getElementList();
int parIndex = elementList.indexOf(parCons);
CompoundStatement compStmt = FrontEnd.parseAndNormalize("{}", CompoundStatement.class);
CompoundStatement parBody = (CompoundStatement) parCons.getInfo().getCFGInfo().getBody();
parCons.getInfo().getCFGInfo().setBody(compStmt);
if (elementList.size() == 1) {
func.getInfo().getCFGInfo().setBody(parBody);
} else {
NodeReplacer.replaceNodes(parCons, parBody);
}
if (elementList.size() > 1) {
BarrierDirective barrier = FrontEnd.parseAndNormalize("#pragma omp barrier \n", BarrierDirective.class);
scope.getInfo().getCFGInfo().addElement(parIndex + 1, barrier);
}
for (CallStatement callStmt : func.getInfo().getCallersOfThis()) {
List<OmpClause> newClauses = new ArrayList<>();
for (OmpClause oldClause : parCons.getInfo().getOmpClauseList()) {
if (oldClause instanceof IfClause || oldClause instanceof NumThreadsClause
|| oldClause instanceof OmpFirstPrivateClause || oldClause instanceof OmpCopyinClause
|| oldClause instanceof OmpReductionClause) {
OmpClause newClause = OmpClauseInfo.getCopy(oldClause);
NodeReplacer.replaceNodes(newClause, oldClause); 
newClauses.add(newClause);
}
}
assert (callStmt.getInfo().getCalledDefinitions().size() == 1);
ParallelConstruct parConsNew = FrontEnd.parseAndNormalize("#pragma omp parallel \n {}",
ParallelConstruct.class);
InsertImmediatePredecessor.insert(callStmt, parConsNew);
NodeRemover.removeNode(callStmt);
for (OmpClause newClause : newClauses) {
OmpClause newClauseCopy = OmpClauseInfo.getCopy(newClause);
parConsNew.getInfo().addClause(newClauseCopy);
}
CompoundStatement newParBody = (CompoundStatement) parConsNew.getInfo().getCFGInfo().getBody();
newParBody.getInfo().getCFGInfo().addElement(callStmt);
}
return true;
}
public static boolean remainingElementsFusableAcrossFunction(FunctionDefinition funcDef,
ParallelConstruct parCons) {
CompoundStatement funcBody = funcDef.getInfo().getCFGInfo().getBody();
List<Node> elementList = funcBody.getInfo().getCFGInfo().getElementList();
int parIndex = elementList.indexOf(parCons);
assert (parIndex >= 0);
for (int i = parIndex + 1; i < elementList.size(); i++) {
Node nextNode = elementList.get(i);
if (nextNode instanceof Declaration || nextNode instanceof UnknownCpp || nextNode instanceof UnknownPragma
|| nextNode instanceof OmpDirective || nextNode instanceof OmpConstruct) {
return false;
}
CellSet readCells = nextNode.getInfo().getSymbolReads();
List<OmpClause> clauseList = parCons.getInfo().getOmpClauseList();
for (OmpClause clause : clauseList) {
Set<String> listedNames = clause.getInfo().getListedNames();
if (clause instanceof OmpReductionClause) {
for (Cell c : readCells) {
if (c instanceof Symbol) {
Symbol sym = (Symbol) c;
if (listedNames.contains(sym.getName())) {
return false;
}
}
}
}
}
Set<Node> contents = nextNode.getInfo().getCFGInfo().getLexicalCFGLeafContents();
for (Node node : contents) {
if (node instanceof PreCallNode) {
return false;
}
if (node instanceof BeginNode) {
BeginNode bn = (BeginNode) node;
if (bn.getParent() instanceof OmpConstruct) {
return false;
}
}
if (node instanceof OmpDirective || node instanceof UnknownCpp || node instanceof UnknownPragma) {
return false;
}
if (node.getInfo().mayWrite()) {
return false;
}
}
}
List<OmpClause> clauseList = parCons.getInfo().getOmpClauseList();
for (int i = parIndex - 1; i >= 0; i--) {
Node prevNode = elementList.get(i);
if (prevNode instanceof Declaration) {
return false;
}
if (prevNode instanceof UnknownCpp || prevNode instanceof UnknownPragma || prevNode instanceof OmpDirective
|| prevNode instanceof OmpConstruct) {
return false;
}
CellSet readCells = prevNode.getInfo().getSymbolReads();
for (OmpClause clause : clauseList) {
Set<String> listedNames = clause.getInfo().getListedNames();
if (clause instanceof OmpReductionClause) {
for (Cell c : readCells) {
if (c instanceof Symbol) {
Symbol sym = (Symbol) c;
if (listedNames.contains(sym.getName())) {
return false;
}
}
}
}
}
Set<Node> contents = prevNode.getInfo().getCFGInfo().getLexicalCFGLeafContents();
for (Node node : contents) {
if (node instanceof PreCallNode) {
return false;
}
if (node instanceof BeginNode) {
BeginNode bn = (BeginNode) node;
if (bn.getParent() instanceof OmpConstruct) {
return false;
}
}
if (node instanceof OmpDirective || node instanceof UnknownCpp || node instanceof UnknownPragma) {
return false;
}
if (node.getInfo().mayWrite()) {
return false;
}
}
}
return true;
}
public static boolean remainingElementsFusableAcrossIteration(IterationStatement itStmt,
ParallelConstruct parCons) {
CompoundStatement itBody = null;
itStmt = (IterationStatement) Misc.getCFGNodeFor(itStmt);
if (itStmt instanceof WhileStatement) {
itBody = (CompoundStatement) ((WhileStatement) itStmt).getInfo().getCFGInfo().getBody();
} else if (itStmt instanceof DoStatement) {
itBody = (CompoundStatement) ((DoStatement) itStmt).getInfo().getCFGInfo().getBody();
} else if (itStmt instanceof ForStatement) {
itBody = (CompoundStatement) ((ForStatement) itStmt).getInfo().getCFGInfo().getBody();
} else {
return false;
}
List<Node> elementList = itBody.getInfo().getCFGInfo().getElementList();
int parIndex = elementList.indexOf(parCons);
assert (parIndex >= 0);
for (int i = parIndex + 1; i < elementList.size(); i++) {
Node nextNode = elementList.get(i);
if (nextNode instanceof Declaration || nextNode instanceof UnknownCpp || nextNode instanceof UnknownPragma
|| nextNode instanceof OmpDirective || nextNode instanceof OmpConstruct) {
return false;
}
CellSet readCells = nextNode.getInfo().getSymbolReads();
List<OmpClause> clauseList = parCons.getInfo().getOmpClauseList();
for (OmpClause clause : clauseList) {
if (clause instanceof OmpReductionClause) {
for (Cell c : readCells) {
Set<String> listedNames = clause.getInfo().getListedNames();
if (c instanceof Symbol) {
Symbol sym = (Symbol) c;
if (listedNames.contains(sym.getName())) {
return false;
}
}
}
}
}
Set<Node> contents = nextNode.getInfo().getCFGInfo().getLexicalCFGLeafContents();
for (Node node : contents) {
if (node instanceof JumpStatement) {
if (node.getInfo().getCFGInfo().getSuccBlocks().stream().anyMatch(s -> !contents.contains(s))) {
return false;
}
}
if (node instanceof PreCallNode) {
return false;
}
if (node instanceof BeginNode) {
BeginNode bn = (BeginNode) node;
if (bn.getParent() instanceof OmpConstruct) {
return false;
}
}
if (node instanceof OmpDirective || node instanceof UnknownCpp || node instanceof UnknownPragma) {
return false;
}
if (node.getInfo().mayWrite()) {
return false;
}
}
}
List<OmpClause> clauseList = parCons.getInfo().getOmpClauseList();
for (int i = parIndex - 1; i >= 0; i--) {
Node prevNode = elementList.get(i);
if (prevNode instanceof Declaration) {
return false;
}
if (prevNode instanceof UnknownCpp || prevNode instanceof UnknownPragma || prevNode instanceof OmpDirective
|| prevNode instanceof OmpConstruct) {
return false;
}
CellSet readCells = prevNode.getInfo().getSymbolReads();
for (OmpClause clause : clauseList) {
if (clause instanceof OmpReductionClause) {
Set<String> listedNames = clause.getInfo().getListedNames();
for (Cell c : readCells) {
if (c instanceof Symbol) {
Symbol sym = (Symbol) c;
if (listedNames.contains(sym.getName())) {
return false;
}
}
}
}
}
Set<Node> contents = prevNode.getInfo().getCFGInfo().getLexicalCFGLeafContents();
for (Node node : contents) {
if (node instanceof JumpStatement) {
if (node.getInfo().getCFGInfo().getSuccBlocks().stream().anyMatch(s -> !contents.contains(s))) {
return false;
}
}
if (node instanceof PreCallNode) {
return false;
}
if (node instanceof BeginNode) {
BeginNode bn = (BeginNode) node;
if (bn.getParent() instanceof OmpConstruct) {
return false;
}
}
if (node instanceof OmpDirective || node instanceof UnknownCpp || node instanceof UnknownPragma) {
return false;
}
if (node.getInfo().mayWrite()) {
return false;
}
}
}
return true;
}
private static boolean mergeDataAttributeListsBelow(ParallelConstruct parConsAbove,
ParallelConstruct parConsBelow) {
boolean changed = false;
List<OmpClause> thisClauses = parConsAbove.getInfo().getOmpClauseList();
List<OmpClause> nextClauses = parConsBelow.getInfo().getOmpClauseList();
for (OmpClause thisClause : thisClauses) {
if (thisClause instanceof IfClause || thisClause instanceof NumThreadsClause
|| thisClause instanceof OmpFirstPrivateClause || thisClause instanceof OmpLastPrivateClause
|| thisClause instanceof OmpCopyinClause || thisClause instanceof OmpReductionClause) {
return changed;
}
}
for (OmpClause nextClause : nextClauses) {
if (nextClause instanceof IfClause || nextClause instanceof NumThreadsClause
|| nextClause instanceof OmpFirstPrivateClause || nextClause instanceof OmpLastPrivateClause
|| nextClause instanceof OmpCopyinClause || nextClause instanceof OmpReductionClause) {
return changed;
}
}
OmpClause thisDfltClause = null;
OmpClause nextDfltClause = null;
for (OmpClause thisClause : thisClauses) {
if (thisClause instanceof OmpDfltSharedClause || thisClause instanceof OmpDfltNoneClause) {
thisDfltClause = thisClause;
break;
}
}
for (OmpClause nextClause : nextClauses) {
if (nextClause instanceof OmpDfltSharedClause || nextClause instanceof OmpDfltNoneClause) {
nextDfltClause = nextClause;
break;
}
}
if (thisDfltClause == null) {
if (nextDfltClause != null) {
return changed;
}
}
if (nextDfltClause == null) {
if (thisDfltClause != null) {
return changed;
}
}
if (thisDfltClause instanceof OmpDfltSharedClause || thisDfltClause instanceof OmpDfltNoneClause) {
if (nextDfltClause == null) {
return changed;
}
String thisDfltName = thisDfltClause.getClass().getSimpleName();
String nextDfltName = nextDfltClause.getClass().getSimpleName();
if (!thisDfltName.contentEquals(nextDfltName)) {
return changed;
}
}
List<String> thisSharedIDs = new ArrayList<>();
List<String> thisPrivateIDs = new ArrayList<>();
List<String> nextSharedIDs = new ArrayList<>();
List<String> nextPrivateIDs = new ArrayList<>();
OmpSharedClause thisSharedClause = null;
OmpPrivateClause thisPrivateClause = null;
OmpSharedClause nextSharedClause = null;
OmpPrivateClause nextPrivateClause = null;
for (OmpClause thisClause : thisClauses) {
if (thisClause instanceof OmpSharedClause) {
thisSharedClause = (OmpSharedClause) thisClause;
VariableList vL = thisSharedClause.getF2();
thisSharedIDs.addAll(Misc.obtainVarNames(vL));
} else if (thisClause instanceof OmpPrivateClause) {
thisPrivateClause = (OmpPrivateClause) thisClause;
VariableList vL = thisPrivateClause.getF2();
thisPrivateIDs.addAll(Misc.obtainVarNames(vL));
}
}
for (OmpClause nextClause : nextClauses) {
if (nextClause instanceof OmpSharedClause) {
nextSharedClause = (OmpSharedClause) nextClause;
VariableList vL = nextSharedClause.getF2();
nextSharedIDs.addAll(Misc.obtainVarNames(vL));
} else if (nextClause instanceof OmpPrivateClause) {
nextPrivateClause = (OmpPrivateClause) nextClause;
VariableList vL = nextPrivateClause.getF2();
nextPrivateIDs.addAll(Misc.obtainVarNames(vL));
}
}
if (Misc.doIntersect(thisSharedIDs, nextPrivateIDs) || Misc.doIntersect(thisPrivateIDs, nextSharedIDs)) {
return changed;
}
if (nextSharedClause != null) {
VariableList vL = nextSharedClause.getF2();
Misc.addVarNames(vL, thisSharedIDs);
} else {
if (thisSharedClause != null) {
parConsBelow.getInfo().addClause(thisSharedClause);
}
}
if (nextPrivateClause != null) {
VariableList vL = nextPrivateClause.getF2();
Misc.addVarNames(vL, thisPrivateIDs);
} else {
if (thisPrivateClause != null) {
parConsBelow.getInfo().addClause(thisPrivateClause);
}
}
changed = true;
return changed;
}
private static boolean merge(ParallelConstruct parConsAbove, ParallelConstruct parConsBelow) {
boolean changed = false;
List<OmpClause> thisClauses = parConsAbove.getInfo().getOmpClauseList();
List<OmpClause> nextClauses = parConsBelow.getInfo().getOmpClauseList();
for (OmpClause thisClause : thisClauses) {
if (thisClause instanceof IfClause || thisClause instanceof NumThreadsClause
|| thisClause instanceof OmpFirstPrivateClause || thisClause instanceof OmpLastPrivateClause
|| thisClause instanceof OmpCopyinClause || thisClause instanceof OmpReductionClause) {
return changed;
}
}
for (OmpClause nextClause : nextClauses) {
if (nextClause instanceof IfClause || nextClause instanceof NumThreadsClause
|| nextClause instanceof OmpFirstPrivateClause || nextClause instanceof OmpLastPrivateClause
|| nextClause instanceof OmpCopyinClause || nextClause instanceof OmpReductionClause) {
return changed;
}
}
OmpClause thisDfltClause = null;
OmpClause nextDfltClause = null;
for (OmpClause thisClause : thisClauses) {
if (thisClause instanceof OmpDfltSharedClause || thisClause instanceof OmpDfltNoneClause) {
thisDfltClause = thisClause;
break;
}
}
for (OmpClause nextClause : nextClauses) {
if (nextClause instanceof OmpDfltSharedClause || nextClause instanceof OmpDfltNoneClause) {
nextDfltClause = nextClause;
break;
}
}
if (thisDfltClause == null) {
if (nextDfltClause != null) {
return changed;
}
}
if (nextDfltClause == null) {
if (thisDfltClause != null) {
return changed;
}
}
if (thisDfltClause instanceof OmpDfltSharedClause || thisDfltClause instanceof OmpDfltNoneClause) {
if (nextDfltClause == null) {
return changed;
}
String thisDfltName = thisDfltClause.getClass().getSimpleName();
String nextDfltName = nextDfltClause.getClass().getSimpleName();
if (!thisDfltName.contentEquals(nextDfltName)) {
return changed;
}
}
List<String> thisSharedIDs = new ArrayList<>();
List<String> thisPrivateIDs = new ArrayList<>();
List<String> nextSharedIDs = new ArrayList<>();
List<String> nextPrivateIDs = new ArrayList<>();
OmpSharedClause thisSharedClause = null;
OmpPrivateClause thisPrivateClause = null;
OmpSharedClause nextSharedClause = null;
OmpPrivateClause nextPrivateClause = null;
for (OmpClause thisClause : thisClauses) {
if (thisClause instanceof OmpSharedClause) {
thisSharedClause = (OmpSharedClause) thisClause;
VariableList vL = thisSharedClause.getF2();
thisSharedIDs.addAll(Misc.obtainVarNames(vL));
} else if (thisClause instanceof OmpPrivateClause) {
thisPrivateClause = (OmpPrivateClause) thisClause;
VariableList vL = thisPrivateClause.getF2();
thisPrivateIDs.addAll(Misc.obtainVarNames(vL));
}
}
for (OmpClause nextClause : nextClauses) {
if (nextClause instanceof OmpSharedClause) {
nextSharedClause = (OmpSharedClause) nextClause;
VariableList vL = nextSharedClause.getF2();
nextSharedIDs.addAll(Misc.obtainVarNames(vL));
} else if (nextClause instanceof OmpPrivateClause) {
nextPrivateClause = (OmpPrivateClause) nextClause;
VariableList vL = nextPrivateClause.getF2();
nextPrivateIDs.addAll(Misc.obtainVarNames(vL));
}
}
if (Misc.doIntersect(thisSharedIDs, nextPrivateIDs) || Misc.doIntersect(thisPrivateIDs, nextSharedIDs)) {
return changed;
}
if (thisSharedClause != null) {
VariableList vL = thisSharedClause.getF2();
Misc.addVarNames(vL, nextSharedIDs);
} else {
if (nextSharedClause != null) {
parConsAbove.getInfo().addClause(nextSharedClause);
}
}
if (thisPrivateClause != null) {
VariableList vL = thisPrivateClause.getF2();
Misc.addVarNames(vL, nextPrivateIDs);
} else {
if (nextPrivateClause != null) {
parConsAbove.getInfo().addClause(nextPrivateClause);
}
}
CompoundStatement nextBody = (CompoundStatement) parConsBelow.getInfo().getCFGInfo().getBody();
CompoundStatement thisBody = (CompoundStatement) parConsAbove.getInfo().getCFGInfo().getBody();
CompoundStatement scope = (CompoundStatement) Misc.getEnclosingCFGNonLeafNode(parConsAbove);
parConsBelow.getInfo().getCFGInfo().setBody(FrontEnd.parseAndNormalize(";", Statement.class));
scope.getInfo().getCFGInfo().removeElement(parConsBelow);
ProfileSS.insertCP();
BarrierDirective barr = FrontEnd.parseAndNormalize("#pragma omp barrier \n", BarrierDirective.class);
thisBody.getInfo().getCFGInfo().addAtLast(barr);
ProfileSS.insertCP();
thisBody.getInfo().getCFGInfo().addAtLast(nextBody);
ProfileSS.insertCP();
thisBody.getInfo().removeExtraScopes();
ProfileSS.insertCP();
return true;
}
private static boolean mergeDataAttributeLists(ParallelConstruct parConsAbove, ParallelConstruct parConsBelow) {
boolean changed = false;
List<OmpClause> thisClauses = parConsAbove.getInfo().getOmpClauseList();
List<OmpClause> nextClauses = parConsBelow.getInfo().getOmpClauseList();
for (OmpClause thisClause : thisClauses) {
if (thisClause instanceof IfClause || thisClause instanceof NumThreadsClause
|| thisClause instanceof OmpFirstPrivateClause || thisClause instanceof OmpLastPrivateClause
|| thisClause instanceof OmpCopyinClause || thisClause instanceof OmpReductionClause) {
return changed;
}
}
for (OmpClause nextClause : nextClauses) {
if (nextClause instanceof IfClause || nextClause instanceof NumThreadsClause
|| nextClause instanceof OmpFirstPrivateClause || nextClause instanceof OmpLastPrivateClause
|| nextClause instanceof OmpCopyinClause || nextClause instanceof OmpReductionClause) {
return changed;
}
}
OmpClause thisDfltClause = null;
OmpClause nextDfltClause = null;
for (OmpClause thisClause : thisClauses) {
if (thisClause instanceof OmpDfltSharedClause || thisClause instanceof OmpDfltNoneClause) {
thisDfltClause = thisClause;
break;
}
}
for (OmpClause nextClause : nextClauses) {
if (nextClause instanceof OmpDfltSharedClause || nextClause instanceof OmpDfltNoneClause) {
nextDfltClause = nextClause;
break;
}
}
if (thisDfltClause == null) {
if (nextDfltClause != null) {
return changed;
}
}
if (nextDfltClause == null) {
if (thisDfltClause != null) {
return changed;
}
}
if (thisDfltClause instanceof OmpDfltSharedClause || thisDfltClause instanceof OmpDfltNoneClause) {
if (nextDfltClause == null) {
return changed;
}
String thisDfltName = thisDfltClause.getClass().getSimpleName();
String nextDfltName = nextDfltClause.getClass().getSimpleName();
if (!thisDfltName.contentEquals(nextDfltName)) {
return changed;
}
}
List<String> thisSharedIDs = new ArrayList<>();
List<String> thisPrivateIDs = new ArrayList<>();
List<String> nextSharedIDs = new ArrayList<>();
List<String> nextPrivateIDs = new ArrayList<>();
OmpSharedClause thisSharedClause = null;
OmpPrivateClause thisPrivateClause = null;
OmpSharedClause nextSharedClause = null;
OmpPrivateClause nextPrivateClause = null;
for (OmpClause thisClause : thisClauses) {
if (thisClause instanceof OmpSharedClause) {
thisSharedClause = (OmpSharedClause) thisClause;
VariableList vL = thisSharedClause.getF2();
thisSharedIDs.addAll(Misc.obtainVarNames(vL));
} else if (thisClause instanceof OmpPrivateClause) {
thisPrivateClause = (OmpPrivateClause) thisClause;
VariableList vL = thisPrivateClause.getF2();
thisPrivateIDs.addAll(Misc.obtainVarNames(vL));
}
}
for (OmpClause nextClause : nextClauses) {
if (nextClause instanceof OmpSharedClause) {
nextSharedClause = (OmpSharedClause) nextClause;
VariableList vL = nextSharedClause.getF2();
nextSharedIDs.addAll(Misc.obtainVarNames(vL));
} else if (nextClause instanceof OmpPrivateClause) {
nextPrivateClause = (OmpPrivateClause) nextClause;
VariableList vL = nextPrivateClause.getF2();
nextPrivateIDs.addAll(Misc.obtainVarNames(vL));
}
}
if (Misc.doIntersect(thisSharedIDs, nextPrivateIDs) || Misc.doIntersect(thisPrivateIDs, nextSharedIDs)) {
return changed;
}
if (thisSharedClause != null) {
VariableList vL = thisSharedClause.getF2();
Misc.addVarNames(vL, nextSharedIDs);
} else {
if (nextSharedClause != null) {
parConsAbove.getInfo().addClause(nextSharedClause);
}
}
if (thisPrivateClause != null) {
VariableList vL = thisPrivateClause.getF2();
Misc.addVarNames(vL, nextPrivateIDs);
} else {
if (nextPrivateClause != null) {
parConsAbove.getInfo().addClause(nextPrivateClause);
}
}
changed = true;
return changed;
}
}
