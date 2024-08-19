package imop.lib.transform.percolate;
import imop.ast.info.cfgNodeInfo.BarrierDirectiveInfo;
import imop.ast.node.external.*;
import imop.ast.node.internal.*;
import imop.lib.analysis.flowanalysis.Cell;
import imop.lib.analysis.flowanalysis.Symbol;
import imop.lib.analysis.mhp.AbstractPhase;
import imop.lib.analysis.mhp.incMHP.BeginPhasePoint;
import imop.lib.analysis.mhp.incMHP.EndPhasePoint;
import imop.lib.analysis.mhp.incMHP.Phase;
import imop.lib.analysis.typesystem.ArrayType;
import imop.lib.analysis.typesystem.FunctionType;
import imop.lib.cfg.CFGLinkFinder;
import imop.lib.cfg.info.CompoundStatementCFGInfo;
import imop.lib.cfg.link.node.CFGLink;
import imop.lib.cg.CallStack;
import imop.lib.cg.NodeWithStack;
import imop.lib.transform.simplify.RedundantSynchronizationRemoval;
import imop.lib.transform.updater.InsertImmediatePredecessor;
import imop.lib.transform.updater.NodeRemover;
import imop.lib.transform.updater.NodeReplacer;
import imop.lib.transform.updater.sideeffect.AddedDFDPredecessor;
import imop.lib.transform.updater.sideeffect.RemovedDFDSuccessor;
import imop.lib.transform.updater.sideeffect.SideEffect;
import imop.lib.util.*;
import imop.parser.FrontEnd;
import imop.parser.Program;
import java.util.*;
import java.util.stream.Collectors;
public class LoopInstructionsRescheduler {
public static int counter = 0;
public static int recCounter = 0;
public static void resetStaticFields() {
LoopInstructionsRescheduler.counter = LoopInstructionsRescheduler.recCounter = 0;
}
public static void tryDifferentSchedulings(IterationStatement itStmt, int minPhaseCount) {
boolean changed;
int finalPhaseCount;
do {
changed = false;
DriverModule.performCopyElimination(itStmt);
for (ParallelConstruct parCons : Misc.getExactEnclosee(Program.getRoot(), ParallelConstruct.class)) {
UpwardPercolater.initPercolate(parCons);
}
int initialPhaseCount = DriverModule.phaseCountPerExecution(itStmt);
if (initialPhaseCount <= minPhaseCount) {
System.out.println("New phase count in the main body: " + DriverModule.phaseCountPerExecution(itStmt));
return;
}
if (!(itStmt instanceof WhileStatement)) {
System.out.println("New phase count in the main body: " + DriverModule.phaseCountPerExecution(itStmt));
return;
}
WhileStatement whileStmt = (WhileStatement) itStmt;
CellList readCellList = new CellList();
CellList writtenCellList = new CellList();
Set<Node> internalNodes = whileStmt.getInfo().getCFGInfo().getIntraTaskCFGLeafContents();
internalNodes.stream().forEach(n -> {
readCellList.addAll(n.getInfo().getSharedReads());
writtenCellList.addAll(n.getInfo().getSharedWrites());
});
CellSet multiAccessedCells = new CellSet();
for (Cell c : writtenCellList) {
if (c == Cell.getNullCell()) {
continue;
}
if (c instanceof Symbol) {
Symbol s = (Symbol) c;
if (s.getType() instanceof FunctionType || s.getType() instanceof ArrayType) {
continue;
}
}
int freq = Collections.frequency(writtenCellList.getReadOnlyInternal(), c)
+ Collections.frequency(readCellList.getReadOnlyInternal(), c);
if (freq < 2) {
continue;
}
multiAccessedCells.add(c);
}
CellMap<List<Node>> freeInstMap = new CellMap<>();
for (Cell c : new CellSet(multiAccessedCells)) {
freeInstMap.put(c, getFreeInstructionsInIteration(itStmt, c, multiAccessedCells, internalNodes));
}
for (Cell cell : multiAccessedCells) {
List<Node> freeInstructionNodes = freeInstMap.get(cell);
if (freeInstructionNodes.isEmpty()) {
continue;
}
List<Node> orderedNodes = LoopInstructionsRescheduler.obtainOrderedNodes(freeInstructionNodes,
whileStmt, cell);
for (Node freeInst : orderedNodes) {
System.err.println("\t\tMoving the following free instruction: "
+ (Misc.isCFGLeafNode(freeInst) ? freeInst.toString()
: freeInst.getClass().getSimpleName() + " for " + cell));
if (moveFreeInstructionAgainstControlFlowInLoop(freeInst, cell, multiAccessedCells, itStmt,
internalNodes)) {
changed = true;
RedundantSynchronizationRemoval.removeBarriers(itStmt);
}
}
}
finalPhaseCount = DriverModule.phaseCountPerExecution(itStmt);
if (finalPhaseCount <= minPhaseCount) {
break;
}
while (DriverModule.performCopyElimination(itStmt)) {
changed = true;
}
} while (changed);
System.out.println("New phase count in the main body: " + finalPhaseCount);
}
private static List<Node> obtainOrderedNodes(List<Node> freeInstructionNodes, WhileStatement whileStmt, Cell cell) {
List<Node> orderedFreeNodes = new ArrayList<>();
Set<Node> internalNodes = whileStmt.getInfo().getCFGInfo().getIntraTaskCFGLeafContents();
if (freeInstructionNodes.isEmpty()) {
return orderedFreeNodes;
}
HashMap<Node, Node> freeCFGNodes = new HashMap<>();
for (Node firstFree : freeInstructionNodes) {
Node firstFreeLeaf;
if (Misc.isCFGLeafNode(firstFree)) {
firstFreeLeaf = firstFree;
} else {
assert (Misc.isCFGNode(firstFree));
firstFreeLeaf = firstFree.getInfo().getCFGInfo().getNestedCFG().getBegin();
}
if (firstFreeLeaf.getInfo().getNodePhaseInfo().getPhaseSet().isEmpty()) {
continue;
}
freeCFGNodes.put(firstFree, firstFreeLeaf);
}
Node firstFree = null;
for (Node first : freeInstructionNodes) {
if (!first.getInfo().getNodePhaseInfo().getPhaseSet().isEmpty()) {
firstFree = first;
break;
}
}
for (Node n : freeCFGNodes.keySet()) {
if (internalNodes.contains(freeCFGNodes.get(n))) {
firstFree = n;
break;
}
}
if (firstFree == null) {
return orderedFreeNodes;
}
orderedFreeNodes.add(firstFree);
Set<Phase> firstPhases = (Set<Phase>) freeCFGNodes.get(firstFree).getInfo().getNodePhaseInfo().getPhaseSet();
Phase insidePhase = null;
outer: for (Phase ph : firstPhases) {
for (EndPhasePoint epp : ph.getEndPoints()) {
if (!(epp.getNode() instanceof BarrierDirective)) {
continue;
}
BarrierDirective barrier = (BarrierDirective) epp.getNode();
if (internalNodes.contains(barrier)) {
insidePhase = ph;
break outer;
}
}
}
assert (insidePhase != null);
Phase currentPhase = insidePhase;
Set<Phase> visitedPhases = new HashSet<>();
boolean foundConstrained = false;
int fixedIndex = 0;
while (currentPhase != null) {
if (!visitedPhases.contains(currentPhase)) {
visitedPhases.add(currentPhase);
} else {
break;
}
if (!freeInstructionNodes.stream().anyMatch(n -> !orderedFreeNodes.contains(n))) {
break;
}
for (Node accessNode : currentPhase.getNodeSet().stream()
.filter(n -> n.getInfo().getSharedAccesses().contains(cell)).collect(Collectors.toList())) {
if (!freeInstructionNodes.stream().anyMatch(
f -> f.getInfo().getCFGInfo().getReachableIntraTaskCFGLeafContents().contains(accessNode))) {
foundConstrained = true;
}
}
for (Node freeOuterNode : freeInstructionNodes) {
if (orderedFreeNodes.contains(freeOuterNode)) {
continue;
}
Node freeCFGNode = freeCFGNodes.get(freeOuterNode);
if (freeCFGNode.getInfo().getNodePhaseInfo().getPhaseSet().contains(currentPhase)) {
if (!foundConstrained) {
orderedFreeNodes.add(freeOuterNode);
} else {
orderedFreeNodes.add(fixedIndex++, freeOuterNode);
}
}
}
currentPhase = currentPhase.getSuccPhase();
}
return orderedFreeNodes;
}
private static List<Node> getFreeInstructionsInIteration(IterationStatement itStmt, Cell cell,
CellSet multiAccessedCells, Set<Node> internalNodes) {
assert (multiAccessedCells.contains(cell));
List<Node> instructionNodes = new ArrayList<>();
List<Node> constrainedNodes = new ArrayList<>();
if (!(itStmt instanceof WhileStatement)) {
return instructionNodes;
}
WhileStatement whileStmt = (WhileStatement) itStmt;
Expression predicate = whileStmt.getInfo().getCFGInfo().getPredicate();
if (!predicate.getInfo().evaluatesToConstant()) {
return instructionNodes;
}
CompoundStatement loopBody = (CompoundStatement) whileStmt.getInfo().getCFGInfo().getBody();
List<Node> loopElements = loopBody.getInfo().getCFGInfo().getElementList();
Set<Phase> visitedPhases = new HashSet<>();
for (AbstractPhase<?, ?> absPh : predicate.getInfo().getNodePhaseInfo().getPhaseSet()) {
Phase ph = (Phase) absPh;
List<Node> thisPhaseNodeList;
Set<BarrierDirective> visitedBarriers = new HashSet<>();
for (BeginPhasePoint bpp : ph.getBeginPoints()) {
if (!(bpp.getNode() instanceof BarrierDirective)) {
continue;
}
BarrierDirective barrier = (BarrierDirective) bpp.getNode();
if (visitedBarriers.contains(barrier)) {
continue;
} else {
visitedBarriers.add(barrier);
}
if (!internalNodes.contains(barrier)) {
continue;
}
thisPhaseNodeList = new ArrayList<>();
CFGLink barrCFGLink = CFGLinkFinder.getCFGLinkFor(barrier);
CompoundStatement barrCompStmt = (CompoundStatement) barrCFGLink.getEnclosingNode();
List<Node> compList = barrCompStmt.getInfo().getCFGInfo().getElementList();
int barrIndex = compList.indexOf(barrier);
for (int i = barrIndex + 1; i < compList.size(); i++) {
Node element = compList.get(i);
if (element instanceof BarrierDirective) {
break;
}
if (Misc.getInheritedEnclosee(element, ContinueStatement.class).stream()
.anyMatch(c -> c.getInfo().getTargetPredicate() == predicate)) {
thisPhaseNodeList.add(element);
break;
}
thisPhaseNodeList.add(element);
}
for (int i = 0; i < loopElements.size(); i++) {
Node element = loopElements.get(i);
if (element instanceof BarrierDirective) {
break;
}
if (element instanceof DummyFlushDirective) {
DummyFlushDirective dfd = (DummyFlushDirective) element;
if (dfd.getDummyFlushType() == DummyFlushType.BARRIER_START) {
break;
}
}
if (Misc.isCFGNode(element)) {
Set<NodeWithStack> elementInternalNodes = element.getInfo().getCFGInfo()
.getIntraTaskCFGLeafContentsOfSameParLevel(new CallStack());
if (elementInternalNodes.stream().filter(ns -> ns.getNode() instanceof BarrierDirective)
.count() > 0) {
if (element instanceof IfStatement) {
Set<NodeWithStack> endingElements = new HashSet<>();
Node startingEndNode = element.getInfo().getCFGInfo().getNestedCFG().getEnd();
Node endingBeginNode = element.getInfo().getCFGInfo().getNestedCFG().getBegin();
Set<Node> outJumpSources = element.getInfo().getOutJumpSources();
CollectorVisitor.collectNodeSetInGenericGraph(
new NodeWithStack(startingEndNode, new CallStack()), endingElements, (n) -> {
if (outJumpSources.stream().anyMatch(nS -> nS == n.getNode())
|| n.getNode() == endingBeginNode) {
return true;
}
if (n.getNode() instanceof BarrierDirective) {
return true;
}
return false;
},
(n) -> n.getNode().getInfo().getCFGInfo()
.getParallelConstructFreeInterProceduralLeafPredecessors(
n.getCallStack()));
if (endingElements.stream().anyMatch(nS -> nS.getNode() instanceof BarrierDirective)) {
break;
}
} else {
break;
}
}
}
thisPhaseNodeList.add(element);
}
processPhaseNodeList(thisPhaseNodeList, instructionNodes, constrainedNodes, cell, multiAccessedCells);
}
Phase thisPhase = ph;
visitedPhases = new HashSet<>();
visitedPhases.add(ph);
while (true) {
thisPhase = thisPhase.getSuccPhase();
if (thisPhase == null) {
break;
}
if (!thisPhase.getEndPoints().parallelStream().anyMatch(n -> internalNodes.contains(n.getNode()))) {
break;
}
if (thisPhase.getNodeSet().parallelStream().anyMatch(n -> n == predicate)) {
break;
}
if (visitedPhases.contains(thisPhase)) {
break;
} else {
visitedPhases.add(thisPhase);
}
for (BeginPhasePoint bpp : thisPhase.getBeginPoints()) {
thisPhaseNodeList = new ArrayList<>();
if (!(bpp.getNode() instanceof BarrierDirective)) {
continue;
}
BarrierDirective barrier = (BarrierDirective) bpp.getNode();
if (!internalNodes.contains(barrier)) {
continue;
}
CFGLink barrCFGLink = CFGLinkFinder.getCFGLinkFor(barrier);
if (barrCFGLink == null) {
continue;
}
CompoundStatement barrCompStmt = (CompoundStatement) barrCFGLink.getEnclosingNode();
List<Node> compList = barrCompStmt.getInfo().getCFGInfo().getElementList();
int barrIndex = compList.indexOf(barrier);
for (int i = barrIndex + 1; i < loopElements.size(); i++) {
Node element = loopElements.get(i);
if (element instanceof BarrierDirective) {
break;
}
if (element instanceof DummyFlushDirective) {
DummyFlushDirective dfd = (DummyFlushDirective) element;
if (dfd.getDummyFlushType() == DummyFlushType.BARRIER_START) {
break;
}
}
if (Misc.isCFGNode(element)) {
Set<NodeWithStack> elementInternalNodes = element.getInfo().getCFGInfo()
.getIntraTaskCFGLeafContentsOfSameParLevel(new CallStack());
if (elementInternalNodes.stream().filter(ns -> ns.getNode() instanceof BarrierDirective)
.count() > 0) {
if (element instanceof IfStatement) {
Set<NodeWithStack> endingElements = new HashSet<>();
Node startingEndNode = element.getInfo().getCFGInfo().getNestedCFG().getEnd();
Node endingBeginNode = element.getInfo().getCFGInfo().getNestedCFG().getBegin();
Set<Node> outJumpSources = element.getInfo().getOutJumpSources();
CollectorVisitor.collectNodeSetInGenericGraph(
new NodeWithStack(startingEndNode, new CallStack()), endingElements,
(n) -> {
if (outJumpSources.stream().anyMatch(nS -> nS == n.getNode())
|| n.getNode() == endingBeginNode) {
return true;
}
if (n.getNode() instanceof BarrierDirective) {
return true;
}
return false;
},
(n) -> n.getNode().getInfo().getCFGInfo()
.getParallelConstructFreeInterProceduralLeafPredecessors(
n.getCallStack()));
if (endingElements.stream()
.anyMatch(nS -> nS.getNode() instanceof BarrierDirective)) {
break;
}
} else {
break;
}
}
}
thisPhaseNodeList.add(element);
}
processPhaseNodeList(thisPhaseNodeList, instructionNodes, constrainedNodes, cell,
multiAccessedCells);
}
}
}
if (constrainedNodes.isEmpty()) {
multiAccessedCells.remove(cell);
return new ArrayList<>();
} else {
instructionNodes.removeAll(constrainedNodes);
if (instructionNodes.isEmpty()) {
multiAccessedCells.remove(cell);
}
return instructionNodes;
}
}
private static void processPhaseNodeList(List<Node> thisPhaseNodeList, List<Node> instructionNodes,
List<Node> constrainedNodes, Cell cell, CellSet multiAccessedCells) {
List<Node> thisInstructionNodes = new ArrayList<>();
for (Node inst : thisPhaseNodeList) {
if (inst.getInfo().getSharedAccesses().contains(cell)) {
thisInstructionNodes.add(inst);
}
}
if (thisInstructionNodes.isEmpty()) {
return;
}
thisInstructionNodes.removeAll(thisInstructionNodes.stream().filter(n -> !n.getInfo().isControlConfined())
.collect(Collectors.toSet()));
if (thisInstructionNodes.isEmpty()) {
return;
}
if (thisInstructionNodes.size() > 1) {
constrainedNodes.addAll(thisInstructionNodes);
return;
}
Node instructionNode = thisInstructionNodes.get(0);
if (constrainedNodes.contains(instructionNode)) {
return;
}
for (Node otherNode : thisPhaseNodeList) {
if (otherNode instanceof Statement) {
Statement otherStmt = (Statement) otherNode;
if (!otherStmt.getInfo().getLabelAnnotations().isEmpty()) {
if (!constrainedNodes.contains(instructionNode)) {
constrainedNodes.add(instructionNode);
}
return;
}
}
if (Misc.isCFGNode(otherNode)) {
if (!otherNode.getInfo().getInJumpDestinations().isEmpty()) {
if (!constrainedNodes.contains(instructionNode)) {
constrainedNodes.add(instructionNode);
}
return;
}
}
}
if (!LoopInstructionsRescheduler.testFrontReachability(thisPhaseNodeList, instructionNode, cell,
multiAccessedCells)) {
if (!constrainedNodes.contains(instructionNode)) {
constrainedNodes.add(instructionNode);
}
return;
}
if (!LoopInstructionsRescheduler.testBackReachability(thisPhaseNodeList, instructionNode, cell,
multiAccessedCells)) {
if (!constrainedNodes.contains(instructionNode)) {
constrainedNodes.add(instructionNode);
}
return;
}
instructionNodes.add(instructionNode);
}
private static boolean testFrontReachability(List<Node> nodeList, Node inst, Cell cell,
CellSet multiAccessedCells) {
if (!inst.getInfo().isControlConfined()) {
return false;
}
int instIndex = nodeList.indexOf(inst);
CellSet instReadSet = new CellSet(inst.getInfo().getReads());
CellSet instWriteSet = new CellSet(inst.getInfo().getWrites());
for (int i = 0; i < instIndex; i++) {
Node ancestor = nodeList.get(i);
boolean canJump = true;
CellSet ancestorReadSet = null;
CellSet ancestorWriteSet = null;
boolean controlConfined = ancestor.getInfo().isControlConfined();
if (controlConfined) {
ancestorReadSet = new CellSet(ancestor.getInfo().getReads());
ancestorWriteSet = new CellSet(ancestor.getInfo().getWrites());
if (Misc.doIntersect(instReadSet, ancestorWriteSet) || Misc.doIntersect(instWriteSet, ancestorReadSet)
|| Misc.doIntersect(instWriteSet, ancestorWriteSet)) {
canJump = false;
}
} else {
ancestorReadSet = new CellSet();
ancestorWriteSet = new CellSet();
if (!ancestor.getInfo().getInJumpDestinations().isEmpty()) {
return false;
}
for (Node bothWaysReachableLeaf : ancestor.getInfo().getCFGInfo()
.getBothBeginAndEndReachableIntraTaskLeafNodes()) {
ancestorReadSet.addAll(bothWaysReachableLeaf.getInfo().getReads());
ancestorWriteSet.addAll(bothWaysReachableLeaf.getInfo().getWrites());
}
if (Misc.doIntersect(instReadSet, ancestorWriteSet) || Misc.doIntersect(instWriteSet, ancestorReadSet)
|| Misc.doIntersect(instWriteSet, ancestorWriteSet)) {
canJump = false;
}
if (canJump && BarrierDirectiveInfo.killedIfPushedUpwards(ancestor, inst)) {
canJump = false;
}
}
if (canJump) {
continue;
} else {
if (ancestor.getInfo().getAccesses().getReadOnlyInternal().stream()
.anyMatch(c -> c != cell && multiAccessedCells.contains(c))) {
return false;
}
return testFrontReachability(nodeList, ancestor, cell, multiAccessedCells);
}
}
return true;
}
private static boolean testBackReachability(List<Node> nodeList, Node inst, Cell cell, CellSet multiAccessedCells) {
if (!inst.getInfo().isControlConfined()) {
return false;
}
int instIndex = nodeList.indexOf(inst);
CellSet instReadSet = new CellSet(inst.getInfo().getReads());
CellSet instWriteSet = new CellSet(inst.getInfo().getWrites());
for (int i = nodeList.size() - 1; i > instIndex; i--) {
Node descendant = nodeList.get(i);
boolean canJump = true;
CellSet descendantReadSet = new CellSet(descendant.getInfo().getReads());
CellSet descendantWriteSet = new CellSet(descendant.getInfo().getWrites());
if (BarrierDirectiveInfo.killedIfPushedUpwards(inst, descendant)) {
canJump = false;
}
if (Misc.doIntersect(instReadSet, descendantWriteSet) || Misc.doIntersect(instWriteSet, descendantReadSet)
|| Misc.doIntersect(instWriteSet, descendantWriteSet)) {
canJump = false;
}
if (canJump) {
continue;
} else {
if (descendant.getInfo().getAccesses().getReadOnlyInternal().stream()
.anyMatch(c -> c != cell && multiAccessedCells.contains(c))) {
return false;
}
return testBackReachability(nodeList, descendant, cell, multiAccessedCells);
}
}
return true;
}
private static boolean moveFreeInstructionAgainstControlFlowInLoop(final Node freeInst, Cell cell,
CellSet multiAccessedCells, IterationStatement itStmt, Set<Node> internalNodes) {
if (!(itStmt instanceof WhileStatement)) {
return false;
}
boolean hasFreeInstMoved = false;
CompoundStatementCFGInfo compStmtCFGInfo = ((CompoundStatement) Misc.getEnclosingBlock(freeInst)).getInfo()
.getCFGInfo();
List<Node> elemList = compStmtCFGInfo.getElementList();
int instPointer = elemList.indexOf(freeInst);
int indexOfBarrier = -1;
int i;
for (i = instPointer - 1; i >= 0; i--) {
Node elem = elemList.get(i);
if (elem instanceof BarrierDirective) {
indexOfBarrier = i;
break;
}
if (elementContainsBarrierForNext(elem)) {
return false;
}
}
WhileStatement whileStmt = (WhileStatement) itStmt;
if (indexOfBarrier == -1) {
if (whileStmt.getInfo().getCFGInfo().getBody() == compStmtCFGInfo.getOwner()) {
boolean couldMove = moveInstructionBeyondLoopHeader(freeInst, itStmt, internalNodes);
if (couldMove) {
hasFreeInstMoved |= moveFreeInstructionAgainstControlFlowInLoop(freeInst, cell, multiAccessedCells,
itStmt, internalNodes);
Expression predicate = whileStmt.getInfo().getCFGInfo().getPredicate();
if (predicate.toString().equals("1")) {
Node outsideNode = null;
outer: for (Node predecessor : predicate.getInfo().getCFGInfo()
.getInterProceduralLeafPredecessors()) {
if (!(predecessor instanceof BeginNode)) {
continue;
}
final String elementStr = freeInst.toString();
Set<Node> endNodes = new HashSet<>();
CollectorVisitor.collectNodeSetInGenericGraph(predecessor, endNodes, n -> {
if (n instanceof EndNode && n.getParent().toString().equals(elementStr)
|| n.toString().equals(elementStr)) {
return true;
} else {
return false;
}
}, n -> new HashSet<>(n.getInfo().getCFGInfo().getLeafPredecessors()));
for (Node endNode : endNodes) {
if (endNode.toString().equals(elementStr)) {
if (endNode != freeInst) {
outsideNode = endNode;
break outer;
}
} else if (endNode instanceof EndNode
&& endNode.getParent().toString().equals(elementStr)) {
outsideNode = endNode.getParent();
}
}
}
boolean outsideNodeMoved = false;
if (outsideNode != null) {
outsideNodeMoved = moveInstructionAgainstControlFlow(outsideNode, cell, multiAccessedCells);
}
if (!outsideNodeMoved) {
peelFirstBarrier(itStmt);
}
}
return hasFreeInstMoved;
} else {
return false;
}
} else {
return false;
}
}
int movingPointer = indexOfBarrier + 1;
BarrierDirective barrier = (BarrierDirective) elemList.get(indexOfBarrier);
outer: while (movingPointer <= instPointer) {
Node element = elemList.get(movingPointer);
if (element instanceof FlushDirective) {
movingPointer++;
continue outer;
}
if (!element.getInfo().isControlConfined()) {
movingPointer++;
continue outer;
}
if (element.getInfo().getAccesses().getReadOnlyInternal().stream()
.anyMatch(c -> c != cell && multiAccessedCells.contains(c))) {
movingPointer++;
continue outer;
}
if (!Phase.isNodeAllowedInNewPhasesAbove(element, barrier)) {
movingPointer++;
continue outer;
}
if (!BarrierDirectiveInfo.checkPotentialToCrossUpwards(elemList, indexOfBarrier, movingPointer)) {
movingPointer++;
continue outer;
}
List<SideEffect> sideEffects = NodeRemover.removeNode(element);
if (!Misc.changePerformed(sideEffects)) {
movingPointer++;
continue outer;
}
for (SideEffect sideEffect : sideEffects) {
if (sideEffect.getClass().getSimpleName().equals("IndexIncremented")
|| sideEffect instanceof AddedDFDPredecessor) {
movingPointer++;
}
}
elemList = compStmtCFGInfo.getElementList();
sideEffects = compStmtCFGInfo.addElement(indexOfBarrier, element);
if (!Misc.changePerformed(sideEffects)) {
Misc.exitDueToError("Cannot add a removed element: " + element);
movingPointer++;
continue outer;
} else {
if (element == freeInst) {
hasFreeInstMoved = true;
}
}
elemList = compStmtCFGInfo.getElementList();
indexOfBarrier++;
movingPointer++;
}
if (hasFreeInstMoved) {
moveFreeInstructionAgainstControlFlowInLoop(freeInst, cell, multiAccessedCells, itStmt, internalNodes);
return true; 
} else {
return false;
}
}
public static void peelFirstBarrier(IterationStatement itStmt) {
if (!(itStmt instanceof WhileStatement)) {
return;
}
WhileStatement whileStmt = (WhileStatement) itStmt;
Set<Node> internalNodes = whileStmt.getInfo().getCFGInfo().getIntraTaskCFGLeafContents();
CompoundStatementCFGInfo whileBodyInfo = ((CompoundStatement) whileStmt.getInfo().getCFGInfo().getBody())
.getInfo().getCFGInfo();
List<Node> whileBodyElements = whileBodyInfo.getElementList();
int indexOfFirstBarrier = -1;
for (int i = 0; i < whileBodyElements.size(); i++) {
Node element = whileBodyElements.get(i);
if (element instanceof BarrierDirective) {
indexOfFirstBarrier = i;
break;
} else if (LoopInstructionsRescheduler.elementContainsBarrierForNext(element)) {
return;
}
if (!element.getInfo().isControlConfined()) {
return;
}
}
if (indexOfFirstBarrier == -1) {
return;
}
BarrierDirective barrier = (BarrierDirective) whileBodyElements.get(indexOfFirstBarrier);
for (int i = 0; i < indexOfFirstBarrier - 1; i++) {
Node element = whileBodyElements.get(i);
CellSet sharedAccesses = element.getInfo().getSharedAccesses();
if (!sharedAccesses.isEmpty() && !Phase.isNodeAllowedInNewPhasesBelow(element, barrier)) {
if (moveInstructionBeyondLoopHeader(element, itStmt, internalNodes)) {
peelFirstBarrier(itStmt);
return;
} else {
return;
}
}
}
System.err.println("\t\tPeeling off the top barrier.");
Expression predicate = whileStmt.getInfo().getCFGInfo().getPredicate();
if (!predicate.toString().trim().equals("1")) {
System.err.println("Unable to peel barrier for loops that do not have their predicate as 1.");
return;
}
List<SideEffect> sideEffects = whileBodyInfo.removeElement(barrier);
if (!Misc.changePerformed(sideEffects)) {
return;
}
boolean first = true;
for (Node predecessor : predicate.getInfo().getCFGInfo().getInterProceduralLeafPredecessors()) {
if (predecessor instanceof BeginNode) {
BarrierDirective tempBarrier = FrontEnd.parseAndNormalize("#pragma omp barrier\n",
BarrierDirective.class);
sideEffects = InsertImmediatePredecessor.insert(predecessor, tempBarrier);
if (!Misc.changePerformed(sideEffects)) {
Misc.exitDueToError("Unable to insert a removed barrier!");
return;
}
continue;
} else {
if (first) {
first = false;
} else {
barrier = FrontEnd.parseAndNormalize("#pragma omp barrier\n", BarrierDirective.class);
}
sideEffects = InsertImmediatePredecessor.insert(predecessor, barrier);
if (!Misc.changePerformed(sideEffects)) {
Misc.exitDueToError("Unable to insert a removed barrier!");
return;
}
}
}
return;
}
private static Node replaceCopyAtLoopEndWith(final Node element, WhileStatement whileStmt) {
Expression predicate = whileStmt.getInfo().getCFGInfo().getPredicate();
Node copiedNode = null;
outer: for (Node predecessor : predicate.getInfo().getCFGInfo().getInterProceduralLeafPredecessors()) {
if (predecessor instanceof BeginNode) {
continue;
}
final String elementStr = element.toString();
Set<Node> endNodes = new HashSet<>();
CollectorVisitor.collectNodeSetInGenericGraph(predecessor, endNodes, n -> {
if (n instanceof EndNode && n.getParent().toString().equals(elementStr)
|| n.toString().equals(elementStr)) {
return true;
} else {
return false;
}
}, n -> new HashSet<>(n.getInfo().getCFGInfo().getLeafPredecessors()));
for (Node endNode : endNodes) {
if (endNode.toString().equals(elementStr)) {
if (endNode != element) {
copiedNode = endNode;
break outer;
}
} else if (endNode instanceof EndNode && endNode.getParent().toString().equals(elementStr)) {
copiedNode = endNode.getParent();
}
}
}
if (copiedNode == null) {
Misc.warnDueToLackOfFeature("Could not find the internal copy of the following element: " + element, null);
return null;
}
Statement tempStmt = FrontEnd.parseAndNormalize(";", Statement.class);
List<SideEffect> sideEffects = NodeReplacer.replaceNodes(element, tempStmt);
sideEffects.addAll(NodeReplacer.replaceNodes(copiedNode, element));
sideEffects.addAll(NodeReplacer.replaceNodes(tempStmt, copiedNode));
if (!Misc.changePerformed(sideEffects)) {
Misc.warnDueToLackOfFeature(
"Could not replace the existing internal copy of the following element: " + element, null);
return null;
} else {
return copiedNode;
}
}
public static boolean elementContainsBarrierForNext(Node element) {
element = Misc.getCFGNodeFor(element);
if (element instanceof BarrierDirective) {
return true;
}
if (Misc.isCFGNode(element)) {
Set<NodeWithStack> elementInternalNodes = element.getInfo().getCFGInfo()
.getIntraTaskCFGLeafContentsOfSameParLevel(new CallStack());
if (elementInternalNodes.stream().filter(ns -> ns.getNode() instanceof BarrierDirective).count() > 0) {
if (element instanceof IfStatement) {
Set<NodeWithStack> endingElements = new HashSet<>();
Node startingEndNode = element.getInfo().getCFGInfo().getNestedCFG().getEnd();
Node endingBeginNode = element.getInfo().getCFGInfo().getNestedCFG().getBegin();
Set<Node> outJumpSources = element.getInfo().getOutJumpSources();
CollectorVisitor.collectNodeSetInGenericGraph(new NodeWithStack(startingEndNode, new CallStack()),
endingElements, (n) -> {
if (outJumpSources.stream().anyMatch(nS -> nS == n.getNode())
|| n.getNode() == endingBeginNode) {
return true;
}
if (n.getNode() instanceof BarrierDirective) {
return true;
}
return false;
}, (n) -> n.getNode().getInfo().getCFGInfo()
.getParallelConstructFreeInterProceduralLeafPredecessors(n.getCallStack()));
if (endingElements.stream().anyMatch(nS -> nS.getNode() instanceof BarrierDirective)) {
return true;
} else {
return false;
}
}
return true;
} else {
return false;
}
} else {
return false;
}
}
private static boolean moveInstructionBeyondLoopHeader(final Node freeInst, IterationStatement itStmt,
Set<Node> internalNodes) {
if (!(itStmt instanceof WhileStatement)) {
return false;
}
WhileStatement whileStmt = (WhileStatement) itStmt;
CompoundStatementCFGInfo compStmtCFGInfo = ((CompoundStatement) Misc.getEnclosingBlock(freeInst)).getInfo()
.getCFGInfo();
if (whileStmt.getInfo().getCFGInfo().getBody() != compStmtCFGInfo.getOwner()) {
return false;
}
Expression predicate = whileStmt.getInfo().getCFGInfo().getPredicate();
Set<Phase> visitedPhases = new HashSet<>();
Node leafFreeInst = Misc.getCFGNodeFor(freeInst);
leafFreeInst = Misc.isCFGLeafNode(leafFreeInst) ? leafFreeInst
: leafFreeInst.getInfo().getCFGInfo().getNestedCFG().getBegin();
for (AbstractPhase<?, ?> absPh : predicate.getInfo().getNodePhaseInfo().getPhaseSet()) {
Phase ph = (Phase) absPh;
if (visitedPhases.contains(ph)) {
continue;
} else {
visitedPhases.add(ph);
}
for (BeginPhasePoint bpp : ph.getBeginPoints()) {
if (!(bpp.getNode() instanceof BarrierDirective)) {
continue;
}
BarrierDirective barrier = (BarrierDirective) bpp.getNode();
if (!bpp.getReachableNodes().contains(leafFreeInst) || !internalNodes.contains(barrier)) {
continue;
}
if (!Phase.isNodeAllowedInNewPhasesAbove(freeInst, barrier)) {
return false;
}
}
}
int moveUptil = obtainCodeIndexToBeMoved(freeInst, whileStmt, internalNodes);
if (moveUptil == -1) {
return false;
}
System.err.println("\t\tAttempting to move first " + (moveUptil + 1) + " statement(s) across the loop.");
int ptr = 0;
Set<Node> sharedNodesMoved = new HashSet<>();
for (int i = 0; i <= moveUptil; i++) {
Node elementToBeMoved = compStmtCFGInfo.getElementList().get(ptr);
List<SideEffect> sideEffects = compStmtCFGInfo.removeElement(elementToBeMoved);
if (elementToBeMoved instanceof DummyFlushDirective) {
ptr++;
continue;
}
if (sideEffects.stream().anyMatch(s -> s instanceof RemovedDFDSuccessor)) {
moveUptil--;
ptr--;
}
if (!elementToBeMoved.getInfo().getSharedAccesses().isEmpty()) {
sharedNodesMoved.add(elementToBeMoved);
}
sideEffects = InsertImmediatePredecessor.insert(predicate, elementToBeMoved);
elementToBeMoved.getInfo().getJumpedPredicates().add(predicate);
assert (Misc.changePerformed(sideEffects)) : sideEffects + " while attempting to move " + elementToBeMoved;
}
for (Node shared : sharedNodesMoved) {
Node copy = replaceCopyAtLoopEndWith(shared, whileStmt);
if (copy != null) {
moveAsHighAsPossible(copy, Cell.getNullCell(), new CellSet());
}
}
for (Node elem : compStmtCFGInfo.getElementList()) {
if (elem instanceof Statement && elem.toString().trim().equals(";")) {
compStmtCFGInfo.removeElement(elem);
}
}
return true;
}
private static void moveAsHighAsPossible(Node inst, Cell cell, CellSet multiAccessedCells) {
Node cfgNode;
if (Misc.isCFGLeafNode(inst)) {
cfgNode = inst;
} else {
assert (Misc.isCFGNode(inst));
cfgNode = inst.getInfo().getCFGInfo().getNestedCFG().getBegin();
}
Set<Phase> allPhases = (Set<Phase>) cfgNode.getInfo().getNodePhaseInfo().getPhaseSet();
do {
boolean moved = moveInstructionAgainstControlFlow(inst, cell, multiAccessedCells);
if (!moved) {
break;
}
Set<Phase> thisPhases = (Set<Phase>) cfgNode.getInfo().getNodePhaseInfo().getPhaseSet();
if (Misc.doIntersect(allPhases, thisPhases)) {
return;
} else {
allPhases.addAll(thisPhases);
continue;
}
} while (true);
}
private static boolean moveInstructionAgainstControlFlow(Node inst, Cell cell, CellSet multiAccessedCells) {
boolean isFreeInstMoved = false;
CompoundStatementCFGInfo compStmtCFGInfo = ((CompoundStatement) Misc.getEnclosingBlock(inst)).getInfo()
.getCFGInfo();
List<Node> elemList = compStmtCFGInfo.getElementList();
int instPointer = elemList.indexOf(inst);
int indexOfBarrier = -1;
int i;
for (i = instPointer - 1; i >= 0; i--) {
Node elem = elemList.get(i);
if (elem instanceof BarrierDirective) {
indexOfBarrier = i;
break;
}
if (Misc.isCFGNode(elem)) {
Set<NodeWithStack> elementInternalNodes = elem.getInfo().getCFGInfo()
.getIntraTaskCFGLeafContentsOfSameParLevel(new CallStack());
if (elementInternalNodes.stream().filter(ns -> ns.getNode() instanceof BarrierDirective).count() > 0) {
if (elem instanceof IfStatement) {
Set<NodeWithStack> endingElements = new HashSet<>();
Node startingEndNode = elem.getInfo().getCFGInfo().getNestedCFG().getEnd();
Node endingBeginNode = elem.getInfo().getCFGInfo().getNestedCFG().getBegin();
Set<Node> outJumpSources = elem.getInfo().getOutJumpSources();
CollectorVisitor.collectNodeSetInGenericGraph(
new NodeWithStack(startingEndNode, new CallStack()), endingElements, (n) -> {
if (outJumpSources.stream().anyMatch(nS -> nS == n.getNode())
|| n.getNode() == endingBeginNode) {
return true;
}
if (n.getNode() instanceof BarrierDirective) {
return true;
}
return false;
}, (n) -> n.getNode().getInfo().getCFGInfo()
.getParallelConstructFreeInterProceduralLeafPredecessors(n.getCallStack()));
if (endingElements.stream().anyMatch(nS -> nS.getNode() instanceof BarrierDirective)) {
indexOfBarrier = i;
return false;
}
} else {
indexOfBarrier = i;
return false;
}
}
}
}
if (indexOfBarrier == -1) {
return false;
}
int movingPointer = indexOfBarrier + 1;
BarrierDirective barrier = (BarrierDirective) elemList.get(indexOfBarrier);
outer: while (movingPointer <= instPointer) {
Node element = elemList.get(movingPointer);
if (element instanceof FlushDirective) {
movingPointer++;
continue outer;
}
if (!element.getInfo().isControlConfined()) {
movingPointer++;
continue outer;
}
if (element.getInfo().getAccesses().getReadOnlyInternal().stream()
.anyMatch(c -> c != cell && multiAccessedCells.contains(c))) {
movingPointer++;
continue outer;
}
if (!Phase.isNodeAllowedInNewPhasesAbove(element, barrier)) {
movingPointer++;
continue outer;
}
if (!BarrierDirectiveInfo.checkPotentialToCrossUpwards(elemList, indexOfBarrier, movingPointer)) {
movingPointer++;
continue outer;
}
List<SideEffect> sideEffects = NodeRemover.removeNode(element);
if (!Misc.changePerformed(sideEffects)) {
movingPointer++;
continue outer;
}
for (SideEffect sideEffect : sideEffects) {
if (sideEffect.getClass().getSimpleName().equals("IndexIncremented")
|| sideEffect instanceof AddedDFDPredecessor) {
movingPointer++;
}
}
elemList = compStmtCFGInfo.getElementList();
sideEffects = compStmtCFGInfo.addElement(indexOfBarrier, element);
if (!Misc.changePerformed(sideEffects)) {
Misc.exitDueToError("Cannot add a removed element: " + element);
movingPointer++;
continue outer;
} else {
if (element == inst) {
isFreeInstMoved = true;
}
}
elemList = compStmtCFGInfo.getElementList();
indexOfBarrier++;
movingPointer++;
}
if (isFreeInstMoved) {
moveInstructionAgainstControlFlow(inst, cell, multiAccessedCells);
return true; 
} else {
return false;
}
}
private static int obtainCodeIndexToBeMoved(Node freeInst, WhileStatement whileStmt, Set<Node> internalNodes) {
CompoundStatement compStmt = (CompoundStatement) whileStmt.getInfo().getCFGInfo().getBody();
CompoundStatementCFGInfo compStmtCFGInfo = compStmt.getInfo().getCFGInfo();
List<Node> elemList = compStmtCFGInfo.getElementList();
Expression predicate = whileStmt.getInfo().getCFGInfo().getPredicate();
int instIndex = elemList.indexOf(freeInst);
if (instIndex == -1) {
return -1;
}
int stackPointer = -1; 
int movingPointer = 0;
outer: while (movingPointer <= instIndex) {
Node element = elemList.get(movingPointer);
if (element.getInfo().getJumpedPredicates().contains(predicate)) {
movingPointer++;
continue outer;
}
if (element instanceof BarrierDirective) {
assert (false);
return stackPointer;
} else if (element instanceof Declaration) {
movingPointer++;
continue outer;
} else if (element instanceof FlushDirective) {
movingPointer++;
continue outer;
}
if (!element.getInfo().isControlConfined()) {
movingPointer++;
continue outer;
}
if (!BarrierDirectiveInfo.checkPotentialToCrossUpwards(elemList, stackPointer, movingPointer)) {
movingPointer++;
continue outer;
}
List<SideEffect> sideEffects = NodeRemover.removeNode(element);
if (!Misc.changePerformed(sideEffects)) {
movingPointer++;
continue outer;
}
for (SideEffect sideEffect : sideEffects) {
if (sideEffect.getClass().getSimpleName().equals("IndexIncremented")
|| sideEffect instanceof AddedDFDPredecessor) {
movingPointer++;
}
}
elemList = compStmtCFGInfo.getElementList();
sideEffects = compStmtCFGInfo.addElement(stackPointer + 1, element);
if (!Misc.changePerformed(sideEffects)) {
movingPointer++;
continue outer;
}
elemList = compStmtCFGInfo.getElementList();
stackPointer = elemList.indexOf(element); 
movingPointer++;
}
return stackPointer;
}
@SuppressWarnings("unused")
@Deprecated
private static boolean canInstructionMoveFreely(IterationStatement itStmt, final Node inst, Cell cell,
CellSet multiAccessedCells, Set<Node> internalNodes) {
assert (inst.getInfo().getSharedAccesses().contains(cell));
Node compElement = null;
if (inst instanceof JumpStatement) {
return false;
} else if (inst instanceof ParameterDeclaration) {
return false;
} else if (inst instanceof PreCallNode || inst instanceof PostCallNode) {
compElement = inst.getParent();
assert (compElement instanceof CallStatement);
} else if (inst instanceof IfClause || inst instanceof NumThreadsClause || inst instanceof FinalClause
|| inst instanceof OmpForReinitExpression || inst instanceof OmpForInitExpression
|| inst instanceof OmpForCondition || inst instanceof Expression) {
CFGLink link = CFGLinkFinder.getCFGLinkFor(inst);
compElement = link.getEnclosingNode();
} else {
compElement = inst;
}
CompoundStatement barrLevelCompStmt = null;
for (AbstractPhase<?, ?> absPh : compElement.getInfo().getNodePhaseInfo().getPhaseSet()) {
Phase ph = (Phase) absPh;
for (BeginPhasePoint bpp : ph.getBeginPoints()) {
if (!(bpp.getNode() instanceof BarrierDirective)) {
continue;
}
BarrierDirective barrier = (BarrierDirective) bpp.getNode();
if (internalNodes.contains(barrier) && bpp.getReachableNodes().contains(inst)) {
CompoundStatement bppEncloser = (CompoundStatement) Misc.getEnclosingBlock(barrier);
if (barrLevelCompStmt == null) {
barrLevelCompStmt = bppEncloser;
} else {
if (barrLevelCompStmt != bppEncloser) {
return false;
}
}
}
}
for (EndPhasePoint epp : ph.getEndPoints()) {
if (!(epp.getNode() instanceof BarrierDirective)) {
continue;
}
BarrierDirective barrier = (BarrierDirective) epp.getNode();
if (internalNodes.contains(barrier) && ph.getBeginPoints().stream().anyMatch(
bpp -> bpp.getReachableNodes().contains(inst) && bpp.getNextBarriers().contains(epp))) {
CompoundStatement eppEncloser = (CompoundStatement) Misc.getEnclosingBlock(barrier);
if (barrLevelCompStmt == null) {
barrLevelCompStmt = eppEncloser;
} else {
if (barrLevelCompStmt != eppEncloser) {
return false;
}
}
}
}
}
Node barrLevelCompElement = null;
for (Node tempElem : barrLevelCompStmt.getInfo().getCFGInfo().getElementList()) {
if (tempElem.getInfo().getCFGInfo().getReachableIntraTaskCFGLeafContents().contains(inst)) {
if (tempElem.getInfo().getCFGInfo().getIntraTaskCFGLeafContentsOfSameParLevel().stream()
.anyMatch(n -> n.getNode() instanceof BarrierDirective)) {
System.out.println(
"The preceding barrier and succeeding barrier are not in the same level of compound statement for "
+ inst);
return false;
} else {
barrLevelCompElement = tempElem;
break;
}
}
}
assert (barrLevelCompElement != null);
if (!canInstructionMoveUpwards(itStmt, barrLevelCompElement, barrLevelCompStmt, cell, multiAccessedCells)) {
return false;
}
if (!canInstructionMoveDownwards(itStmt, barrLevelCompElement, barrLevelCompStmt, cell, multiAccessedCells)) {
return false;
}
return true;
}
@Deprecated
private static boolean canInstructionMoveUpwards(IterationStatement itStmt, Node compElement,
CompoundStatement compStmt, Cell cell, CellSet multiAccessedCells) {
CompoundStatementCFGInfo compStmtInfo = compStmt.getInfo().getCFGInfo();
assert (compStmtInfo.getElementList().contains(compElement));
assert (compElement.getInfo().getSharedAccesses().contains(cell));
return false;
}
@Deprecated
private static boolean canInstructionMoveDownwards(IterationStatement itStmt, Node compElement,
CompoundStatement compStmt, Cell cell, CellSet multiAccessedCells) {
CompoundStatementCFGInfo compStmtInfo = compStmt.getInfo().getCFGInfo();
assert (compStmtInfo.getElementList().contains(compElement));
assert (compElement.getInfo().getSharedAccesses().contains(cell));
return false;
}
}
