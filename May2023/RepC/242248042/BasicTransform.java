package imop.lib.transform;
import imop.ast.node.external.*;
import imop.ast.node.internal.*;
import imop.deprecated.Deprecated_ImmediatePredecessorInserter;
import imop.lib.analysis.typesystem.Type;
import imop.lib.builder.Builder;
import imop.lib.cfg.CFGGenerator;
import imop.lib.cfg.info.CFGInfo;
import imop.lib.cfg.info.CompoundStatementCFGInfo;
import imop.lib.cfg.info.ForStatementCFGInfo;
import imop.lib.getter.StringGetter;
import imop.lib.transform.simplify.InsertDummyFlushDirectives;
import imop.lib.transform.updater.InsertImmediatePredecessor;
import imop.lib.transform.updater.InsertImmediateSuccessor;
import imop.lib.transform.updater.NodeRemover;
import imop.lib.transform.updater.NodeReplacer;
import imop.lib.transform.updater.sideeffect.MissingCFGParent;
import imop.lib.transform.updater.sideeffect.SideEffect;
import imop.lib.transform.updater.sideeffect.SyntacticConstraint;
import imop.lib.util.Misc;
import imop.parser.FrontEnd;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Vector;
public class BasicTransform {
public static TranslationUnit root; 
public static boolean crudeInsertNewBeforeBase(Node baseNode, Node newNode) {
List<Node> containingList = Misc.getContainingList(baseNode);
Node tempNewNode = Misc.getWrapperInList(newNode);
baseNode = Misc.getWrapperInList(baseNode);
if (tempNewNode == null) {
newNode = FrontEnd.parseAlone(newNode.getInfo().getString(), baseNode.getClass());
} else {
newNode = tempNewNode;
}
assert (newNode != null && baseNode != null);
int indexOfM = containingList.indexOf(baseNode);
containingList.add(indexOfM, newNode);
return true;
}
public static boolean crudeInsertNewAfterBase(Node baseNode, Node newNode) {
List<Node> containingList = Misc.getContainingList(baseNode);
if (containingList == null) {
return false;
}
Node tempNewNode = Misc.getWrapperInList(newNode);
baseNode = Misc.getWrapperInList(baseNode);
if (tempNewNode == null) {
newNode = FrontEnd.parseAlone(newNode.getInfo().getString(), baseNode.getClass());
} else {
newNode = tempNewNode;
}
assert (newNode != null && baseNode != null);
int indexOfM = containingList.indexOf(baseNode);
containingList.add(indexOfM + 1, newNode);
return true;
}
public static boolean crudeReplaceBaseWithNewInList(Node baseNode, Node newNode) {
List<Node> containingList = Misc.getContainingList(baseNode);
if (containingList == null) {
return false;
}
Node tempNewNode = Misc.getWrapperInList(newNode);
baseNode = Misc.getWrapperInList(baseNode);
if (tempNewNode == null) {
newNode = FrontEnd.parseAlone(newNode.getInfo().getString(), baseNode.getClass());
} else {
newNode = tempNewNode;
}
assert (newNode != null && baseNode != null);
int indexOfM = containingList.indexOf(baseNode);
containingList.add(indexOfM, newNode);
containingList.remove(baseNode);
return true;
}
public static boolean crudeSwapBaseWithNew(Node baseNode, Node newNode) {
baseNode = Misc.getInternalFirstCFGNode(baseNode);
newNode = Misc.getInternalFirstCFGNode(newNode);
List<Node> containingList = Misc.getContainingList(baseNode);
if (containingList == null) {
return false;
}
baseNode = Misc.getWrapperInList(baseNode);
newNode = Misc.getWrapperInList(newNode);
int indexOfM = containingList.indexOf(baseNode);
int indexOfN = containingList.indexOf(newNode);
if (indexOfM > indexOfN) {
Node tempNode = baseNode;
baseNode = newNode;
newNode = tempNode;
int tempIndex = indexOfM;
indexOfM = indexOfN;
indexOfN = tempIndex;
}
assert (newNode != null && baseNode != null && baseNode != newNode);
containingList.remove(newNode);
containingList.add(indexOfM, newNode);
containingList.remove(baseNode);
containingList.add(indexOfN, baseNode);
baseNode = Misc.getInternalFirstCFGNode(baseNode);
newNode = Misc.getInternalFirstCFGNode(newNode);
CFGInfo baseNodeCFG = baseNode.getInfo().getCFGInfo();
CFGInfo newNodeCFG = newNode.getInfo().getCFGInfo();
Vector<Node> x1List = new Vector<>(baseNodeCFG.getPredBlocks());
Vector<Node> x2List = new Vector<>(newNodeCFG.getSuccBlocks());
newNodeCFG.getPredBlocks().remove(baseNode);
baseNodeCFG.getSuccBlocks().remove(newNode);
for (Node x1 : x1List) {
CFGInfo x1CFG = x1.getInfo().getCFGInfo();
x1CFG.getSuccBlocks().remove(baseNode);
x1CFG.getSuccBlocks().add(newNode);
newNodeCFG.getPredBlocks().add(x1);
}
for (Node x2 : x2List) {
CFGInfo x2CFG = x2.getInfo().getCFGInfo();
x2CFG.getPredBlocks().remove(newNode);
x2CFG.getPredBlocks().add(baseNode);
baseNodeCFG.getSuccBlocks().add(x2);
}
newNodeCFG.getSuccBlocks().add(baseNode);
baseNodeCFG.getPredBlocks().add(newNode);
return true;
}
public static <T extends Node> boolean crudeReplaceOldWithNew(T baseNode, T newNode)
throws IllegalArgumentException, IllegalAccessException {
return CrudeReplaceNode.replace(baseNode, newNode);
}
public static boolean splitCompoundStatement(CompoundStatement node) {
Vector<CompoundStatementElement> newList = new Vector<>();
if ((!node.getF1().present()) || node.getF1().getNodes().size() == 1) {
return false;
}
for (Node element : node.getF1().getNodes()) {
CompoundStatementElement newCSE = FrontEnd.parseAlone("{" + element.getInfo().getString() + "}",
CompoundStatementElement.class);
newCSE.getF0().setChoice(element);
newList.add(newCSE);
}
List<Node> containingList = Misc.getContainingList(node);
int indexOfNode = Misc.getIndexInContainingList(node);
if (containingList == null) {
return false;
}
containingList.remove(Misc.getWrapperInList(node));
containingList.addAll(indexOfNode, newList);
for (Node newNode : newList) {
CFGInfo cfgInfo = Misc.getInternalFirstCFGNode(newNode).getInfo().getCFGInfo();
cfgInfo.getPredBlocks().clear();
cfgInfo.getSuccBlocks().clear();
}
Node firstNode = Misc.getInternalFirstCFGNode(newList.firstElement());
for (Node pred : node.getInfo().getCFGInfo().getPredBlocks()) {
CFGInfo predInfo = pred.getInfo().getCFGInfo();
predInfo.getSuccBlocks().remove(node);
predInfo.getSuccBlocks().add(firstNode);
firstNode.getInfo().getCFGInfo().getPredBlocks().add(pred);
}
Node lastNode = Misc.getInternalFirstCFGNode(newList.lastElement());
for (Node succ : node.getInfo().getCFGInfo().getSuccBlocks()) {
CFGInfo succInfo = succ.getInfo().getCFGInfo();
succInfo.getPredBlocks().remove(node);
succInfo.getPredBlocks().add(lastNode);
lastNode.getInfo().getCFGInfo().getSuccBlocks().add(succ);
}
for (Node tempNode : newList) {
Node newNode = Misc.getInternalFirstCFGNode(tempNode);
if (tempNode == newList.firstElement()) {
continue;
} else {
Node prevTempNode = newList.get(newList.indexOf(tempNode) - 1);
Node prevNode = Misc.getInternalFirstCFGNode(prevTempNode);
prevNode.getInfo().getCFGInfo().getSuccBlocks().add(newNode);
newNode.getInfo().getCFGInfo().getPredBlocks().add(prevNode);
}
}
return true;
}
public static Node obtainRenamedNode(Node node, HashMap<String, String> nameMap) {
Node replacementNode = node;
if (node instanceof DummyFlushDirective) {
assert (false);
return null;
} else if (node instanceof Statement) {
String replacementString = StringGetter.getRenamedString(node, nameMap);
replacementNode = FrontEnd.parseAndNormalize(replacementString, Statement.class);
} else if (node instanceof PreCallNode) {
PreCallNode oldPre = (PreCallNode) node;
List<SimplePrimaryExpression> newSPEList = new ArrayList<>();
for (SimplePrimaryExpression oldSPE : oldPre.getArgumentList()) {
SimplePrimaryExpression newSPE = null;
if (oldSPE.isAConstant()) {
newSPE = new SimplePrimaryExpression(oldSPE.getConstant());
} else {
String oldIdStr = oldSPE.getIdentifier().getTokenImage();
String newIdStr = nameMap.get(oldIdStr);
NodeToken newId = null;
if (newIdStr == null) {
newId = oldSPE.getIdentifier();
} else {
newId = new NodeToken(newIdStr);
}
newSPE = new SimplePrimaryExpression(newId);
}
newSPEList.add(newSPE);
}
PreCallNode newPre = new PreCallNode(newSPEList);
CallStatement oldCallStmt = oldPre.getParent();
replacementNode = new CallStatement(oldCallStmt.getFunctionDesignatorNode(), newPre,
oldCallStmt.getPostCallNode());
} else if (node instanceof PostCallNode) {
PostCallNode oldPost = (PostCallNode) node;
SimplePrimaryExpression oldSPE = oldPost.getReturnReceiver();
SimplePrimaryExpression newSPE = null;
if (oldSPE.isAConstant()) {
newSPE = new SimplePrimaryExpression(oldSPE.getConstant());
} else {
String oldIdStr = oldSPE.getIdentifier().getTokenImage();
String newIdStr = nameMap.get(oldIdStr);
NodeToken newId = null;
if (newIdStr == null) {
newId = oldSPE.getIdentifier();
} else {
newId = new NodeToken(newIdStr);
}
newSPE = new SimplePrimaryExpression(newId);
}
PostCallNode newPost = new PostCallNode(newSPE);
CallStatement oldCallStmt = oldPost.getParent();
replacementNode = new CallStatement(oldCallStmt.getFunctionDesignatorNode(), oldCallStmt.getPreCallNode(),
newPost);
} else {
String replacementString = StringGetter.getRenamedString(node, nameMap);
replacementNode = FrontEnd.parseAndNormalize(replacementString, node.getClass());
}
return replacementNode;
}
public static Node renameIdsInNodes(Node node, HashMap<String, String> nameMap) {
validateRenamingMap(nameMap);
boolean found = false;
String str = node.toString();
for (String key : nameMap.keySet()) {
if (str.contains(key)) {
found = true;
break;
}
}
if (!found) {
return node;
}
Node replacementNode = node;
if (node instanceof Statement) {
String replacementString = StringGetter.getRenamedString(node, nameMap);
replacementNode = FrontEnd.parseAndNormalize(replacementString, Statement.class);
NodeReplacer.replaceNodes(node, replacementNode);
} else if (node instanceof PreCallNode) {
PreCallNode oldPre = (PreCallNode) node;
List<SimplePrimaryExpression> newSPEList = new ArrayList<>();
for (SimplePrimaryExpression oldSPE : oldPre.getArgumentList()) {
SimplePrimaryExpression newSPE = null;
if (oldSPE.isAConstant()) {
newSPE = new SimplePrimaryExpression(oldSPE.getConstant());
} else {
String oldIdStr = oldSPE.getIdentifier().getTokenImage();
String newIdStr = nameMap.get(oldIdStr);
NodeToken newId = null;
if (newIdStr == null) {
newId = oldSPE.getIdentifier();
} else {
newId = new NodeToken(newIdStr);
}
newSPE = new SimplePrimaryExpression(newId);
}
newSPEList.add(newSPE);
}
PreCallNode newPre = new PreCallNode(newSPEList);
CallStatement oldCallStmt = oldPre.getParent();
replacementNode = new CallStatement(oldCallStmt.getFunctionDesignatorNode(), newPre,
oldCallStmt.getPostCallNode());
NodeReplacer.replaceNodes(oldCallStmt, replacementNode);
} else if (node instanceof PostCallNode) {
PostCallNode oldPost = (PostCallNode) node;
SimplePrimaryExpression oldSPE = oldPost.getReturnReceiver();
SimplePrimaryExpression newSPE = null;
if (oldSPE.isAConstant()) {
newSPE = new SimplePrimaryExpression(oldSPE.getConstant());
} else {
String oldIdStr = oldSPE.getIdentifier().getTokenImage();
String newIdStr = nameMap.get(oldIdStr);
NodeToken newId = new NodeToken(newIdStr);
newSPE = new SimplePrimaryExpression(newId);
}
PostCallNode newPost = new PostCallNode(newSPE);
CallStatement oldCallStmt = oldPost.getParent();
replacementNode = new CallStatement(oldCallStmt.getFunctionDesignatorNode(), oldCallStmt.getPreCallNode(),
newPost);
NodeReplacer.replaceNodes(oldCallStmt, replacementNode);
} else {
String replacementString = StringGetter.getRenamedString(node, nameMap);
replacementNode = FrontEnd.parseAndNormalize(replacementString, node.getClass());
NodeReplacer.replaceNodes(node, replacementNode);
}
return replacementNode;
}
private static void validateRenamingMap(HashMap<String, String> nameMap) {
for (String strSrc : nameMap.keySet()) {
for (String strDest : nameMap.values()) {
if (strSrc.equals(strDest)) {
Thread.dumpStack();
Misc.exitDueToError("Cannot perform conlifcting renaming for " + strSrc + " to "
+ nameMap.get(strSrc) + " when another renaming has that name as the destination.");
}
}
}
}
public static List<SideEffect> pushDeclarationUp(Declaration decl) {
List<SideEffect> sideEffects = new ArrayList<>();
CompoundStatement scope = (CompoundStatement) Misc.getEnclosingBlock(decl);
if (scope == null) {
Misc.warnDueToLackOfFeature(
"Could not find any enclosing block for the declaration that needs to be moved higher.", null);
sideEffects.add(new MissingCFGParent(decl));
return sideEffects;
}
CompoundStatementCFGInfo scopeInfo = scope.getInfo().getCFGInfo();
List<Node> elementList = scopeInfo.getElementList();
int index = elementList.indexOf(decl);
if (index == 0) {
return sideEffects;
}
int finalIndex = index;
for (int i = index; i >= 1; i--) {
Node prevNode = elementList.get(i - 1);
if (decl.getInfo().hasInitializer()) {
if (Misc.haveDataDependences(decl.getInfo().getInitializer(), prevNode)) {
break;
}
}
if (decl.getInfo().clashesSyntacticallyWith(prevNode)) {
break;
}
finalIndex--;
}
if (finalIndex < index) {
sideEffects.addAll(scopeInfo.removeDeclaration(decl));
sideEffects.addAll(scopeInfo.addDeclaration(finalIndex, decl));
} else {
sideEffects.add(new SyntacticConstraint(decl, ""));
}
return sideEffects;
}
public static boolean removeSideEffectsFromInitializer(Declaration decl) {
Initializer init = decl.getInfo().getInitializer();
if (init == null) {
return false;
}
if (!init.getInfo().mayWrite()) {
return false;
}
CompoundStatement scope = (CompoundStatement) Misc.getEnclosingBlock(decl);
if (scope == null) {
return false;
}
Type typeOfInit = Type.getType(init);
String tempName = Builder.getNewTempName(decl.getInfo().getDeclaredName() + "Init");
Declaration simplifyingTemp = typeOfInit.getDeclaration(tempName);
InsertImmediatePredecessor.insert(decl, simplifyingTemp);
Statement simplifyingExp = FrontEnd.parseAndNormalize(tempName + " = " + init.toString() + ";",
Statement.class);
InsertImmediatePredecessor.insert(decl, simplifyingExp);
Declarator declarator = decl.getInfo().getDeclarator(decl.getInfo().getDeclaredName());
String newDeclString = decl.getF0().toString() + " " + declarator + " = " + tempName + ";";
Declaration newDecl = FrontEnd.parseAndNormalize(newDeclString, Declaration.class);
NodeReplacer.replaceNodes(decl, newDecl);
return true;
}
public static List<SideEffect> simplifyPredicate(IterationStatement itStmt) {
List<SideEffect> sideEffectList = new ArrayList<>();
if (itStmt instanceof WhileStatement) {
WhileStatement whileStmt = (WhileStatement) itStmt;
return whileStmt.getInfo().changePredicateToConstantTrue();
} else if (itStmt instanceof DoStatement) {
assert (false);
} else if (itStmt instanceof ForStatement) {
assert (false);
}
return sideEffectList;
}
public static DoStatement convertAllDoWhileToWhile(Node root) {
for (DoStatement doStmt : Misc.getExactEnclosee(root, DoStatement.class)) {
}
return null;
}
public static IterationStatement convertToWhile(IterationStatement itStmt) {
if (itStmt instanceof WhileStatement) {
return itStmt;
} else if (itStmt instanceof DoStatement) {
return itStmt;
} else {
ForStatementCFGInfo cfgInfo = (ForStatementCFGInfo) itStmt.getInfo().getCFGInfo();
if (cfgInfo.hasStepExpression()) {
Expression stepExpr = cfgInfo.getStepExpression();
for (Node pred : stepExpr.getInfo().getCFGInfo().getPredBlocks()) {
Statement expStmt = FrontEnd.parseAndNormalize(stepExpr + ";", Statement.class);
InsertImmediatePredecessor.insert(pred, expStmt);
}
cfgInfo.removeStepExpression();
}
StringBuilder whileStr = new StringBuilder();
if (cfgInfo.hasInitExpression()) {
Statement initStmt = FrontEnd.parseAndNormalize(cfgInfo.getInitExpression() + ";", Statement.class);
InsertImmediatePredecessor.insert(itStmt, initStmt);
cfgInfo.removeInitExpression();
}
whileStr.append("while (").append(cfgInfo.getTerminationExpression()).append(")").append(cfgInfo.getBody());
WhileStatement whileStmt = FrontEnd.parseAndNormalize(whileStr.toString(), WhileStatement.class);
NodeReplacer.replaceNodes(itStmt, whileStmt);
return whileStmt;
}
}
public static void removeEmptyConstructs(Node node) {
for (Node openNode : Misc.getExactEnclosee(node, OmpConstruct.class)) {
Statement stmt = openNode.getInfo().getCFGInfo().getBody();
if (stmt == null || !(stmt instanceof CompoundStatement)) {
continue;
}
CompoundStatement compStmt = (CompoundStatement) stmt;
if (compStmt.getInfo().getCFGInfo().getElementList().isEmpty()) {
boolean needsBarrier = false;
boolean needsFlush = false;
if (openNode instanceof SingleConstruct) {
SingleConstruct singleCons = (SingleConstruct) openNode;
if (!singleCons.getInfo().hasNowaitClause()) {
needsBarrier = true;
}
} else if (openNode instanceof ForConstruct) {
ForConstruct forCons = (ForConstruct) openNode;
if (!forCons.getInfo().hasNowaitClause()) {
needsBarrier = true;
}
} else if (openNode instanceof SectionsConstruct) {
SectionsConstruct secCons = (SectionsConstruct) openNode;
if (!secCons.getInfo().hasNowaitClause()) {
needsBarrier = true;
}
}
if (InsertDummyFlushDirectives.requiresRemovalOfDFDs(openNode)) {
needsFlush = true;
}
if (needsBarrier) {
BarrierDirective barrDir = FrontEnd.parseAndNormalize("#pragma omp barrier\n",
BarrierDirective.class);
InsertImmediateSuccessor.insert(openNode, barrDir);
}
if (needsFlush) {
FlushDirective flushDir = FrontEnd.parseAndNormalize("#pragma omp flush\n", FlushDirective.class);
InsertImmediatePredecessor.insert(openNode, flushDir);
}
NodeRemover.removeNode(openNode);
}
}
}
@Deprecated
public static void deprecated_insertImmediatePredecessor(Node baseNode, Node predecessor) {
baseNode = Misc.getCFGNodeFor(baseNode);
Node parentNode = Misc.getEnclosingCFGNonLeafNode(baseNode);
CFGGenerator.createCFGEdgesIn(predecessor);
Deprecated_ImmediatePredecessorInserter updater = new Deprecated_ImmediatePredecessorInserter(baseNode,
predecessor);
parentNode.accept(updater);
}
@Deprecated
public static void deprecated_insertImmediateSuccessor(Node baseNode, Node successor) {
baseNode = Misc.getCFGNodeFor(baseNode);
Node parentNode = Misc.getEnclosingCFGNonLeafNode(baseNode);
Deprecated_ImmediatePredecessorInserter updater = new Deprecated_ImmediatePredecessorInserter(baseNode,
successor);
parentNode.accept(updater);
}
}
