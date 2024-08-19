package imop.lib.transform.simplify;
import imop.ast.annotation.Label;
import imop.ast.info.OmpConstructInfo;
import imop.ast.node.external.*;
import imop.lib.transform.updater.sideeffect.AddedEnclosingBlock;
import imop.lib.transform.updater.sideeffect.AddedExplicitBarrier;
import imop.lib.transform.updater.sideeffect.AddedNowaitClause;
import imop.lib.transform.updater.sideeffect.SideEffect;
import imop.lib.util.Misc;
import imop.parser.FrontEnd;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
public class ImplicitBarrierRemover {
public static void removeImplicitBarrierDuringParsing(Node newNode) {
for (CompoundStatement enclosingComp : Misc.getInheritedPostOrderEnclosee(newNode, CompoundStatement.class)) {
for (Node stmtNode : enclosingComp.getInfo().getCFGInfo().getElementList()) {
int indexOfNode = -1;
if (stmtNode instanceof ForConstruct) {
ForConstruct forCons = (ForConstruct) stmtNode;
OmpConstructInfo constructInfo = forCons.getInfo();
if (!constructInfo.hasNowaitClause()) {
indexOfNode = enclosingComp.getInfo().getCFGInfo().getElementList().indexOf(stmtNode);
assert (indexOfNode != -1)
: "Could not find proper enclosing compound-statement for the node: " + stmtNode;
forCons.getInfo().addNowaitClause();
new AddedNowaitClause(forCons);
} else {
continue;
}
} else if (stmtNode instanceof SectionsConstruct) {
SectionsConstruct secCons = (SectionsConstruct) stmtNode;
OmpConstructInfo constructInfo = secCons.getInfo();
if (!constructInfo.hasNowaitClause()) {
indexOfNode = enclosingComp.getInfo().getCFGInfo().getElementList().indexOf(stmtNode);
assert (indexOfNode != -1)
: "Could not find proper enclosing compound-statement for the node: " + stmtNode;
secCons.getInfo().addNowaitClause();
new AddedNowaitClause(secCons);
} else {
continue;
}
} else if (stmtNode instanceof SingleConstruct) {
SingleConstruct singleCons = (SingleConstruct) stmtNode;
OmpConstructInfo constructInfo = singleCons.getInfo();
if (!constructInfo.hasNowaitClause()) {
indexOfNode = enclosingComp.getInfo().getCFGInfo().getElementList().indexOf(stmtNode);
assert (indexOfNode != -1)
: "Could not find proper enclosing compound-statement for the node: " + stmtNode;
singleCons.getInfo().addNowaitClause();
new AddedNowaitClause(singleCons);
} else {
continue;
}
} else {
continue;
}
BarrierDirective newBarrierDirective = FrontEnd.parseAndNormalize("#pragma omp barrier\n",
BarrierDirective.class);
enclosingComp.getInfo().getCFGInfo().addElement(indexOfNode + 1, newBarrierDirective);
new AddedExplicitBarrier(newBarrierDirective);
}
}
}
public static Statement makeBarrierExplicitForNode(Statement inNode, List<SideEffect> sideEffectList) {
Statement node = (Statement) Misc.getCFGNodeFor(inNode);
boolean changed = false;
if (node instanceof ForConstruct) {
ForConstruct forCons = (ForConstruct) node;
OmpConstructInfo constructInfo = forCons.getInfo();
if (!constructInfo.hasNowaitClause()) {
forCons.getInfo().addNowaitClause();
sideEffectList.add(new AddedNowaitClause(node));
changed = true;
}
} else if (node instanceof SectionsConstruct) {
SectionsConstruct secCons = (SectionsConstruct) node;
OmpConstructInfo constructInfo = secCons.getInfo();
if (!constructInfo.hasNowaitClause()) {
secCons.getInfo().addNowaitClause();
sideEffectList.add(new AddedNowaitClause(node));
changed = true;
}
} else if (node instanceof SingleConstruct) {
SingleConstruct singleCons = (SingleConstruct) node;
OmpConstructInfo constructInfo = singleCons.getInfo();
if (!constructInfo.hasNowaitClause()) {
singleCons.getInfo().addNowaitClause();
sideEffectList.add(new AddedNowaitClause(node));
changed = true;
}
} else {
return inNode;
}
if (!changed) {
return inNode;
}
CompoundStatement compStmt = FrontEnd.parseAndNormalize("{}", CompoundStatement.class);
sideEffectList.addAll(compStmt.getInfo().getCFGInfo().addElement(0, node));
BarrierDirective newBarrierDirective = FrontEnd.parseAndNormalize("#pragma omp barrier\n",
BarrierDirective.class);
sideEffectList.addAll(compStmt.getInfo().getCFGInfo().addAtLast(newBarrierDirective));
List<Node> elements = compStmt.getInfo().getCFGInfo().getElementList();
sideEffectList.add(new AddedExplicitBarrier(elements.get(elements.size() - 1)));
sideEffectList.add(new AddedEnclosingBlock(compStmt));
return Misc.getStatementWrapper(compStmt);
}
@Deprecated
public static void deprecated_removeImplicitBarrier(Node root) {
Set<Class<? extends Node>> classList = new HashSet<>();
classList.add(ForConstruct.class);
classList.add(SectionsConstruct.class);
classList.add(SingleConstruct.class);
for (Node stmtNode : Misc.getInheritedPostOrderEnclosee(root, classList)) {
if (stmtNode == root) {
continue;
}
Statement oldStmt = (Statement) stmtNode;
ImplicitBarrierRemover.makeBarrierExplicitForNode(oldStmt, new ArrayList<>());
}
}
@Deprecated
public static Statement deprecated_obtainNormalizedNode(Statement inNode, List<SideEffect> sideEffectList) {
Statement retNode = inNode;
Statement node = (Statement) Misc.getCFGNodeFor(inNode);
boolean changed = false;
if (node instanceof ForConstruct) {
ForConstruct forCons = (ForConstruct) node;
OmpConstructInfo constructInfo = forCons.getInfo();
if (!constructInfo.hasNowaitClause()) {
String newConstruct = "";
for (Label lab : constructInfo.getLabelAnnotations()) {
newConstruct += lab.getString();
}
newConstruct += "{";
newConstruct += forCons.getF0().getInfo().getString();
newConstruct += forCons.getF1().getF0().getInfo().getString();
newConstruct += forCons.getF1().getF1().getInfo().getString();
newConstruct += " nowait\n";
newConstruct += forCons.getF2().getInfo().getString();
newConstruct += forCons.getF3().getInfo().getString();
newConstruct += "# pragma omp barrier \n";
newConstruct += "}";
retNode = FrontEnd.parseAndNormalize(newConstruct, Statement.class);
changed = true;
}
} else if (node instanceof SectionsConstruct) {
SectionsConstruct secCons = (SectionsConstruct) node;
OmpConstructInfo constructInfo = secCons.getInfo();
if (!constructInfo.hasNowaitClause()) {
String newConstruct = "";
for (Label lab : constructInfo.getLabelAnnotations()) {
newConstruct += lab.getString();
}
newConstruct += "{";
newConstruct += secCons.getF0().getInfo().getString();
newConstruct += secCons.getF1().getInfo().getString();
newConstruct += secCons.getF2().getInfo().getString();
newConstruct += " nowait ";
newConstruct += secCons.getF3().getInfo().getString();
newConstruct += secCons.getF4().getInfo().getString();
newConstruct += "# pragma omp barrier \n";
newConstruct += "}";
retNode = FrontEnd.parseAndNormalize(newConstruct, Statement.class);
changed = true;
}
} else if (node instanceof SingleConstruct) {
SingleConstruct singleCons = (SingleConstruct) node;
OmpConstructInfo constructInfo = singleCons.getInfo();
if (!constructInfo.hasNowaitClause()) {
String newConstruct = "";
for (Label lab : constructInfo.getLabelAnnotations()) {
newConstruct += lab.getString();
}
newConstruct += "{";
newConstruct += singleCons.getF0().getInfo().getString();
newConstruct += singleCons.getF1().getInfo().getString();
newConstruct += singleCons.getF2().getInfo().getString();
newConstruct += " nowait ";
newConstruct += singleCons.getF3().getInfo().getString();
newConstruct += singleCons.getF4().getInfo().getString();
newConstruct += "# pragma omp barrier \n";
newConstruct += "}";
retNode = FrontEnd.parseAndNormalize(newConstruct, Statement.class);
changed = true;
}
}
if (changed) {
CompoundStatement compStmt = (CompoundStatement) Misc.getCFGNodeFor(retNode);
List<Node> elements = compStmt.getInfo().getCFGInfo().getElementList();
for (int i = 0; i < elements.size(); i++) {
Node elem = elements.get(i);
if (elem instanceof ForConstruct || elem instanceof SectionsConstruct
|| elem instanceof SingleConstruct) {
sideEffectList.add(new AddedNowaitClause(elem));
break;
}
}
sideEffectList.add(new AddedExplicitBarrier(elements.get(elements.size() - 1)));
sideEffectList.add(new AddedEnclosingBlock(compStmt));
}
return retNode;
}
}
