package imop.lib.transform.simplify;
import imop.ast.annotation.Label;
import imop.ast.node.external.*;
import imop.ast.node.internal.*;
import imop.baseVisitor.GJNoArguDepthFirstProcess;
import imop.lib.analysis.typesystem.Type;
import imop.lib.builder.Builder;
import imop.lib.getter.ExpressionTypeGetter;
import imop.lib.getter.StructUnionOrEnumInfoGetter;
import imop.lib.util.Misc;
import imop.parser.FrontEnd;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
public class ExpressionSimplifier extends GJNoArguDepthFirstProcess<ExpressionSimplifier.SimplificationString> {
public static final int INIT = 2;
public static class SimplificationString {
private StringBuilder replacementString;
private StringBuilder prelude;
private List<Declaration> temporaryDeclarations;
public SimplificationString() {
}
public StringBuilder getPrelude() {
if (prelude == null) {
prelude = new StringBuilder(INIT);
}
return prelude;
}
public void setPrelude(StringBuilder prelude) {
this.prelude = prelude;
}
public List<Declaration> getTemporaryDeclarations() {
if (temporaryDeclarations == null) {
temporaryDeclarations = new ArrayList<>();
}
return temporaryDeclarations;
}
public void setTemporaryDeclarations(List<Declaration> temporaryDeclarations) {
this.temporaryDeclarations = temporaryDeclarations;
}
public StringBuilder getReplacementString() {
if (replacementString == null) {
replacementString = new StringBuilder(INIT);
}
return replacementString;
}
public void setReplacementString(StringBuilder replacementString) {
this.replacementString = replacementString;
}
public boolean hasNoTempDeclarations() {
if (temporaryDeclarations == null) {
return true;
}
return temporaryDeclarations.isEmpty();
}
public int hasPrelude() {
if (prelude == null) {
return 0;
}
return prelude.length();
}
}
Set<CompoundStatement> originalCS = new HashSet<>();
public ExpressionSimplifier(Set<CompoundStatement> originalCS) {
this.originalCS = originalCS;
}
private void extractFunctionCall(SimplificationString ret, Expression retExp) {
if (!Misc.isACall(ret.getReplacementString())) {
return;
}
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(retExp).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
@Override
public SimplificationString visit(NodeList n) {
SimplificationString ret = new SimplificationString();
for (Node node : n.getNodes()) {
SimplificationString elementSS = node.accept(this);
ret.getPrelude().append(elementSS.getPrelude());
ret.getTemporaryDeclarations().addAll(elementSS.getTemporaryDeclarations());
ret.getReplacementString().append(" " + elementSS.getReplacementString() + " ");
}
return ret;
}
@Override
public SimplificationString visit(NodeListOptional n) {
SimplificationString ret = new SimplificationString();
if (n.present()) {
for (Node node : n.getNodes()) {
SimplificationString elementSS = node.accept(this);
ret.getPrelude().append(elementSS.getPrelude());
ret.getTemporaryDeclarations().addAll(elementSS.getTemporaryDeclarations());
ret.getReplacementString().append(" " + elementSS.getReplacementString() + " ");
}
}
return ret;
}
@Override
public SimplificationString visit(NodeOptional n) {
SimplificationString ret = new SimplificationString();
if (n.present()) {
ret = n.getNode().accept(this);
}
return ret;
}
@Override
public SimplificationString visit(NodeToken n) {
SimplificationString ret = new SimplificationString();
ret.setReplacementString(new StringBuilder(n.getTokenImage()));
return ret;
}
@Override
public SimplificationString visit(NodeSequence n) {
SimplificationString ret = new SimplificationString();
if (n.getNodes().isEmpty()) {
return ret;
}
for (Node node : n.getNodes()) {
SimplificationString tempSS = node.accept(this);
if (tempSS == null) {
System.err.println(node.getClass().getSimpleName());
System.exit(0);
}
ret.getPrelude().append(tempSS.getPrelude());
ret.getTemporaryDeclarations().addAll(tempSS.getTemporaryDeclarations());
ret.getReplacementString().append(" " + tempSS.getReplacementString());
}
return ret;
}
@Override
public SimplificationString visit(TranslationUnit n) {
SimplificationString ret = new SimplificationString();
for (Node element : n.getF0().getNodes()) {
SimplificationString elementSS = element.accept(this);
for (Declaration tempDecl : elementSS.getTemporaryDeclarations()) {
ret.getReplacementString().append(tempDecl.getInfo().getString() + " ");
}
ret.getReplacementString().append(elementSS.getPrelude());
ret.getReplacementString().append(" " + elementSS.getReplacementString() + " ");
}
return ret;
}
@Override
public SimplificationString visit(ElementsOfTranslation n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(ExternalDeclaration n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(FunctionDefinition n) {
SimplificationString ret = new SimplificationString();
ret.setReplacementString(new StringBuilder(n.getF0().getInfo().getString() + " "));
ret.getReplacementString().append(this.getVoidFixedString(n.getF1()));
ret.getReplacementString().append(n.getF2().getInfo().getString() + " ");
SimplificationString compSS = n.getF3().accept(this);
assert (compSS.hasNoTempDeclarations());
assert (compSS.hasPrelude() == 0);
ret.getReplacementString().append(compSS.getReplacementString());
return ret;
}
private String getVoidFixedString(Declarator decl) {
String retStr = decl + " ";
if (Misc.getInheritedEnclosee(decl, Declarator.class).size() != 1) {
return retStr;
}
List<Node> decOpList = decl.getF1().getF1().getF0().getNodes();
if (decOpList.size() != 1
|| !(((ADeclaratorOp) decOpList.get(0)).getF0().getChoice() instanceof ParameterTypeListClosed)) {
return retStr;
}
if (decOpList.get(0).toString().trim().equals("(void )")) {
retStr = (decl.getF0().present() ? (decl.getF0() + " ") : "") + decl.getF1().getF0() + " ()" + " ";
}
return retStr;
}
@Override
public SimplificationString visit(Declaration n) {
SimplificationString ret = new SimplificationString();
if (!n.getF1().present()) {
ret.setReplacementString(new StringBuilder(n.getF0() + ";"));
return ret;
}
StringBuilder declarationSpecifierString = new StringBuilder(n.getF0() + " ");
StructUnionOrEnumInfoGetter aggregateInfoGetter = new StructUnionOrEnumInfoGetter();
n.accept(aggregateInfoGetter);
if (aggregateInfoGetter.isUserDefinedDefinitionOrAssociatedTypedef) {
StringBuilder declarationSpecifierStringSimple = new StringBuilder(aggregateInfoGetter.simpleDeclaration);
if (n.getInfo().isTypedef()) {
StringBuilder str = new StringBuilder(INIT);
for (Node aDeclSpecNode : n.getF0().getF0().getNodes()) {
ADeclarationSpecifier aDeclSpec = (ADeclarationSpecifier) aDeclSpecNode;
if (aDeclSpec.getF0().getChoice() instanceof StorageClassSpecifier) {
StorageClassSpecifier stoClaSpec = (StorageClassSpecifier) aDeclSpec.getF0().getChoice();
if (stoClaSpec.getF0().getChoice() instanceof NodeToken) {
NodeToken nodeToken = (NodeToken) stoClaSpec.getF0().getChoice();
if (nodeToken.getTokenImage().equals("typedef")) {
} else {
str.append(aDeclSpec + " ");
}
} else {
assert (false);
}
} else {
str.append(aDeclSpec + " ");
}
}
ret.setPrelude(str.append(";"));
} else {
ret.setPrelude(new StringBuilder(declarationSpecifierString + ";"));
}
InitDeclaratorList initDeclList = (InitDeclaratorList) n.getF1().getNode();
SimplificationString declSS = initDeclList.getF0().accept(this);
ret.getTemporaryDeclarations().addAll(declSS.getTemporaryDeclarations());
ret.getPrelude().append(declSS.getPrelude());
if (!initDeclList.getF1().present()) {
ret.setReplacementString(new StringBuilder(
declarationSpecifierStringSimple + " " + declSS.getReplacementString() + ";"));
return ret;
} else {
ret.getPrelude().append(declarationSpecifierStringSimple + " " + declSS.getReplacementString() + ";");
Node lastElement = null;
List<Node> allButFirst = initDeclList.getF1().getNodes();
for (Node declNode : allButFirst) {
if (declNode == allButFirst.get(allButFirst.size() - 1)) {
declNode = ((NodeSequence) declNode).getNodes().get(1);
lastElement = declNode;
break;
}
declNode = ((NodeSequence) declNode).getNodes().get(1);
declSS = declNode.accept(this);
ret.getTemporaryDeclarations().addAll(declSS.getTemporaryDeclarations());
ret.getPrelude().append(declSS.getPrelude());
ret.getPrelude()
.append(declarationSpecifierStringSimple + " " + declSS.getReplacementString() + ";");
}
declSS = lastElement.accept(this);
ret.getTemporaryDeclarations().addAll(declSS.getTemporaryDeclarations());
ret.getPrelude().append(declSS.getPrelude());
ret.setReplacementString(new StringBuilder(
declarationSpecifierStringSimple + " " + declSS.getReplacementString() + ";"));
return ret;
}
} else {
InitDeclaratorList initDeclList = (InitDeclaratorList) n.getF1().getNode();
SimplificationString declSS = initDeclList.getF0().accept(this);
ret.getTemporaryDeclarations().addAll(declSS.getTemporaryDeclarations());
ret.setPrelude(declSS.getPrelude());
if (!initDeclList.getF1().present()) {
ret.getReplacementString()
.append(declarationSpecifierString + " " + declSS.getReplacementString() + ";");
} else {
ret.getPrelude().append(declarationSpecifierString + " " + declSS.getReplacementString() + ";");
Node lastElement = null;
List<Node> allButFirst = initDeclList.getF1().getNodes();
for (Node declNode : allButFirst) {
if (declNode == allButFirst.get(allButFirst.size() - 1)) {
declNode = ((NodeSequence) declNode).getNodes().get(1);
lastElement = declNode;
break;
}
declNode = ((NodeSequence) declNode).getNodes().get(1);
declSS = declNode.accept(this);
ret.getTemporaryDeclarations().addAll(declSS.getTemporaryDeclarations());
ret.getPrelude().append(declSS.getPrelude());
ret.getPrelude().append(declarationSpecifierString + " " + declSS.getReplacementString() + ";");
}
declSS = lastElement.accept(this);
ret.getTemporaryDeclarations().addAll(declSS.getTemporaryDeclarations());
ret.getPrelude().append(declSS.getPrelude());
ret.setReplacementString(
new StringBuilder(declarationSpecifierString + " " + declSS.getReplacementString() + ";"));
}
}
return ret;
}
@Override
public SimplificationString visit(DeclarationList n) {
SimplificationString ret = new SimplificationString();
for (Node declNode : n.getF0().getNodes()) {
SimplificationString elementSS = declNode.accept(this);
for (Declaration tempDecl : elementSS.getTemporaryDeclarations()) {
ret.getReplacementString().append(tempDecl.getInfo().getString() + " ");
}
ret.getReplacementString().append(elementSS.getPrelude());
ret.getReplacementString().append(elementSS.getReplacementString());
}
return ret;
}
@Override
public SimplificationString visit(DeclarationSpecifiers n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(ADeclarationSpecifier n) {
SimplificationString ret = n.getF0().accept(this);
ret.getReplacementString().append(" ");
return ret;
}
@Override
public SimplificationString visit(InitDeclarator n) {
SimplificationString ret = new SimplificationString();
if (n.getF1().present()) {
Initializer initNode = (Initializer) ((NodeSequence) n.getF1().getNode()).getNodes().get(1);
ret = initNode.accept(this);
this.extractFunctionCall(ret, initNode);
ret.setReplacementString(
new StringBuilder(n.getF0().getInfo().getString()).append(" = " + ret.getReplacementString()));
} else {
ret.setReplacementString(new StringBuilder(n.getF0().getInfo().getString() + " "));
}
return ret;
}
@Override
public SimplificationString visit(Initializer n) {
if (n.getF0().getChoice() instanceof AssignmentExpression) {
SimplificationString ret = n.getF0().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType((AssignmentExpression) n.getF0().getChoice()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
return ret;
}
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(ArrayInitializer n) {
SimplificationString ret = n.getF1().accept(this);
ret.setReplacementString(
new StringBuilder("{" + ret.getReplacementString() + n.getF2().getInfo().getString() + " }"));
return ret;
}
@Override
public SimplificationString visit(InitializerList n) {
SimplificationString ret = n.getF0().accept(this);
for (Node initNodeSeq : n.getF1().getNodes()) {
Node init = ((NodeSequence) initNodeSeq).getNodes().get(1);
SimplificationString initSS = init.accept(this);
ret.getTemporaryDeclarations().addAll(initSS.getTemporaryDeclarations());
ret.getPrelude().append(initSS.getPrelude());
ret.getReplacementString().append(", " + initSS.getReplacementString());
}
return ret;
}
@Override
public SimplificationString visit(TypeName n) {
SimplificationString ret0 = n.getF0().accept(this);
SimplificationString ret1 = n.getF1().accept(this);
if (ret1.getReplacementString().toString().isEmpty()) {
ret0.getReplacementString().append(" ");
} else {
ret0.getReplacementString().append(" " + ret1.getReplacementString().toString() + " ");
}
return ret0;
}
@Override
public SimplificationString visit(AbstractDeclarator n) {
SimplificationString ret = n.getF0().accept(this);
ret.getReplacementString().append(" ");
return ret;
}
@Override
public SimplificationString visit(TypedefName n) {
SimplificationString ret = new SimplificationString();
ret.getReplacementString().append(" " + n.getF0().getTokenImage() + " ");
return ret;
}
@Override
public SimplificationString visit(Statement n) {
SimplificationString ret = n.getStmtF0().accept(this);
assert (ret.hasNoTempDeclarations());
assert (ret.hasPrelude() == 0);
return ret;
}
@Override
public SimplificationString visit(UnknownCpp n) {
SimplificationString ret = new SimplificationString();
ret.setReplacementString(new StringBuilder(INIT));
List<Label> labels = n.getInfo().getLabelAnnotations();
for (Label label : labels) {
ret.getReplacementString().append(label.getString());
}
ret.getReplacementString().append("\n#" + n.getF1() + " \n");
return ret;
}
@Override
public SimplificationString visit(OmpConstruct n) {
return n.getOmpConsF0().accept(this);
}
@Override
public SimplificationString visit(OmpDirective n) {
return n.getOmpDirF0().accept(this);
}
@Override
public SimplificationString visit(UnknownPragma n) {
SimplificationString ret = new SimplificationString();
ret.setReplacementString(new StringBuilder(INIT));
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
ret.getReplacementString().append("\n#pragma " + n.getF2() + "\n");
return ret;
}
@Override
public SimplificationString visit(ParallelConstruct n) {
SimplificationString ret = new SimplificationString();
SimplificationString stmtSS = n.getParConsF2().accept(this);
assert (stmtSS.hasNoTempDeclarations());
assert (stmtSS.hasPrelude() == 0);
SimplificationString parDirSS = n.getParConsF1().accept(this);
if (parDirSS.hasPrelude() == 0) {
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
ret.getReplacementString().append("\n#pragma omp ");
ret.getReplacementString().append(parDirSS.getReplacementString());
ret.getReplacementString().append(stmtSS.getReplacementString());
} else {
boolean needsEncapsulation = this.needsEncapsulation(n);
if (needsEncapsulation) {
ret.getReplacementString().append("{");
}
for (Declaration tempDecl : parDirSS.getTemporaryDeclarations()) {
ret.getReplacementString().append(tempDecl.getInfo().getString() + " ");
}
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
ret.getReplacementString().append(parDirSS.getPrelude());
ret.getReplacementString().append("\n#pragma omp ");
ret.getReplacementString().append(parDirSS.getReplacementString());
ret.getReplacementString().append(stmtSS.getReplacementString());
if (needsEncapsulation) {
ret.getReplacementString().append("}");
}
}
return ret;
}
@Override
public SimplificationString visit(ParallelDirective n) {
SimplificationString ret = n.getF1().accept(this);
ret.setReplacementString(
new StringBuilder(" parallel " + ret.getReplacementString() + n.getF2().getInfo().getString() + " "));
return ret;
}
@Override
public SimplificationString visit(UniqueParallelOrDataClauseList n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(AUniqueParallelOrDataClause n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(UniqueParallelClause n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(IfClause n) {
SimplificationString ret = n.getF2().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF2()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(new StringBuilder(" if (" + ret.getReplacementString() + ") "));
return ret;
}
@Override
public SimplificationString visit(NumThreadsClause n) {
SimplificationString ret = n.getF2().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF2()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(new StringBuilder(" num_threads (" + ret.getReplacementString() + ") "));
return ret;
}
@Override
public SimplificationString visit(DataClause n) {
SimplificationString ret = new SimplificationString();
ret.setReplacementString(new StringBuilder(n.getF0().toString()));
ret.getReplacementString().append(" ");
return ret;
}
@Override
public SimplificationString visit(ForConstruct n) {
SimplificationString ret = new SimplificationString();
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
SimplificationString forDirSS = n.getF1().accept(this);
SimplificationString initExpSS = n.getF2().getF2().accept(this);
SimplificationString forCondSS = n.getF2().getF4().accept(this);
SimplificationString reinitExpSS = n.getF2().getF6().accept(this);
SimplificationString stmtSS = n.getF3().accept(this);
if (forCondSS.hasPrelude() == 0) {
StringBuilder preString = new StringBuilder(INIT);
for (Declaration tempDecl : forDirSS.getTemporaryDeclarations()) {
preString.append(tempDecl.getInfo().getString() + " ");
}
for (Declaration tempDecl : initExpSS.getTemporaryDeclarations()) {
preString.append(tempDecl.getInfo().getString() + " ");
}
preString.append(forDirSS.getPrelude());
preString.append(initExpSS.getPrelude());
if (preString.length() != 0) {
boolean needsEncapsulation = this.needsEncapsulation(n);
if (needsEncapsulation) {
ret.getReplacementString().append("{");
}
ret.getReplacementString().append(preString);
}
ret.getReplacementString().append("\n#pragma omp ");
ret.getReplacementString().append(forDirSS.getReplacementString() + " ");
ret.getReplacementString().append(" for (" + initExpSS.getReplacementString() + "; "
+ forCondSS.getReplacementString() + ";" + reinitExpSS.getReplacementString() + ")");
if (reinitExpSS.hasPrelude() == 0) {
ret.getReplacementString().append(stmtSS.getReplacementString());
} else {
ret.getReplacementString().append("{");
for (Declaration tempDecl : reinitExpSS.getTemporaryDeclarations()) {
ret.getReplacementString().append(tempDecl.getInfo().getString() + " ");
}
ret.getReplacementString().append(stmtSS.getReplacementString());
ret.getReplacementString().append(reinitExpSS.getPrelude());
ret.getReplacementString().append("}");
}
if (preString.length() != 0) {
boolean needsEncapsulation = this.needsEncapsulation(n);
if (needsEncapsulation) {
ret.getReplacementString().append("}");
}
}
} else {
StringBuilder preString = new StringBuilder(INIT);
for (Declaration tempDecl : forDirSS.getTemporaryDeclarations()) {
preString.append(tempDecl.getInfo().getString() + " ");
}
for (Declaration tempDecl : initExpSS.getTemporaryDeclarations()) {
preString.append(tempDecl.getInfo().getString() + " ");
}
for (Declaration tempDecl : forCondSS.getTemporaryDeclarations()) {
preString.append(tempDecl.getInfo().getString() + " ");
}
preString.append(forDirSS.getPrelude());
preString.append(initExpSS.getPrelude());
preString.append(initExpSS.getReplacementString());
preString.append(forCondSS.getPrelude());
if (preString.length() != 0) {
boolean needsEncapsulation = this.needsEncapsulation(n);
if (needsEncapsulation) {
ret.getReplacementString().append("{");
}
ret.getReplacementString().append(preString);
}
ret.getReplacementString().append("\n#pragma omp ");
ret.getReplacementString().append(forDirSS.getReplacementString() + " ");
ret.getReplacementString().append(" for ( ;" + forCondSS.getReplacementString() + ";)");
ret.getReplacementString().append("{");
for (Declaration tempDecl : reinitExpSS.getTemporaryDeclarations()) {
ret.getReplacementString().append(tempDecl.getInfo().getString() + " ");
}
ret.getReplacementString().append(" " + stmtSS.getReplacementString());
ret.getReplacementString().append(" " + reinitExpSS.getPrelude());
ret.getReplacementString().append(" " + reinitExpSS.getReplacementString());
ret.getReplacementString().append(" " + forCondSS.getPrelude());
ret.getReplacementString().append("}");
if (preString.length() != 0) {
boolean needsEncapsulation = this.needsEncapsulation(n);
if (needsEncapsulation) {
ret.getReplacementString().append("}");
}
}
}
return ret;
}
@Override
public SimplificationString visit(ForDirective n) {
SimplificationString ret = n.getF1().accept(this);
ret.setReplacementString(
new StringBuilder(" for " + ret.getReplacementString() + " " + n.getF2().getInfo().getString() + " "));
return ret;
}
@Override
public SimplificationString visit(UniqueForOrDataOrNowaitClauseList n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(AUniqueForOrDataOrNowaitClause n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(NowaitClause n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(UniqueForClause n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(UniqueForCollapse n) {
SimplificationString ret = n.getF2().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF2()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(new StringBuilder(" collapse(" + ret.getReplacementString() + ")"));
return ret;
}
@Override
public SimplificationString visit(UniqueForClauseSchedule n) {
SimplificationString ret = new SimplificationString();
if (n.getF3().present()) {
Expression exp = (Expression) ((NodeSequence) n.getF3().getNode()).getNodes().get(1);
ret = exp.accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(exp).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(new StringBuilder(
" schedule (" + n.getF2().getInfo().getString() + " , " + ret.getReplacementString() + ")"));
} else {
ret.setReplacementString(new StringBuilder(" schedule (" + n.getF2().getInfo().getString() + ")"));
}
return ret;
}
@Override
public SimplificationString visit(OmpForInitExpression n) {
SimplificationString ret = n.getF2().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF2()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(
new StringBuilder(n.getF0().getInfo().getString() + " = " + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(OmpForCondition n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(OmpForLTCondition n) {
SimplificationString ret = n.getF2().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF2()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(
new StringBuilder(n.getF0().getInfo().getString() + " < " + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(OmpForLECondition n) {
SimplificationString ret = n.getF2().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF2()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(
new StringBuilder(n.getF0().getInfo().getString() + " <= " + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(OmpForGTCondition n) {
SimplificationString ret = n.getF2().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF2()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(
new StringBuilder(n.getF0().getInfo().getString() + " > " + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(OmpForGECondition n) {
SimplificationString ret = n.getF2().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF2()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(
new StringBuilder(n.getF0().getInfo().getString() + " >= " + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(OmpForReinitExpression n) {
return n.getOmpForReinitF0().accept(this);
}
@Override
public SimplificationString visit(PostIncrementId n) {
SimplificationString ret = new SimplificationString();
ret.setReplacementString(new StringBuilder(n.getF0().getTokenImage() + "++ "));
return ret;
}
@Override
public SimplificationString visit(PostDecrementId n) {
SimplificationString ret = new SimplificationString();
ret.setReplacementString(new StringBuilder(n.getF0().getTokenImage() + "-- "));
return ret;
}
@Override
public SimplificationString visit(PreIncrementId n) {
SimplificationString ret = new SimplificationString();
ret.setReplacementString(new StringBuilder("++" + n.getF1().getTokenImage() + " "));
return ret;
}
@Override
public SimplificationString visit(PreDecrementId n) {
SimplificationString ret = new SimplificationString();
ret.setReplacementString(new StringBuilder("--" + n.getF1().getTokenImage() + " "));
return ret;
}
@Override
public SimplificationString visit(ShortAssignPlus n) {
SimplificationString ret = n.getF2().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF2()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(
new StringBuilder(n.getF0().getInfo().getString() + " += " + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(ShortAssignMinus n) {
SimplificationString ret = n.getF2().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF2()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(
new StringBuilder(n.getF0().getInfo().getString() + " -= " + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(OmpForAdditive n) {
SimplificationString ret = n.getF4().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF4()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(new StringBuilder(n.getF0().getInfo().getString() + " = "
+ n.getF2().getInfo().getString() + " + " + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(OmpForSubtractive n) {
SimplificationString ret = n.getF4().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF4()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(new StringBuilder(n.getF0().getInfo().getString() + " = "
+ n.getF2().getInfo().getString() + " - " + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(OmpForMultiplicative n) {
SimplificationString ret = n.getF2().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF2()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(new StringBuilder(n.getF0().getInfo().getString() + " = " + ret.getReplacementString()
+ " + " + n.getF4().getInfo().getString()));
return ret;
}
@Override
public SimplificationString visit(SectionsConstruct n) {
SimplificationString ret = new SimplificationString();
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
SimplificationString clauseSS = n.getF2().accept(this);
SimplificationString sectionsSS = n.getF4().accept(this);
assert (sectionsSS.hasNoTempDeclarations());
assert (sectionsSS.hasPrelude() == 0);
if (clauseSS.hasPrelude() == 0) {
ret.getReplacementString().append("\n#pragma omp sections ");
ret.getReplacementString()
.append(clauseSS.getReplacementString() + " " + n.getF3().getInfo().getString() + " ");
ret.getReplacementString().append(sectionsSS.getReplacementString());
} else {
boolean needsEncapsulation = this.needsEncapsulation(n);
if (needsEncapsulation) {
ret.getReplacementString().append("{");
}
for (Declaration tempDecl : clauseSS.getTemporaryDeclarations()) {
ret.getReplacementString().append(tempDecl.getInfo().getString() + " ");
}
ret.getReplacementString().append(clauseSS.getPrelude() + "\n");
ret.getReplacementString().append("\n#pragma omp sections ");
ret.getReplacementString()
.append(clauseSS.getReplacementString() + " " + n.getF3().getInfo().getString() + " ");
ret.getReplacementString().append(sectionsSS.getReplacementString());
if (needsEncapsulation) {
ret.getReplacementString().append("}");
}
}
return ret;
}
@Override
public SimplificationString visit(NowaitDataClauseList n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(ANowaitDataClause n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(SectionsScope n) {
SimplificationString ret = n.getF1().accept(this);
SimplificationString sectionSS = n.getF2().accept(this);
ret.setReplacementString(
new StringBuilder("{" + ret.getReplacementString() + sectionSS.getReplacementString() + "}"));
return ret;
}
@Override
public SimplificationString visit(ASection n) {
SimplificationString ret = n.getF3().accept(this);
ret.setReplacementString(new StringBuilder(n.getF0().getInfo().getString() + " section "
+ n.getF2().getInfo().getString() + " " + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(SingleConstruct n) {
SimplificationString ret = new SimplificationString();
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
SimplificationString clauseSS = n.getF2().accept(this);
SimplificationString stmtSS = n.getF4().accept(this);
if (clauseSS.hasPrelude() == 0) {
ret.getReplacementString().append("\n#pragma omp single " + clauseSS.getReplacementString()
+ n.getF3().getInfo().getString() + " " + stmtSS.getReplacementString());
} else {
boolean needsEncapsulation = this.needsEncapsulation(n);
if (needsEncapsulation) {
ret.getReplacementString().append("{");
}
for (Declaration tempDecl : clauseSS.getTemporaryDeclarations()) {
ret.getReplacementString().append(tempDecl.getInfo().getString() + " ");
}
ret.getReplacementString().append(clauseSS.getPrelude());
ret.getReplacementString().append("\n#pragma omp single " + clauseSS.getReplacementString()
+ n.getF3().getInfo().getString() + " " + stmtSS.getReplacementString());
if (needsEncapsulation) {
ret.getReplacementString().append("}");
}
}
return ret;
}
@Override
public SimplificationString visit(SingleClauseList n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(ASingleClause n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(OmpCopyPrivateClause n) {
SimplificationString ret = new SimplificationString();
ret.setReplacementString(new StringBuilder("copyprivate(" + n.getF2() + ") "));
return ret;
}
@Override
public SimplificationString visit(TaskConstruct n) {
SimplificationString ret = new SimplificationString();
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
SimplificationString clauseSS = n.getF2().accept(this);
SimplificationString stmtSS = n.getF4().accept(this);
if (clauseSS.hasPrelude() == 0) {
ret.getReplacementString().append("\n#pragma omp task " + clauseSS.getReplacementString()
+ n.getF3().getInfo().getString() + " " + stmtSS.getReplacementString());
} else {
boolean needsEncapsulation = this.needsEncapsulation(n);
if (needsEncapsulation) {
ret.getReplacementString().append("{");
}
for (Declaration tempDecl : clauseSS.getTemporaryDeclarations()) {
ret.getReplacementString().append(tempDecl.getInfo().getString() + " ");
}
ret.getReplacementString().append(clauseSS.getPrelude() + "\n");
ret.getReplacementString().append("\n#pragma omp task " + clauseSS.getReplacementString()
+ n.getF3().getInfo().getString() + " " + stmtSS.getReplacementString());
if (needsEncapsulation) {
ret.getReplacementString().append("}");
}
}
return ret;
}
@Override
public SimplificationString visit(TaskClause n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(UniqueTaskClause n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(FinalClause n) {
SimplificationString ret = n.getF2().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF2()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(new StringBuilder(" final(" + ret.getReplacementString() + ")"));
return ret;
}
@Override
public SimplificationString visit(UntiedClause n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(MergeableClause n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(ParallelForConstruct n) {
SimplificationString ret = new SimplificationString();
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
SimplificationString forDirSS = n.getF3().accept(this);
SimplificationString initExpSS = n.getF5().getF2().accept(this);
SimplificationString forCondSS = n.getF5().getF4().accept(this);
SimplificationString reinitExpSS = n.getF5().getF6().accept(this);
SimplificationString stmtSS = n.getF6().accept(this);
StringBuilder forSpecificClauses = new StringBuilder(INIT);
StringBuilder parSpecificClauses = new StringBuilder(INIT);
List<OmpClause> clauseList = Misc.getClauseList(n);
for (OmpClause clause : clauseList) {
if (clause instanceof IfClause || clause instanceof NumThreadsClause || clause instanceof OmpPrivateClause
|| clause instanceof OmpFirstPrivateClause || clause instanceof OmpLastPrivateClause
|| clause instanceof OmpSharedClause || clause instanceof OmpCopyinClause
|| clause instanceof OmpDfltSharedClause || clause instanceof OmpDfltNoneClause
) {
parSpecificClauses.append(" " + clause.getInfo().getString() + " ");
} else {
forSpecificClauses.append(" " + clause.getInfo().getString() + " ");
}
}
if (forCondSS.hasPrelude() == 0) {
StringBuilder preString = new StringBuilder(INIT);
for (Declaration tempDecl : forDirSS.getTemporaryDeclarations()) {
preString.append(tempDecl.getInfo().getString() + " ");
}
for (Declaration tempDecl : initExpSS.getTemporaryDeclarations()) {
preString.append(tempDecl.getInfo().getString() + " ");
}
preString.append(forDirSS.getPrelude());
preString.append(initExpSS.getPrelude());
if (preString.length() != 0) {
ret.getReplacementString().append("{"); 
ret.getReplacementString().append(preString);
}
ret.getReplacementString().append("\n#pragma omp parallel " + parSpecificClauses);
ret.getReplacementString().append("\n{"); 
ret.getReplacementString().append("\n#pragma omp for " + forSpecificClauses);
ret.getReplacementString().append(n.getF4().getInfo().getString() + " ");
ret.getReplacementString().append(" for (" + initExpSS.getReplacementString() + "; "
+ forCondSS.getReplacementString() + ";" + reinitExpSS.getReplacementString() + ")");
if (reinitExpSS.hasPrelude() == 0) {
ret.getReplacementString().append(stmtSS.getReplacementString());
} else {
ret.getReplacementString().append("{");
for (Declaration tempDecl : reinitExpSS.getTemporaryDeclarations()) {
ret.getReplacementString().append(tempDecl.getInfo().getString() + " ");
}
ret.getReplacementString().append(stmtSS.getReplacementString());
ret.getReplacementString().append(reinitExpSS.getPrelude());
ret.getReplacementString().append("}");
}
ret.getReplacementString().append("\n}\n"); 
if (preString.length() != 0) {
ret.getReplacementString().append("}"); 
}
} else {
StringBuilder preString = new StringBuilder(INIT);
for (Declaration tempDecl : forDirSS.getTemporaryDeclarations()) {
preString.append(tempDecl.getInfo().getString() + " ");
}
for (Declaration tempDecl : initExpSS.getTemporaryDeclarations()) {
preString.append(tempDecl.getInfo().getString() + " ");
}
for (Declaration tempDecl : forCondSS.getTemporaryDeclarations()) {
preString.append(tempDecl.getInfo().getString() + " ");
}
preString.append(forDirSS.getPrelude());
preString.append(initExpSS.getPrelude());
preString.append(initExpSS.getReplacementString());
preString.append(forCondSS.getPrelude());
if (preString.length() != 0) {
ret.getReplacementString().append("{"); 
ret.getReplacementString().append(preString);
}
ret.getReplacementString().append("\n#pragma omp parallel " + parSpecificClauses);
ret.getReplacementString().append("\n{\n"); 
ret.getReplacementString().append("#pragma omp for " + forSpecificClauses);
ret.getReplacementString().append(n.getF4().getInfo().getString() + " ");
ret.getReplacementString().append(" for ( ;" + forCondSS.getReplacementString() + ";)");
ret.getReplacementString().append("{"); 
for (Declaration tempDecl : reinitExpSS.getTemporaryDeclarations()) {
ret.getReplacementString().append(tempDecl.getInfo().getString() + " ");
}
ret.getReplacementString().append(" " + stmtSS.getReplacementString());
ret.getReplacementString().append(" " + reinitExpSS.getPrelude());
ret.getReplacementString().append(" " + reinitExpSS.getReplacementString());
ret.getReplacementString().append(" " + forCondSS.getPrelude());
ret.getReplacementString().append("}"); 
ret.getReplacementString().append("\n}\n"); 
if (preString.length() != 0) {
ret.getReplacementString().append("}"); 
}
}
return ret;
}
@Override
public SimplificationString visit(UniqueParallelOrUniqueForOrDataClauseList n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(AUniqueParallelOrUniqueForOrDataClause n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(ParallelSectionsConstruct n) {
SimplificationString ret = new SimplificationString();
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
SimplificationString clauseSS = n.getF3().accept(this);
SimplificationString sectionsSS = n.getF5().accept(this);
StringBuilder parSpecificClauses = new StringBuilder(INIT);
StringBuilder secSpecificClauses = clauseSS.getReplacementString();
assert (sectionsSS.hasNoTempDeclarations());
assert (sectionsSS.hasPrelude() == 0);
if (clauseSS.hasPrelude() == 0) {
ret.getReplacementString().append("\n#pragma omp parallel " + parSpecificClauses);
ret.getReplacementString().append("\n{\n#pragma omp sections " + secSpecificClauses);
ret.getReplacementString().append(" " + n.getF4().getInfo().getString() + " ");
ret.getReplacementString().append(sectionsSS.getReplacementString());
ret.getReplacementString().append("}");
} else {
ret.getReplacementString().append("{");
for (Declaration tempDecl : clauseSS.getTemporaryDeclarations()) {
ret.getReplacementString().append(tempDecl.getInfo().getString() + " ");
}
ret.getReplacementString().append(clauseSS.getPrelude() + "\n");
ret.getReplacementString().append("\n#pragma omp parallel " + parSpecificClauses);
ret.getReplacementString().append("\n{\n#pragma omp sections " + secSpecificClauses);
ret.getReplacementString().append(" " + n.getF4().getInfo().getString() + " ");
ret.getReplacementString().append(sectionsSS.getReplacementString());
ret.getReplacementString().append("}"); 
ret.getReplacementString().append("}"); 
}
return ret;
}
@Override
public SimplificationString visit(MasterConstruct n) {
SimplificationString ret = new SimplificationString();
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
ret.getReplacementString().append("\n#pragma omp master \n");
ret.getReplacementString().append(n.getF3().accept(this).getReplacementString());
return ret;
}
@Override
public SimplificationString visit(CriticalConstruct n) {
SimplificationString ret = new SimplificationString();
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
ret.getReplacementString().append("\n#pragma omp critical " + n.getF2().getInfo().getString() + " " + "\n");
ret.getReplacementString().append(n.getF4().accept(this).getReplacementString());
return ret;
}
@Override
public SimplificationString visit(AtomicConstruct n) {
SimplificationString ret = new SimplificationString();
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
assert (n.getF4() instanceof ExpressionStatement);
ExpressionStatement expStmt = (ExpressionStatement) n.getF4();
assert (expStmt.getF0().present());
Expression exp = (Expression) expStmt.getF0().getNode();
SimplificationString retExp = exp.accept(this);
if (!retExp.hasNoTempDeclarations()) {
for (Declaration tempDecl : retExp.getTemporaryDeclarations()) {
ret.getReplacementString().append(tempDecl.toString() + " ");
}
}
if (retExp.hasPrelude() != 0) {
ret.getReplacementString().append(retExp.getPrelude());
}
if (retExp.hasPrelude() == 0) {
ret.getReplacementString().append("\n#pragma omp atomic " + n.getF2() + "\n" + n.getF4());
} else {
ret.getReplacementString()
.append("\n#pragma omp atomic " + n.getF2() + "\n" + retExp.getReplacementString() + "; ");
}
return ret;
}
@Override
public SimplificationString visit(FlushDirective n) {
SimplificationString ret = new SimplificationString();
ret.setReplacementString(new StringBuilder(INIT));
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
ret.getReplacementString().append("\n#pragma omp flush " + n.getF2() + "\n");
return ret;
}
@Override
public SimplificationString visit(OrderedConstruct n) {
SimplificationString retStmt = n.getF3().accept(this);
SimplificationString ret = new SimplificationString();
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
ret.getReplacementString().append("\n#pragma omp ordered\n" + retStmt.getReplacementString());
return ret;
}
@Override
public SimplificationString visit(BarrierDirective n) {
SimplificationString ret = new SimplificationString();
ret.setReplacementString(new StringBuilder(INIT));
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
ret.getReplacementString().append("\n#pragma omp barrier\n");
return ret;
}
@Override
public SimplificationString visit(TaskwaitDirective n) {
SimplificationString ret = new SimplificationString();
ret.setReplacementString(new StringBuilder(INIT));
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
ret.getReplacementString().append("\n#pragma omp taskwait\n");
return ret;
}
@Override
public SimplificationString visit(TaskyieldDirective n) {
SimplificationString ret = new SimplificationString();
ret.setReplacementString(new StringBuilder(INIT));
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
ret.getReplacementString().append("\n#pragma omp taskyield\n");
return ret;
}
@Override
public SimplificationString visit(ThreadPrivateDirective n) {
SimplificationString ret = new SimplificationString();
ret.setReplacementString(new StringBuilder("\n#pragma omp threadprivate(" + n.getF3() + ")\n"));
return ret;
}
@Override
public SimplificationString visit(DeclareReductionDirective n) {
SimplificationString ret = new SimplificationString();
SimplificationString combinerSS = n.getF8().accept(this);
if (Misc.isACall(combinerSS.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF8()).getDeclaration(tempName);
combinerSS.getTemporaryDeclarations().add(decl);
combinerSS.getPrelude().append(tempName + " = " + combinerSS.getReplacementString() + ";");
combinerSS.setReplacementString(new StringBuilder(" " + tempName + " "));
}
SimplificationString initClauseSS = n.getF10().accept(this);
StringBuilder preString = new StringBuilder(INIT);
for (Declaration tempDecl : combinerSS.getTemporaryDeclarations()) {
preString.append(tempDecl.getInfo().getString() + " ");
}
for (Declaration tempDecl : initClauseSS.getTemporaryDeclarations()) {
preString.append(tempDecl.getInfo().getString() + " ");
}
preString.append(combinerSS.getPrelude());
preString.append(initClauseSS.getPrelude());
if (preString.length() != 0) {
ret.setReplacementString(new StringBuilder("{" + preString));
}
ret.getReplacementString()
.append(n.getF0().getInfo().getString() + " " + " declare reduction (" + n.getF4().getInfo().getString()
+ " " + ":" + n.getF6().getInfo().getString() + " " + ":" + combinerSS.getReplacementString()
+ ")" + initClauseSS.getReplacementString() + "\n");
if (preString.length() != 0) {
ret.getReplacementString().append("}");
}
return ret;
}
@Override
public SimplificationString visit(ReductionTypeList n) {
SimplificationString ret = new SimplificationString();
ret.setReplacementString(new StringBuilder(n.getF0() + " "));
return ret;
}
@Override
public SimplificationString visit(InitializerClause n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(AssignInitializerClause n) {
SimplificationString ret = n.getF4().accept(this);
ret.setReplacementString(new StringBuilder(
n.getF0().getTokenImage() + "(" + n.getF2().getTokenImage() + "=" + ret.getReplacementString() + ")"));
return ret;
}
@Override
public SimplificationString visit(ArgumentInitializerClause n) {
SimplificationString ret = n.getF4().accept(this);
ret.setReplacementString(new StringBuilder(
n.getF0().getTokenImage() + "(" + n.getF2().getTokenImage() + "(" + ret.getReplacementString() + "))"));
return ret;
}
@Override
public SimplificationString visit(ReductionOp n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(VariableList n) {
SimplificationString ret = new SimplificationString();
ret.setReplacementString(new StringBuilder(" " + n.getF0() + n.getF1() + " "));
return ret;
}
@Override
public SimplificationString visit(LabeledStatement n) {
return n.getLabStmtF0().accept(this);
}
@Override
public SimplificationString visit(SimpleLabeledStatement n) {
SimplificationString ret = n.getF2().accept(this);
ret.setReplacementString(new StringBuilder(n.getF0().getTokenImage() + ": " + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(CaseLabeledStatement n) {
SimplificationString ret = n.getF3().accept(this);
assert (ret.hasNoTempDeclarations());
assert (ret.hasPrelude() == 0);
ret.setReplacementString(
new StringBuilder("case " + n.getF1().getInfo().getString() + " " + ": " + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(DefaultLabeledStatement n) {
SimplificationString ret = n.getF2().accept(this);
assert (ret.hasNoTempDeclarations());
assert (ret.hasPrelude() == 0);
ret.setReplacementString(new StringBuilder("default :" + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(ExpressionStatement n) {
SimplificationString ret = n.getF0().accept(this);
if (ret.hasPrelude() != 0) {
StringBuilder expString = ret.getReplacementString();
boolean needsEncapsulation = this.needsEncapsulation(n);
if (needsEncapsulation) {
ret.setReplacementString(new StringBuilder("{"));
} else {
ret.setReplacementString(new StringBuilder(INIT));
}
for (Declaration tempDecl : ret.getTemporaryDeclarations()) {
ret.getReplacementString().append(tempDecl.getInfo().getString() + " ");
}
List<Label> labels = n.getInfo().getLabelAnnotations();
for (Label label : labels) {
ret.getReplacementString().append(label.getString());
}
ret.getReplacementString().append(ret.getPrelude());
ret.getReplacementString().append(expString + ";");
if (needsEncapsulation) {
ret.getReplacementString().append("}");
}
} else {
StringBuilder expString = ret.getReplacementString();
ret.setReplacementString(new StringBuilder(INIT));
List<Label> labels = n.getInfo().getLabelAnnotations();
for (Label label : labels) {
ret.getReplacementString().append(label.getString());
}
ret.getReplacementString().append(expString);
ret.getReplacementString().append(";");
}
ret.setPrelude(new StringBuilder(INIT));
ret.getTemporaryDeclarations().clear();
return ret;
}
@Override
public SimplificationString visit(CompoundStatement n) {
SimplificationString ret = new SimplificationString();
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
boolean needsEncapsulation = this.needsEncapsulation(n);
if (needsEncapsulation) {
ret.getReplacementString().append("{");
}
for (Node element : n.getF1().getNodes()) {
boolean needsBraces = this.originalCS.contains(Misc.getCFGNodeFor(element));
if (needsBraces) {
ret.getReplacementString().append("{");
}
SimplificationString elementSS = element.accept(this);
for (Declaration tempDecl : elementSS.getTemporaryDeclarations()) {
ret.getReplacementString().append(tempDecl.getInfo().getString() + " ");
}
ret.getReplacementString().append(elementSS.getPrelude());
ret.getReplacementString().append(elementSS.getReplacementString());
if (needsBraces) {
ret.getReplacementString().append("}");
}
}
if (needsEncapsulation) {
ret.getReplacementString().append("}");
}
return ret;
}
@Override
public SimplificationString visit(CompoundStatementElement n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(SelectionStatement n) {
return n.getSelStmtF0().accept(this);
}
@Override
public SimplificationString visit(IfStatement n) {
SimplificationString ret = new SimplificationString();
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
SimplificationString expSS = n.getF2().accept(this);
if (Misc.isACall(expSS.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF2()).getDeclaration(tempName);
expSS.getTemporaryDeclarations().add(decl);
expSS.getPrelude().append(tempName + " = " + expSS.getReplacementString() + ";");
expSS.setReplacementString(new StringBuilder(" " + tempName + " "));
}
SimplificationString thenSS = n.getF4().accept(this);
SimplificationString elseSS = new SimplificationString();
if (n.getF5().present()) {
elseSS = ((NodeSequence) n.getF5().getNode()).getNodes().get(1).accept(this);
}
boolean needsEncapsulation = this.needsEncapsulation(n);
if (needsEncapsulation) {
ret.getReplacementString().append("{");
}
for (Declaration tempDecl : expSS.getTemporaryDeclarations()) {
ret.getReplacementString().append(tempDecl.getInfo().getString() + " ");
}
ret.getReplacementString().append(expSS.getPrelude());
ret.getReplacementString().append("if (" + expSS.getReplacementString() + ")");
assert (thenSS.hasNoTempDeclarations());
assert (thenSS.hasPrelude() == 0);
ret.getReplacementString().append(thenSS.getReplacementString());
if (elseSS.getReplacementString().length() != 0) {
ret.getReplacementString().append("else ");
assert (elseSS.hasNoTempDeclarations());
assert (elseSS.hasPrelude() == 0);
ret.getReplacementString().append(elseSS.getReplacementString());
}
if (needsEncapsulation) {
ret.getReplacementString().append("}");
}
return ret;
}
@Override
public SimplificationString visit(SwitchStatement n) {
SimplificationString ret = new SimplificationString();
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
SimplificationString expSS = n.getF2().accept(this);
if (Misc.isACall(expSS.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF2()).getDeclaration(tempName);
expSS.getTemporaryDeclarations().add(decl);
expSS.getPrelude().append(tempName + " = " + expSS.getReplacementString() + ";");
expSS.setReplacementString(new StringBuilder(" " + tempName + " "));
}
SimplificationString stmtSS = n.getF4().accept(this);
boolean needsEncapsulation = this.needsEncapsulation(n);
if (needsEncapsulation) {
ret.getReplacementString().append("{");
}
for (Declaration tempDecl : expSS.getTemporaryDeclarations()) {
ret.getReplacementString().append(tempDecl.getInfo().getString() + " ");
}
ret.getReplacementString().append(expSS.getPrelude());
ret.getReplacementString().append(" switch (" + expSS.getReplacementString() + ") {");
assert (stmtSS.hasNoTempDeclarations());
assert (stmtSS.hasPrelude() == 0);
ret.getReplacementString().append(stmtSS.getReplacementString() + "}");
if (needsEncapsulation) {
ret.getReplacementString().append("}");
}
return ret;
}
@Override
public SimplificationString visit(IterationStatement n) {
return n.getItStmtF0().accept(this);
}
@Override
public SimplificationString visit(WhileStatement n) {
SimplificationString ret = new SimplificationString();
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
SimplificationString expSS = n.getF2().accept(this);
if (Misc.isACall(expSS.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF2()).getDeclaration(tempName);
expSS.getTemporaryDeclarations().add(decl);
expSS.getPrelude().append(tempName + " = " + expSS.getReplacementString() + ";");
expSS.setReplacementString(new StringBuilder(" " + tempName + " "));
}
SimplificationString stmtSS = n.getF4().accept(this);
boolean needsEncapsulation = this.needsEncapsulation(n);
if (needsEncapsulation) {
ret.getReplacementString().append("{");
}
for (Declaration tempDecl : expSS.getTemporaryDeclarations()) {
ret.getReplacementString().append(tempDecl.getInfo().getString() + " ");
}
ret.getReplacementString().append(expSS.getPrelude());
ret.getReplacementString().append(" while (" + expSS.getReplacementString() + ") {");
for (Declaration tempDecl : stmtSS.getTemporaryDeclarations()) {
ret.getReplacementString().append(tempDecl.getInfo().getString() + " ");
}
ret.getReplacementString().append(stmtSS.getPrelude());
StringBuilder central = stmtSS.getReplacementString();
if (central.charAt(0) == '{') {
central = new StringBuilder(central.substring(1, central.length() - 1));
}
ret.getReplacementString().append(central);
ret.getReplacementString().append(expSS.getPrelude() + "}");
if (needsEncapsulation) {
ret.getReplacementString().append("}");
}
return ret;
}
@Override
public SimplificationString visit(DoStatement n) {
SimplificationString ret = new SimplificationString();
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
SimplificationString stmtSS = n.getF1().accept(this);
SimplificationString expSS = n.getF4().accept(this);
if (Misc.isACall(expSS.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF4()).getDeclaration(tempName);
expSS.getTemporaryDeclarations().add(decl);
expSS.getPrelude().append(tempName + " = " + expSS.getReplacementString() + ";");
expSS.setReplacementString(new StringBuilder(" " + tempName + " "));
}
boolean needsEncapsulation = this.needsEncapsulation(n);
if (needsEncapsulation) {
ret.getReplacementString().append("{");
}
for (Declaration tempDecl : expSS.getTemporaryDeclarations()) {
ret.getReplacementString().append(tempDecl.getInfo().getString() + " ");
}
ret.getReplacementString().append("do {");
assert (stmtSS.hasNoTempDeclarations());
ret.getReplacementString().append(stmtSS.getPrelude());
ret.getReplacementString().append(stmtSS.getReplacementString());
ret.getReplacementString().append(expSS.getPrelude());
ret.getReplacementString().append("} while (" + expSS.getReplacementString() + ");");
if (needsEncapsulation) {
ret.getReplacementString().append("}");
}
return ret;
}
@Override
public SimplificationString visit(ForStatement n) {
SimplificationString ret = new SimplificationString();
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
SimplificationString e1SS = n.getF2().accept(this);
if (n.getF2().present()) {
if (Misc.isACall(e1SS.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType((Expression) n.getF2().getNode()).getDeclaration(tempName);
e1SS.getTemporaryDeclarations().add(decl);
e1SS.getPrelude().append(tempName + " = " + e1SS.getReplacementString() + ";");
e1SS.setReplacementString(new StringBuilder(" " + tempName + " "));
}
}
SimplificationString e2SS = n.getF4().accept(this);
if (n.getF4().present()) {
if (Misc.isACall(e2SS.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType((Expression) n.getF4().getNode()).getDeclaration(tempName);
e2SS.getTemporaryDeclarations().add(decl);
e2SS.getPrelude().append(tempName + " = " + e2SS.getReplacementString() + ";");
e2SS.setReplacementString(new StringBuilder(" " + tempName + " "));
}
}
SimplificationString e3SS = n.getF6().accept(this);
if (n.getF6().present()) {
if (Misc.isACall(e3SS.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType((Expression) n.getF6().getNode()).getDeclaration(tempName);
e3SS.getTemporaryDeclarations().add(decl);
e3SS.getPrelude().append(tempName + " = " + e3SS.getReplacementString() + ";");
e3SS.setReplacementString(new StringBuilder(" " + tempName + " "));
}
}
SimplificationString stmtSS = n.getF8().accept(this);
boolean needsEncapsulation = this.needsEncapsulation(n);
if (needsEncapsulation) {
ret.getReplacementString().append("{");
}
for (Declaration tempDecl : e1SS.getTemporaryDeclarations()) {
ret.getReplacementString().append(tempDecl.getInfo().getString() + " ");
}
for (Declaration tempDecl : e2SS.getTemporaryDeclarations()) {
ret.getReplacementString().append(tempDecl.getInfo().getString() + " ");
}
ret.getReplacementString().append(e1SS.getPrelude());
if (e2SS.hasPrelude() == 0) {
ret.getReplacementString().append("for(" + e1SS.getReplacementString() + ";");
ret.getReplacementString().append(e2SS.getReplacementString() + ";" + e3SS.getReplacementString() + ")");
if (!e3SS.hasNoTempDeclarations() || e3SS.hasPrelude() != 0) {
ret.getReplacementString().append("{");
for (Declaration tempDeclaration : e3SS.getTemporaryDeclarations()) {
ret.getReplacementString().append(tempDeclaration.getInfo().getString() + " ");
}
}
assert (stmtSS.hasPrelude() == 0);
assert (stmtSS.hasNoTempDeclarations());
ret.getReplacementString().append(stmtSS.getReplacementString());
if (!e3SS.hasNoTempDeclarations() || e3SS.hasPrelude() != 0) {
ret.getReplacementString().append(e3SS.getPrelude());
ret.getReplacementString().append("}");
}
} else {
ret.getReplacementString().append(e1SS.getReplacementString() + ";");
ret.getReplacementString().append(e2SS.getPrelude());
ret.getReplacementString().append("for (;" + e2SS.getReplacementString() + ";)");
if (!e3SS.hasNoTempDeclarations() || e3SS.hasPrelude() != 0 || e2SS.hasPrelude() != 0) {
ret.getReplacementString().append("{");
for (Declaration tempDeclaration : e3SS.getTemporaryDeclarations()) {
ret.getReplacementString().append(tempDeclaration.getInfo().getString() + " ");
}
}
assert (stmtSS.hasPrelude() == 0);
assert (stmtSS.hasNoTempDeclarations());
ret.getReplacementString().append(stmtSS.getReplacementString());
ret.getReplacementString().append(e3SS.getPrelude());
ret.getReplacementString().append(e3SS.getReplacementString() + ";");
if (!e3SS.hasNoTempDeclarations() || e3SS.hasPrelude() != 0 || e2SS.hasPrelude() != 0) {
ret.getReplacementString().append(e2SS.getPrelude());
ret.getReplacementString().append("}");
}
}
if (needsEncapsulation) {
ret.getReplacementString().append("}");
}
return ret;
}
@Override
public SimplificationString visit(JumpStatement n) {
return n.getJumpStmtF0().accept(this);
}
@Override
public SimplificationString visit(GotoStatement n) {
SimplificationString ret = new SimplificationString();
ret.setReplacementString(new StringBuilder(INIT));
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
ret.getReplacementString().append("goto " + n.getF1().getTokenImage() + "; ");
return ret;
}
@Override
public SimplificationString visit(ContinueStatement n) {
SimplificationString ret = new SimplificationString();
ret.setReplacementString(new StringBuilder(INIT));
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
ret.getReplacementString().append("continue; ");
return ret;
}
@Override
public SimplificationString visit(BreakStatement n) {
SimplificationString ret = new SimplificationString();
ret.setReplacementString(new StringBuilder(INIT));
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
ret.getReplacementString().append("break; ");
return ret;
}
@Override
public SimplificationString visit(ReturnStatement n) {
SimplificationString ret = new SimplificationString();
SimplificationString expSS = n.getF1().accept(this);
if (n.getF1().present()) {
if (Misc.isACall(expSS.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType((Expression) n.getF1().getNode()).getDeclaration(tempName);
expSS.getTemporaryDeclarations().add(decl);
expSS.getPrelude().append(tempName + " = " + expSS.getReplacementString() + ";");
expSS.setReplacementString(new StringBuilder(" " + tempName + " "));
}
}
if (expSS.hasPrelude() == 0) {
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
ret.getReplacementString().append("return " + expSS.getReplacementString() + ";");
} else {
boolean needsEncapsulation = this.needsEncapsulation(n);
if (needsEncapsulation) {
ret.setReplacementString(new StringBuilder("{"));
} else {
ret.setReplacementString(new StringBuilder(INIT));
}
for (Declaration tempDecl : expSS.getTemporaryDeclarations()) {
ret.getReplacementString().append(tempDecl.getInfo().getString() + " ");
}
for (Label label : n.getInfo().getLabelAnnotations()) {
ret.getReplacementString().append(label.getString());
}
ret.getReplacementString().append(expSS.getPrelude() + "return " + expSS.getReplacementString() + ";");
if (needsEncapsulation) {
ret.getReplacementString().append("}");
}
}
return ret;
}
@Override
public SimplificationString visit(Expression n) {
SimplificationString ret = n.getExpF0().accept(this);
if (!n.getExpF1().present()) {
return ret;
}
ret.getPrelude().append(ret.getReplacementString() + ";");
SimplificationString elementSS = null;
for (Node nodeSeq : n.getExpF1().getNodes()) {
AssignmentExpression assignExp = (AssignmentExpression) ((NodeSequence) nodeSeq).getNodes().get(1);
elementSS = assignExp.accept(this);
ret.getTemporaryDeclarations().addAll(elementSS.getTemporaryDeclarations());
ret.getPrelude().append(elementSS.getPrelude());
if (nodeSeq != n.getExpF1().getNodes().get(n.getExpF1().getNodes().size() - 1)) {
ret.getPrelude().append(elementSS.getReplacementString() + ";");
}
}
ret.setReplacementString(elementSS.getReplacementString());
return ret;
}
@Override
public SimplificationString visit(AssignmentExpression n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(NonConditionalExpression n) {
SimplificationString rhsSS = n.getF2().accept(this);
StringBuilder opSS = new StringBuilder(n.getF1().getInfo().getString() + " ");
if (!opSS.toString().equals("= ")) {
if (Misc.isACall(rhsSS.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF2()).getDeclaration(tempName);
rhsSS.getTemporaryDeclarations().add(decl);
rhsSS.getPrelude().append(tempName + " = " + rhsSS.getReplacementString() + ";");
rhsSS.setReplacementString(new StringBuilder(" " + tempName + " "));
}
}
SimplificationString lhsSS = n.getF0().accept(this);
rhsSS.getPrelude().append(lhsSS.getPrelude());
rhsSS.getTemporaryDeclarations().addAll(lhsSS.getTemporaryDeclarations());
rhsSS.setReplacementString(
new StringBuilder(lhsSS.getReplacementString() + " " + opSS + " " + rhsSS.getReplacementString()));
return rhsSS;
}
@Override
public SimplificationString visit(AssignmentOperator n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(ConditionalExpression n) {
SimplificationString ret = n.getF0().accept(this);
if (!n.getF1().present()) {
return ret;
}
Expression e1 = n.getF0();
Expression e2 = (Expression) (((NodeSequence) n.getF1().getNode()).getNodes().get(1));
Expression e3 = (Expression) (((NodeSequence) n.getF1().getNode()).getNodes().get(3));
Type typeOfE1 = Type.getType(e1);
Declaration t1Decl = typeOfE1.getDeclaration(Builder.getNewTempName());
Type typeOfE2 = Type.getType(n);
Declaration t2Decl = typeOfE2.getDeclaration(Builder.getNewTempName());
ret.getTemporaryDeclarations().add(t1Decl);
ret.getTemporaryDeclarations().add(t2Decl);
StringBuilder t1Str = new StringBuilder(t1Decl.getInfo().getIDNameList().get(0));
StringBuilder t2Str = new StringBuilder(t2Decl.getInfo().getIDNameList().get(0));
ret.getPrelude().append(t1Str + " = " + ret.getReplacementString() + ";");
ret.getPrelude().append("if (" + t1Str + ") {");
SimplificationString tempStr = e2.accept(this);
ret.getTemporaryDeclarations().addAll(tempStr.getTemporaryDeclarations());
ret.getPrelude().append(tempStr.getPrelude());
ret.getPrelude().append(t2Str + " = " + tempStr.getReplacementString() + ";");
ret.getPrelude().append("} else {");
tempStr = e3.accept(this);
ret.getTemporaryDeclarations().addAll(tempStr.getTemporaryDeclarations());
ret.getPrelude().append(tempStr.getPrelude());
ret.getPrelude().append(t2Str + " = " + tempStr.getReplacementString() + ";");
ret.getPrelude().append("}");
ret.setReplacementString(t2Str);
return ret;
}
@Override
public SimplificationString visit(ConstantExpression n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(LogicalORExpression n) {
SimplificationString ret = n.getF0().accept(this);
if (!n.getF1().present()) {
return n.getF0().accept(this);
}
Type typeOfExp = Type.getType(n.getF0());
Declaration tempDeclaration = typeOfExp.getDeclaration(Builder.getNewTempName());
ret.getTemporaryDeclarations().add(tempDeclaration);
StringBuilder tempName = new StringBuilder(tempDeclaration.getInfo().getIDNameList().get(0));
ret.getPrelude().append(" " + tempName + " = " + ret.getReplacementString() + ";");
ret.getPrelude().append("if (!" + tempName + ") {");
SimplificationString tempSS = ((NodeSequence) n.getF1().getNode()).getNodes().get(1).accept(this);
ret.getPrelude().append(tempSS.getPrelude());
ret.getPrelude().append(tempName + " = " + tempSS.getReplacementString() + ";");
ret.getPrelude().append("}");
ret.getTemporaryDeclarations().addAll(tempSS.getTemporaryDeclarations());
ret.setReplacementString(tempName);
return ret;
}
@Override
public SimplificationString visit(LogicalANDExpression n) {
SimplificationString ret = n.getF0().accept(this);
if (!n.getF1().present()) {
return ret;
}
Type typeOfExp = Type.getType(n.getF0());
Declaration tempDeclaration = typeOfExp.getDeclaration(Builder.getNewTempName());
ret.getTemporaryDeclarations().add(tempDeclaration);
StringBuilder tempName = new StringBuilder(tempDeclaration.getInfo().getIDNameList().get(0));
ret.getPrelude().append(" " + tempName + " = " + ret.getReplacementString() + ";");
ret.getPrelude().append("if (" + tempName + ") {");
SimplificationString tempSS = ((NodeSequence) n.getF1().getNode()).getNodes().get(1).accept(this);
ret.getPrelude().append(tempSS.getPrelude());
ret.getPrelude().append(tempName + " = " + tempSS.getReplacementString() + ";");
ret.getPrelude().append("}");
ret.getTemporaryDeclarations().addAll(tempSS.getTemporaryDeclarations());
ret.setReplacementString(tempName);
return ret;
}
@Override
public SimplificationString visit(InclusiveORExpression n) {
SimplificationString ret = n.getF0().accept(this);
if (n.getF1().present()) {
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF0()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
SimplificationString childSS = ((NodeSequence) n.getF1().getNode()).getNodes().get(1).accept(this);
if (Misc.isACall(childSS.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type
.getType(((InclusiveORExpression) ((NodeSequence) n.getF1().getNode()).getNodes().get(1)))
.getDeclaration(tempName);
childSS.getTemporaryDeclarations().add(decl);
childSS.getPrelude().append(tempName + " = " + childSS.getReplacementString() + ";");
childSS.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.getPrelude().append(childSS.getPrelude());
ret.getTemporaryDeclarations().addAll(childSS.getTemporaryDeclarations());
ret.getReplacementString().append(" | " + childSS.getReplacementString());
}
return ret;
}
@Override
public SimplificationString visit(ExclusiveORExpression n) {
SimplificationString ret = n.getF0().accept(this);
if (n.getF1().present()) {
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF0()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
SimplificationString childSS = ((NodeSequence) n.getF1().getNode()).getNodes().get(1).accept(this);
if (Misc.isACall(childSS.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type
.getType(((ExclusiveORExpression) ((NodeSequence) n.getF1().getNode()).getNodes().get(1)))
.getDeclaration(tempName);
childSS.getTemporaryDeclarations().add(decl);
childSS.getPrelude().append(tempName + " = " + childSS.getReplacementString() + ";");
childSS.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.getPrelude().append(childSS.getPrelude());
ret.getTemporaryDeclarations().addAll(childSS.getTemporaryDeclarations());
ret.getReplacementString().append(" ^ " + childSS.getReplacementString());
}
return ret;
}
@Override
public SimplificationString visit(ANDExpression n) {
SimplificationString ret = n.getF0().accept(this);
if (n.getF1().present()) {
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF0()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
SimplificationString childSS = ((NodeSequence) n.getF1().getNode()).getNodes().get(1).accept(this);
if (Misc.isACall(childSS.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type
.getType(((ANDExpression) ((NodeSequence) n.getF1().getNode()).getNodes().get(1)))
.getDeclaration(tempName);
childSS.getTemporaryDeclarations().add(decl);
childSS.getPrelude().append(tempName + " = " + childSS.getReplacementString() + ";");
childSS.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.getPrelude().append(childSS.getPrelude());
ret.getTemporaryDeclarations().addAll(childSS.getTemporaryDeclarations());
ret.getReplacementString().append(" & " + childSS.getReplacementString());
}
return ret;
}
@Override
public SimplificationString visit(EqualityExpression n) {
SimplificationString ret = n.getF0().accept(this);
if (n.getF1().present()) {
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF0()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
}
SimplificationString childSS = n.getF1().accept(this);
ret.getPrelude().append(childSS.getPrelude());
ret.getTemporaryDeclarations().addAll(childSS.getTemporaryDeclarations());
ret.getReplacementString().append(childSS.getReplacementString());
return ret;
}
@Override
public SimplificationString visit(EqualOptionalExpression n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(EqualExpression n) {
SimplificationString ret = n.getF1().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF1()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(new StringBuilder("==" + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(NonEqualExpression n) {
SimplificationString ret = n.getF1().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF1()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(new StringBuilder("!=" + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(RelationalExpression n) {
SimplificationString ret = n.getRelExpF0().accept(this);
if (n.getRelExpF1().present()) {
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getRelExpF0()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
}
SimplificationString childSS = n.getRelExpF1().accept(this);
ret.getPrelude().append(childSS.getPrelude());
ret.getTemporaryDeclarations().addAll(childSS.getTemporaryDeclarations());
ret.getReplacementString().append(childSS.getReplacementString());
return ret;
}
@Override
public SimplificationString visit(RelationalOptionalExpression n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(RelationalLTExpression n) {
SimplificationString ret = n.getF1().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF1()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(new StringBuilder("<" + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(RelationalGTExpression n) {
SimplificationString ret = n.getF1().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF1()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(new StringBuilder(">" + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(RelationalLEExpression n) {
SimplificationString ret = n.getF1().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF1()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(new StringBuilder("<=" + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(RelationalGEExpression n) {
SimplificationString ret = n.getF1().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF1()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(new StringBuilder(">=" + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(ShiftExpression n) {
SimplificationString ret = n.getF0().accept(this);
if (n.getF1().present()) {
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF0()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
}
SimplificationString childSS = n.getF1().accept(this);
ret.getPrelude().append(childSS.getPrelude());
ret.getTemporaryDeclarations().addAll(childSS.getTemporaryDeclarations());
ret.getReplacementString().append(childSS.getReplacementString());
return ret;
}
@Override
public SimplificationString visit(ShiftOptionalExpression n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(ShiftLeftExpression n) {
SimplificationString ret = n.getF1().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF1()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(new StringBuilder(">>" + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(ShiftRightExpression n) {
SimplificationString ret = n.getF1().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF1()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(new StringBuilder("<<" + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(AdditiveExpression n) {
SimplificationString ret = n.getF0().accept(this);
if (n.getF1().present()) {
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF0()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
}
SimplificationString childSS = n.getF1().accept(this);
ret.getPrelude().append(childSS.getPrelude());
ret.getTemporaryDeclarations().addAll(childSS.getTemporaryDeclarations());
ret.getReplacementString().append(childSS.getReplacementString());
return ret;
}
@Override
public SimplificationString visit(AdditiveOptionalExpression n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(AdditivePlusExpression n) {
SimplificationString ret = n.getF1().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF1()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(new StringBuilder("+" + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(AdditiveMinusExpression n) {
SimplificationString ret = n.getF1().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF1()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(new StringBuilder("-" + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(MultiplicativeExpression n) {
SimplificationString ret = n.getF0().accept(this);
if (n.getF1().present()) {
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF0()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
}
SimplificationString childSS = n.getF1().accept(this);
ret.getPrelude().append(childSS.getPrelude());
ret.getTemporaryDeclarations().addAll(childSS.getTemporaryDeclarations());
ret.getReplacementString().append(childSS.getReplacementString());
return ret;
}
@Override
public SimplificationString visit(MultiplicativeOptionalExpression n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(MultiplicativeMultiExpression n) {
SimplificationString ret = n.getF1().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF1()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(new StringBuilder("*" + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(MultiplicativeDivExpression n) {
SimplificationString ret = n.getF1().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF1()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(new StringBuilder("/" + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(MultiplicativeModExpression n) {
SimplificationString ret = n.getF1().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF1()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(new StringBuilder("%" + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(CastExpression n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(CastExpressionTyped n) {
SimplificationString ret = n.getF3().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF3()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(
new StringBuilder("(" + n.getF1().getInfo().getString() + " " + ")" + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(UnaryExpression n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(UnaryExpressionPreIncrement n) {
SimplificationString ret = n.getF1().accept(this);
ret.setReplacementString(new StringBuilder(" ++" + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(UnaryExpressionPreDecrement n) {
SimplificationString ret = n.getF1().accept(this);
ret.setReplacementString(new StringBuilder(" --" + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(UnaryCastExpression n) {
SimplificationString ret = n.getF1().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF1()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(n.getF0().toString() + " "));
ret.getReplacementString().append(tempName + " ");
} else {
ret.setReplacementString(
new StringBuilder(n.getF0().getInfo().getString() + " " + ret.getReplacementString()));
}
return ret;
}
@Override
public SimplificationString visit(UnarySizeofExpression n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(SizeofUnaryExpression n) {
SimplificationString ret = new SimplificationString();
ret.setReplacementString(new StringBuilder("sizeof " + n.getF1() + " "));
return ret;
}
@Override
public SimplificationString visit(SizeofTypeName n) {
SimplificationString ret = new SimplificationString();
ret.setReplacementString(new StringBuilder("sizeof(" + n.getF2() + ") "));
return ret;
}
@Override
public SimplificationString visit(PostfixExpression n) {
SimplificationString ret = n.getF0().accept(this);
List<Node> postFixOpList = n.getF1().getF0().getNodes();
for (Node postFixOpNode : postFixOpList) {
Node postFixOp = ((APostfixOperation) postFixOpNode).getF0().getChoice();
if (postFixOp instanceof ArgumentList) {
Expression newE1 = FrontEnd.parseAlone(ret.getReplacementString().toString(), Expression.class);
if (!Misc.isSimplePrimaryExpression(newE1)) {
Type e1Type = ExpressionTypeGetter.getHalfPostfixExpressionType(n,
(APostfixOperation) postFixOpNode);
String newTempName = Builder.getNewTempName();
Declaration e1Decl = e1Type.getDeclaration(newTempName);
ret.getTemporaryDeclarations().add(e1Decl);
ret.getPrelude().append(newTempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + newTempName + " "));
}
}
SimplificationString elementSS = postFixOp.accept(this);
ret.getPrelude().append(elementSS.getPrelude());
ret.getTemporaryDeclarations().addAll(elementSS.getTemporaryDeclarations());
ret.getReplacementString().append(elementSS.getReplacementString());
if (postFixOpNode != postFixOpList.get(postFixOpList.size() - 1)) {
if (Misc.isACall(ret.getReplacementString())) {
APostfixOperation nextOperation = (APostfixOperation) postFixOpList
.get(postFixOpList.indexOf(postFixOpNode) + 1);
Type thisType = ExpressionTypeGetter.getHalfPostfixExpressionType(n, nextOperation);
String tempName = Builder.getNewTempName();
Declaration decl = thisType.getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
}
}
return ret;
}
@Override
public SimplificationString visit(PostfixOperationsList n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(APostfixOperation n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(PlusPlus n) {
SimplificationString ret = new SimplificationString();
ret.setReplacementString(new StringBuilder("++ "));
return ret;
}
@Override
public SimplificationString visit(MinusMinus n) {
SimplificationString ret = new SimplificationString();
ret.setReplacementString(new StringBuilder("-- "));
return ret;
}
@Override
public SimplificationString visit(BracketExpression n) {
SimplificationString ret = n.getF1().accept(this);
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF1()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
ret.setReplacementString(new StringBuilder("[" + ret.getReplacementString() + "]"));
return ret;
}
@Override
public SimplificationString visit(ArgumentList n) {
SimplificationString ret = n.getF1().accept(this);
ret.setReplacementString(new StringBuilder("(" + ret.getReplacementString() + ")"));
return ret;
}
@Override
public SimplificationString visit(DotId n) {
SimplificationString ret = n.getF1().accept(this);
ret.setReplacementString(new StringBuilder("." + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(ArrowId n) {
SimplificationString ret = n.getF1().accept(this);
ret.setReplacementString(new StringBuilder("->" + ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(PrimaryExpression n) {
return n.getF0().accept(this);
}
@Override
public SimplificationString visit(ExpressionClosed n) {
SimplificationString ret = n.getF1().accept(this);
Expression newExp = FrontEnd.parseAlone(ret.getReplacementString().toString(), Expression.class);
if (Misc.isSimplePrimaryExpression(newExp)) {
ret.setReplacementString(new StringBuilder(" " + ret.getReplacementString() + " "));
} else {
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = Type.getType(n.getF1()).getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
newExp = FrontEnd.parseAlone(ret.getReplacementString().toString(), Expression.class);
if (Misc.isSimplePrimaryExpression(newExp)) {
ret.setReplacementString(new StringBuilder(" " + ret.getReplacementString() + " "));
} else {
ret.setReplacementString(new StringBuilder("(" + ret.getReplacementString() + ")"));
}
}
return ret;
}
@Override
public SimplificationString visit(ExpressionList n) {
SimplificationString ret = new SimplificationString();
List<AssignmentExpression> reverseList = new ArrayList<>();
reverseList.add(n.getF0());
for (Node listElem : n.getF1().getNodes()) {
AssignmentExpression assignmentExp = (AssignmentExpression) ((NodeSequence) listElem).getNodes().get(1);
reverseList.add(0, assignmentExp);
}
StringBuilder[] idNameList = new StringBuilder[reverseList.size()];
int i = 0;
for (AssignmentExpression assignExp : reverseList) {
SimplificationString childSS = assignExp.accept(this);
ret.getTemporaryDeclarations().addAll(childSS.getTemporaryDeclarations());
ret.getPrelude().append(childSS.getPrelude());
AssignmentExpression newArguExp = FrontEnd.parseAlone(childSS.getReplacementString().toString(),
AssignmentExpression.class);
StringBuilder tempName;
if (!Misc.isSimplePrimaryExpression(newArguExp)) {
Type argumentType = Type.getType(assignExp);
Declaration declaration = argumentType.getDeclaration(Builder.getNewTempName());
ret.getTemporaryDeclarations().add(declaration);
tempName = new StringBuilder(declaration.getInfo().getIDNameList().get(0));
StringBuilder assignStr = childSS.getReplacementString();
ret.getPrelude().append(tempName + " = " + assignStr + ";");
} else {
tempName = new StringBuilder(newArguExp.getInfo().getString() + " ");
}
idNameList[i++] = tempName;
}
for (i = 0; i < idNameList.length - 1; i++) {
ret.setReplacementString(new StringBuilder(", " + idNameList[i] + ret.getReplacementString()));
}
ret.setReplacementString(new StringBuilder(idNameList[i]).append(ret.getReplacementString()));
return ret;
}
@Override
public SimplificationString visit(Constant n) {
SimplificationString ret = new SimplificationString();
ret.setReplacementString(new StringBuilder(" " + n.getF0() + " "));
return ret;
}
@Override
public SimplificationString visit(CallStatement n) {
SimplificationString ret = new SimplificationString();
ret.getReplacementString().append(" " + n.toString() + " ");
return ret;
}
@Override
public SimplificationString visit(PreCallNode n) {
assert (false);
return null;
}
@Override
public SimplificationString visit(PostCallNode n) {
assert (false);
return null;
}
@Deprecated
public void collapseCalls(SimplificationString ret, Type t1) {
if (Misc.isACall(ret.getReplacementString())) {
String tempName = Builder.getNewTempName();
Declaration decl = t1.getDeclaration(tempName);
ret.getTemporaryDeclarations().add(decl);
ret.getPrelude().append(tempName + " = " + ret.getReplacementString() + ";");
ret.setReplacementString(new StringBuilder(" " + tempName + " "));
}
}
public boolean needsEncapsulation(Statement s) {
s = (Statement) Misc.getCFGNodeFor(s);
if (s.getInfo().hasLabelAnnotations()) {
return true;
}
Statement stmt = Misc.getEnclosingNode(s, Statement.class);
if (stmt == null) {
return true;
}
if (stmt.getParent() instanceof NodeChoice) {
NodeChoice nodeChoice = (NodeChoice) stmt.getParent();
if (nodeChoice.getParent() instanceof CompoundStatementElement) {
return false;
}
}
return true;
}
}
