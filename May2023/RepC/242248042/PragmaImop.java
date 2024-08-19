package imop.ast.annotation;
import imop.ast.node.external.*;
public class PragmaImop extends Annotation {
private String pragmaString;
private Node annotatedNode;
public PragmaImop(String pragmaString) {
this.pragmaString = pragmaString;
}
public PragmaImop(String pragmaString, Node annotatedNode) {
assert (annotatedNode != null);
this.pragmaString = pragmaString;
this.annotatedNode = annotatedNode;
}
public String getPragmaString() {
return pragmaString;
}
public Node getAnnotatedNode() {
return annotatedNode;
}
@Override
public String toString() {
return "\n#pragma imop " + this.pragmaString + "\n";
}
}
