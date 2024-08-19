package imop.parser;
public interface CParserConstants {
int EOF = 0;
int INTEGER_LITERAL = 8;
int DECIMAL_LITERAL = 9;
int HEX_LITERAL = 10;
int OCTAL_LITERAL = 11;
int FLOATING_POINT_LITERAL = 12;
int EXPONENT = 13;
int CHARACTER_LITERAL = 14;
int STRING_LITERAL = 15;
int RESTRICT = 23;
int CONTINUE = 24;
int VOLATILE = 25;
int REGISTER = 26;
int CCONST = 27;
int INLINE = 28;
int CINLINED = 29;
int CINLINED2 = 30;
int CSIGNED = 31;
int CSIGNED2 = 32;
int UNSIGNED = 33;
int TYPEDEF = 34;
int DFLT = 35;
int DOUBLE = 36;
int SWITCH = 37;
int RETURN = 38;
int EXTERN = 39;
int STRUCT = 40;
int STATIC = 41;
int SIGNED = 42;
int WHILE = 43;
int BREAK = 44;
int UNION = 45;
int CONST = 46;
int FLOAT = 47;
int SHORT = 48;
int ELSE = 49;
int CASE = 50;
int LONG = 51;
int ENUM = 52;
int AUTO = 53;
int VOID = 54;
int CHAR = 55;
int GOTO = 56;
int FOR = 57;
int INT = 58;
int IF = 59;
int DO = 60;
int SIZEOF = 61;
int EXTENSION = 62;
int CATOMIC = 63;
int COMPLEX = 64;
int ELLIPSIS = 65;
int OP_SLASS = 66;
int OP_SRASS = 67;
int OP_EQ = 68;
int OP_AND = 69;
int OP_OR = 70;
int OP_MULASS = 71;
int OP_DIVASS = 72;
int OP_MODASS = 73;
int OP_ADDASS = 74;
int OP_SUBASS = 75;
int OP_ANDASS = 76;
int OP_XORASS = 77;
int OP_ORASS = 78;
int OP_SL = 79;
int OP_SR = 80;
int OP_NEQ = 81;
int OP_GE = 82;
int OP_LE = 83;
int OP_DEREF = 84;
int OP_INCR = 85;
int OP_DECR = 86;
int OP_GT = 87;
int OP_LT = 88;
int OP_ADD = 89;
int OP_SUB = 90;
int OP_MUL = 91;
int OP_DIV = 92;
int OP_MOD = 93;
int OP_ASS = 94;
int OP_BITAND = 95;
int OP_BITOR = 96;
int OP_BITXOR = 97;
int OP_NOT = 98;
int OP_BITNOT = 99;
int COLON = 100;
int SEMICOLON = 101;
int QUESTION = 102;
int DOT = 103;
int LEFTPAREN = 104;
int RIGHTPAREN = 105;
int LEFTBRACKET = 106;
int RIGHTBRACKET = 107;
int LEFTBRACE = 108;
int RIGHTBRACE = 109;
int COMMA = 110;
int CROSSBAR = 111;
int UNKNOWN_CPP = 112;
int PRAGMA = 113;
int OMP_NL = 134;
int OMP_CR = 135;
int OMP = 140;
int PARALLEL = 142;
int SECTIONS = 143;
int SECTION = 144;
int SINGLE = 145;
int ORDERED = 146;
int MASTER = 147;
int CRITICAL = 148;
int ATOMIC = 149;
int BARRIER = 150;
int FLUSH = 151;
int NOWAIT = 152;
int SCHEDULE = 153;
int DYNAMIC = 154;
int GUIDED = 155;
int RUNTIME = 156;
int NONE = 157;
int REDUCTION = 158;
int PRIVATE = 159;
int FIRSTPRIVATE = 160;
int LASTPRIVATE = 161;
int COPYPRIVATE = 162;
int SHARED = 163;
int COPYIN = 164;
int THREADPRIVATE = 165;
int NUM_THREADS = 166;
int COLLAPSE = 167;
int READ = 168;
int WRITE = 169;
int UPDATE = 170;
int CAPTURE = 171;
int TASK = 172;
int TASKWAIT = 173;
int DECLARE = 174;
int TASKYIELD = 175;
int UNTIED = 176;
int MERGEABLE = 177;
int INITIALIZER = 178;
int FINAL = 179;
int IDENTIFIER = 180;
int LETTER = 181;
int DIGIT = 182;
int DEFAULT = 0;
int AfterCrossbar = 1;
int Pragma = 2;
int Omp = 3;
int AfterAttrib = 4;
int Cpp = 5;
String[] tokenImage = { "<EOF>", "\" \"", "\"\\t\"", "\"\\n\"", "\"\\r\"", "\"\\f\"", "<token of kind 6>",
"<token of kind 7>", "<INTEGER_LITERAL>", "<DECIMAL_LITERAL>", "<HEX_LITERAL>", "<OCTAL_LITERAL>",
"<FLOATING_POINT_LITERAL>", "<EXPONENT>", "<CHARACTER_LITERAL>", "<STRING_LITERAL>", "\"__attribute__\"",
"\"__asm\"", "\"__asm__\"", "\"asm\"", "\"(\"", "\")\"", "<token of kind 22>", "<RESTRICT>", "\"continue\"",
"\"volatile\"", "\"register\"", "\"__const\"", "\"inline\"", "\"__inline\"", "\"__inline__\"",
"\"__signed\"", "\"__signed__\"", "\"unsigned\"", "\"typedef\"", "\"default\"", "\"double\"", "\"switch\"",
"\"return\"", "\"extern\"", "\"struct\"", "\"static\"", "\"signed\"", "\"while\"", "\"break\"", "\"union\"",
"\"const\"", "\"float\"", "\"short\"", "\"else\"", "\"case\"", "\"long\"", "\"enum\"", "\"auto\"",
"\"void\"", "\"char\"", "\"goto\"", "\"for\"", "\"int\"", "\"if\"", "\"do\"", "\"sizeof\"",
"\"__extension__\"", "\"_Atomic\"", "\"_Complex\"", "\"...\"", "\"<<=\"", "\">>=\"", "\"==\"", "\"&&\"",
"\"||\"", "\"*=\"", "\"/=\"", "\"%=\"", "\"+=\"", "\"-=\"", "\"&=\"", "\"^=\"", "\"|=\"", "\"<<\"",
"\">>\"", "\"!=\"", "\">=\"", "\"<=\"", "\"->\"", "\"++\"", "\"--\"", "\">\"", "\"<\"", "\"+\"", "\"-\"",
"\"*\"", "\"/\"", "\"%\"", "\"=\"", "\"&\"", "\"|\"", "\"^\"", "\"!\"", "\"~\"", "\":\"", "\";\"", "\"?\"",
"\".\"", "\"(\"", "\")\"", "\"[\"", "\"]\"", "\"{\"", "\"}\"", "\",\"", "\"#\"", "<UNKNOWN_CPP>",
"\"pragma\"", "\"include\"", "\"import\"", "\"define\"", "\"ifndef\"", "\"ident\"", "\"undef\"",
"\"ifdef\"", "\"endif\"", "\"line\"", "\"else\"", "\"if\"", "\"elif\"", "<token of kind 126>",
"<token of kind 127>", "\" \"", "\"\\t\"", "<token of kind 130>", "<token of kind 131>", "\"\\n\"",
"\"\\r\"", "\"\\n\"", "\"\\r\"", "\" \"", "\"\\t\"", "<token of kind 138>", "<token of kind 139>",
"\"omp\"", "<token of kind 141>", "\"parallel\"", "\"sections\"", "\"section\"", "\"single\"",
"\"ordered\"", "\"master\"", "\"critical\"", "\"atomic\"", "\"barrier\"", "\"flush\"", "\"nowait\"",
"\"schedule\"", "\"dynamic\"", "\"guided\"", "\"runtime\"", "\"none\"", "\"reduction\"", "\"private\"",
"\"firstprivate\"", "\"lastprivate\"", "\"copyprivate\"", "\"shared\"", "\"copyin\"", "\"threadprivate\"",
"\"num_threads\"", "\"collapse\"", "\"read\"", "\"write\"", "\"update\"", "\"capture\"", "\"task\"",
"\"taskwait\"", "\"declare\"", "\"taskyield\"", "\"untied\"", "\"mergeable\"", "\"initializer\"",
"\"final\"", "<IDENTIFIER>", "<LETTER>", "<DIGIT>", };
}
