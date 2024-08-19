
#ifndef LLVM_CLANG_BASIC_TARGETINFO_H
#define LLVM_CLANG_BASIC_TARGETINFO_H

#include "clang/Basic/AddressSpaces.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/TargetCXXABI.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/VersionTuple.h"
#include <cassert>
#include <string>
#include <vector>

namespace llvm {
struct fltSemantics;
}

namespace clang {
class DiagnosticsEngine;
class LangOptions;
class CodeGenOptions;
class MacroBuilder;
class QualType;
class SourceLocation;
class SourceManager;

namespace Builtin { struct Info; }

class TargetInfo : public RefCountedBase<TargetInfo> {
std::shared_ptr<TargetOptions> TargetOpts;
llvm::Triple Triple;
protected:
bool BigEndian;
bool TLSSupported;
bool VLASupported;
bool NoAsmVariants;  
bool HasLegalHalfType; 
bool HasFloat128;
unsigned char PointerWidth, PointerAlign;
unsigned char BoolWidth, BoolAlign;
unsigned char IntWidth, IntAlign;
unsigned char HalfWidth, HalfAlign;
unsigned char FloatWidth, FloatAlign;
unsigned char DoubleWidth, DoubleAlign;
unsigned char LongDoubleWidth, LongDoubleAlign, Float128Align;
unsigned char LargeArrayMinWidth, LargeArrayAlign;
unsigned char LongWidth, LongAlign;
unsigned char LongLongWidth, LongLongAlign;

unsigned char ShortAccumWidth, ShortAccumAlign;
unsigned char AccumWidth, AccumAlign;
unsigned char LongAccumWidth, LongAccumAlign;
unsigned char ShortFractWidth, ShortFractAlign;
unsigned char FractWidth, FractAlign;
unsigned char LongFractWidth, LongFractAlign;

bool PaddingOnUnsignedFixedPoint;

unsigned char ShortAccumScale;
unsigned char AccumScale;
unsigned char LongAccumScale;

unsigned char SuitableAlign;
unsigned char DefaultAlignForAttributeAligned;
unsigned char MinGlobalAlign;
unsigned char MaxAtomicPromoteWidth, MaxAtomicInlineWidth;
unsigned short MaxVectorAlign;
unsigned short MaxTLSAlign;
unsigned short SimdDefaultAlign;
unsigned short NewAlign;
std::unique_ptr<llvm::DataLayout> DataLayout;
const char *MCountName;
const llvm::fltSemantics *HalfFormat, *FloatFormat, *DoubleFormat,
*LongDoubleFormat, *Float128Format;
unsigned char RegParmMax, SSERegParmMax;
TargetCXXABI TheCXXABI;
const LangASMap *AddrSpaceMap;

mutable StringRef PlatformName;
mutable VersionTuple PlatformMinVersion;

unsigned HasAlignMac68kSupport : 1;
unsigned RealTypeUsesObjCFPRet : 3;
unsigned ComplexLongDoubleUsesFP2Ret : 1;

unsigned HasBuiltinMSVaList : 1;

unsigned IsRenderScriptTarget : 1;

TargetInfo(const llvm::Triple &T);

void resetDataLayout(StringRef DL) {
DataLayout.reset(new llvm::DataLayout(DL));
}

public:
static TargetInfo *
CreateTargetInfo(DiagnosticsEngine &Diags,
const std::shared_ptr<TargetOptions> &Opts);

virtual ~TargetInfo();

TargetOptions &getTargetOpts() const {
assert(TargetOpts && "Missing target options");
return *TargetOpts;
}

enum IntType {
NoInt = 0,
SignedChar,
UnsignedChar,
SignedShort,
UnsignedShort,
SignedInt,
UnsignedInt,
SignedLong,
UnsignedLong,
SignedLongLong,
UnsignedLongLong
};

enum RealType {
NoFloat = 255,
Float = 0,
Double,
LongDouble,
Float128
};

enum BuiltinVaListKind {
CharPtrBuiltinVaList = 0,

VoidPtrBuiltinVaList,

AArch64ABIBuiltinVaList,

PNaClABIBuiltinVaList,

PowerABIBuiltinVaList,

X86_64ABIBuiltinVaList,

AAPCSABIBuiltinVaList,

SystemZBuiltinVaList
};

protected:
IntType SizeType, IntMaxType, PtrDiffType, IntPtrType, WCharType,
WIntType, Char16Type, Char32Type, Int64Type, SigAtomicType,
ProcessIDType;

unsigned UseSignedCharForObjCBool : 1;

unsigned UseBitFieldTypeAlignment : 1;

unsigned UseZeroLengthBitfieldAlignment : 1;

unsigned UseExplicitBitFieldAlignment : 1;

unsigned ZeroLengthBitfieldBoundary;

bool UseAddrSpaceMapMangling;

public:
IntType getSizeType() const { return SizeType; }
IntType getSignedSizeType() const {
switch (SizeType) {
case UnsignedShort:
return SignedShort;
case UnsignedInt:
return SignedInt;
case UnsignedLong:
return SignedLong;
case UnsignedLongLong:
return SignedLongLong;
default:
llvm_unreachable("Invalid SizeType");
}
}
IntType getIntMaxType() const { return IntMaxType; }
IntType getUIntMaxType() const {
return getCorrespondingUnsignedType(IntMaxType);
}
IntType getPtrDiffType(unsigned AddrSpace) const {
return AddrSpace == 0 ? PtrDiffType : getPtrDiffTypeV(AddrSpace);
}
IntType getUnsignedPtrDiffType(unsigned AddrSpace) const {
return getCorrespondingUnsignedType(getPtrDiffType(AddrSpace));
}
IntType getIntPtrType() const { return IntPtrType; }
IntType getUIntPtrType() const {
return getCorrespondingUnsignedType(IntPtrType);
}
IntType getWCharType() const { return WCharType; }
IntType getWIntType() const { return WIntType; }
IntType getChar16Type() const { return Char16Type; }
IntType getChar32Type() const { return Char32Type; }
IntType getInt64Type() const { return Int64Type; }
IntType getUInt64Type() const {
return getCorrespondingUnsignedType(Int64Type);
}
IntType getSigAtomicType() const { return SigAtomicType; }
IntType getProcessIDType() const { return ProcessIDType; }

static IntType getCorrespondingUnsignedType(IntType T) {
switch (T) {
case SignedChar:
return UnsignedChar;
case SignedShort:
return UnsignedShort;
case SignedInt:
return UnsignedInt;
case SignedLong:
return UnsignedLong;
case SignedLongLong:
return UnsignedLongLong;
default:
llvm_unreachable("Unexpected signed integer type");
}
}

unsigned getTypeWidth(IntType T) const;

virtual IntType getIntTypeByWidth(unsigned BitWidth, bool IsSigned) const;

virtual IntType getLeastIntTypeByWidth(unsigned BitWidth,
bool IsSigned) const;

RealType getRealTypeByWidth(unsigned BitWidth) const;

unsigned getTypeAlign(IntType T) const;

static bool isTypeSigned(IntType T);

uint64_t getPointerWidth(unsigned AddrSpace) const {
return AddrSpace == 0 ? PointerWidth : getPointerWidthV(AddrSpace);
}
uint64_t getPointerAlign(unsigned AddrSpace) const {
return AddrSpace == 0 ? PointerAlign : getPointerAlignV(AddrSpace);
}

virtual uint64_t getMaxPointerWidth() const {
return PointerWidth;
}

virtual uint64_t getNullPointerValue(LangAS AddrSpace) const { return 0; }

unsigned getBoolWidth() const { return BoolWidth; }

unsigned getBoolAlign() const { return BoolAlign; }

unsigned getCharWidth() const { return 8; } 
unsigned getCharAlign() const { return 8; } 

unsigned getShortWidth() const { return 16; } 

unsigned getShortAlign() const { return 16; } 

unsigned getIntWidth() const { return IntWidth; }
unsigned getIntAlign() const { return IntAlign; }

unsigned getLongWidth() const { return LongWidth; }
unsigned getLongAlign() const { return LongAlign; }

unsigned getLongLongWidth() const { return LongLongWidth; }
unsigned getLongLongAlign() const { return LongLongAlign; }

unsigned getShortAccumWidth() const { return ShortAccumWidth; }
unsigned getShortAccumAlign() const { return ShortAccumAlign; }

unsigned getAccumWidth() const { return AccumWidth; }
unsigned getAccumAlign() const { return AccumAlign; }

unsigned getLongAccumWidth() const { return LongAccumWidth; }
unsigned getLongAccumAlign() const { return LongAccumAlign; }

unsigned getShortFractWidth() const { return ShortFractWidth; }
unsigned getShortFractAlign() const { return ShortFractAlign; }

unsigned getFractWidth() const { return FractWidth; }
unsigned getFractAlign() const { return FractAlign; }

unsigned getLongFractWidth() const { return LongFractWidth; }
unsigned getLongFractAlign() const { return LongFractAlign; }

unsigned getShortAccumScale() const { return ShortAccumScale; }
unsigned getShortAccumIBits() const {
return ShortAccumWidth - ShortAccumScale - 1;
}

unsigned getAccumScale() const { return AccumScale; }
unsigned getAccumIBits() const { return AccumWidth - AccumScale - 1; }

unsigned getLongAccumScale() const { return LongAccumScale; }
unsigned getLongAccumIBits() const {
return LongAccumWidth - LongAccumScale - 1;
}

unsigned getUnsignedShortAccumScale() const {
return PaddingOnUnsignedFixedPoint ? ShortAccumScale : ShortAccumScale + 1;
}
unsigned getUnsignedShortAccumIBits() const {
return PaddingOnUnsignedFixedPoint
? getShortAccumIBits()
: ShortAccumWidth - getUnsignedShortAccumScale();
}

unsigned getUnsignedAccumScale() const {
return PaddingOnUnsignedFixedPoint ? AccumScale : AccumScale + 1;
}
unsigned getUnsignedAccumIBits() const {
return PaddingOnUnsignedFixedPoint ? getAccumIBits()
: AccumWidth - getUnsignedAccumScale();
}

unsigned getUnsignedLongAccumScale() const {
return PaddingOnUnsignedFixedPoint ? LongAccumScale : LongAccumScale + 1;
}
unsigned getUnsignedLongAccumIBits() const {
return PaddingOnUnsignedFixedPoint
? getLongAccumIBits()
: LongAccumWidth - getUnsignedLongAccumScale();
}

unsigned getShortFractScale() const { return ShortFractWidth - 1; }

unsigned getFractScale() const { return FractWidth - 1; }

unsigned getLongFractScale() const { return LongFractWidth - 1; }

unsigned getUnsignedShortFractScale() const {
return PaddingOnUnsignedFixedPoint ? getShortFractScale()
: getShortFractScale() + 1;
}

unsigned getUnsignedFractScale() const {
return PaddingOnUnsignedFixedPoint ? getFractScale() : getFractScale() + 1;
}

unsigned getUnsignedLongFractScale() const {
return PaddingOnUnsignedFixedPoint ? getLongFractScale()
: getLongFractScale() + 1;
}

virtual bool hasInt128Type() const {
return (getPointerWidth(0) >= 64) || getTargetOpts().ForceEnableInt128;
} 

virtual bool hasLegalHalfType() const { return HasLegalHalfType; }

virtual bool hasFloat128Type() const { return HasFloat128; }

unsigned getSuitableAlign() const { return SuitableAlign; }

unsigned getDefaultAlignForAttributeAligned() const {
return DefaultAlignForAttributeAligned;
}

unsigned getMinGlobalAlign() const { return MinGlobalAlign; }

unsigned getNewAlign() const {
return NewAlign ? NewAlign : std::max(LongDoubleAlign, LongLongAlign);
}

unsigned getWCharWidth() const { return getTypeWidth(WCharType); }
unsigned getWCharAlign() const { return getTypeAlign(WCharType); }

unsigned getChar16Width() const { return getTypeWidth(Char16Type); }
unsigned getChar16Align() const { return getTypeAlign(Char16Type); }

unsigned getChar32Width() const { return getTypeWidth(Char32Type); }
unsigned getChar32Align() const { return getTypeAlign(Char32Type); }

unsigned getHalfWidth() const { return HalfWidth; }
unsigned getHalfAlign() const { return HalfAlign; }
const llvm::fltSemantics &getHalfFormat() const { return *HalfFormat; }

unsigned getFloatWidth() const { return FloatWidth; }
unsigned getFloatAlign() const { return FloatAlign; }
const llvm::fltSemantics &getFloatFormat() const { return *FloatFormat; }

unsigned getDoubleWidth() const { return DoubleWidth; }
unsigned getDoubleAlign() const { return DoubleAlign; }
const llvm::fltSemantics &getDoubleFormat() const { return *DoubleFormat; }

unsigned getLongDoubleWidth() const { return LongDoubleWidth; }
unsigned getLongDoubleAlign() const { return LongDoubleAlign; }
const llvm::fltSemantics &getLongDoubleFormat() const {
return *LongDoubleFormat;
}

unsigned getFloat128Width() const { return 128; }
unsigned getFloat128Align() const { return Float128Align; }
const llvm::fltSemantics &getFloat128Format() const {
return *Float128Format;
}

virtual bool useFloat128ManglingForLongDouble() const { return false; }

virtual unsigned getFloatEvalMethod() const { return 0; }

unsigned getLargeArrayMinWidth() const { return LargeArrayMinWidth; }
unsigned getLargeArrayAlign() const { return LargeArrayAlign; }

unsigned getMaxAtomicPromoteWidth() const { return MaxAtomicPromoteWidth; }
unsigned getMaxAtomicInlineWidth() const { return MaxAtomicInlineWidth; }
virtual void setMaxAtomicWidth() {}
virtual bool hasBuiltinAtomic(uint64_t AtomicSizeInBits,
uint64_t AlignmentInBits) const {
return AtomicSizeInBits <= AlignmentInBits &&
AtomicSizeInBits <= getMaxAtomicInlineWidth() &&
(AtomicSizeInBits <= getCharWidth() ||
llvm::isPowerOf2_64(AtomicSizeInBits / getCharWidth()));
}

unsigned getMaxVectorAlign() const { return MaxVectorAlign; }
unsigned getSimdDefaultAlign() const { return SimdDefaultAlign; }

unsigned getIntMaxTWidth() const {
return getTypeWidth(IntMaxType);
}

virtual unsigned getUnwindWordWidth() const { return getPointerWidth(0); }

virtual unsigned getRegisterWidth() const {
return PointerWidth;
}

const char *getMCountName() const {
return MCountName;
}

bool useSignedCharForObjCBool() const {
return UseSignedCharForObjCBool;
}
void noSignedCharForObjCBool() {
UseSignedCharForObjCBool = false;
}

bool useBitFieldTypeAlignment() const {
return UseBitFieldTypeAlignment;
}

bool useZeroLengthBitfieldAlignment() const {
return UseZeroLengthBitfieldAlignment;
}

unsigned getZeroLengthBitfieldBoundary() const {
return ZeroLengthBitfieldBoundary;
}

bool useExplicitBitFieldAlignment() const {
return UseExplicitBitFieldAlignment;
}

bool hasAlignMac68kSupport() const {
return HasAlignMac68kSupport;
}

static const char *getTypeName(IntType T);

const char *getTypeConstantSuffix(IntType T) const;

static const char *getTypeFormatModifier(IntType T);

bool useObjCFPRetForRealType(RealType T) const {
return RealTypeUsesObjCFPRet & (1 << T);
}

bool useObjCFP2RetForComplexLongDouble() const {
return ComplexLongDoubleUsesFP2Ret;
}

virtual bool useFP16ConversionIntrinsics() const {
return true;
}

bool useAddressSpaceMapMangling() const {
return UseAddrSpaceMapMangling;
}


virtual void getTargetDefines(const LangOptions &Opts,
MacroBuilder &Builder) const = 0;


virtual ArrayRef<Builtin::Info> getTargetBuiltins() const = 0;

virtual bool isCLZForZeroUndef() const { return true; }

virtual BuiltinVaListKind getBuiltinVaListKind() const = 0;

bool hasBuiltinMSVaList() const { return HasBuiltinMSVaList; }

bool isRenderScriptTarget() const { return IsRenderScriptTarget; }

bool isValidClobber(StringRef Name) const;

virtual bool isValidGCCRegisterName(StringRef Name) const;

StringRef getNormalizedGCCRegisterName(StringRef Name,
bool ReturnCanonical = false) const;

virtual StringRef getConstraintRegister(StringRef Constraint,
StringRef Expression) const {
return "";
}

struct ConstraintInfo {
enum {
CI_None = 0x00,
CI_AllowsMemory = 0x01,
CI_AllowsRegister = 0x02,
CI_ReadWrite = 0x04,         
CI_HasMatchingInput = 0x08,  
CI_ImmediateConstant = 0x10, 
CI_EarlyClobber = 0x20,      
};
unsigned Flags;
int TiedOperand;
struct {
int Min;
int Max;
} ImmRange;
llvm::SmallSet<int, 4> ImmSet;

std::string ConstraintStr;  
std::string Name;           
public:
ConstraintInfo(StringRef ConstraintStr, StringRef Name)
: Flags(0), TiedOperand(-1), ConstraintStr(ConstraintStr.str()),
Name(Name.str()) {
ImmRange.Min = ImmRange.Max = 0;
}

const std::string &getConstraintStr() const { return ConstraintStr; }
const std::string &getName() const { return Name; }
bool isReadWrite() const { return (Flags & CI_ReadWrite) != 0; }
bool earlyClobber() { return (Flags & CI_EarlyClobber) != 0; }
bool allowsRegister() const { return (Flags & CI_AllowsRegister) != 0; }
bool allowsMemory() const { return (Flags & CI_AllowsMemory) != 0; }

bool hasMatchingInput() const { return (Flags & CI_HasMatchingInput) != 0; }

bool hasTiedOperand() const { return TiedOperand != -1; }
unsigned getTiedOperand() const {
assert(hasTiedOperand() && "Has no tied operand!");
return (unsigned)TiedOperand;
}

bool requiresImmediateConstant() const {
return (Flags & CI_ImmediateConstant) != 0;
}
bool isValidAsmImmediate(const llvm::APInt &Value) const {
return (Value.sge(ImmRange.Min) && Value.sle(ImmRange.Max)) ||
ImmSet.count(Value.getZExtValue()) != 0;
}

void setIsReadWrite() { Flags |= CI_ReadWrite; }
void setEarlyClobber() { Flags |= CI_EarlyClobber; }
void setAllowsMemory() { Flags |= CI_AllowsMemory; }
void setAllowsRegister() { Flags |= CI_AllowsRegister; }
void setHasMatchingInput() { Flags |= CI_HasMatchingInput; }
void setRequiresImmediate(int Min, int Max) {
Flags |= CI_ImmediateConstant;
ImmRange.Min = Min;
ImmRange.Max = Max;
}
void setRequiresImmediate(llvm::ArrayRef<int> Exacts) {
Flags |= CI_ImmediateConstant;
for (int Exact : Exacts)
ImmSet.insert(Exact);
}
void setRequiresImmediate(int Exact) {
Flags |= CI_ImmediateConstant;
ImmSet.insert(Exact);
}
void setRequiresImmediate() {
Flags |= CI_ImmediateConstant;
ImmRange.Min = INT_MIN;
ImmRange.Max = INT_MAX;
}

void setTiedOperand(unsigned N, ConstraintInfo &Output) {
Output.setHasMatchingInput();
Flags = Output.Flags;
TiedOperand = N;
}
};

virtual bool validateGlobalRegisterVariable(StringRef RegName,
unsigned RegSize,
bool &HasSizeMismatch) const {
HasSizeMismatch = false;
return true;
}

bool validateOutputConstraint(ConstraintInfo &Info) const;
bool validateInputConstraint(MutableArrayRef<ConstraintInfo> OutputConstraints,
ConstraintInfo &info) const;

virtual bool validateOutputSize(StringRef ,
unsigned ) const {
return true;
}

virtual bool validateInputSize(StringRef ,
unsigned ) const {
return true;
}
virtual bool
validateConstraintModifier(StringRef ,
char ,
unsigned ,
std::string &) const {
return true;
}
virtual bool
validateAsmConstraint(const char *&Name,
TargetInfo::ConstraintInfo &info) const = 0;

bool resolveSymbolicName(const char *&Name,
ArrayRef<ConstraintInfo> OutputConstraints,
unsigned &Index) const;

virtual std::string convertConstraint(const char *&Constraint) const {
if (*Constraint == 'p')
return std::string("r");
return std::string(1, *Constraint);
}

virtual const char *getClobbers() const = 0;

virtual bool isNan2008() const {
return true;
}

const llvm::Triple &getTriple() const {
return Triple;
}

const llvm::DataLayout &getDataLayout() const {
assert(DataLayout && "Uninitialized DataLayout!");
return *DataLayout;
}

struct GCCRegAlias {
const char * const Aliases[5];
const char * const Register;
};

struct AddlRegName {
const char * const Names[5];
const unsigned RegNum;
};

virtual bool hasProtectedVisibility() const { return true; }

virtual std::string isValidSectionSpecifier(StringRef SR) const {
return "";
}

virtual void adjust(LangOptions &Opts);

virtual void adjustTargetOptions(const CodeGenOptions &CGOpts,
TargetOptions &TargetOpts) const {}

virtual bool initFeatureMap(llvm::StringMap<bool> &Features,
DiagnosticsEngine &Diags, StringRef CPU,
const std::vector<std::string> &FeatureVec) const;

virtual StringRef getABI() const { return StringRef(); }

TargetCXXABI getCXXABI() const {
return TheCXXABI;
}

virtual bool setCPU(const std::string &Name) {
return false;
}

virtual void fillValidCPUList(SmallVectorImpl<StringRef> &Values) const {}

virtual bool isValidCPUName(StringRef Name) const {
return true;
}

virtual bool setABI(const std::string &Name) {
return false;
}

virtual bool setFPMath(StringRef Name) {
return false;
}

virtual void setFeatureEnabled(llvm::StringMap<bool> &Features,
StringRef Name,
bool Enabled) const {
Features[Name] = Enabled;
}

virtual bool isValidFeatureName(StringRef Feature) const {
return true;
}

virtual bool handleTargetFeatures(std::vector<std::string> &Features,
DiagnosticsEngine &Diags) {
return true;
}

virtual bool hasFeature(StringRef Feature) const {
return false;
}

virtual bool supportsMultiVersioning() const { return false; }

virtual bool validateCpuSupports(StringRef Name) const { return false; }

virtual unsigned multiVersionSortPriority(StringRef Name) const {
return 0;
}

virtual bool validateCpuIs(StringRef Name) const { return false; }

virtual bool validateCPUSpecificCPUDispatch(StringRef Name) const {
return false;
}

virtual char CPUSpecificManglingCharacter(StringRef Name) const {
llvm_unreachable(
"cpu_specific Multiversioning not implemented on this target");
}

virtual void getCPUSpecificCPUDispatchFeatures(
StringRef Name, llvm::SmallVectorImpl<StringRef> &Features) const {
llvm_unreachable(
"cpu_specific Multiversioning not implemented on this target");
}

unsigned getRegParmMax() const {
assert(RegParmMax < 7 && "RegParmMax value is larger than AST can handle");
return RegParmMax;
}

bool isTLSSupported() const {
return TLSSupported;
}

unsigned short getMaxTLSAlign() const {
return MaxTLSAlign;
}

bool isVLASupported() const { return VLASupported; }

bool isSEHTrySupported() const {
return getTriple().isOSWindows() &&
(getTriple().getArch() == llvm::Triple::x86 ||
getTriple().getArch() == llvm::Triple::x86_64 ||
getTriple().getArch() == llvm::Triple::aarch64);
}

bool hasNoAsmVariants() const {
return NoAsmVariants;
}

virtual int getEHDataRegisterNumber(unsigned RegNo) const {
return -1;
}

virtual const char *getStaticInitSectionSpecifier() const {
return nullptr;
}

const LangASMap &getAddressSpaceMap() const { return *AddrSpaceMap; }

virtual llvm::Optional<LangAS> getConstantAddressSpace() const {
return LangAS::Default;
}

StringRef getPlatformName() const { return PlatformName; }

VersionTuple getPlatformMinVersion() const { return PlatformMinVersion; }

bool isBigEndian() const { return BigEndian; }
bool isLittleEndian() const { return !BigEndian; }

enum CallingConvMethodType {
CCMT_Unknown,
CCMT_Member,
CCMT_NonMember
};

virtual CallingConv getDefaultCallingConv(CallingConvMethodType MT) const {
return CC_C;
}

enum CallingConvCheckResult {
CCCR_OK,
CCCR_Warning,
CCCR_Ignore,
};

virtual CallingConvCheckResult checkCallingConvention(CallingConv CC) const {
switch (CC) {
default:
return CCCR_Warning;
case CC_C:
return CCCR_OK;
}
}

enum CallingConvKind {
CCK_Default,
CCK_ClangABI4OrPS4,
CCK_MicrosoftWin64
};

virtual CallingConvKind getCallingConvKind(bool ClangABICompat4) const;

virtual bool hasSjLjLowering() const {
return false;
}

virtual bool
checkCFProtectionBranchSupported(DiagnosticsEngine &Diags) const;

virtual bool
checkCFProtectionReturnSupported(DiagnosticsEngine &Diags) const;

virtual bool allowsLargerPreferedTypeAlignment() const { return true; }

virtual void setSupportedOpenCLOpts() {}

virtual void setOpenCLExtensionOpts() {
for (const auto &Ext : getTargetOpts().OpenCLExtensionsAsWritten) {
getTargetOpts().SupportedOpenCLOptions.support(Ext);
}
}

OpenCLOptions &getSupportedOpenCLOpts() {
return getTargetOpts().SupportedOpenCLOptions;
}

const OpenCLOptions &getSupportedOpenCLOpts() const {
return getTargetOpts().SupportedOpenCLOptions;
}

enum OpenCLTypeKind {
OCLTK_Default,
OCLTK_ClkEvent,
OCLTK_Event,
OCLTK_Image,
OCLTK_Pipe,
OCLTK_Queue,
OCLTK_ReserveID,
OCLTK_Sampler,
};

virtual LangAS getOpenCLTypeAddrSpace(OpenCLTypeKind TK) const;

virtual unsigned getVtblPtrAddressSpace() const {
return 0;
}

virtual Optional<unsigned> getDWARFAddressSpace(unsigned AddressSpace) const {
return None;
}

virtual bool validateTarget(DiagnosticsEngine &Diags) const {
return true;
}

protected:
virtual uint64_t getPointerWidthV(unsigned AddrSpace) const {
return PointerWidth;
}
virtual uint64_t getPointerAlignV(unsigned AddrSpace) const {
return PointerAlign;
}
virtual enum IntType getPtrDiffTypeV(unsigned AddrSpace) const {
return PtrDiffType;
}
virtual ArrayRef<const char *> getGCCRegNames() const = 0;
virtual ArrayRef<GCCRegAlias> getGCCRegAliases() const = 0;
virtual ArrayRef<AddlRegName> getGCCAddlRegNames() const {
return None;
}

private:
void CheckFixedPointBits() const;
};

}  

#endif
