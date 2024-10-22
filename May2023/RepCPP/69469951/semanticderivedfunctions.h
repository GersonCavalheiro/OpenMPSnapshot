


#pragma once

#include <unordered_map>

#include "semanticderived.h"

class DerivedAdd: public SemanticDerived
{
public:
typedef enum
{
MAXPARAM = 0
} TParam;

DerivedAdd()
{
setDefaultParam();
}

~DerivedAdd()
{}

virtual TParamIndex getMaxParam() const override
{
return MAXPARAM;
}

virtual bool isControlDerived() override
{
return controlDerived;
}

virtual TSemanticValue execute( const SemanticInfo *info ) override;

virtual void init( KTimeline *whichWindow ) override
{}

virtual std::string getName() override
{
return DerivedAdd::name;
}

virtual SemanticFunction *clone() override
{
return new DerivedAdd( *this );
}

protected:
virtual const bool getMyInitFromBegin() override
{
return initFromBegin;
}
virtual TParamValue getDefaultParam( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return ( TParamValue ) 0;
}
virtual std::string getDefaultParamName( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return "";
}

private:
static const bool initFromBegin = false;
static const bool controlDerived = false;
static std::string name;
};


class DerivedProduct: public SemanticDerived
{
public:
typedef enum
{
MAXPARAM = 0
} TParam;

DerivedProduct()
{
setDefaultParam();
}

~DerivedProduct()
{}

virtual TParamIndex getMaxParam() const override
{
return MAXPARAM;
}

virtual bool isControlDerived() override
{
return controlDerived;
}

virtual TSemanticValue execute( const SemanticInfo *info ) override;

virtual void init( KTimeline *whichWindow ) override
{}

virtual std::string getName() override
{
return DerivedProduct::name;
}

virtual SemanticFunction *clone() override
{
return new DerivedProduct( *this );
}


protected:
virtual const bool getMyInitFromBegin() override
{
return initFromBegin;
}
virtual TParamValue getDefaultParam( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return ( TParamValue ) 0;
}
virtual std::string getDefaultParamName( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return "";
}

private:
static const bool initFromBegin = false;
static const bool controlDerived = false;
static std::string name;

};


class DerivedSubstract: public SemanticDerived
{
public:
typedef enum
{
MAXPARAM = 0
} TParam;

DerivedSubstract()
{
setDefaultParam();
}

~DerivedSubstract()
{}

virtual TParamIndex getMaxParam() const override
{
return MAXPARAM;
}

virtual bool isControlDerived() override
{
return controlDerived;
}

virtual TSemanticValue execute( const SemanticInfo *info ) override;

virtual void init( KTimeline *whichWindow ) override
{}

virtual std::string getName() override
{
return DerivedSubstract::name;
}

virtual SemanticFunction *clone() override
{
return new DerivedSubstract( *this );
}


protected:
virtual const bool getMyInitFromBegin() override
{
return initFromBegin;
}
virtual TParamValue getDefaultParam( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return ( TParamValue ) 0;
}
virtual std::string getDefaultParamName( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return "";
}

private:
static const bool initFromBegin = false;
static const bool controlDerived = false;
static std::string name;

};


class DerivedDivide: public SemanticDerived
{
public:
typedef enum
{
MAXPARAM = 0
} TParam;

DerivedDivide()
{
setDefaultParam();
}

~DerivedDivide()
{}

virtual TParamIndex getMaxParam() const override
{
return MAXPARAM;
}

virtual bool isControlDerived() override
{
return controlDerived;
}

virtual TSemanticValue execute( const SemanticInfo *info ) override;
virtual void init( KTimeline *whichWindow ) override
{}

virtual std::string getName() override
{
return DerivedDivide::name;
}

virtual SemanticFunction *clone() override
{
return new DerivedDivide( *this );
}


protected:
virtual const bool getMyInitFromBegin() override
{
return initFromBegin;
}
virtual TParamValue getDefaultParam( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return ( TParamValue ) 0;
}
virtual std::string getDefaultParamName( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return "";
}

private:
static const bool initFromBegin = false;
static const bool controlDerived = false;
static std::string name;

};


class DerivedMaximum: public SemanticDerived
{
public:
typedef enum
{
MAXPARAM = 0
} TParam;

DerivedMaximum()
{
setDefaultParam();
}

~DerivedMaximum()
{}

virtual TParamIndex getMaxParam() const override
{
return MAXPARAM;
}

virtual bool isControlDerived() override
{
return controlDerived;
}

virtual TSemanticValue execute( const SemanticInfo *info ) override;
virtual void init( KTimeline *whichWindow ) override
{}

virtual std::string getName() override
{
return DerivedMaximum::name;
}

virtual SemanticFunction *clone() override
{
return new DerivedMaximum( *this );
}


protected:
virtual const bool getMyInitFromBegin() override
{
return initFromBegin;
}
virtual TParamValue getDefaultParam( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return ( TParamValue ) 0;
}
virtual std::string getDefaultParamName( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return "";
}

private:
static const bool initFromBegin = false;
static const bool controlDerived = false;
static std::string name;

};


class DerivedMinimum: public SemanticDerived
{
public:
typedef enum
{
MAXPARAM = 0
} TParam;

DerivedMinimum()
{
setDefaultParam();
}

~DerivedMinimum()
{}

virtual TParamIndex getMaxParam() const override
{
return MAXPARAM;
}

virtual bool isControlDerived() override
{
return controlDerived;
}

virtual TSemanticValue execute( const SemanticInfo *info ) override;
virtual void init( KTimeline *whichWindow ) override
{}

virtual std::string getName() override
{
return DerivedMinimum::name;
}

virtual SemanticFunction *clone() override
{
return new DerivedMinimum( *this );
}


protected:
virtual const bool getMyInitFromBegin() override
{
return initFromBegin;
}
virtual TParamValue getDefaultParam( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return ( TParamValue ) 0;
}
virtual std::string getDefaultParamName( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return "";
}

private:
static const bool initFromBegin = false;
static const bool controlDerived = false;
static std::string name;

};


class DerivedDifferent: public SemanticDerived
{
public:
typedef enum
{
MAXPARAM = 0
} TParam;

DerivedDifferent()
{
setDefaultParam();
}

~DerivedDifferent()
{}

virtual TParamIndex getMaxParam() const override
{
return MAXPARAM;
}

virtual bool isControlDerived() override
{
return controlDerived;
}

virtual TSemanticValue execute( const SemanticInfo *info ) override;
virtual void init( KTimeline *whichWindow ) override
{}

virtual std::string getName() override
{
return DerivedDifferent::name;
}

virtual SemanticFunction *clone() override
{
return new DerivedDifferent( *this );
}


protected:
virtual const bool getMyInitFromBegin() override
{
return initFromBegin;
}
virtual TParamValue getDefaultParam( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return ( TParamValue ) 0;
}
virtual std::string getDefaultParamName( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return "";
}

private:
static const bool initFromBegin = false;
static const bool controlDerived = false;
static std::string name;

};


class ControlDerivedClearBy: public SemanticDerived
{
public:
typedef enum
{
MAXPARAM = 0
} TParam;

ControlDerivedClearBy()
{
setDefaultParam();
}

~ControlDerivedClearBy()
{}

virtual TParamIndex getMaxParam() const override
{
return MAXPARAM;
}

virtual bool isControlDerived() override
{
return controlDerived;
}

virtual TSemanticValue execute( const SemanticInfo *info ) override;
virtual void init( KTimeline *whichWindow ) override;

virtual std::string getName() override
{
return ControlDerivedClearBy::name;
}

virtual SemanticFunction *clone() override
{
return new ControlDerivedClearBy( *this );
}


protected:
virtual const bool getMyInitFromBegin() override
{
return initFromBegin;
}
virtual TParamValue getDefaultParam( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return ( TParamValue ) 0;
}
virtual std::string getDefaultParamName( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return "";
}

private:
static const bool initFromBegin = true;
static const bool controlDerived = false;
static std::string name;

std::unordered_map<TObjectOrder, TSemanticValue> lastControlValue;
std::unordered_map<TObjectOrder, TRecordTime> lastDataBeginTime;

};


class ControlDerivedMaximum: public SemanticDerived
{
public:
typedef enum
{
MAXPARAM = 0
} TParam;

ControlDerivedMaximum()
{
setDefaultParam();
}

~ControlDerivedMaximum()
{}

virtual TParamIndex getMaxParam() const override
{
return MAXPARAM;
}

virtual bool isControlDerived() override
{
return controlDerived;
}

virtual TSemanticValue execute( const SemanticInfo *info ) override;
virtual void init( KTimeline *whichWindow ) override
{}

virtual std::string getName() override
{
return ControlDerivedMaximum::name;
}

virtual SemanticFunction *clone() override
{
return new ControlDerivedMaximum( *this );
}


protected:
virtual const bool getMyInitFromBegin() override
{
return initFromBegin;
}
virtual TParamValue getDefaultParam( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return ( TParamValue ) 0;
}
virtual std::string getDefaultParamName( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return "";
}

private:
static const bool initFromBegin = false;
static const bool controlDerived = true;
static std::string name;

};


class ControlDerivedAdd: public SemanticDerived
{
public:
typedef enum
{
MAXPARAM = 0
} TParam;

ControlDerivedAdd()
{
setDefaultParam();
}

~ControlDerivedAdd()
{}

virtual TParamIndex getMaxParam() const override
{
return MAXPARAM;
}

virtual bool isControlDerived() override
{
return controlDerived;
}

virtual TSemanticValue execute( const SemanticInfo *info ) override;
virtual void init( KTimeline *whichWindow ) override
{}

virtual std::string getName() override
{
return ControlDerivedAdd::name;
}

virtual SemanticFunction *clone() override
{
return new ControlDerivedAdd( *this );
}


protected:
virtual const bool getMyInitFromBegin() override
{
return initFromBegin;
}
virtual TParamValue getDefaultParam( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return ( TParamValue ) 0;
}
virtual std::string getDefaultParamName( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return "";
}

private:
static const bool initFromBegin = false;
static const bool controlDerived = true;
static std::string name;

};


class ControlDerivedEnumerate: public SemanticDerived
{
public:
typedef enum
{
MAXPARAM = 0
} TParam;

ControlDerivedEnumerate()
{
setDefaultParam();
}

~ControlDerivedEnumerate()
{}

virtual TParamIndex getMaxParam() const override
{
return MAXPARAM;
}

virtual bool isControlDerived() override
{
return controlDerived;
}

virtual TSemanticValue execute( const SemanticInfo *info ) override;
virtual void init( KTimeline *whichWindow ) override;

virtual std::string getName() override
{
return ControlDerivedEnumerate::name;
}

virtual SemanticFunction *clone() override
{
return new ControlDerivedEnumerate( *this );
}

virtual SemanticInfoType getSemanticInfoType() const override
{
return NO_TYPE;
}

protected:
virtual const bool getMyInitFromBegin() override
{
return initFromBegin;
}
virtual TParamValue getDefaultParam( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return ( TParamValue ) 0;
}
virtual std::string getDefaultParamName( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return "";
}

private:
static const bool initFromBegin = true;
static const bool controlDerived = false;
static std::string name;

std::unordered_map<TObjectOrder, TSemanticValue> prevControlValue;
std::unordered_map<TObjectOrder, TRecordTime> prevDataTime;
std::unordered_map<TObjectOrder, TSemanticValue> myEnumerate;

};


class ControlDerivedAverage: public SemanticDerived
{
public:
typedef enum
{
MAXPARAM = 0
} TParam;

ControlDerivedAverage()
{
setDefaultParam();
}

~ControlDerivedAverage()
{}

virtual TParamIndex getMaxParam() const override
{
return MAXPARAM;
}

virtual bool isControlDerived() override
{
return controlDerived;
}

virtual TSemanticValue execute( const SemanticInfo *info ) override;
virtual void init( KTimeline *whichWindow ) override;

virtual std::string getName() override
{
return ControlDerivedAverage::name;
}

virtual SemanticFunction *clone() override
{
return new ControlDerivedAverage( *this );
}

virtual SemanticInfoType getSemanticInfoType() const override
{
return NO_TYPE;
}

protected:
virtual const bool getMyInitFromBegin() override
{
return initFromBegin;
}
virtual TParamValue getDefaultParam( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return ( TParamValue ) 0;
}
virtual std::string getDefaultParamName( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return "";
}

private:
static const bool initFromBegin = false;
static const bool controlDerived = true;
static std::string name;

std::unordered_map<TObjectOrder, TSemanticValue> totalValue;
std::unordered_map<TObjectOrder, TRecordTime> totalTime;

};


