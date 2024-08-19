


#pragma once


#include "semanticcpu.h"

class ActiveThread: public SemanticCPU
{
public:
typedef enum
{
MAXPARAM = 0
} TParam;

ActiveThread()
{
setDefaultParam();
}

~ActiveThread()
{}

virtual TParamIndex getMaxParam() const override
{
return MAXPARAM;
}

virtual TSemanticValue execute( const SemanticInfo *info ) override;
virtual void init( KTimeline *whichWindow ) override
{}

virtual std::string getName() override
{
return ActiveThread::name;
}

virtual SemanticFunction *clone() override
{
return new ActiveThread( *this );
}

virtual SemanticInfoType getSemanticInfoType() const override
{
return SAME_TYPE;
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
static std::string name;
};


class ActiveThreadSign: public SemanticCPU
{
public:
typedef enum
{
MAXPARAM = 0
} TParam;

ActiveThreadSign()
{
setDefaultParam();
}

~ActiveThreadSign()
{}

virtual TParamIndex getMaxParam() const override
{
return MAXPARAM;
}

virtual TSemanticValue execute( const SemanticInfo *info ) override;
virtual void init( KTimeline *whichWindow ) override
{}

virtual std::string getName() override
{
return ActiveThreadSign::name;
}

virtual SemanticFunction *clone() override
{
return new ActiveThreadSign( *this );
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
static std::string name;

};


class ActiveThreadValues: public SemanticCPU
{
public:
typedef enum
{
VALUES = 0,
MAXPARAM
} TParam;

ActiveThreadValues()
{
setDefaultParam();
}

~ActiveThreadValues()
{}

virtual TParamIndex getMaxParam() const override
{
return MAXPARAM;
}

virtual TSemanticValue execute( const SemanticInfo *info ) override;
virtual void init( KTimeline *whichWindow ) override
{}

virtual std::string getName() override
{
return ActiveThreadValues::name;
}

virtual SemanticFunction *clone() override
{
return new ActiveThreadValues( *this );
}

virtual SemanticInfoType getSemanticInfoType() const override
{
return SAME_TYPE;
}

protected:
virtual const bool getMyInitFromBegin() override
{
return initFromBegin;
}
virtual TParamValue getDefaultParam( TParamIndex whichParam ) override
{
TParamValue tmp;

if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
if ( whichParam == VALUES )
{
tmp.push_back( 1 );
}
return tmp;
}
virtual std::string getDefaultParamName( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return "Values";
}

private:
static const bool initFromBegin = false;
static std::string name;

};


class ActiveThreadValuesSign: public SemanticCPU
{
public:
typedef enum
{
VALUES = 0,
MAXPARAM
} TParam;

ActiveThreadValuesSign()
{
setDefaultParam();
}

~ActiveThreadValuesSign()
{}

virtual TParamIndex getMaxParam() const override
{
return MAXPARAM;
}

virtual TSemanticValue execute( const SemanticInfo *info ) override;

virtual void init( KTimeline *whichWindow ) override
{}

virtual std::string getName() override
{
return ActiveThreadValuesSign::name;
}

virtual SemanticFunction *clone() override
{
return new ActiveThreadValuesSign( *this );
}


protected:
virtual const bool getMyInitFromBegin() override
{
return initFromBegin;
}
virtual TParamValue getDefaultParam( TParamIndex whichParam ) override
{
TParamValue tmp;

if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
if ( whichParam == VALUES )
{
tmp.push_back( 1 );
}
return tmp;
}
virtual std::string getDefaultParamName( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return "Values";
}

private:
static const bool initFromBegin = false;
static std::string name;

};


