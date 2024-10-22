


#pragma once


#include "semanticnotthread.h"

class Adding: public SemanticNotThread
{
public:
typedef enum
{
MAXPARAM = 0
} TParam;

Adding()
{
setDefaultParam();
}

~Adding()
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
return Adding::name;
}

virtual SemanticFunction *clone() override
{
return new Adding( *this );
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


class AddingSign: public SemanticNotThread
{
public:
typedef enum
{
MAXPARAM = 0
} TParam;

AddingSign()
{
setDefaultParam();
}

~AddingSign()
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
return AddingSign::name;
}

virtual SemanticFunction *clone() override
{
return new AddingSign( *this );
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


class Average: public SemanticNotThread
{
public:
typedef enum
{
MAXPARAM = 0
} TParam;

Average()
{
setDefaultParam();
}

~Average()
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
return Average::name;
}

virtual SemanticFunction *clone() override
{
return new Average( *this );
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


class Maximum: public SemanticNotThread
{
public:
typedef enum
{
MAXPARAM = 0
} TParam;

Maximum()
{
setDefaultParam();
}

~Maximum()
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
return Maximum::name;
}

virtual SemanticFunction *clone() override
{
return new Maximum( *this );
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


class Minimum: public SemanticNotThread
{
public:
typedef enum
{
MAXPARAM = 0
} TParam;

Minimum()
{
setDefaultParam();
}

~Minimum()
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
return Minimum::name;
}

virtual SemanticFunction *clone() override
{
return new Minimum( *this );
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


class Activity: public SemanticNotThread
{
public:
typedef enum
{
VALUES = 0,
MAXPARAM
} TParam;

Activity()
{
setDefaultParam();
}

~Activity()
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
return Activity::name;
}

virtual SemanticFunction *clone() override
{
return new Activity( *this );
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
else if ( whichParam == VALUES )
tmp.push_back( 1 );

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


class InActivity: public SemanticNotThread
{
public:
typedef enum
{
VALUES = 0,
MAXPARAM
} TParam;

InActivity()
{
setDefaultParam();
}

~InActivity()
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
return InActivity::name;
}

virtual SemanticFunction *clone() override
{
return new InActivity( *this );
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
else if ( whichParam == VALUES )
tmp.push_back( 1 );

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


class NotThreadMode: public SemanticNotThread
{
public:
typedef enum
{
MAXPARAM = 0
} TParam;

NotThreadMode()
{
setDefaultParam();
}

~NotThreadMode()
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
return NotThreadMode::name;
}

virtual SemanticFunction *clone() override
{
return new NotThreadMode( *this );
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


class ObjectI: public SemanticNotThread
{
public:
typedef enum
{
OBJECT = 0,
MAXPARAM
} TParam;

ObjectI()
{
setDefaultParam();
}

~ObjectI()
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
return ObjectI::name;
}

virtual SemanticFunction *clone() override
{
return new ObjectI( *this );
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
else if ( whichParam == OBJECT )
tmp.push_back( 1 );

return tmp;
}
virtual std::string getDefaultParamName( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return "Object";
}

private:
static const bool initFromBegin = false;
static std::string name;

};


class AddObjectsI: public SemanticNotThread
{
public:
typedef enum
{
OBJECTS = 0,
MAXPARAM
} TParam;

AddObjectsI()
{
setDefaultParam();
}

~AddObjectsI()
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
return AddObjectsI::name;
}

virtual SemanticFunction *clone() override
{
return new AddObjectsI( *this );
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
else if ( whichParam == OBJECTS )
tmp.push_back( 0 );

return tmp;
}
virtual std::string getDefaultParamName( TParamIndex whichParam ) override
{
if ( whichParam >= getMaxParam() )
throw SemanticException( TSemanticErrorCode::maxParamExceeded );
return "Objects";
}

private:
static const bool initFromBegin = false;
static std::string name;

};


class ChangedValue: public SemanticNotThread
{
public:
typedef enum
{
MAXPARAM = 0
} TParam;

ChangedValue()
{
setDefaultParam();
}

~ChangedValue()
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
return ChangedValue::name;
}

virtual SemanticFunction *clone() override
{
return new ChangedValue( *this );
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
static const bool initFromBegin = true;
static std::string name;

};


