


#pragma once


#include "semantichigh.h"

class SemanticDerived: public SemanticHigh
{
public:
SemanticDerived()
{}

~SemanticDerived()
{}

virtual bool isControlDerived() = 0;

virtual SemanticInfoType getSemanticInfoType() const override
{
return SAME_TYPE;
}

protected:

private:

};



