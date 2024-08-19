#pragma once

#include <ostream>
#include <streambuf>

class NullStreambuf : public std::streambuf {
char dummyBuffer[64];

protected:
virtual int overflow(int c) override final;
};

class NullOStream final : private NullStreambuf, public std::ostream {
public:
NullOStream() : std::ostream(this) {}
NullStreambuf *rdbuf() { return this; }
virtual void avoidOutOfLineVirtualCompilerWarning();
};

