
#pragma once

#include <exception>
#include <iostream>
#include <sstream>

#define _ping_ __FILE__, __LINE__




namespace dg
{

class Message
{
private:
std::stringstream sstream_;
Message( const Message&); 
Message& operator=(const Message&);
public:
Message(){}

Message(const char* file, const int line){
sstream_ << "\n    Message from file **"<<file<<"** in line **" <<line<<"**:\n    ";
}

Message( std::string m){ sstream_<<m;}
~Message(){}
template<class T>
Message & operator << (const T& value)
{
sstream_ << value;
return *this;
}
std::string str() const {return sstream_.str();}
friend std::ostream& operator<<(std::ostream& os, const Message& m)
{
os<<m.str();
return os;
}
};



class Error : public std::exception
{
private:
std::string m;
public:


Error(const Message& message){
m = message.str();
}

std::string get_message( ) const{return m;}


void append( const Message& message)
{
m+= message.str();
}

void append_line( const Message& message)
{
m+= "\n"+message.str();
}
virtual const char* what() const throw()
{
return m.c_str();
}
virtual ~Error() throw(){}
};



struct Fail : public Error
{


Fail( double eps): Fail(eps, Message("")){}


Fail( double eps, const Message& m): Error(
Message("\n    FAILED to converge to ")<< eps << "! "<<m),
eps( eps) {}

double epsilon() const { return eps;}
virtual ~Fail() throw(){}
private:
double eps;
};


static inline void abort_program(int code = -1){
#ifdef MPI_VERSION
MPI_Abort(MPI_COMM_WORLD, code);
#endif 
exit( code);
}


}
