

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <boost/wave.hpp>

#include <boost/wave/cpplexer/cpp_lex_token.hpp>    
#include <boost/wave/cpplexer/cpp_lex_iterator.hpp> 

#include "preprocess_pragma_output.hpp"

int main(int argc, char *argv[])
{
if (2 != argc) {
std::cerr << "Usage: preprocess_pragma_output infile" << std::endl;
return -1;
}

boost::wave::util::file_position_type current_position;

try {
std::ifstream instream(argv[1]);
std::string instring;

if (!instream.is_open()) {
std::cerr << "Could not open input file: " << argv[1] << std::endl;
return -2;
}
instream.unsetf(std::ios::skipws);
instring = std::string(std::istreambuf_iterator<char>(instream.rdbuf()),
std::istreambuf_iterator<char>());

typedef boost::wave::cpplexer::lex_token<> token_type;

typedef boost::wave::cpplexer::lex_iterator<token_type> lex_iterator_type;

typedef boost::wave::context<
std::string::iterator, lex_iterator_type,
boost::wave::iteration_context_policies::load_file_to_string,
preprocess_pragma_output_hooks>
context_type;

context_type ctx (instring.begin(), instring.end(), argv[1]);

context_type::iterator_type first = ctx.begin();
context_type::iterator_type last = ctx.end();

while (first != last) {
current_position = (*first).get_position();
std::cout << (*first).get_value();
++first;
}
}
catch (boost::wave::cpp_exception const& e) {
std::cerr 
<< e.file_name() << "(" << e.line_no() << "): "
<< e.description() << std::endl;
return 2;
}
catch (std::exception const& e) {
std::cerr 
<< current_position.get_file() 
<< "(" << current_position.get_line() << "): "
<< "exception caught: " << e.what()
<< std::endl;
return 3;
}
catch (...) {
std::cerr 
<< current_position.get_file() 
<< "(" << current_position.get_line() << "): "
<< "unexpected exception caught." << std::endl;
return 4;
}
return 0;
}
