

#if !defined(BOOST_WAVE_SAMPLE_PREPROCESS_PRAGMA_OUTPUT_APR_03_2008_0813AM)
#define BOOST_WAVE_SAMPLE_PREPROCESS_PRAGMA_OUTPUT_APR_03_2008_0813AM

template <typename String, typename Iterator>
inline String
as_unescaped_string(Iterator it, Iterator const& end)
{
using namespace boost::wave;

String result;
for (; it != end; ++it) 
{
switch (token_id(*it)) {
case T_STRINGLIT:
{
String val (util::impl::unescape_lit((*it).get_value()).c_str());
val.erase(val.size()-1);
val.erase(0, 1);
result += val;
}
break;

default:    
break;
}
}
return result;
}

template <typename String, typename Container>
inline String
as_unescaped_string(Container const &token_sequence)
{
return as_unescaped_string<String>(token_sequence.begin(), 
token_sequence.end());
}

class preprocess_pragma_output_hooks
:   public boost::wave::context_policies::default_preprocessing_hooks
{
public:
preprocess_pragma_output_hooks() {}

template <typename Context>
struct reset_language_support
{
reset_language_support(Context& ctx)
: ctx_(ctx), lang_(ctx.get_language())
{
ctx.set_language(boost::wave::enable_single_line(lang_), false);
}
~reset_language_support()
{
ctx_.set_language(lang_, false);
}

Context& ctx_;
boost::wave::language_support lang_;
};

template <typename Context, typename Container>
bool 
interpret_pragma(Context& ctx, Container &pending, 
typename Context::token_type const& option, 
Container const& values, typename Context::token_type const& act_token)
{
typedef typename Context::iterator_type iterator_type;

if (option.get_value() == "pp")  {

try {
std::string s (as_unescaped_string<std::string>(values)); 
reset_language_support<Context> lang(ctx);

using namespace boost::wave;

Container pragma;
iterator_type end = ctx.end();
for (iterator_type it = ctx.begin(s.begin(), s.end()); 
it != end && token_id(*it) != T_EOF;
std::advance(it, 2))  
{
pragma.push_back(*it);
}

pending.splice(pending.begin(), pragma);
}
catch (boost::wave::preprocess_exception const& ) {
return false;
}
return true;
}

return false;   
}
};


#endif

