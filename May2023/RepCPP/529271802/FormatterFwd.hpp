
#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "StringTools.hpp"

namespace CLI {

class Option;
class App;


enum class AppFormatMode {
Normal,  
All,     
Sub,     
};

class FormatterBase {
protected:

std::size_t column_width_{30};

std::map<std::string, std::string> labels_{};


public:
FormatterBase() = default;
FormatterBase(const FormatterBase &) = default;
FormatterBase(FormatterBase &&) = default;

virtual ~FormatterBase() noexcept {}  

virtual std::string make_help(const App *, std::string, AppFormatMode) const = 0;


void label(std::string key, std::string val) { labels_[key] = val; }

void column_width(std::size_t val) { column_width_ = val; }


std::string get_label(std::string key) const {
if(labels_.find(key) == labels_.end())
return key;
else
return labels_.at(key);
}

std::size_t get_column_width() const { return column_width_; }

};

class FormatterLambda final : public FormatterBase {
using funct_t = std::function<std::string(const App *, std::string, AppFormatMode)>;

funct_t lambda_;

public:
explicit FormatterLambda(funct_t funct) : lambda_(std::move(funct)) {}

~FormatterLambda() noexcept override {}  

std::string make_help(const App *app, std::string name, AppFormatMode mode) const override {
return lambda_(app, name, mode);
}
};

class Formatter : public FormatterBase {
public:
Formatter() = default;
Formatter(const Formatter &) = default;
Formatter(Formatter &&) = default;


virtual std::string make_group(std::string group, bool is_positional, std::vector<const Option *> opts) const;

virtual std::string make_positionals(const App *app) const;

std::string make_groups(const App *app, AppFormatMode mode) const;

virtual std::string make_subcommands(const App *app, AppFormatMode mode) const;

virtual std::string make_subcommand(const App *sub) const;

virtual std::string make_expanded(const App *sub) const;

virtual std::string make_footer(const App *app) const;

virtual std::string make_description(const App *app) const;

virtual std::string make_usage(const App *app, std::string name) const;

std::string make_help(const App * , std::string, AppFormatMode) const override;


virtual std::string make_option(const Option *opt, bool is_positional) const {
std::stringstream out;
detail::format_help(
out, make_option_name(opt, is_positional) + make_option_opts(opt), make_option_desc(opt), column_width_);
return out.str();
}

virtual std::string make_option_name(const Option *, bool) const;

virtual std::string make_option_opts(const Option *) const;

virtual std::string make_option_desc(const Option *) const;

virtual std::string make_option_usage(const Option *opt) const;

};

}  
