
#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ConfigFwd.hpp"
#include "Error.hpp"
#include "FormatterFwd.hpp"
#include "Macros.hpp"
#include "Option.hpp"
#include "Split.hpp"
#include "StringTools.hpp"
#include "TypeTools.hpp"

namespace CLI {

#ifndef CLI11_PARSE
#define CLI11_PARSE(app, argc, argv)                                                                                   \
try {                                                                                                              \
(app).parse((argc), (argv));                                                                                   \
} catch(const CLI::ParseError &e) {                                                                                \
return (app).exit(e);                                                                                          \
}
#endif

namespace detail {
enum class Classifier { NONE, POSITIONAL_MARK, SHORT, LONG, WINDOWS_STYLE, SUBCOMMAND, SUBCOMMAND_TERMINATOR };
struct AppFriend;
}  

namespace FailureMessage {
std::string simple(const App *app, const Error &e);
std::string help(const App *app, const Error &e);
}  


enum class config_extras_mode : char { error = 0, ignore, ignore_all, capture };

class App;

using App_p = std::shared_ptr<App>;

class Option_group;

class App {
friend Option;
friend detail::AppFriend;

protected:


std::string name_{};

std::string description_{};

bool allow_extras_{false};

config_extras_mode allow_config_extras_{config_extras_mode::ignore};

bool prefix_command_{false};

bool has_automatic_name_{false};

bool required_{false};

bool disabled_{false};

bool pre_parse_called_{false};

bool immediate_callback_{false};

std::function<void(std::size_t)> pre_parse_callback_{};

std::function<void()> parse_complete_callback_{};

std::function<void()> final_callback_{};


OptionDefaults option_defaults_{};

std::vector<Option_p> options_{};


std::string footer_{};

std::function<std::string()> footer_callback_{};

Option *help_ptr_{nullptr};

Option *help_all_ptr_{nullptr};

Option *version_ptr_{nullptr};

std::shared_ptr<FormatterBase> formatter_{new Formatter()};

std::function<std::string(const App *, const Error &e)> failure_message_{FailureMessage::simple};


using missing_t = std::vector<std::pair<detail::Classifier, std::string>>;

missing_t missing_{};

std::vector<Option *> parse_order_{};

std::vector<App *> parsed_subcommands_{};

std::set<App *> exclude_subcommands_{};

std::set<Option *> exclude_options_{};

std::set<App *> need_subcommands_{};

std::set<Option *> need_options_{};


std::vector<App_p> subcommands_{};

bool ignore_case_{false};

bool ignore_underscore_{false};

bool fallthrough_{false};

bool allow_windows_style_options_{
#ifdef _WIN32
true
#else
false
#endif
};
bool positionals_at_end_{false};

enum class startup_mode : char { stable, enabled, disabled };
startup_mode default_startup{startup_mode::stable};

bool configurable_{false};

bool validate_positionals_{false};

bool silent_{false};

std::uint32_t parsed_{0U};

std::size_t require_subcommand_min_{0};

std::size_t require_subcommand_max_{0};

std::size_t require_option_min_{0};

std::size_t require_option_max_{0};

App *parent_{nullptr};

std::string group_{"Subcommands"};

std::vector<std::string> aliases_{};


Option *config_ptr_{nullptr};

std::shared_ptr<Config> config_formatter_{new ConfigTOML()};


App(std::string app_description, std::string app_name, App *parent)
: name_(std::move(app_name)), description_(std::move(app_description)), parent_(parent) {
if(parent_ != nullptr) {
if(parent_->help_ptr_ != nullptr)
set_help_flag(parent_->help_ptr_->get_name(false, true), parent_->help_ptr_->get_description());
if(parent_->help_all_ptr_ != nullptr)
set_help_all_flag(parent_->help_all_ptr_->get_name(false, true),
parent_->help_all_ptr_->get_description());

option_defaults_ = parent_->option_defaults_;

failure_message_ = parent_->failure_message_;
allow_extras_ = parent_->allow_extras_;
allow_config_extras_ = parent_->allow_config_extras_;
prefix_command_ = parent_->prefix_command_;
immediate_callback_ = parent_->immediate_callback_;
ignore_case_ = parent_->ignore_case_;
ignore_underscore_ = parent_->ignore_underscore_;
fallthrough_ = parent_->fallthrough_;
validate_positionals_ = parent_->validate_positionals_;
configurable_ = parent_->configurable_;
allow_windows_style_options_ = parent_->allow_windows_style_options_;
group_ = parent_->group_;
footer_ = parent_->footer_;
formatter_ = parent_->formatter_;
config_formatter_ = parent_->config_formatter_;
require_subcommand_max_ = parent_->require_subcommand_max_;
}
}

public:

explicit App(std::string app_description = "", std::string app_name = "")
: App(app_description, app_name, nullptr) {
set_help_flag("-h,--help", "Print this help message and exit");
}

App(const App &) = delete;
App &operator=(const App &) = delete;

virtual ~App() = default;

App *callback(std::function<void()> app_callback) {
if(immediate_callback_) {
parse_complete_callback_ = std::move(app_callback);
} else {
final_callback_ = std::move(app_callback);
}
return this;
}

App *final_callback(std::function<void()> app_callback) {
final_callback_ = std::move(app_callback);
return this;
}

App *parse_complete_callback(std::function<void()> pc_callback) {
parse_complete_callback_ = std::move(pc_callback);
return this;
}

App *preparse_callback(std::function<void(std::size_t)> pp_callback) {
pre_parse_callback_ = std::move(pp_callback);
return this;
}

App *name(std::string app_name = "") {

if(parent_ != nullptr) {
auto oname = name_;
name_ = app_name;
auto &res = _compare_subcommand_names(*this, *_get_fallthrough_parent());
if(!res.empty()) {
name_ = oname;
throw(OptionAlreadyAdded(app_name + " conflicts with existing subcommand names"));
}
} else {
name_ = app_name;
}
has_automatic_name_ = false;
return this;
}

App *alias(std::string app_name) {
if(app_name.empty() || !detail::valid_alias_name_string(app_name)) {
throw IncorrectConstruction("Aliases may not be empty or contain newlines or null characters");
}
if(parent_ != nullptr) {
aliases_.push_back(app_name);
auto &res = _compare_subcommand_names(*this, *_get_fallthrough_parent());
if(!res.empty()) {
aliases_.pop_back();
throw(OptionAlreadyAdded("alias already matches an existing subcommand: " + app_name));
}
} else {
aliases_.push_back(app_name);
}

return this;
}

App *allow_extras(bool allow = true) {
allow_extras_ = allow;
return this;
}

App *required(bool require = true) {
required_ = require;
return this;
}

App *disabled(bool disable = true) {
disabled_ = disable;
return this;
}

App *silent(bool silence = true) {
silent_ = silence;
return this;
}

App *disabled_by_default(bool disable = true) {
if(disable) {
default_startup = startup_mode::disabled;
} else {
default_startup = (default_startup == startup_mode::enabled) ? startup_mode::enabled : startup_mode::stable;
}
return this;
}

App *enabled_by_default(bool enable = true) {
if(enable) {
default_startup = startup_mode::enabled;
} else {
default_startup =
(default_startup == startup_mode::disabled) ? startup_mode::disabled : startup_mode::stable;
}
return this;
}

App *immediate_callback(bool immediate = true) {
immediate_callback_ = immediate;
if(immediate_callback_) {
if(final_callback_ && !(parse_complete_callback_)) {
std::swap(final_callback_, parse_complete_callback_);
}
} else if(!(final_callback_) && parse_complete_callback_) {
std::swap(final_callback_, parse_complete_callback_);
}
return this;
}

App *validate_positionals(bool validate = true) {
validate_positionals_ = validate;
return this;
}

App *allow_config_extras(bool allow = true) {
if(allow) {
allow_config_extras_ = config_extras_mode::capture;
allow_extras_ = true;
} else {
allow_config_extras_ = config_extras_mode::error;
}
return this;
}

App *allow_config_extras(config_extras_mode mode) {
allow_config_extras_ = mode;
return this;
}

App *prefix_command(bool allow = true) {
prefix_command_ = allow;
return this;
}

App *ignore_case(bool value = true) {
if(value && !ignore_case_) {
ignore_case_ = true;
auto *p = (parent_ != nullptr) ? _get_fallthrough_parent() : this;
auto &match = _compare_subcommand_names(*this, *p);
if(!match.empty()) {
ignore_case_ = false;  
throw OptionAlreadyAdded("ignore case would cause subcommand name conflicts: " + match);
}
}
ignore_case_ = value;
return this;
}

App *allow_windows_style_options(bool value = true) {
allow_windows_style_options_ = value;
return this;
}

App *positionals_at_end(bool value = true) {
positionals_at_end_ = value;
return this;
}

App *configurable(bool value = true) {
configurable_ = value;
return this;
}

App *ignore_underscore(bool value = true) {
if(value && !ignore_underscore_) {
ignore_underscore_ = true;
auto *p = (parent_ != nullptr) ? _get_fallthrough_parent() : this;
auto &match = _compare_subcommand_names(*this, *p);
if(!match.empty()) {
ignore_underscore_ = false;
throw OptionAlreadyAdded("ignore underscore would cause subcommand name conflicts: " + match);
}
}
ignore_underscore_ = value;
return this;
}

App *formatter(std::shared_ptr<FormatterBase> fmt) {
formatter_ = fmt;
return this;
}

App *formatter_fn(std::function<std::string(const App *, std::string, AppFormatMode)> fmt) {
formatter_ = std::make_shared<FormatterLambda>(fmt);
return this;
}

App *config_formatter(std::shared_ptr<Config> fmt) {
config_formatter_ = fmt;
return this;
}

bool parsed() const { return parsed_ > 0; }

OptionDefaults *option_defaults() { return &option_defaults_; }


Option *add_option(std::string option_name,
callback_t option_callback,
std::string option_description = "",
bool defaulted = false,
std::function<std::string()> func = {}) {
Option myopt{option_name, option_description, option_callback, this};

if(std::find_if(std::begin(options_), std::end(options_), [&myopt](const Option_p &v) {
return *v == myopt;
}) == std::end(options_)) {
options_.emplace_back();
Option_p &option = options_.back();
option.reset(new Option(option_name, option_description, option_callback, this));

option->default_function(func);

if(defaulted)
option->capture_default_str();

option_defaults_.copy_to(option.get());

if(!defaulted && option->get_always_capture_default())
option->capture_default_str();

return option.get();
}
for(auto &opt : options_) {
auto &matchname = opt->matching_name(myopt);
if(!matchname.empty()) {
throw(OptionAlreadyAdded("added option matched existing option name: " + matchname));
}
}
throw(OptionAlreadyAdded("added option matched existing option name"));  
}

template <typename AssignTo,
typename ConvertTo = AssignTo,
enable_if_t<!std::is_const<ConvertTo>::value, detail::enabler> = detail::dummy>
Option *add_option(std::string option_name,
AssignTo &variable,  
std::string option_description = "") {

auto fun = [&variable](const CLI::results_t &res) {  
return detail::lexical_conversion<AssignTo, ConvertTo>(res, variable);
};

Option *opt = add_option(option_name, fun, option_description, false, [&variable]() {
return CLI::detail::checked_to_string<AssignTo, ConvertTo>(variable);
});
opt->type_name(detail::type_name<ConvertTo>());
auto Tcount = detail::type_count<AssignTo>::value;
auto XCcount = detail::type_count<ConvertTo>::value;
opt->type_size(detail::type_count_min<ConvertTo>::value, (std::max)(Tcount, XCcount));
opt->expected(detail::expected_count<ConvertTo>::value);
opt->run_callback_for_default();
return opt;
}

template <typename AssignTo, enable_if_t<!std::is_const<AssignTo>::value, detail::enabler> = detail::dummy>
Option *add_option_no_stream(std::string option_name,
AssignTo &variable,  
std::string option_description = "") {

auto fun = [&variable](const CLI::results_t &res) {  
return detail::lexical_conversion<AssignTo, AssignTo>(res, variable);
};

Option *opt = add_option(option_name, fun, option_description, false, []() { return std::string{}; });
opt->type_name(detail::type_name<AssignTo>());
opt->type_size(detail::type_count_min<AssignTo>::value, detail::type_count<AssignTo>::value);
opt->expected(detail::expected_count<AssignTo>::value);
opt->run_callback_for_default();
return opt;
}

template <typename ArgType>
Option *add_option_function(std::string option_name,
const std::function<void(const ArgType &)> &func,  
std::string option_description = "") {

auto fun = [func](const CLI::results_t &res) {
ArgType variable;
bool result = detail::lexical_conversion<ArgType, ArgType>(res, variable);
if(result) {
func(variable);
}
return result;
};

Option *opt = add_option(option_name, std::move(fun), option_description, false);
opt->type_name(detail::type_name<ArgType>());
opt->type_size(detail::type_count_min<ArgType>::value, detail::type_count<ArgType>::value);
opt->expected(detail::expected_count<ArgType>::value);
return opt;
}

Option *add_option(std::string option_name) {
return add_option(option_name, CLI::callback_t{}, std::string{}, false);
}

template <typename T,
enable_if_t<std::is_const<T>::value && std::is_constructible<std::string, T>::value, detail::enabler> =
detail::dummy>
Option *add_option(std::string option_name, T &option_description) {
return add_option(option_name, CLI::callback_t(), option_description, false);
}

Option *set_help_flag(std::string flag_name = "", const std::string &help_description = "") {
if(help_ptr_ != nullptr) {
remove_option(help_ptr_);
help_ptr_ = nullptr;
}

if(!flag_name.empty()) {
help_ptr_ = add_flag(flag_name, help_description);
help_ptr_->configurable(false);
}

return help_ptr_;
}

Option *set_help_all_flag(std::string help_name = "", const std::string &help_description = "") {
if(help_all_ptr_ != nullptr) {
remove_option(help_all_ptr_);
help_all_ptr_ = nullptr;
}

if(!help_name.empty()) {
help_all_ptr_ = add_flag(help_name, help_description);
help_all_ptr_->configurable(false);
}

return help_all_ptr_;
}

Option *set_version_flag(std::string flag_name = "",
const std::string &versionString = "",
const std::string &version_help = "Display program version information and exit") {
if(version_ptr_ != nullptr) {
remove_option(version_ptr_);
version_ptr_ = nullptr;
}

if(!flag_name.empty()) {
version_ptr_ = add_flag_callback(
flag_name, [versionString]() { throw(CLI::CallForVersion(versionString, 0)); }, version_help);
version_ptr_->configurable(false);
}

return version_ptr_;
}
Option *set_version_flag(std::string flag_name,
std::function<std::string()> vfunc,
const std::string &version_help = "Display program version information and exit") {
if(version_ptr_ != nullptr) {
remove_option(version_ptr_);
version_ptr_ = nullptr;
}

if(!flag_name.empty()) {
version_ptr_ = add_flag_callback(
flag_name, [vfunc]() { throw(CLI::CallForVersion(vfunc(), 0)); }, version_help);
version_ptr_->configurable(false);
}

return version_ptr_;
}

private:
Option *_add_flag_internal(std::string flag_name, CLI::callback_t fun, std::string flag_description) {
Option *opt;
if(detail::has_default_flag_values(flag_name)) {
auto flag_defaults = detail::get_default_flag_values(flag_name);
detail::remove_default_flag_values(flag_name);
opt = add_option(std::move(flag_name), std::move(fun), std::move(flag_description), false);
for(const auto &fname : flag_defaults)
opt->fnames_.push_back(fname.first);
opt->default_flag_values_ = std::move(flag_defaults);
} else {
opt = add_option(std::move(flag_name), std::move(fun), std::move(flag_description), false);
}
if(opt->get_positional()) {
auto pos_name = opt->get_name(true);
remove_option(opt);
throw IncorrectConstruction::PositionalFlag(pos_name);
}
opt->multi_option_policy(MultiOptionPolicy::TakeLast);
opt->expected(0);
opt->required(false);
return opt;
}

public:
Option *add_flag(std::string flag_name) { return _add_flag_internal(flag_name, CLI::callback_t(), std::string{}); }

template <typename T,
enable_if_t<std::is_const<T>::value && std::is_constructible<std::string, T>::value, detail::enabler> =
detail::dummy>
Option *add_flag(std::string flag_name, T &flag_description) {
return _add_flag_internal(flag_name, CLI::callback_t(), flag_description);
}

template <typename T,
enable_if_t<std::is_constructible<T, std::int64_t>::value && !is_bool<T>::value, detail::enabler> =
detail::dummy>
Option *add_flag(std::string flag_name,
T &flag_count,  
std::string flag_description = "") {
flag_count = 0;
CLI::callback_t fun = [&flag_count](const CLI::results_t &res) {
try {
detail::sum_flag_vector(res, flag_count);
} catch(const std::invalid_argument &) {
return false;
}
return true;
};
return _add_flag_internal(flag_name, std::move(fun), std::move(flag_description))
->multi_option_policy(MultiOptionPolicy::TakeAll);
}

template <typename T,
enable_if_t<!detail::is_mutable_container<T>::value && !std::is_const<T>::value &&
(!std::is_constructible<T, std::int64_t>::value || is_bool<T>::value) &&
!std::is_constructible<std::function<void(int)>, T>::value,
detail::enabler> = detail::dummy>
Option *add_flag(std::string flag_name,
T &flag_result,  
std::string flag_description = "") {

CLI::callback_t fun = [&flag_result](const CLI::results_t &res) {
return CLI::detail::lexical_cast(res[0], flag_result);
};
return _add_flag_internal(flag_name, std::move(fun), std::move(flag_description))->run_callback_for_default();
}

template <typename T,
enable_if_t<!std::is_assignable<std::function<void(std::int64_t)> &, T>::value, detail::enabler> =
detail::dummy>
Option *add_flag(std::string flag_name,
std::vector<T> &flag_results,  
std::string flag_description = "") {
CLI::callback_t fun = [&flag_results](const CLI::results_t &res) {
bool retval = true;
for(const auto &elem : res) {
flag_results.emplace_back();
retval &= detail::lexical_cast(elem, flag_results.back());
}
return retval;
};
return _add_flag_internal(flag_name, std::move(fun), std::move(flag_description))
->multi_option_policy(MultiOptionPolicy::TakeAll)
->run_callback_for_default();
}

Option *add_flag_callback(std::string flag_name,
std::function<void(void)> function,  
std::string flag_description = "") {

CLI::callback_t fun = [function](const CLI::results_t &res) {
bool trigger{false};
auto result = CLI::detail::lexical_cast(res[0], trigger);
if(result && trigger) {
function();
}
return result;
};
return _add_flag_internal(flag_name, std::move(fun), std::move(flag_description));
}

Option *add_flag_function(std::string flag_name,
std::function<void(std::int64_t)> function,  
std::string flag_description = "") {

CLI::callback_t fun = [function](const CLI::results_t &res) {
std::int64_t flag_count = 0;
detail::sum_flag_vector(res, flag_count);
function(flag_count);
return true;
};
return _add_flag_internal(flag_name, std::move(fun), std::move(flag_description))
->multi_option_policy(MultiOptionPolicy::TakeAll);
}

#ifdef CLI11_CPP14
Option *add_flag(std::string flag_name,
std::function<void(std::int64_t)> function,  
std::string flag_description = "") {
return add_flag_function(std::move(flag_name), std::move(function), std::move(flag_description));
}
#endif

Option *set_config(std::string option_name = "",
std::string default_filename = "",
const std::string &help_message = "Read an ini file",
bool config_required = false) {

if(config_ptr_ != nullptr) {
remove_option(config_ptr_);
config_ptr_ = nullptr;  
}

if(!option_name.empty()) {
config_ptr_ = add_option(option_name, help_message);
if(config_required) {
config_ptr_->required();
}
if(!default_filename.empty()) {
config_ptr_->default_str(std::move(default_filename));
}
config_ptr_->configurable(false);
}

return config_ptr_;
}

bool remove_option(Option *opt) {
for(Option_p &op : options_) {
op->remove_needs(opt);
op->remove_excludes(opt);
}

if(help_ptr_ == opt)
help_ptr_ = nullptr;
if(help_all_ptr_ == opt)
help_all_ptr_ = nullptr;

auto iterator =
std::find_if(std::begin(options_), std::end(options_), [opt](const Option_p &v) { return v.get() == opt; });
if(iterator != std::end(options_)) {
options_.erase(iterator);
return true;
}
return false;
}

template <typename T = Option_group>
T *add_option_group(std::string group_name, std::string group_description = "") {
if(!detail::valid_alias_name_string(group_name)) {
throw IncorrectConstruction("option group names may not contain newlines or null characters");
}
auto option_group = std::make_shared<T>(std::move(group_description), group_name, this);
auto ptr = option_group.get();
App_p app_ptr = std::dynamic_pointer_cast<App>(option_group);
add_subcommand(std::move(app_ptr));
return ptr;
}


App *add_subcommand(std::string subcommand_name = "", std::string subcommand_description = "") {
if(!subcommand_name.empty() && !detail::valid_name_string(subcommand_name)) {
if(!detail::valid_first_char(subcommand_name[0])) {
throw IncorrectConstruction(
"Subcommand name starts with invalid character, '!' and '-' are not allowed");
}
for(auto c : subcommand_name) {
if(!detail::valid_later_char(c)) {
throw IncorrectConstruction(std::string("Subcommand name contains invalid character ('") + c +
"'), all characters are allowed except"
"'=',':','{','}', and ' '");
}
}
}
CLI::App_p subcom = std::shared_ptr<App>(new App(std::move(subcommand_description), subcommand_name, this));
return add_subcommand(std::move(subcom));
}

App *add_subcommand(CLI::App_p subcom) {
if(!subcom)
throw IncorrectConstruction("passed App is not valid");
auto ckapp = (name_.empty() && parent_ != nullptr) ? _get_fallthrough_parent() : this;
auto &mstrg = _compare_subcommand_names(*subcom, *ckapp);
if(!mstrg.empty()) {
throw(OptionAlreadyAdded("subcommand name or alias matches existing subcommand: " + mstrg));
}
subcom->parent_ = this;
subcommands_.push_back(std::move(subcom));
return subcommands_.back().get();
}

bool remove_subcommand(App *subcom) {
for(App_p &sub : subcommands_) {
sub->remove_excludes(subcom);
sub->remove_needs(subcom);
}

auto iterator = std::find_if(
std::begin(subcommands_), std::end(subcommands_), [subcom](const App_p &v) { return v.get() == subcom; });
if(iterator != std::end(subcommands_)) {
subcommands_.erase(iterator);
return true;
}
return false;
}
App *get_subcommand(const App *subcom) const {
if(subcom == nullptr)
throw OptionNotFound("nullptr passed");
for(const App_p &subcomptr : subcommands_)
if(subcomptr.get() == subcom)
return subcomptr.get();
throw OptionNotFound(subcom->get_name());
}

App *get_subcommand(std::string subcom) const {
auto subc = _find_subcommand(subcom, false, false);
if(subc == nullptr)
throw OptionNotFound(subcom);
return subc;
}
App *get_subcommand(int index = 0) const {
if(index >= 0) {
auto uindex = static_cast<unsigned>(index);
if(uindex < subcommands_.size())
return subcommands_[uindex].get();
}
throw OptionNotFound(std::to_string(index));
}

CLI::App_p get_subcommand_ptr(App *subcom) const {
if(subcom == nullptr)
throw OptionNotFound("nullptr passed");
for(const App_p &subcomptr : subcommands_)
if(subcomptr.get() == subcom)
return subcomptr;
throw OptionNotFound(subcom->get_name());
}

CLI::App_p get_subcommand_ptr(std::string subcom) const {
for(const App_p &subcomptr : subcommands_)
if(subcomptr->check_name(subcom))
return subcomptr;
throw OptionNotFound(subcom);
}

CLI::App_p get_subcommand_ptr(int index = 0) const {
if(index >= 0) {
auto uindex = static_cast<unsigned>(index);
if(uindex < subcommands_.size())
return subcommands_[uindex];
}
throw OptionNotFound(std::to_string(index));
}

App *get_option_group(std::string group_name) const {
for(const App_p &app : subcommands_) {
if(app->name_.empty() && app->group_ == group_name) {
return app.get();
}
}
throw OptionNotFound(group_name);
}

std::size_t count() const { return parsed_; }

std::size_t count_all() const {
std::size_t cnt{0};
for(auto &opt : options_) {
cnt += opt->count();
}
for(auto &sub : subcommands_) {
cnt += sub->count_all();
}
if(!get_name().empty()) {  
cnt += parsed_;
}
return cnt;
}

App *group(std::string group_name) {
group_ = group_name;
return this;
}

App *require_subcommand() {
require_subcommand_min_ = 1;
require_subcommand_max_ = 0;
return this;
}

App *require_subcommand(int value) {
if(value < 0) {
require_subcommand_min_ = 0;
require_subcommand_max_ = static_cast<std::size_t>(-value);
} else {
require_subcommand_min_ = static_cast<std::size_t>(value);
require_subcommand_max_ = static_cast<std::size_t>(value);
}
return this;
}

App *require_subcommand(std::size_t min, std::size_t max) {
require_subcommand_min_ = min;
require_subcommand_max_ = max;
return this;
}

App *require_option() {
require_option_min_ = 1;
require_option_max_ = 0;
return this;
}

App *require_option(int value) {
if(value < 0) {
require_option_min_ = 0;
require_option_max_ = static_cast<std::size_t>(-value);
} else {
require_option_min_ = static_cast<std::size_t>(value);
require_option_max_ = static_cast<std::size_t>(value);
}
return this;
}

App *require_option(std::size_t min, std::size_t max) {
require_option_min_ = min;
require_option_max_ = max;
return this;
}

App *fallthrough(bool value = true) {
fallthrough_ = value;
return this;
}

explicit operator bool() const { return parsed_ > 0; }


virtual void pre_callback() {}

void clear() {

parsed_ = 0;
pre_parse_called_ = false;

missing_.clear();
parsed_subcommands_.clear();
for(const Option_p &opt : options_) {
opt->clear();
}
for(const App_p &subc : subcommands_) {
subc->clear();
}
}

void parse(int argc, const char *const *argv) {
if(name_.empty() || has_automatic_name_) {
has_automatic_name_ = true;
name_ = argv[0];
}

std::vector<std::string> args;
args.reserve(static_cast<std::size_t>(argc) - 1);
for(int i = argc - 1; i > 0; i--)
args.emplace_back(argv[i]);
parse(std::move(args));
}

void parse(std::string commandline, bool program_name_included = false) {

if(program_name_included) {
auto nstr = detail::split_program_name(commandline);
if((name_.empty()) || (has_automatic_name_)) {
has_automatic_name_ = true;
name_ = nstr.first;
}
commandline = std::move(nstr.second);
} else {
detail::trim(commandline);
}
if(!commandline.empty()) {
commandline = detail::find_and_modify(commandline, "=", detail::escape_detect);
if(allow_windows_style_options_)
commandline = detail::find_and_modify(commandline, ":", detail::escape_detect);
}

auto args = detail::split_up(std::move(commandline));
args.erase(std::remove(args.begin(), args.end(), std::string{}), args.end());
std::reverse(args.begin(), args.end());

parse(std::move(args));
}

void parse(std::vector<std::string> &args) {
if(parsed_ > 0)
clear();

parsed_ = 1;
_validate();
_configure();
parent_ = nullptr;
parsed_ = 0;

_parse(args);
run_callback();
}

void parse(std::vector<std::string> &&args) {
if(parsed_ > 0)
clear();

parsed_ = 1;
_validate();
_configure();
parent_ = nullptr;
parsed_ = 0;

_parse(std::move(args));
run_callback();
}

void parse_from_stream(std::istream &input) {
if(parsed_ == 0) {
_validate();
_configure();
}

_parse_stream(input);
run_callback();
}
void failure_message(std::function<std::string(const App *, const Error &e)> function) {
failure_message_ = function;
}

int exit(const Error &e, std::ostream &out = std::cout, std::ostream &err = std::cerr) const {

if(e.get_name() == "RuntimeError")
return e.get_exit_code();

if(e.get_name() == "CallForHelp") {
out << help();
return e.get_exit_code();
}

if(e.get_name() == "CallForAllHelp") {
out << help("", AppFormatMode::All);
return e.get_exit_code();
}

if(e.get_name() == "CallForVersion") {
out << e.what() << std::endl;
return e.get_exit_code();
}

if(e.get_exit_code() != static_cast<int>(ExitCodes::Success)) {
if(failure_message_)
err << failure_message_(this, e) << std::flush;
}

return e.get_exit_code();
}


std::size_t count(std::string option_name) const { return get_option(option_name)->count(); }

std::vector<App *> get_subcommands() const { return parsed_subcommands_; }

std::vector<const App *> get_subcommands(const std::function<bool(const App *)> &filter) const {
std::vector<const App *> subcomms(subcommands_.size());
std::transform(std::begin(subcommands_), std::end(subcommands_), std::begin(subcomms), [](const App_p &v) {
return v.get();
});

if(filter) {
subcomms.erase(std::remove_if(std::begin(subcomms),
std::end(subcomms),
[&filter](const App *app) { return !filter(app); }),
std::end(subcomms));
}

return subcomms;
}

std::vector<App *> get_subcommands(const std::function<bool(App *)> &filter) {
std::vector<App *> subcomms(subcommands_.size());
std::transform(std::begin(subcommands_), std::end(subcommands_), std::begin(subcomms), [](const App_p &v) {
return v.get();
});

if(filter) {
subcomms.erase(
std::remove_if(std::begin(subcomms), std::end(subcomms), [&filter](App *app) { return !filter(app); }),
std::end(subcomms));
}

return subcomms;
}

bool got_subcommand(const App *subcom) const {
return get_subcommand(subcom)->parsed_ > 0;
}

bool got_subcommand(std::string subcommand_name) const { return get_subcommand(subcommand_name)->parsed_ > 0; }

App *excludes(Option *opt) {
if(opt == nullptr) {
throw OptionNotFound("nullptr passed");
}
exclude_options_.insert(opt);
return this;
}

App *excludes(App *app) {
if(app == nullptr) {
throw OptionNotFound("nullptr passed");
}
if(app == this) {
throw OptionNotFound("cannot self reference in needs");
}
auto res = exclude_subcommands_.insert(app);
if(res.second) {
app->exclude_subcommands_.insert(this);
}
return this;
}

App *needs(Option *opt) {
if(opt == nullptr) {
throw OptionNotFound("nullptr passed");
}
need_options_.insert(opt);
return this;
}

App *needs(App *app) {
if(app == nullptr) {
throw OptionNotFound("nullptr passed");
}
if(app == this) {
throw OptionNotFound("cannot self reference in needs");
}
need_subcommands_.insert(app);
return this;
}

bool remove_excludes(Option *opt) {
auto iterator = std::find(std::begin(exclude_options_), std::end(exclude_options_), opt);
if(iterator == std::end(exclude_options_)) {
return false;
}
exclude_options_.erase(iterator);
return true;
}

bool remove_excludes(App *app) {
auto iterator = std::find(std::begin(exclude_subcommands_), std::end(exclude_subcommands_), app);
if(iterator == std::end(exclude_subcommands_)) {
return false;
}
auto other_app = *iterator;
exclude_subcommands_.erase(iterator);
other_app->remove_excludes(this);
return true;
}

bool remove_needs(Option *opt) {
auto iterator = std::find(std::begin(need_options_), std::end(need_options_), opt);
if(iterator == std::end(need_options_)) {
return false;
}
need_options_.erase(iterator);
return true;
}

bool remove_needs(App *app) {
auto iterator = std::find(std::begin(need_subcommands_), std::end(need_subcommands_), app);
if(iterator == std::end(need_subcommands_)) {
return false;
}
need_subcommands_.erase(iterator);
return true;
}


App *footer(std::string footer_string) {
footer_ = std::move(footer_string);
return this;
}
App *footer(std::function<std::string()> footer_function) {
footer_callback_ = std::move(footer_function);
return this;
}
std::string config_to_str(bool default_also = false, bool write_description = false) const {
return config_formatter_->to_config(this, default_also, write_description, "");
}

std::string help(std::string prev = "", AppFormatMode mode = AppFormatMode::Normal) const {
if(prev.empty())
prev = get_name();
else
prev += " " + get_name();

auto selected_subcommands = get_subcommands();
if(!selected_subcommands.empty()) {
return selected_subcommands.at(0)->help(prev, mode);
}
return formatter_->make_help(this, prev, mode);
}

std::string version() const {
std::string val;
if(version_ptr_ != nullptr) {
auto rv = version_ptr_->results();
version_ptr_->clear();
version_ptr_->add_result("true");
try {
version_ptr_->run_callback();
} catch(const CLI::CallForVersion &cfv) {
val = cfv.what();
}
version_ptr_->clear();
version_ptr_->add_result(rv);
}
return val;
}

std::shared_ptr<FormatterBase> get_formatter() const { return formatter_; }

std::shared_ptr<Config> get_config_formatter() const { return config_formatter_; }

std::shared_ptr<ConfigBase> get_config_formatter_base() const {
#if defined(__cpp_rtti) || (defined(__GXX_RTTI) && __GXX_RTTI) || (defined(_HAS_STATIC_RTTI) && (_HAS_STATIC_RTTI == 0))
return std::dynamic_pointer_cast<ConfigBase>(config_formatter_);
#else
return std::static_pointer_cast<ConfigBase>(config_formatter_);
#endif
}

std::string get_description() const { return description_; }

App *description(std::string app_description) {
description_ = std::move(app_description);
return this;
}

std::vector<const Option *> get_options(const std::function<bool(const Option *)> filter = {}) const {
std::vector<const Option *> options(options_.size());
std::transform(std::begin(options_), std::end(options_), std::begin(options), [](const Option_p &val) {
return val.get();
});

if(filter) {
options.erase(std::remove_if(std::begin(options),
std::end(options),
[&filter](const Option *opt) { return !filter(opt); }),
std::end(options));
}

return options;
}

std::vector<Option *> get_options(const std::function<bool(Option *)> filter = {}) {
std::vector<Option *> options(options_.size());
std::transform(std::begin(options_), std::end(options_), std::begin(options), [](const Option_p &val) {
return val.get();
});

if(filter) {
options.erase(
std::remove_if(std::begin(options), std::end(options), [&filter](Option *opt) { return !filter(opt); }),
std::end(options));
}

return options;
}

Option *get_option_no_throw(std::string option_name) noexcept {
for(Option_p &opt : options_) {
if(opt->check_name(option_name)) {
return opt.get();
}
}
for(auto &subc : subcommands_) {
if(subc->get_name().empty()) {
auto opt = subc->get_option_no_throw(option_name);
if(opt != nullptr) {
return opt;
}
}
}
return nullptr;
}

const Option *get_option_no_throw(std::string option_name) const noexcept {
for(const Option_p &opt : options_) {
if(opt->check_name(option_name)) {
return opt.get();
}
}
for(const auto &subc : subcommands_) {
if(subc->get_name().empty()) {
auto opt = subc->get_option_no_throw(option_name);
if(opt != nullptr) {
return opt;
}
}
}
return nullptr;
}

const Option *get_option(std::string option_name) const {
auto opt = get_option_no_throw(option_name);
if(opt == nullptr) {
throw OptionNotFound(option_name);
}
return opt;
}

Option *get_option(std::string option_name) {
auto opt = get_option_no_throw(option_name);
if(opt == nullptr) {
throw OptionNotFound(option_name);
}
return opt;
}

const Option *operator[](const std::string &option_name) const { return get_option(option_name); }

const Option *operator[](const char *option_name) const { return get_option(option_name); }

bool get_ignore_case() const { return ignore_case_; }

bool get_ignore_underscore() const { return ignore_underscore_; }

bool get_fallthrough() const { return fallthrough_; }

bool get_allow_windows_style_options() const { return allow_windows_style_options_; }

bool get_positionals_at_end() const { return positionals_at_end_; }

bool get_configurable() const { return configurable_; }

const std::string &get_group() const { return group_; }

std::string get_footer() const { return (footer_callback_) ? footer_callback_() + '\n' + footer_ : footer_; }

std::size_t get_require_subcommand_min() const { return require_subcommand_min_; }

std::size_t get_require_subcommand_max() const { return require_subcommand_max_; }

std::size_t get_require_option_min() const { return require_option_min_; }

std::size_t get_require_option_max() const { return require_option_max_; }

bool get_prefix_command() const { return prefix_command_; }

bool get_allow_extras() const { return allow_extras_; }

bool get_required() const { return required_; }

bool get_disabled() const { return disabled_; }

bool get_silent() const { return silent_; }

bool get_immediate_callback() const { return immediate_callback_; }

bool get_disabled_by_default() const { return (default_startup == startup_mode::disabled); }

bool get_enabled_by_default() const { return (default_startup == startup_mode::enabled); }
bool get_validate_positionals() const { return validate_positionals_; }

config_extras_mode get_allow_config_extras() const { return allow_config_extras_; }

Option *get_help_ptr() { return help_ptr_; }

const Option *get_help_ptr() const { return help_ptr_; }

const Option *get_help_all_ptr() const { return help_all_ptr_; }

Option *get_config_ptr() { return config_ptr_; }

const Option *get_config_ptr() const { return config_ptr_; }

Option *get_version_ptr() { return version_ptr_; }

const Option *get_version_ptr() const { return version_ptr_; }

App *get_parent() { return parent_; }

const App *get_parent() const { return parent_; }

const std::string &get_name() const { return name_; }

const std::vector<std::string> &get_aliases() const { return aliases_; }

App *clear_aliases() {
aliases_.clear();
return this;
}

std::string get_display_name(bool with_aliases = false) const {
if(name_.empty()) {
return std::string("[Option Group: ") + get_group() + "]";
}
if(aliases_.empty() || !with_aliases) {
return name_;
}
std::string dispname = name_;
for(const auto &lalias : aliases_) {
dispname.push_back(',');
dispname.push_back(' ');
dispname.append(lalias);
}
return dispname;
}

bool check_name(std::string name_to_check) const {
std::string local_name = name_;
if(ignore_underscore_) {
local_name = detail::remove_underscore(name_);
name_to_check = detail::remove_underscore(name_to_check);
}
if(ignore_case_) {
local_name = detail::to_lower(name_);
name_to_check = detail::to_lower(name_to_check);
}

if(local_name == name_to_check) {
return true;
}
for(auto les : aliases_) {
if(ignore_underscore_) {
les = detail::remove_underscore(les);
}
if(ignore_case_) {
les = detail::to_lower(les);
}
if(les == name_to_check) {
return true;
}
}
return false;
}

std::vector<std::string> get_groups() const {
std::vector<std::string> groups;

for(const Option_p &opt : options_) {
if(std::find(groups.begin(), groups.end(), opt->get_group()) == groups.end()) {
groups.push_back(opt->get_group());
}
}

return groups;
}

const std::vector<Option *> &parse_order() const { return parse_order_; }

std::vector<std::string> remaining(bool recurse = false) const {
std::vector<std::string> miss_list;
for(const std::pair<detail::Classifier, std::string> &miss : missing_) {
miss_list.push_back(std::get<1>(miss));
}
if(recurse) {
if(!allow_extras_) {
for(const auto &sub : subcommands_) {
if(sub->name_.empty() && !sub->missing_.empty()) {
for(const std::pair<detail::Classifier, std::string> &miss : sub->missing_) {
miss_list.push_back(std::get<1>(miss));
}
}
}
}

for(const App *sub : parsed_subcommands_) {
std::vector<std::string> output = sub->remaining(recurse);
std::copy(std::begin(output), std::end(output), std::back_inserter(miss_list));
}
}
return miss_list;
}

std::vector<std::string> remaining_for_passthrough(bool recurse = false) const {
std::vector<std::string> miss_list = remaining(recurse);
std::reverse(std::begin(miss_list), std::end(miss_list));
return miss_list;
}

std::size_t remaining_size(bool recurse = false) const {
auto remaining_options = static_cast<std::size_t>(std::count_if(
std::begin(missing_), std::end(missing_), [](const std::pair<detail::Classifier, std::string> &val) {
return val.first != detail::Classifier::POSITIONAL_MARK;
}));

if(recurse) {
for(const App_p &sub : subcommands_) {
remaining_options += sub->remaining_size(recurse);
}
}
return remaining_options;
}


protected:
void _validate() const {
auto pcount = std::count_if(std::begin(options_), std::end(options_), [](const Option_p &opt) {
return opt->get_items_expected_max() >= detail::expected_max_vector_size && !opt->nonpositional();
});
if(pcount > 1) {
auto pcount_req = std::count_if(std::begin(options_), std::end(options_), [](const Option_p &opt) {
return opt->get_items_expected_max() >= detail::expected_max_vector_size && !opt->nonpositional() &&
opt->get_required();
});
if(pcount - pcount_req > 1) {
throw InvalidError(name_);
}
}

std::size_t nameless_subs{0};
for(const App_p &app : subcommands_) {
app->_validate();
if(app->get_name().empty())
++nameless_subs;
}

if(require_option_min_ > 0) {
if(require_option_max_ > 0) {
if(require_option_max_ < require_option_min_) {
throw(InvalidError("Required min options greater than required max options",
ExitCodes::InvalidError));
}
}
if(require_option_min_ > (options_.size() + nameless_subs)) {
throw(InvalidError("Required min options greater than number of available options",
ExitCodes::InvalidError));
}
}
}

void _configure() {
if(default_startup == startup_mode::enabled) {
disabled_ = false;
} else if(default_startup == startup_mode::disabled) {
disabled_ = true;
}
for(const App_p &app : subcommands_) {
if(app->has_automatic_name_) {
app->name_.clear();
}
if(app->name_.empty()) {
app->fallthrough_ = false;  
app->prefix_command_ = false;
}
app->parent_ = this;
app->_configure();
}
}

void run_callback(bool final_mode = false, bool suppress_final_callback = false) {
pre_callback();
if(!final_mode && parse_complete_callback_) {
parse_complete_callback_();
}
for(App *subc : get_subcommands()) {
subc->run_callback(true, suppress_final_callback);
}
for(auto &subc : subcommands_) {
if(subc->name_.empty() && subc->count_all() > 0) {
subc->run_callback(true, suppress_final_callback);
}
}

if(final_callback_ && (parsed_ > 0) && (!suppress_final_callback)) {
if(!name_.empty() || count_all() > 0 || parent_ == nullptr) {
final_callback_();
}
}
}

bool _valid_subcommand(const std::string &current, bool ignore_used = true) const {
if(require_subcommand_max_ != 0 && parsed_subcommands_.size() >= require_subcommand_max_) {
return parent_ != nullptr && parent_->_valid_subcommand(current, ignore_used);
}
auto com = _find_subcommand(current, true, ignore_used);
if(com != nullptr) {
return true;
}
return parent_ != nullptr && parent_->_valid_subcommand(current, ignore_used);
}

detail::Classifier _recognize(const std::string &current, bool ignore_used_subcommands = true) const {
std::string dummy1, dummy2;

if(current == "--")
return detail::Classifier::POSITIONAL_MARK;
if(_valid_subcommand(current, ignore_used_subcommands))
return detail::Classifier::SUBCOMMAND;
if(detail::split_long(current, dummy1, dummy2))
return detail::Classifier::LONG;
if(detail::split_short(current, dummy1, dummy2)) {
if(dummy1[0] >= '0' && dummy1[0] <= '9') {
if(get_option_no_throw(std::string{'-', dummy1[0]}) == nullptr) {
return detail::Classifier::NONE;
}
}
return detail::Classifier::SHORT;
}
if((allow_windows_style_options_) && (detail::split_windows_style(current, dummy1, dummy2)))
return detail::Classifier::WINDOWS_STYLE;
if((current == "++") && !name_.empty() && parent_ != nullptr)
return detail::Classifier::SUBCOMMAND_TERMINATOR;
return detail::Classifier::NONE;
}


void _process_config_file() {
if(config_ptr_ != nullptr) {
bool config_required = config_ptr_->get_required();
auto file_given = config_ptr_->count() > 0;
auto config_files = config_ptr_->as<std::vector<std::string>>();
if(config_files.empty() || config_files.front().empty()) {
if(config_required) {
throw FileError::Missing("no specified config file");
}
return;
}
for(auto rit = config_files.rbegin(); rit != config_files.rend(); ++rit) {
const auto &config_file = *rit;
auto path_result = detail::check_path(config_file.c_str());
if(path_result == detail::path_type::file) {
try {
std::vector<ConfigItem> values = config_formatter_->from_file(config_file);
_parse_config(values);
if(!file_given) {
config_ptr_->add_result(config_file);
}
} catch(const FileError &) {
if(config_required || file_given)
throw;
}
} else if(config_required || file_given) {
throw FileError::Missing(config_file);
}
}
}
}

void _process_env() {
for(const Option_p &opt : options_) {
if(opt->count() == 0 && !opt->envname_.empty()) {
char *buffer = nullptr;
std::string ename_string;

#ifdef _MSC_VER
std::size_t sz = 0;
if(_dupenv_s(&buffer, &sz, opt->envname_.c_str()) == 0 && buffer != nullptr) {
ename_string = std::string(buffer);
free(buffer);
}
#else
buffer = std::getenv(opt->envname_.c_str());
if(buffer != nullptr)
ename_string = std::string(buffer);
#endif

if(!ename_string.empty()) {
opt->add_result(ename_string);
}
}
}

for(App_p &sub : subcommands_) {
if(sub->get_name().empty() || !sub->parse_complete_callback_)
sub->_process_env();
}
}

void _process_callbacks() {

for(App_p &sub : subcommands_) {
if(sub->get_name().empty() && sub->parse_complete_callback_) {
if(sub->count_all() > 0) {
sub->_process_callbacks();
sub->run_callback();
}
}
}

for(const Option_p &opt : options_) {
if((*opt) && !opt->get_callback_run()) {
opt->run_callback();
}
}
for(App_p &sub : subcommands_) {
if(!sub->parse_complete_callback_) {
sub->_process_callbacks();
}
}
}

void _process_help_flags(bool trigger_help = false, bool trigger_all_help = false) const {
const Option *help_ptr = get_help_ptr();
const Option *help_all_ptr = get_help_all_ptr();

if(help_ptr != nullptr && help_ptr->count() > 0)
trigger_help = true;
if(help_all_ptr != nullptr && help_all_ptr->count() > 0)
trigger_all_help = true;

if(!parsed_subcommands_.empty()) {
for(const App *sub : parsed_subcommands_)
sub->_process_help_flags(trigger_help, trigger_all_help);

} else if(trigger_all_help) {
throw CallForAllHelp();
} else if(trigger_help) {
throw CallForHelp();
}
}

void _process_requirements() {
bool excluded{false};
std::string excluder;
for(auto &opt : exclude_options_) {
if(opt->count() > 0) {
excluded = true;
excluder = opt->get_name();
}
}
for(auto &subc : exclude_subcommands_) {
if(subc->count_all() > 0) {
excluded = true;
excluder = subc->get_display_name();
}
}
if(excluded) {
if(count_all() > 0) {
throw ExcludesError(get_display_name(), excluder);
}
return;
}

bool missing_needed{false};
std::string missing_need;
for(auto &opt : need_options_) {
if(opt->count() == 0) {
missing_needed = true;
missing_need = opt->get_name();
}
}
for(auto &subc : need_subcommands_) {
if(subc->count_all() == 0) {
missing_needed = true;
missing_need = subc->get_display_name();
}
}
if(missing_needed) {
if(count_all() > 0) {
throw RequiresError(get_display_name(), missing_need);
}
return;
}

std::size_t used_options = 0;
for(const Option_p &opt : options_) {

if(opt->count() != 0) {
++used_options;
}
if(opt->get_required() && opt->count() == 0) {
throw RequiredError(opt->get_name());
}
for(const Option *opt_req : opt->needs_)
if(opt->count() > 0 && opt_req->count() == 0)
throw RequiresError(opt->get_name(), opt_req->get_name());
for(const Option *opt_ex : opt->excludes_)
if(opt->count() > 0 && opt_ex->count() != 0)
throw ExcludesError(opt->get_name(), opt_ex->get_name());
}
if(require_subcommand_min_ > 0) {
auto selected_subcommands = get_subcommands();
if(require_subcommand_min_ > selected_subcommands.size())
throw RequiredError::Subcommand(require_subcommand_min_);
}


for(App_p &sub : subcommands_) {
if(sub->disabled_)
continue;
if(sub->name_.empty() && sub->count_all() > 0) {
++used_options;
}
}

if(require_option_min_ > used_options || (require_option_max_ > 0 && require_option_max_ < used_options)) {
auto option_list = detail::join(options_, [this](const Option_p &ptr) {
if(ptr.get() == help_ptr_ || ptr.get() == help_all_ptr_) {
return std::string{};
}
return ptr->get_name(false, true);
});

auto subc_list = get_subcommands([](App *app) { return ((app->get_name().empty()) && (!app->disabled_)); });
if(!subc_list.empty()) {
option_list += "," + detail::join(subc_list, [](const App *app) { return app->get_display_name(); });
}
throw RequiredError::Option(require_option_min_, require_option_max_, used_options, option_list);
}

for(App_p &sub : subcommands_) {
if(sub->disabled_)
continue;
if(sub->name_.empty() && sub->required_ == false) {
if(sub->count_all() == 0) {
if(require_option_min_ > 0 && require_option_min_ <= used_options) {
continue;
}
if(require_option_max_ > 0 && used_options >= require_option_min_) {
continue;
}
}
}
if(sub->count() > 0 || sub->name_.empty()) {
sub->_process_requirements();
}

if(sub->required_ && sub->count_all() == 0) {
throw(CLI::RequiredError(sub->get_display_name()));
}
}
}

void _process() {
CLI::FileError fe("ne");
bool caught_error{false};
try {
_process_config_file();
_process_env();
} catch(const CLI::FileError &fe2) {
fe = fe2;
caught_error = true;
}
_process_callbacks();
_process_help_flags();

if(caught_error) {
throw CLI::FileError(std::move(fe));
}

_process_requirements();
}

void _process_extras() {
if(!(allow_extras_ || prefix_command_)) {
std::size_t num_left_over = remaining_size();
if(num_left_over > 0) {
throw ExtrasError(name_, remaining(false));
}
}

for(App_p &sub : subcommands_) {
if(sub->count() > 0)
sub->_process_extras();
}
}

void _process_extras(std::vector<std::string> &args) {
if(!(allow_extras_ || prefix_command_)) {
std::size_t num_left_over = remaining_size();
if(num_left_over > 0) {
args = remaining(false);
throw ExtrasError(name_, args);
}
}

for(App_p &sub : subcommands_) {
if(sub->count() > 0)
sub->_process_extras(args);
}
}

void increment_parsed() {
++parsed_;
for(App_p &sub : subcommands_) {
if(sub->get_name().empty())
sub->increment_parsed();
}
}
void _parse(std::vector<std::string> &args) {
increment_parsed();
_trigger_pre_parse(args.size());
bool positional_only = false;

while(!args.empty()) {
if(!_parse_single(args, positional_only)) {
break;
}
}

if(parent_ == nullptr) {
_process();

_process_extras(args);

args = remaining_for_passthrough(false);
} else if(parse_complete_callback_) {
_process_env();
_process_callbacks();
_process_help_flags();
_process_requirements();
run_callback(false, true);
}
}

void _parse(std::vector<std::string> &&args) {
increment_parsed();
_trigger_pre_parse(args.size());
bool positional_only = false;

while(!args.empty()) {
_parse_single(args, positional_only);
}
_process();

_process_extras();
}

void _parse_stream(std::istream &input) {
auto values = config_formatter_->from_config(input);
_parse_config(values);
increment_parsed();
_trigger_pre_parse(values.size());
_process();

_process_extras();
}

void _parse_config(const std::vector<ConfigItem> &args) {
for(const ConfigItem &item : args) {
if(!_parse_single_config(item) && allow_config_extras_ == config_extras_mode::error)
throw ConfigError::Extras(item.fullname());
}
}

bool _parse_single_config(const ConfigItem &item, std::size_t level = 0) {
if(level < item.parents.size()) {
try {
auto subcom = get_subcommand(item.parents.at(level));
auto result = subcom->_parse_single_config(item, level + 1);

return result;
} catch(const OptionNotFound &) {
return false;
}
}
if(item.name == "++") {
if(configurable_) {
increment_parsed();
_trigger_pre_parse(2);
if(parent_ != nullptr) {
parent_->parsed_subcommands_.push_back(this);
}
}
return true;
}
if(item.name == "--") {
if(configurable_) {
_process_callbacks();
_process_requirements();
run_callback();
}
return true;
}
Option *op = get_option_no_throw("--" + item.name);
if(op == nullptr) {
if(item.name.size() == 1) {
op = get_option_no_throw("-" + item.name);
}
}
if(op == nullptr) {
op = get_option_no_throw(item.name);
}
if(op == nullptr) {
if(get_allow_config_extras() == config_extras_mode::capture)
missing_.emplace_back(detail::Classifier::NONE, item.fullname());
return false;
}

if(!op->get_configurable()) {
if(get_allow_config_extras() == config_extras_mode::ignore_all) {
return false;
}
throw ConfigError::NotConfigurable(item.fullname());
}

if(op->empty()) {
if(op->get_expected_min() == 0) {
auto res = config_formatter_->to_flag(item);
res = op->get_flag_value(item.name, res);

op->add_result(res);

} else {
op->add_result(item.inputs);
op->run_callback();
}
}

return true;
}

bool _parse_single(std::vector<std::string> &args, bool &positional_only) {
bool retval = true;
detail::Classifier classifier = positional_only ? detail::Classifier::NONE : _recognize(args.back());
switch(classifier) {
case detail::Classifier::POSITIONAL_MARK:
args.pop_back();
positional_only = true;
if((!_has_remaining_positionals()) && (parent_ != nullptr)) {
retval = false;
} else {
_move_to_missing(classifier, "--");
}
break;
case detail::Classifier::SUBCOMMAND_TERMINATOR:
args.pop_back();
retval = false;
break;
case detail::Classifier::SUBCOMMAND:
retval = _parse_subcommand(args);
break;
case detail::Classifier::LONG:
case detail::Classifier::SHORT:
case detail::Classifier::WINDOWS_STYLE:
_parse_arg(args, classifier);
break;
case detail::Classifier::NONE:
retval = _parse_positional(args, false);
if(retval && positionals_at_end_) {
positional_only = true;
}
break;
default:
throw HorribleError("unrecognized classifier (you should not see this!)");
}
return retval;
}

std::size_t _count_remaining_positionals(bool required_only = false) const {
std::size_t retval = 0;
for(const Option_p &opt : options_) {
if(opt->get_positional() && (!required_only || opt->get_required())) {
if(opt->get_items_expected_min() > 0 &&
static_cast<int>(opt->count()) < opt->get_items_expected_min()) {
retval += static_cast<std::size_t>(opt->get_items_expected_min()) - opt->count();
}
}
}
return retval;
}

bool _has_remaining_positionals() const {
for(const Option_p &opt : options_) {
if(opt->get_positional() && ((static_cast<int>(opt->count()) < opt->get_items_expected_min()))) {
return true;
}
}

return false;
}

bool _parse_positional(std::vector<std::string> &args, bool haltOnSubcommand) {

const std::string &positional = args.back();

if(positionals_at_end_) {
auto arg_rem = args.size();
auto remreq = _count_remaining_positionals(true);
if(arg_rem <= remreq) {
for(const Option_p &opt : options_) {
if(opt->get_positional() && opt->required_) {
if(static_cast<int>(opt->count()) < opt->get_items_expected_min()) {
if(validate_positionals_) {
std::string pos = positional;
pos = opt->_validate(pos, 0);
if(!pos.empty()) {
continue;
}
}
opt->add_result(positional);
parse_order_.push_back(opt.get());
args.pop_back();
return true;
}
}
}
}
}
for(const Option_p &opt : options_) {
if(opt->get_positional() &&
(static_cast<int>(opt->count()) < opt->get_items_expected_min() || opt->get_allow_extra_args())) {
if(validate_positionals_) {
std::string pos = positional;
pos = opt->_validate(pos, 0);
if(!pos.empty()) {
continue;
}
}
opt->add_result(positional);
parse_order_.push_back(opt.get());
args.pop_back();
return true;
}
}

for(auto &subc : subcommands_) {
if((subc->name_.empty()) && (!subc->disabled_)) {
if(subc->_parse_positional(args, false)) {
if(!subc->pre_parse_called_) {
subc->_trigger_pre_parse(args.size());
}
return true;
}
}
}
if(parent_ != nullptr && fallthrough_)
return _get_fallthrough_parent()->_parse_positional(args, static_cast<bool>(parse_complete_callback_));

auto com = _find_subcommand(args.back(), true, false);
if(com != nullptr && (require_subcommand_max_ == 0 || require_subcommand_max_ > parsed_subcommands_.size())) {
if(haltOnSubcommand) {
return false;
}
args.pop_back();
com->_parse(args);
return true;
}
auto parent_app = (parent_ != nullptr) ? _get_fallthrough_parent() : this;
com = parent_app->_find_subcommand(args.back(), true, false);
if(com != nullptr && (com->parent_->require_subcommand_max_ == 0 ||
com->parent_->require_subcommand_max_ > com->parent_->parsed_subcommands_.size())) {
return false;
}

if(positionals_at_end_) {
throw CLI::ExtrasError(name_, args);
}
if(parent_ != nullptr && name_.empty()) {
return false;
}
_move_to_missing(detail::Classifier::NONE, positional);
args.pop_back();
if(prefix_command_) {
while(!args.empty()) {
_move_to_missing(detail::Classifier::NONE, args.back());
args.pop_back();
}
}

return true;
}

App *_find_subcommand(const std::string &subc_name, bool ignore_disabled, bool ignore_used) const noexcept {
for(const App_p &com : subcommands_) {
if(com->disabled_ && ignore_disabled)
continue;
if(com->get_name().empty()) {
auto subc = com->_find_subcommand(subc_name, ignore_disabled, ignore_used);
if(subc != nullptr) {
return subc;
}
}
if(com->check_name(subc_name)) {
if((!*com) || !ignore_used)
return com.get();
}
}
return nullptr;
}

bool _parse_subcommand(std::vector<std::string> &args) {
if(_count_remaining_positionals( true) > 0) {
_parse_positional(args, false);
return true;
}
auto com = _find_subcommand(args.back(), true, true);
if(com != nullptr) {
args.pop_back();
if(!com->silent_) {
parsed_subcommands_.push_back(com);
}
com->_parse(args);
auto parent_app = com->parent_;
while(parent_app != this) {
parent_app->_trigger_pre_parse(args.size());
if(!com->silent_) {
parent_app->parsed_subcommands_.push_back(com);
}
parent_app = parent_app->parent_;
}
return true;
}

if(parent_ == nullptr)
throw HorribleError("Subcommand " + args.back() + " missing");
return false;
}

bool _parse_arg(std::vector<std::string> &args, detail::Classifier current_type) {

std::string current = args.back();

std::string arg_name;
std::string value;
std::string rest;

switch(current_type) {
case detail::Classifier::LONG:
if(!detail::split_long(current, arg_name, value))
throw HorribleError("Long parsed but missing (you should not see this):" + args.back());
break;
case detail::Classifier::SHORT:
if(!detail::split_short(current, arg_name, rest))
throw HorribleError("Short parsed but missing! You should not see this");
break;
case detail::Classifier::WINDOWS_STYLE:
if(!detail::split_windows_style(current, arg_name, value))
throw HorribleError("windows option parsed but missing! You should not see this");
break;
case detail::Classifier::SUBCOMMAND:
case detail::Classifier::SUBCOMMAND_TERMINATOR:
case detail::Classifier::POSITIONAL_MARK:
case detail::Classifier::NONE:
default:
throw HorribleError("parsing got called with invalid option! You should not see this");
}

auto op_ptr =
std::find_if(std::begin(options_), std::end(options_), [arg_name, current_type](const Option_p &opt) {
if(current_type == detail::Classifier::LONG)
return opt->check_lname(arg_name);
if(current_type == detail::Classifier::SHORT)
return opt->check_sname(arg_name);
return opt->check_lname(arg_name) || opt->check_sname(arg_name);
});

if(op_ptr == std::end(options_)) {
for(auto &subc : subcommands_) {
if(subc->name_.empty() && !subc->disabled_) {
if(subc->_parse_arg(args, current_type)) {
if(!subc->pre_parse_called_) {
subc->_trigger_pre_parse(args.size());
}
return true;
}
}
}
if(parent_ != nullptr && fallthrough_)
return _get_fallthrough_parent()->_parse_arg(args, current_type);
if(parent_ != nullptr && name_.empty()) {
return false;
}
args.pop_back();
_move_to_missing(current_type, current);
return true;
}

args.pop_back();

Option_p &op = *op_ptr;
if(op->get_inject_separator()) {
if(!op->results().empty() && !op->results().back().empty()) {
op->add_result(std::string{});
}
}
if(op->get_trigger_on_parse() && op->current_option_state_ == Option::option_state::callback_run) {
op->clear();
}
int min_num = (std::min)(op->get_type_size_min(), op->get_items_expected_min());
int max_num = op->get_items_expected_max();
if(max_num >= detail::expected_max_vector_size / 16 && !op->get_allow_extra_args()) {
auto tmax = op->get_type_size_max();
max_num = detail::checked_multiply(tmax, op->get_expected_min()) ? tmax : detail::expected_max_vector_size;
}
int collected = 0;     
int result_count = 0;  
if(max_num == 0) {
auto res = op->get_flag_value(arg_name, value);
op->add_result(res);
parse_order_.push_back(op.get());
} else if(!value.empty()) {  
op->add_result(value, result_count);
parse_order_.push_back(op.get());
collected += result_count;
} else if(!rest.empty()) {
op->add_result(rest, result_count);
parse_order_.push_back(op.get());
rest = "";
collected += result_count;
}

while(min_num > collected && !args.empty()) {
std::string current_ = args.back();
args.pop_back();
op->add_result(current_, result_count);
parse_order_.push_back(op.get());
collected += result_count;
}

if(min_num > collected) {  
throw ArgumentMismatch::TypedAtLeast(op->get_name(), min_num, op->get_type_name());
}

if(max_num > collected || op->get_allow_extra_args()) {  
auto remreqpos = _count_remaining_positionals(true);
while((collected < max_num || op->get_allow_extra_args()) && !args.empty() &&
_recognize(args.back(), false) == detail::Classifier::NONE) {
if(remreqpos >= args.size()) {
break;
}

op->add_result(args.back(), result_count);
parse_order_.push_back(op.get());
args.pop_back();
collected += result_count;
}

if(!args.empty() && _recognize(args.back()) == detail::Classifier::POSITIONAL_MARK)
args.pop_back();
if(min_num == 0 && max_num > 0 && collected == 0) {
auto res = op->get_flag_value(arg_name, std::string{});
op->add_result(res);
parse_order_.push_back(op.get());
}
}

if(min_num > 0 && op->get_type_size_max() != min_num && (collected % op->get_type_size_max()) != 0) {
op->add_result(std::string{});
}
if(op->get_trigger_on_parse()) {
op->run_callback();
}
if(!rest.empty()) {
rest = "-" + rest;
args.push_back(rest);
}
return true;
}

void _trigger_pre_parse(std::size_t remaining_args) {
if(!pre_parse_called_) {
pre_parse_called_ = true;
if(pre_parse_callback_) {
pre_parse_callback_(remaining_args);
}
} else if(immediate_callback_) {
if(!name_.empty()) {
auto pcnt = parsed_;
auto extras = std::move(missing_);
clear();
parsed_ = pcnt;
pre_parse_called_ = true;
missing_ = std::move(extras);
}
}
}

App *_get_fallthrough_parent() {
if(parent_ == nullptr) {
throw(HorribleError("No Valid parent"));
}
auto fallthrough_parent = parent_;
while((fallthrough_parent->parent_ != nullptr) && (fallthrough_parent->get_name().empty())) {
fallthrough_parent = fallthrough_parent->parent_;
}
return fallthrough_parent;
}

const std::string &_compare_subcommand_names(const App &subcom, const App &base) const {
static const std::string estring;
if(subcom.disabled_) {
return estring;
}
for(auto &subc : base.subcommands_) {
if(subc.get() != &subcom) {
if(subc->disabled_) {
continue;
}
if(!subcom.get_name().empty()) {
if(subc->check_name(subcom.get_name())) {
return subcom.get_name();
}
}
if(!subc->get_name().empty()) {
if(subcom.check_name(subc->get_name())) {
return subc->get_name();
}
}
for(const auto &les : subcom.aliases_) {
if(subc->check_name(les)) {
return les;
}
}
for(const auto &les : subc->aliases_) {
if(subcom.check_name(les)) {
return les;
}
}
if(subc->get_name().empty()) {
auto &cmpres = _compare_subcommand_names(subcom, *subc);
if(!cmpres.empty()) {
return cmpres;
}
}
if(subcom.get_name().empty()) {
auto &cmpres = _compare_subcommand_names(*subc, subcom);
if(!cmpres.empty()) {
return cmpres;
}
}
}
}
return estring;
}
void _move_to_missing(detail::Classifier val_type, const std::string &val) {
if(allow_extras_ || subcommands_.empty()) {
missing_.emplace_back(val_type, val);
return;
}
for(auto &subc : subcommands_) {
if(subc->name_.empty() && subc->allow_extras_) {
subc->missing_.emplace_back(val_type, val);
return;
}
}
missing_.emplace_back(val_type, val);
}

public:
void _move_option(Option *opt, App *app) {
if(opt == nullptr) {
throw OptionNotFound("the option is NULL");
}
bool found = false;
for(auto &subc : subcommands_) {
if(app == subc.get()) {
found = true;
}
}
if(!found) {
throw OptionNotFound("The Given app is not a subcommand");
}

if((help_ptr_ == opt) || (help_all_ptr_ == opt))
throw OptionAlreadyAdded("cannot move help options");

if(config_ptr_ == opt)
throw OptionAlreadyAdded("cannot move config file options");

auto iterator =
std::find_if(std::begin(options_), std::end(options_), [opt](const Option_p &v) { return v.get() == opt; });
if(iterator != std::end(options_)) {
const auto &opt_p = *iterator;
if(std::find_if(std::begin(app->options_), std::end(app->options_), [&opt_p](const Option_p &v) {
return (*v == *opt_p);
}) == std::end(app->options_)) {
app->options_.push_back(std::move(*iterator));
options_.erase(iterator);
} else {
throw OptionAlreadyAdded("option was not located: " + opt->get_name());
}
} else {
throw OptionNotFound("could not locate the given Option");
}
}
};  

class Option_group : public App {
public:
Option_group(std::string group_description, std::string group_name, App *parent)
: App(std::move(group_description), "", parent) {
group(group_name);
}
using App::add_option;
Option *add_option(Option *opt) {
if(get_parent() == nullptr) {
throw OptionNotFound("Unable to locate the specified option");
}
get_parent()->_move_option(opt, this);
return opt;
}
void add_options(Option *opt) { add_option(opt); }
template <typename... Args> void add_options(Option *opt, Args... args) {
add_option(opt);
add_options(args...);
}
using App::add_subcommand;
App *add_subcommand(App *subcom) {
App_p subc = subcom->get_parent()->get_subcommand_ptr(subcom);
subc->get_parent()->remove_subcommand(subcom);
add_subcommand(std::move(subc));
return subcom;
}
};
inline void TriggerOn(App *trigger_app, App *app_to_enable) {
app_to_enable->enabled_by_default(false);
app_to_enable->disabled_by_default();
trigger_app->preparse_callback([app_to_enable](std::size_t) { app_to_enable->disabled(false); });
}

inline void TriggerOn(App *trigger_app, std::vector<App *> apps_to_enable) {
for(auto &app : apps_to_enable) {
app->enabled_by_default(false);
app->disabled_by_default();
}

trigger_app->preparse_callback([apps_to_enable](std::size_t) {
for(auto &app : apps_to_enable) {
app->disabled(false);
}
});
}

inline void TriggerOff(App *trigger_app, App *app_to_enable) {
app_to_enable->disabled_by_default(false);
app_to_enable->enabled_by_default();
trigger_app->preparse_callback([app_to_enable](std::size_t) { app_to_enable->disabled(); });
}

inline void TriggerOff(App *trigger_app, std::vector<App *> apps_to_enable) {
for(auto &app : apps_to_enable) {
app->disabled_by_default(false);
app->enabled_by_default();
}

trigger_app->preparse_callback([apps_to_enable](std::size_t) {
for(auto &app : apps_to_enable) {
app->disabled();
}
});
}

inline void deprecate_option(Option *opt, const std::string &replacement = "") {
Validator deprecate_warning{[opt, replacement](std::string &) {
std::cout << opt->get_name() << " is deprecated please use '" << replacement
<< "' instead\n";
return std::string();
},
"DEPRECATED"};
deprecate_warning.application_index(0);
opt->check(deprecate_warning);
if(!replacement.empty()) {
opt->description(opt->get_description() + " DEPRECATED: please use '" + replacement + "' instead");
}
}

inline void deprecate_option(App *app, const std::string &option_name, const std::string &replacement = "") {
auto opt = app->get_option(option_name);
deprecate_option(opt, replacement);
}

inline void deprecate_option(App &app, const std::string &option_name, const std::string &replacement = "") {
auto opt = app.get_option(option_name);
deprecate_option(opt, replacement);
}

inline void retire_option(App *app, Option *opt) {
App temp;
auto option_copy = temp.add_option(opt->get_name(false, true))
->type_size(opt->get_type_size_min(), opt->get_type_size_max())
->expected(opt->get_expected_min(), opt->get_expected_max())
->allow_extra_args(opt->get_allow_extra_args());

app->remove_option(opt);
auto opt2 = app->add_option(option_copy->get_name(false, true), "option has been retired and has no effect")
->type_name("RETIRED")
->default_str("RETIRED")
->type_size(option_copy->get_type_size_min(), option_copy->get_type_size_max())
->expected(option_copy->get_expected_min(), option_copy->get_expected_max())
->allow_extra_args(option_copy->get_allow_extra_args());

Validator retired_warning{[opt2](std::string &) {
std::cout << "WARNING " << opt2->get_name() << " is retired and has no effect\n";
return std::string();
},
""};
retired_warning.application_index(0);
opt2->check(retired_warning);
}

inline void retire_option(App &app, Option *opt) { retire_option(&app, opt); }

inline void retire_option(App *app, const std::string &option_name) {

auto opt = app->get_option_no_throw(option_name);
if(opt != nullptr) {
retire_option(app, opt);
return;
}
auto opt2 = app->add_option(option_name, "option has been retired and has no effect")
->type_name("RETIRED")
->expected(0, 1)
->default_str("RETIRED");
Validator retired_warning{[opt2](std::string &) {
std::cout << "WARNING " << opt2->get_name() << " is retired and has no effect\n";
return std::string();
},
""};
retired_warning.application_index(0);
opt2->check(retired_warning);
}

inline void retire_option(App &app, const std::string &option_name) { retire_option(&app, option_name); }

namespace FailureMessage {

inline std::string simple(const App *app, const Error &e) {
std::string header = std::string(e.what()) + "\n";
std::vector<std::string> names;

if(app->get_help_ptr() != nullptr)
names.push_back(app->get_help_ptr()->get_name());

if(app->get_help_all_ptr() != nullptr)
names.push_back(app->get_help_all_ptr()->get_name());

if(!names.empty())
header += "Run with " + detail::join(names, " or ") + " for more information.\n";

return header;
}

inline std::string help(const App *app, const Error &e) {
std::string header = std::string("ERROR: ") + e.get_name() + ": " + e.what() + "\n";
header += app->help();
return header;
}

}  

namespace detail {
struct AppFriend {
#ifdef CLI11_CPP14

template <typename... Args> static decltype(auto) parse_arg(App *app, Args &&...args) {
return app->_parse_arg(std::forward<Args>(args)...);
}

template <typename... Args> static decltype(auto) parse_subcommand(App *app, Args &&...args) {
return app->_parse_subcommand(std::forward<Args>(args)...);
}
#else
template <typename... Args>
static auto parse_arg(App *app, Args &&...args) ->
typename std::result_of<decltype (&App::_parse_arg)(App, Args...)>::type {
return app->_parse_arg(std::forward<Args>(args)...);
}

template <typename... Args>
static auto parse_subcommand(App *app, Args &&...args) ->
typename std::result_of<decltype (&App::_parse_subcommand)(App, Args...)>::type {
return app->_parse_subcommand(std::forward<Args>(args)...);
}
#endif
static App *get_fallthrough_parent(App *app) { return app->_get_fallthrough_parent(); }
};
}  

}  
