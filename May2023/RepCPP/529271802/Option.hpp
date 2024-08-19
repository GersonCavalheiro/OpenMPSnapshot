
#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "Error.hpp"
#include "Macros.hpp"
#include "Split.hpp"
#include "StringTools.hpp"
#include "Validators.hpp"

namespace CLI {

using results_t = std::vector<std::string>;
using callback_t = std::function<bool(const results_t &)>;

class Option;
class App;

using Option_p = std::unique_ptr<Option>;
enum class MultiOptionPolicy : char {
Throw,      
TakeLast,   
TakeFirst,  
Join,       
TakeAll     
};

template <typename CRTP> class OptionBase {
friend App;

protected:
std::string group_ = std::string("Options");

bool required_{false};

bool ignore_case_{false};

bool ignore_underscore_{false};

bool configurable_{true};

bool disable_flag_override_{false};

char delimiter_{'\0'};

bool always_capture_default_{false};

MultiOptionPolicy multi_option_policy_{MultiOptionPolicy::Throw};

template <typename T> void copy_to(T *other) const {
other->group(group_);
other->required(required_);
other->ignore_case(ignore_case_);
other->ignore_underscore(ignore_underscore_);
other->configurable(configurable_);
other->disable_flag_override(disable_flag_override_);
other->delimiter(delimiter_);
other->always_capture_default(always_capture_default_);
other->multi_option_policy(multi_option_policy_);
}

public:

CRTP *group(const std::string &name) {
if(!detail::valid_alias_name_string(name)) {
throw IncorrectConstruction("Group names may not contain newlines or null characters");
}
group_ = name;
return static_cast<CRTP *>(this);
}

CRTP *required(bool value = true) {
required_ = value;
return static_cast<CRTP *>(this);
}

CRTP *mandatory(bool value = true) { return required(value); }

CRTP *always_capture_default(bool value = true) {
always_capture_default_ = value;
return static_cast<CRTP *>(this);
}


const std::string &get_group() const { return group_; }

bool get_required() const { return required_; }

bool get_ignore_case() const { return ignore_case_; }

bool get_ignore_underscore() const { return ignore_underscore_; }

bool get_configurable() const { return configurable_; }

bool get_disable_flag_override() const { return disable_flag_override_; }

char get_delimiter() const { return delimiter_; }

bool get_always_capture_default() const { return always_capture_default_; }

MultiOptionPolicy get_multi_option_policy() const { return multi_option_policy_; }


CRTP *take_last() {
auto self = static_cast<CRTP *>(this);
self->multi_option_policy(MultiOptionPolicy::TakeLast);
return self;
}

CRTP *take_first() {
auto self = static_cast<CRTP *>(this);
self->multi_option_policy(MultiOptionPolicy::TakeFirst);
return self;
}

CRTP *take_all() {
auto self = static_cast<CRTP *>(this);
self->multi_option_policy(MultiOptionPolicy::TakeAll);
return self;
}

CRTP *join() {
auto self = static_cast<CRTP *>(this);
self->multi_option_policy(MultiOptionPolicy::Join);
return self;
}

CRTP *join(char delim) {
auto self = static_cast<CRTP *>(this);
self->delimiter_ = delim;
self->multi_option_policy(MultiOptionPolicy::Join);
return self;
}

CRTP *configurable(bool value = true) {
configurable_ = value;
return static_cast<CRTP *>(this);
}

CRTP *delimiter(char value = '\0') {
delimiter_ = value;
return static_cast<CRTP *>(this);
}
};

class OptionDefaults : public OptionBase<OptionDefaults> {
public:
OptionDefaults() = default;


OptionDefaults *multi_option_policy(MultiOptionPolicy value = MultiOptionPolicy::Throw) {
multi_option_policy_ = value;
return this;
}

OptionDefaults *ignore_case(bool value = true) {
ignore_case_ = value;
return this;
}

OptionDefaults *ignore_underscore(bool value = true) {
ignore_underscore_ = value;
return this;
}

OptionDefaults *disable_flag_override(bool value = true) {
disable_flag_override_ = value;
return this;
}

OptionDefaults *delimiter(char value = '\0') {
delimiter_ = value;
return this;
}
};

class Option : public OptionBase<Option> {
friend App;

protected:

std::vector<std::string> snames_{};

std::vector<std::string> lnames_{};

std::vector<std::pair<std::string, std::string>> default_flag_values_{};

std::vector<std::string> fnames_{};

std::string pname_{};

std::string envname_{};


std::string description_{};

std::string default_str_{};

std::string option_text_{};

std::function<std::string()> type_name_{[]() { return std::string(); }};

std::function<std::string()> default_function_{};


int type_size_max_{1};
int type_size_min_{1};

int expected_min_{1};
int expected_max_{1};

std::vector<Validator> validators_{};

std::set<Option *> needs_{};

std::set<Option *> excludes_{};


App *parent_{nullptr};

callback_t callback_{};


results_t results_{};
results_t proc_results_{};
enum class option_state : char {
parsing = 0,       
validated = 2,     
reduced = 4,       
callback_run = 6,  
};
option_state current_option_state_{option_state::parsing};
bool allow_extra_args_{false};
bool flag_like_{false};
bool run_callback_for_default_{false};
bool inject_separator_{false};
bool trigger_on_result_{false};
bool force_callback_{false};

Option(std::string option_name, std::string option_description, callback_t callback, App *parent)
: description_(std::move(option_description)), parent_(parent), callback_(std::move(callback)) {
std::tie(snames_, lnames_, pname_) = detail::get_names(detail::split_names(option_name));
}

public:

Option(const Option &) = delete;
Option &operator=(const Option &) = delete;

std::size_t count() const { return results_.size(); }

bool empty() const { return results_.empty(); }

explicit operator bool() const { return !empty() || force_callback_; }

void clear() {
results_.clear();
current_option_state_ = option_state::parsing;
}


Option *expected(int value) {
if(value < 0) {
expected_min_ = -value;
if(expected_max_ < expected_min_) {
expected_max_ = expected_min_;
}
allow_extra_args_ = true;
flag_like_ = false;
} else if(value == detail::expected_max_vector_size) {
expected_min_ = 1;
expected_max_ = detail::expected_max_vector_size;
allow_extra_args_ = true;
flag_like_ = false;
} else {
expected_min_ = value;
expected_max_ = value;
flag_like_ = (expected_min_ == 0);
}
return this;
}

Option *expected(int value_min, int value_max) {
if(value_min < 0) {
value_min = -value_min;
}

if(value_max < 0) {
value_max = detail::expected_max_vector_size;
}
if(value_max < value_min) {
expected_min_ = value_max;
expected_max_ = value_min;
} else {
expected_max_ = value_max;
expected_min_ = value_min;
}

return this;
}
Option *allow_extra_args(bool value = true) {
allow_extra_args_ = value;
return this;
}
bool get_allow_extra_args() const { return allow_extra_args_; }
Option *trigger_on_parse(bool value = true) {
trigger_on_result_ = value;
return this;
}
bool get_trigger_on_parse() const { return trigger_on_result_; }

Option *force_callback(bool value = true) {
force_callback_ = value;
return this;
}
bool get_force_callback() const { return force_callback_; }

Option *run_callback_for_default(bool value = true) {
run_callback_for_default_ = value;
return this;
}
bool get_run_callback_for_default() const { return run_callback_for_default_; }

Option *check(Validator validator, const std::string &validator_name = "") {
validator.non_modifying();
validators_.push_back(std::move(validator));
if(!validator_name.empty())
validators_.back().name(validator_name);
return this;
}

Option *check(std::function<std::string(const std::string &)> Validator,
std::string Validator_description = "",
std::string Validator_name = "") {
validators_.emplace_back(Validator, std::move(Validator_description), std::move(Validator_name));
validators_.back().non_modifying();
return this;
}

Option *transform(Validator Validator, const std::string &Validator_name = "") {
validators_.insert(validators_.begin(), std::move(Validator));
if(!Validator_name.empty())
validators_.front().name(Validator_name);
return this;
}

Option *transform(const std::function<std::string(std::string)> &func,
std::string transform_description = "",
std::string transform_name = "") {
validators_.insert(validators_.begin(),
Validator(
[func](std::string &val) {
val = func(val);
return std::string{};
},
std::move(transform_description),
std::move(transform_name)));

return this;
}

Option *each(const std::function<void(std::string)> &func) {
validators_.emplace_back(
[func](std::string &inout) {
func(inout);
return std::string{};
},
std::string{});
return this;
}
Validator *get_validator(const std::string &Validator_name = "") {
for(auto &Validator : validators_) {
if(Validator_name == Validator.get_name()) {
return &Validator;
}
}
if((Validator_name.empty()) && (!validators_.empty())) {
return &(validators_.front());
}
throw OptionNotFound(std::string{"Validator "} + Validator_name + " Not Found");
}

Validator *get_validator(int index) {
if(index >= 0 && index < static_cast<int>(validators_.size())) {
return &(validators_[static_cast<decltype(validators_)::size_type>(index)]);
}
throw OptionNotFound("Validator index is not valid");
}

Option *needs(Option *opt) {
if(opt != this) {
needs_.insert(opt);
}
return this;
}

template <typename T = App> Option *needs(std::string opt_name) {
auto opt = static_cast<T *>(parent_)->get_option_no_throw(opt_name);
if(opt == nullptr) {
throw IncorrectConstruction::MissingOption(opt_name);
}
return needs(opt);
}

template <typename A, typename B, typename... ARG> Option *needs(A opt, B opt1, ARG... args) {
needs(opt);
return needs(opt1, args...);
}

bool remove_needs(Option *opt) {
auto iterator = std::find(std::begin(needs_), std::end(needs_), opt);

if(iterator == std::end(needs_)) {
return false;
}
needs_.erase(iterator);
return true;
}

Option *excludes(Option *opt) {
if(opt == this) {
throw(IncorrectConstruction("and option cannot exclude itself"));
}
excludes_.insert(opt);

opt->excludes_.insert(this);


return this;
}

template <typename T = App> Option *excludes(std::string opt_name) {
auto opt = static_cast<T *>(parent_)->get_option_no_throw(opt_name);
if(opt == nullptr) {
throw IncorrectConstruction::MissingOption(opt_name);
}
return excludes(opt);
}

template <typename A, typename B, typename... ARG> Option *excludes(A opt, B opt1, ARG... args) {
excludes(opt);
return excludes(opt1, args...);
}

bool remove_excludes(Option *opt) {
auto iterator = std::find(std::begin(excludes_), std::end(excludes_), opt);

if(iterator == std::end(excludes_)) {
return false;
}
excludes_.erase(iterator);
return true;
}

Option *envname(std::string name) {
envname_ = std::move(name);
return this;
}

template <typename T = App> Option *ignore_case(bool value = true) {
if(!ignore_case_ && value) {
ignore_case_ = value;
auto *parent = static_cast<T *>(parent_);
for(const Option_p &opt : parent->options_) {
if(opt.get() == this) {
continue;
}
auto &omatch = opt->matching_name(*this);
if(!omatch.empty()) {
ignore_case_ = false;
throw OptionAlreadyAdded("adding ignore case caused a name conflict with " + omatch);
}
}
} else {
ignore_case_ = value;
}
return this;
}

template <typename T = App> Option *ignore_underscore(bool value = true) {

if(!ignore_underscore_ && value) {
ignore_underscore_ = value;
auto *parent = static_cast<T *>(parent_);
for(const Option_p &opt : parent->options_) {
if(opt.get() == this) {
continue;
}
auto &omatch = opt->matching_name(*this);
if(!omatch.empty()) {
ignore_underscore_ = false;
throw OptionAlreadyAdded("adding ignore underscore caused a name conflict with " + omatch);
}
}
} else {
ignore_underscore_ = value;
}
return this;
}

Option *multi_option_policy(MultiOptionPolicy value = MultiOptionPolicy::Throw) {
if(value != multi_option_policy_) {
if(multi_option_policy_ == MultiOptionPolicy::Throw && expected_max_ == detail::expected_max_vector_size &&
expected_min_ > 1) {  
expected_max_ = expected_min_;
}
multi_option_policy_ = value;
current_option_state_ = option_state::parsing;
}
return this;
}

Option *disable_flag_override(bool value = true) {
disable_flag_override_ = value;
return this;
}

int get_type_size() const { return type_size_min_; }

int get_type_size_min() const { return type_size_min_; }
int get_type_size_max() const { return type_size_max_; }

int get_inject_separator() const { return inject_separator_; }

std::string get_envname() const { return envname_; }

std::set<Option *> get_needs() const { return needs_; }

std::set<Option *> get_excludes() const { return excludes_; }

std::string get_default_str() const { return default_str_; }

callback_t get_callback() const { return callback_; }

const std::vector<std::string> &get_lnames() const { return lnames_; }

const std::vector<std::string> &get_snames() const { return snames_; }

const std::vector<std::string> &get_fnames() const { return fnames_; }
const std::string &get_single_name() const {
if(!lnames_.empty()) {
return lnames_[0];
}
if(!pname_.empty()) {
return pname_;
}
if(!snames_.empty()) {
return snames_[0];
}
return envname_;
}
int get_expected() const { return expected_min_; }

int get_expected_min() const { return expected_min_; }
int get_expected_max() const { return expected_max_; }

int get_items_expected_min() const { return type_size_min_ * expected_min_; }

int get_items_expected_max() const {
int t = type_size_max_;
return detail::checked_multiply(t, expected_max_) ? t : detail::expected_max_vector_size;
}
int get_items_expected() const { return get_items_expected_min(); }

bool get_positional() const { return pname_.length() > 0; }

bool nonpositional() const { return (snames_.size() + lnames_.size()) > 0; }

bool has_description() const { return description_.length() > 0; }

const std::string &get_description() const { return description_; }

Option *description(std::string option_description) {
description_ = std::move(option_description);
return this;
}

Option *option_text(std::string text) {
option_text_ = std::move(text);
return this;
}

const std::string &get_option_text() const { return option_text_; }


std::string get_name(bool positional = false,  
bool all_options = false  
) const {
if(get_group().empty())
return {};  

if(all_options) {

std::vector<std::string> name_list;

if((positional && (!pname_.empty())) || (snames_.empty() && lnames_.empty())) {
name_list.push_back(pname_);
}
if((get_items_expected() == 0) && (!fnames_.empty())) {
for(const std::string &sname : snames_) {
name_list.push_back("-" + sname);
if(check_fname(sname)) {
name_list.back() += "{" + get_flag_value(sname, "") + "}";
}
}

for(const std::string &lname : lnames_) {
name_list.push_back("--" + lname);
if(check_fname(lname)) {
name_list.back() += "{" + get_flag_value(lname, "") + "}";
}
}
} else {
for(const std::string &sname : snames_)
name_list.push_back("-" + sname);

for(const std::string &lname : lnames_)
name_list.push_back("--" + lname);
}

return detail::join(name_list);
}

if(positional)
return pname_;

if(!lnames_.empty())
return std::string(2, '-') + lnames_[0];

if(!snames_.empty())
return std::string(1, '-') + snames_[0];

return pname_;
}


void run_callback() {
if(force_callback_ && results_.empty()) {
add_result(default_str_);
}
if(current_option_state_ == option_state::parsing) {
_validate_results(results_);
current_option_state_ = option_state::validated;
}

if(current_option_state_ < option_state::reduced) {
_reduce_results(proc_results_, results_);
current_option_state_ = option_state::reduced;
}
if(current_option_state_ >= option_state::reduced) {
current_option_state_ = option_state::callback_run;
if(!(callback_)) {
return;
}
const results_t &send_results = proc_results_.empty() ? results_ : proc_results_;
bool local_result = callback_(send_results);

if(!local_result)
throw ConversionError(get_name(), results_);
}
}

const std::string &matching_name(const Option &other) const {
static const std::string estring;
for(const std::string &sname : snames_)
if(other.check_sname(sname))
return sname;
for(const std::string &lname : lnames_)
if(other.check_lname(lname))
return lname;

if(ignore_case_ ||
ignore_underscore_) {  
for(const std::string &sname : other.snames_)
if(check_sname(sname))
return sname;
for(const std::string &lname : other.lnames_)
if(check_lname(lname))
return lname;
}
return estring;
}
bool operator==(const Option &other) const { return !matching_name(other).empty(); }

bool check_name(const std::string &name) const {

if(name.length() > 2 && name[0] == '-' && name[1] == '-')
return check_lname(name.substr(2));
if(name.length() > 1 && name.front() == '-')
return check_sname(name.substr(1));
if(!pname_.empty()) {
std::string local_pname = pname_;
std::string local_name = name;
if(ignore_underscore_) {
local_pname = detail::remove_underscore(local_pname);
local_name = detail::remove_underscore(local_name);
}
if(ignore_case_) {
local_pname = detail::to_lower(local_pname);
local_name = detail::to_lower(local_name);
}
if(local_name == local_pname) {
return true;
}
}

if(!envname_.empty()) {
return (name == envname_);
}
return false;
}

bool check_sname(std::string name) const {
return (detail::find_member(std::move(name), snames_, ignore_case_) >= 0);
}

bool check_lname(std::string name) const {
return (detail::find_member(std::move(name), lnames_, ignore_case_, ignore_underscore_) >= 0);
}

bool check_fname(std::string name) const {
if(fnames_.empty()) {
return false;
}
return (detail::find_member(std::move(name), fnames_, ignore_case_, ignore_underscore_) >= 0);
}

std::string get_flag_value(const std::string &name, std::string input_value) const {
static const std::string trueString{"true"};
static const std::string falseString{"false"};
static const std::string emptyString{"{}"};
if(disable_flag_override_) {
if(!((input_value.empty()) || (input_value == emptyString))) {
auto default_ind = detail::find_member(name, fnames_, ignore_case_, ignore_underscore_);
if(default_ind >= 0) {
if(default_flag_values_[static_cast<std::size_t>(default_ind)].second != input_value) {
throw(ArgumentMismatch::FlagOverride(name));
}
} else {
if(input_value != trueString) {
throw(ArgumentMismatch::FlagOverride(name));
}
}
}
}
auto ind = detail::find_member(name, fnames_, ignore_case_, ignore_underscore_);
if((input_value.empty()) || (input_value == emptyString)) {
if(flag_like_) {
return (ind < 0) ? trueString : default_flag_values_[static_cast<std::size_t>(ind)].second;
} else {
return (ind < 0) ? default_str_ : default_flag_values_[static_cast<std::size_t>(ind)].second;
}
}
if(ind < 0) {
return input_value;
}
if(default_flag_values_[static_cast<std::size_t>(ind)].second == falseString) {
try {
auto val = detail::to_flag_value(input_value);
return (val == 1) ? falseString : (val == (-1) ? trueString : std::to_string(-val));
} catch(const std::invalid_argument &) {
return input_value;
}
} else {
return input_value;
}
}

Option *add_result(std::string s) {
_add_result(std::move(s), results_);
current_option_state_ = option_state::parsing;
return this;
}

Option *add_result(std::string s, int &results_added) {
results_added = _add_result(std::move(s), results_);
current_option_state_ = option_state::parsing;
return this;
}

Option *add_result(std::vector<std::string> s) {
current_option_state_ = option_state::parsing;
for(auto &str : s) {
_add_result(std::move(str), results_);
}
return this;
}

const results_t &results() const { return results_; }

results_t reduced_results() const {
results_t res = proc_results_.empty() ? results_ : proc_results_;
if(current_option_state_ < option_state::reduced) {
if(current_option_state_ == option_state::parsing) {
res = results_;
_validate_results(res);
}
if(!res.empty()) {
results_t extra;
_reduce_results(extra, res);
if(!extra.empty()) {
res = std::move(extra);
}
}
}
return res;
}

template <typename T> void results(T &output) const {
bool retval;
if(current_option_state_ >= option_state::reduced || (results_.size() == 1 && validators_.empty())) {
const results_t &res = (proc_results_.empty()) ? results_ : proc_results_;
retval = detail::lexical_conversion<T, T>(res, output);
} else {
results_t res;
if(results_.empty()) {
if(!default_str_.empty()) {
_add_result(std::string(default_str_), res);
_validate_results(res);
results_t extra;
_reduce_results(extra, res);
if(!extra.empty()) {
res = std::move(extra);
}
} else {
res.emplace_back();
}
} else {
res = reduced_results();
}
retval = detail::lexical_conversion<T, T>(res, output);
}
if(!retval) {
throw ConversionError(get_name(), results_);
}
}

template <typename T> T as() const {
T output;
results(output);
return output;
}

bool get_callback_run() const { return (current_option_state_ == option_state::callback_run); }


Option *type_name_fn(std::function<std::string()> typefun) {
type_name_ = std::move(typefun);
return this;
}

Option *type_name(std::string typeval) {
type_name_fn([typeval]() { return typeval; });
return this;
}

Option *type_size(int option_type_size) {
if(option_type_size < 0) {
type_size_max_ = -option_type_size;
type_size_min_ = -option_type_size;
expected_max_ = detail::expected_max_vector_size;
} else {
type_size_max_ = option_type_size;
if(type_size_max_ < detail::expected_max_vector_size) {
type_size_min_ = option_type_size;
} else {
inject_separator_ = true;
}
if(type_size_max_ == 0)
required_ = false;
}
return this;
}
Option *type_size(int option_type_size_min, int option_type_size_max) {
if(option_type_size_min < 0 || option_type_size_max < 0) {
expected_max_ = detail::expected_max_vector_size;
option_type_size_min = (std::abs)(option_type_size_min);
option_type_size_max = (std::abs)(option_type_size_max);
}

if(option_type_size_min > option_type_size_max) {
type_size_max_ = option_type_size_min;
type_size_min_ = option_type_size_max;
} else {
type_size_min_ = option_type_size_min;
type_size_max_ = option_type_size_max;
}
if(type_size_max_ == 0) {
required_ = false;
}
if(type_size_max_ >= detail::expected_max_vector_size) {
inject_separator_ = true;
}
return this;
}

void inject_separator(bool value = true) { inject_separator_ = value; }

Option *default_function(const std::function<std::string()> &func) {
default_function_ = func;
return this;
}

Option *capture_default_str() {
if(default_function_) {
default_str_ = default_function_();
}
return this;
}

Option *default_str(std::string val) {
default_str_ = std::move(val);
return this;
}

template <typename X> Option *default_val(const X &val) {
std::string val_str = detail::to_string(val);
auto old_option_state = current_option_state_;
results_t old_results{std::move(results_)};
results_.clear();
try {
add_result(val_str);
if(run_callback_for_default_ && !trigger_on_result_) {
run_callback();  
current_option_state_ = option_state::parsing;
} else {
_validate_results(results_);
current_option_state_ = old_option_state;
}
} catch(const CLI::Error &) {
results_ = std::move(old_results);
current_option_state_ = old_option_state;
throw;
}
results_ = std::move(old_results);
default_str_ = std::move(val_str);
return this;
}

std::string get_type_name() const {
std::string full_type_name = type_name_();
if(!validators_.empty()) {
for(auto &Validator : validators_) {
std::string vtype = Validator.get_description();
if(!vtype.empty()) {
full_type_name += ":" + vtype;
}
}
}
return full_type_name;
}

private:
void _validate_results(results_t &res) const {
if(!validators_.empty()) {
if(type_size_max_ > 1) {  
int index = 0;
if(get_items_expected_max() < static_cast<int>(res.size()) &&
multi_option_policy_ == CLI::MultiOptionPolicy::TakeLast) {
index = get_items_expected_max() - static_cast<int>(res.size());
}

for(std::string &result : res) {
if(detail::is_separator(result) && type_size_max_ != type_size_min_ && index >= 0) {
index = 0;  
continue;
}
auto err_msg = _validate(result, (index >= 0) ? (index % type_size_max_) : index);
if(!err_msg.empty())
throw ValidationError(get_name(), err_msg);
++index;
}
} else {
int index = 0;
if(expected_max_ < static_cast<int>(res.size()) &&
multi_option_policy_ == CLI::MultiOptionPolicy::TakeLast) {
index = expected_max_ - static_cast<int>(res.size());
}
for(std::string &result : res) {
auto err_msg = _validate(result, index);
++index;
if(!err_msg.empty())
throw ValidationError(get_name(), err_msg);
}
}
}
}


void _reduce_results(results_t &res, const results_t &original) const {


res.clear();
switch(multi_option_policy_) {
case MultiOptionPolicy::TakeAll:
break;
case MultiOptionPolicy::TakeLast: {
std::size_t trim_size = std::min<std::size_t>(
static_cast<std::size_t>(std::max<int>(get_items_expected_max(), 1)), original.size());
if(original.size() != trim_size) {
res.assign(original.end() - static_cast<results_t::difference_type>(trim_size), original.end());
}
} break;
case MultiOptionPolicy::TakeFirst: {
std::size_t trim_size = std::min<std::size_t>(
static_cast<std::size_t>(std::max<int>(get_items_expected_max(), 1)), original.size());
if(original.size() != trim_size) {
res.assign(original.begin(), original.begin() + static_cast<results_t::difference_type>(trim_size));
}
} break;
case MultiOptionPolicy::Join:
if(results_.size() > 1) {
res.push_back(detail::join(original, std::string(1, (delimiter_ == '\0') ? '\n' : delimiter_)));
}
break;
case MultiOptionPolicy::Throw:
default: {
auto num_min = static_cast<std::size_t>(get_items_expected_min());
auto num_max = static_cast<std::size_t>(get_items_expected_max());
if(num_min == 0) {
num_min = 1;
}
if(num_max == 0) {
num_max = 1;
}
if(original.size() < num_min) {
throw ArgumentMismatch::AtLeast(get_name(), static_cast<int>(num_min), original.size());
}
if(original.size() > num_max) {
throw ArgumentMismatch::AtMost(get_name(), static_cast<int>(num_max), original.size());
}
break;
}
}
}

std::string _validate(std::string &result, int index) const {
std::string err_msg;
if(result.empty() && expected_min_ == 0) {
return err_msg;
}
for(const auto &vali : validators_) {
auto v = vali.get_application_index();
if(v == -1 || v == index) {
try {
err_msg = vali(result);
} catch(const ValidationError &err) {
err_msg = err.what();
}
if(!err_msg.empty())
break;
}
}

return err_msg;
}

int _add_result(std::string &&result, std::vector<std::string> &res) const {
int result_count = 0;
if(allow_extra_args_ && !result.empty() && result.front() == '[' &&
result.back() == ']') {  
result.pop_back();

for(auto &var : CLI::detail::split(result.substr(1), ',')) {
if(!var.empty()) {
result_count += _add_result(std::move(var), res);
}
}
return result_count;
}
if(delimiter_ == '\0') {
res.push_back(std::move(result));
++result_count;
} else {
if((result.find_first_of(delimiter_) != std::string::npos)) {
for(const auto &var : CLI::detail::split(result, delimiter_)) {
if(!var.empty()) {
res.push_back(var);
++result_count;
}
}
} else {
res.push_back(std::move(result));
++result_count;
}
}
return result_count;
}
};  

}  
