

#pragma once

namespace hydra_thrust
{

struct no_traversal_tag {};

struct incrementable_traversal_tag
: no_traversal_tag {};

struct single_pass_traversal_tag
: incrementable_traversal_tag {};

struct forward_traversal_tag
: single_pass_traversal_tag {};

struct bidirectional_traversal_tag
: forward_traversal_tag {};

struct random_access_traversal_tag
: bidirectional_traversal_tag {};

} 

