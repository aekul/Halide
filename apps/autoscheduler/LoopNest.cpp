#include "CSE.h"
#include "LoopNest.h"
#include "Simplify.h"

#include <regex>

namespace Halide {
namespace Internal {

using std::vector;

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

void AllocNode::dump(int indent_level) const {
  std::cout << indent(indent_level) << "alloc " << name;
  //for (size_t i = 0; i < region.size(); i++) {
    //std::cout << "[" << region[i] << "]";
  //}
  std::cout << "[" << size << "]";
  std::cout << "\n";
}

json AllocNode::to_json() const {
  json jdata;
  jdata["type"] = "alloc";
  jdata["name"] = name;

  jdata["features"] = {
    size
    , std::log2(size)
    , bytes_per_point
    , std::log2(bytes_per_point)
    , size * bytes_per_point
    , std::log2(size * bytes_per_point)
    , should_store_on_stack
  };

  return jdata;
}

std::vector<std::shared_ptr<PipelineLoop>> AllocNode::create_pipeline_loop_nest() const {
  return {};
}

std::set<std::string> AllocNode::get_store_funcs() const {
  return {basic_name(name)};
}

int64_t AllocNode::get_sum_of_allocs() const {
  return size;
}

int64_t AllocNode::get_bytes_loaded_from_nested_producers(int64_t prod_outer_extents, const std::set<std::string>& producers) const {
  if (producers.count(name) == 0) {
    return 0;
  }

  return prod_outer_extents * size * bytes_per_point;
}

void BlockNode::add_child(std::unique_ptr<LoopLevelNode> child) {
  children.push_back(std::move(child));
}

void BlockNode::add_child(std::unique_ptr<AllocNode> child) {
  stored_here.insert(child->name);
  child->parent = this;
  children.push_back(std::move(child));
}

void BlockNode::dump(int indent_level) const {
  for (const auto& child : children) {
    child->dump(indent_level);
  }
}

json BlockNode::to_json() const {
  json jdata;
  jdata["type"] = "block";
  for (const auto& child : children) {
    jdata["children"].push_back(child->to_json());
  }
  return jdata;
}

int64_t BlockNode::get_non_unique_bytes_read() const {
  int64_t bytes_read = 0;
  for (const auto& child : children) {
    bytes_read += child->get_non_unique_bytes_read();
  }
  return bytes_read;
}

std::vector<std::shared_ptr<PipelineLoop>> BlockNode::create_pipeline_loop_nest() const {
  std::vector<std::shared_ptr<PipelineLoop>> loops;

  for (const auto& c : children) {
    std::vector<std::shared_ptr<PipelineLoop>> child_loops = c->create_pipeline_loop_nest();

    for (const auto& l : child_loops) {
      loops.push_back(l);
    }
  }

  return loops;
}

std::set<std::string> BlockNode::get_compute_funcs() const {
  std::set<std::string> funcs;
  for (const auto& c : children) {
    std::set<std::string> child_funcs = c->get_compute_funcs();

    for (const auto& f : child_funcs) {
      funcs.insert(f);
    }
  }
  return funcs;
}

std::set<std::string> BlockNode::get_store_funcs() const {
  std::set<std::string> funcs;
  for (const auto& c : children) {
    std::set<std::string> child_funcs = c->get_store_funcs();

    for (const auto& f : child_funcs) {
      funcs.insert(f);
    }
  }
  return funcs;
}

int64_t BlockNode::get_sum_of_allocs() const {
  int64_t sum = 0;
  for (const auto& c : children) {
    sum += c->get_sum_of_allocs();
  }
  return sum;
}

int64_t BlockNode::get_num_bytes_computed() const {
  int64_t sum = 0;
  for (const auto& c : children) {
    sum += c->get_num_bytes_computed();
  }
  return sum;
}

int64_t BlockNode::get_bytes_loaded_from_nested_producers(int64_t prod_outer_extents, const std::set<std::string>& producers) const {
  int64_t sum = 0;
  for (const auto& c : children) {
    sum += c->get_bytes_loaded_from_nested_producers(prod_outer_extents, producers);
  }
  return sum;
}

int64_t BlockNode::get_bytes_loaded_from_external_producers() const {
  int64_t sum = 0;
  for (const auto& c : children) {
    sum += c->get_bytes_loaded_from_external_producers();
  }
  return sum;
}

ComputeNode::ComputeNode(Function func, const Expr& arg, const std::vector<Expr>& values, const BlockNode* parent, const ScheduleFeatures& schedule_features, const PipelineFeatures& pipeline_features, int64_t non_unique_bytes_read_per_point, int64_t bytes_per_point)
  : func{func}
  , arg{arg}
  , values{values}
  , parent{parent}
  , schedule_features{schedule_features}
  , pipeline_features{pipeline_features}
  , non_unique_bytes_read_per_point{non_unique_bytes_read_per_point}
  , bytes_per_point{bytes_per_point}
{
  featurize();
}

void ComputeNode::featurize() {
  auto loops = get_loop_stack();
  LoopNestFeaturizer featurizer(func, features, loops, store_jacobian, load_jacobians);

  memset(&features, 0, sizeof(features));

  for (auto v : values) {
    featurizer.visit_store_args(v.type(), arg);
    v = common_subexpression_elimination(simplify(v)); // Get things into canonical form
    v.accept(&featurizer);
  }

  auto v = common_subexpression_elimination(simplify(arg)); // Get things into canonical form
  v.accept(&featurizer);
} 

std::vector<const LoopNode*> ComputeNode::get_loop_stack() const {
  const LoopNode* loop = parent->parent;

  std::vector<const LoopNode*> loops;

  while (loop) {
    loops.push_back(loop);
    loop = loop->parent->parent;
  }

  return loops;
}

void ComputeNode::dump(int indent_level) const {
  std::cout << indent(indent_level) << func.name() << "(";

  //for (int i = 0, N = args.size(); i < N; i++) {
    std::cout << arg;

    //if (i < N - 1) {
      //std::cout << ", ";
    //}
  //}
  std::cout << ") = ";

  std::cout << values[0];
  std::cout << "\n";
}

json ComputeNode::to_json() const {
  json jdata;
  jdata["type"] = "compute";
  jdata["name"] = func.name();
  jdata["loop_nest_pipeline_features"] = features.json_dump();
  jdata["pipeline_features"] = pipeline_features.json_dump();
  auto schedule_features_vector = schedule_features.to_vector();
  schedule_features_vector.push_back(non_unique_bytes_read_per_point);
  jdata["schedule_features"] = schedule_features_vector;

  for (const auto& s : log2_one_plus(schedule_features_vector)) {
    jdata["schedule_features"].push_back(s);
  }
  //jdata["store_jacobian"] = store_jacobian.to_json();
  //for (const auto& j : load_jacobians) {
    //jdata["load_jacobians"].push_back({
      //{"name", j.first}
      //, {"jacobian", j.second.to_json()}
    //});
  //}
  return jdata;
}

std::vector<std::shared_ptr<PipelineLoop>> ComputeNode::create_pipeline_loop_nest() const {
  return {};
}

std::string basic_name(std::string name) {
  std::regex depth_pattern{"(.*)(?:\\.[0-9]+)$"};
  std::smatch match;
  if (std::regex_match(name, match, depth_pattern)) {
    name = match[1].str();
  }

  std::regex dollar_pattern{"([^$]*)(?:\\$[^.$]*)(.*)"};
  while (std::regex_match(name, match, dollar_pattern)) {
    name = match[1].str() + match[2].str();
  }

  std::regex stage_pattern{"([a-zA-Z0-9_]+)(?:\\.s[0-9]+)?(.*)"};
  std::regex_match(name, match, stage_pattern);
  internal_assert(match.size() == 3);
  return match[1].str() + match[2].str();
}

void basic_name_test() {
  internal_assert(basic_name("f") == "f") << basic_name("f");
  internal_assert(basic_name("f.s0") == "f") << basic_name("f.s0");
  internal_assert(basic_name("f.s0.1") == "f") << basic_name("f.s0.1");
  internal_assert(basic_name("f.s0.x.1") == "f.x") << basic_name("f.s0.x.1");
  internal_assert(basic_name("f.s0.x$18.1") == "f.x") << basic_name("f.s0.x$18.1");
  internal_assert(basic_name("f$2.s0.x$18.1") == "f.x") << basic_name("f$2.s0.x$18.1");
  internal_assert(basic_name("f.x.1") == "f.x") << basic_name("f.x.1");
  internal_assert(basic_name("f.x.10") == "f.x") << basic_name("f.x.10");
  internal_assert(basic_name("f.x_inner.10") == "f.x_inner") << basic_name("f.x_inner.10");
  internal_assert(basic_name("f.x$3.10") == "f.x") << basic_name("f.x$3.10");
  internal_assert(basic_name("f.x.x_inner.1") == "f.x.x_inner") << basic_name("f.x.x_inner.1");
  internal_assert(basic_name("f.r96463$x") == "f.r96463") << basic_name("f.r96463");
}

std::set<std::string> ComputeNode::get_compute_funcs() const {
  return {basic_name(func.name())};
}

int64_t ComputeNode::get_non_unique_bytes_read() const {
  return non_unique_bytes_read_per_point;
}

int64_t ComputeNode::get_num_bytes_computed() const {
  return bytes_per_point;
}

std::string LoopNode::MakeVarName(Function f, int stage_index, int depth, VarOrRVar var, bool parallel) {
  std::ostringstream var_name;
  var_name << f.name();
  var_name << ".s" << stage_index;
  var_name << "." << var.name();
  if (parallel && var.name().rfind("_par") == std::string::npos) {
    var_name << "_par";
  }
  var_name << "." << depth;
  return var_name.str();
}

LoopNode::LoopNode(Function f, int stage_index, int64_t extent, bool vectorized, const BlockNode* parent, int depth, bool parallel, TailStrategy tail_strategy, VarOrRVar var, bool unrolled, int product_of_outer_loops, int64_t bytes_loaded_from_external_producers, const std::set<std::string>& nested_producers)
  : func{f}
  , var_name{MakeVarName(f, stage_index, depth, var, parallel)}
  , var{Variable::make(Int(32), var_name)}
  , stage_index{stage_index}
  , extent{extent}
  , vectorized{vectorized}
  , parent{parent}
  , parallel{parallel}
  , unrolled{unrolled}
  , body{make_unique<BlockNode>()}
  , tail_strategy{tail_strategy}
  , product_of_outer_loops{product_of_outer_loops}
  , bytes_loaded_from_external_producers{bytes_loaded_from_external_producers}
  , nested_producers{nested_producers}
{
  body->parent = this;
}

void LoopNode::dump(int indent_level) const {
  std::cout << indent(indent_level);
  std::cout << "for " << var << " in [0, " << extent << ")";

  if (parallel) {
    std::cout << " parallel";
  }

  if (unrolled) {
    std::cout << " unrolled";
  }
  std::cout << ":\n";
  body->dump(indent_level + 1);
}

json LoopNode::to_json() const {
  json jdata;
  jdata["type"] = "loop";
  std::stringstream s;
  s << var;
  jdata["var"] = s.str();

  int64_t non_unique_bytes_read = get_non_unique_bytes_read();
  int64_t unique_bytes_read = get_bytes_loaded_from_external_producers() + get_bytes_loaded_from_nested_producers(1, nested_producers);
  float bytes_read_ratio = (float)non_unique_bytes_read / (float)(1 + unique_bytes_read);
  int64_t working_set = get_sum_of_allocs();
  int64_t num_bytes_computed = get_num_bytes_computed();


  jdata["features"] = {
    extent
    , std::log2(extent)
    , product_of_outer_loops
    , non_unique_bytes_read
    , std::log2(1 + non_unique_bytes_read)
    , unique_bytes_read
    , std::log2(1 + unique_bytes_read)
    , bytes_read_ratio
    , std::log2(1 + bytes_read_ratio)
    , working_set
    , std::log2(1 + working_set)
    , num_bytes_computed
    , std::log2(num_bytes_computed)
    , parallel
    , unrolled
    , vectorized
    , tail_strategy == TailStrategy::RoundUp
    , tail_strategy == TailStrategy::GuardWithIf
    , tail_strategy == TailStrategy::ShiftInwards
    , tail_strategy == TailStrategy::Auto
  };

  jdata["block"] = body->to_json();
  return jdata;
}

int64_t LoopNode::get_non_unique_bytes_read() const {
  return extent * body->get_non_unique_bytes_read();
}

int64_t LoopNode::get_sum_of_allocs() const {
  return body->get_sum_of_allocs();
}

int64_t LoopNode::get_num_bytes_computed() const {
  return extent * body->get_num_bytes_computed();
}

int64_t LoopNode::get_bytes_loaded_from_nested_producers(int64_t prod_outer_extents, const std::set<std::string>& producers) const {
  return body->get_bytes_loaded_from_nested_producers(extent * prod_outer_extents, producers);
}

int64_t LoopNode::get_bytes_loaded_from_external_producers() const {
  return bytes_loaded_from_external_producers + body->get_bytes_loaded_from_external_producers();
}

std::vector<std::shared_ptr<PipelineLoop>> LoopNode::create_pipeline_loop_nest() const {
  std::vector<std::shared_ptr<PipelineLoop>> loops;
  std::stringstream s;
  s << var;

  std::shared_ptr<PipelineLoop> l = std::make_shared<PipelineLoop>(
    func.name().substr(0, func.name().find("$"))
    , basic_name(s.str())
    , parallel
    , vectorized
    , unrolled
  );

  l->children = body->create_pipeline_loop_nest();
  l->compute_here = body->get_compute_funcs();
  l->store_here = body->get_store_funcs();
  loops.push_back(l);
  return loops;
}

void PipelineLoop::print(int depth) {
  std::cout << std::string(2 * depth, ' ');
  std::cout << "for " << var_name;
  if (min.defined() && extent.defined()) {
    std::cout << " in [" << min << ", " << extent << "]";
  }

  if (parallel) {
    std::cout << " parallel";
  }
  if (unrolled) {
    std::cout << " unrolled";
  }

  std::cout << ":\n";

  for (const auto& s : store_here) {
    std::cout << std::string(2 * (depth + 1), ' ');
    std::cout << "store " << s << "\n";
  }

  for (const auto& c : children) {
    c->print(depth + 1);
  }

  for (const auto& c : compute_here) {
    std::cout << std::string(2 * (depth + 1), ' ');
    std::cout << "compute " << c << "\n";
  }

  for (const auto& c : compute_here_) {
    std::cout << std::string(2 * (depth + 1), ' ');
    std::cout << c.func << "[" << c.index << "] = " << c.value << "\n";
  }
}

std::vector<std::shared_ptr<PipelineLoop>> PipelineLoop::create(const std::vector<std::string>& lines) {
  std::vector<std::pair<std::string, int>> productions;
  std::vector<std::pair<std::shared_ptr<PipelineLoop>, int>> loops;
  std::vector<std::shared_ptr<PipelineLoop>> root_loops;

  std::set<std::string> stored;

  std::string var_pattern{"(?:([a-zA-Z0-9_]*)\\.)*([a-zA-Z0-9_]+)(?: in \\[0, ([0-9]+)\\])?:"};
  std::regex produce_regex("( *)produce ([a-zA-Z0-9_]+):");
  std::regex consume_regex("( *)consume ([a-zA-Z0-9_]+):");
  std::regex store_regex("( *)store ([a-zA-Z0-9_]+):");
  std::regex loop_regex("( *)(parallel|vectorized|unrolled|for) " + var_pattern);
  std::regex compute_regex("( *)([a-zA-Z0-9_]+)\\(\\.\\.\\.\\) = \\.\\.\\.");

  auto create_loop = [&](const std::string& var_name, int depth, bool parallel, bool vectorized, bool unrolled) {
    const auto& func_name = productions.back().first;
    std::pair<std::string, std::string> key{func_name, var_name};

    std::string name{func_name + "." + var_name};
    if (parallel) {
      name += "_par";
    }

    std::shared_ptr<PipelineLoop> l = std::make_shared<PipelineLoop>(
      func_name
      , name 
      , parallel
      , vectorized
      , unrolled
    );

    if (loops.size() > 0) {
      loops.back().first->children.push_back(l);
    } else {
      root_loops.push_back(l);
    }
    loops.push_back({l, depth});
  };

  for (int i = 0; i < (int)lines.size(); i++) {
    const auto& line = lines[i];

    std::smatch match;

    bool parallel = false;
    bool vectorized = false;
    bool unrolled = false;
    bool production = false;
    bool compute = false;
    bool store = false;
    bool loop = false;
    int name_index = 2;
    
    if (std::regex_match(line, match, produce_regex)) {
      production = true;
    } else if (std::regex_match(line, match, consume_regex)) {
    } else if (std::regex_match(line, match, store_regex)) {
      store = true;
    } else if (std::regex_match(line, match, compute_regex)) {
      compute = true;
    } else if (std::regex_match(line, match, loop_regex)) {
      parallel = match[2].str() == "parallel";
      vectorized = match[2].str() == "vectorized";
      unrolled = match[2].str() == "unrolled";
      loop = true;
      name_index = 4;
    } else {
      internal_assert(false);
    }

    int depth = match[1].str().size() / 2;
    auto name = match[name_index].str();

    while (loops.size() > 0 && depth <= loops.back().second) {
      loops.pop_back();
    }

    while (productions.size() > 0 && depth <= productions.back().second) {
      productions.pop_back();
    }

    if (production) {
      // If func hasn't been explicitly stored somewhere, it must be stored at
      // its compute site
      if (stored.count(name) == 0) {
        store = true;
      }
      productions.push_back({name, depth});
    }

    if (loop) {
      create_loop(name, depth, parallel, vectorized, unrolled);
    }

    if (compute) {
      internal_assert(loops.size() > 0);
      loops.back().first->compute_here.insert(name);
    }

    if (store) {
      if (loops.size() > 0) {
        loops.back().first->store_here.insert(name);
      }
      stored.insert(name);
    }
  }

  return root_loops;
}

bool PipelineLoop::match(const PipelineLoop& other) {
  if (func_name != other.func_name) {
    internal_assert(false) << func_name << " " << other.func_name << "\n";
    return false;
  }

  if (parallel != other.parallel || unrolled != other.unrolled || vectorized != other.vectorized) {
    return false;
  }

  if (children.size() != other.children.size()) {
    return false;
  }

  auto other_name = basic_name(other.var_name);
  if (var_name != other_name) {
    internal_assert(false) << var_name << " " << other_name << "\n";
    return false;
  }

  if (compute_here != other.compute_here) {
    return false;
  }

  if (store_here != other.store_here) {
    return false;
  }

  for (int i = 0; i < (int)children.size(); i++) {
    if (!children[i]->match(*other.children[i])) {
      return false;
    }
  }

  return true;
}

bool LoopNestRoot::matches_pipeline_loop_nest(const std::vector<Function> &outputs) {
  std::string loop_nest;
  for (const auto& f : outputs) {
    loop_nest += Func(f).get_loop_nest();
  }

  std::vector<std::string> lines;
  std::stringstream ss{loop_nest};
  for (std::string line; std::getline(ss, line); ) {
    std::cout << line << "\n";
    lines.push_back(line);
  }

  std::cout << "loop nest: \n";
  dump();

  std::vector<std::shared_ptr<PipelineLoop>> lowered_loops = get_lowered_loops(outputs);
  std::cout << "\nlowered: \n";
  for (const auto& l : lowered_loops) {
    l->print();
  }

  const auto& pipeline_loops = PipelineLoop::create(lines);
  const auto& block_loops = block.create_pipeline_loop_nest();

  std::cout << "pipeline: \n";
  for (const auto& l : pipeline_loops) {
    l->print();
  }

  std::cout << "\nblock: \n";
  for (const auto& l : block_loops) {
    l->print();
  }

  if (pipeline_loops.size() != block_loops.size()) {
    return false;
  }

  for (int i = 0; i < (int)pipeline_loops.size(); i++) {
    if (!pipeline_loops[i]->match(*block_loops[i])) {
      return false;
    }
  }

  return true;
}

std::vector<std::shared_ptr<PipelineLoop>> LoopNestRoot::get_lowered_loops(const std::vector<Function> &outputs) {
  std::vector<std::shared_ptr<PipelineLoop>> lowered_loops;

  for (const auto& f : outputs) {
    Module m = Func(f).get_compiled_to_lowered_stmt({});
    LoweredFuncToLoopNest lf_to_loop_nest{output_sizes};
    
    auto name = f.name();
    auto n = name.find("$");
    if (n != std::string::npos) {
      name.replace(n, 1, "_");
    }
    m.get_function_by_name(name).body.accept(&lf_to_loop_nest);
    for (const auto& l : lf_to_loop_nest.root_loops) {
      lowered_loops.push_back(l);
    }
  }

  return lowered_loops;
}

bool LoopNestRoot::matches_lowered_loop_nest(const std::vector<Function> &outputs) {
  std::vector<std::shared_ptr<PipelineLoop>> lowered_loops = get_lowered_loops(outputs);

  const auto& block_loops = block.create_pipeline_loop_nest();

  std::cout << "lowered: \n";
  for (const auto& l : lowered_loops) {
    l->print();
  }

  std::cout << "\nblock: \n";
  for (const auto& l : block_loops) {
    l->print();
  }

  if (lowered_loops.size() != block_loops.size()) {
    return false;
  }

  for (int i = 0; i < (int)lowered_loops.size(); i++) {
    if (!lowered_loops[i]->match(*block_loops[i])) {
      return false;
    }
  }

  return true;
}

void LoopNestRoot::dump() const {
  block.dump();
}

} // namespace Internal
} // namespace Halide
