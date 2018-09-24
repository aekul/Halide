#include "CSE.h"
#include "LoopNest.h"
#include "Simplify.h"

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
  jdata["size"] = size;
  jdata["region"] = region;
  jdata["bytes_per_point"] = bytes_per_point;
  return jdata;
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

ComputeNode::ComputeNode(Function func, const Expr& arg, const std::vector<Expr>& values, const BlockNode* parent, const ScheduleFeatures& schedule_features, const PipelineFeatures& pipeline_features)
  : func{func}
  , arg{arg}
  , values{values}
  , parent{parent}
  , schedule_features{schedule_features}
  , pipeline_features{pipeline_features}
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
  jdata["pipeline_features"] = features.json_dump();
  jdata["schedule_features"] = schedule_features.json_dump();
  jdata["store_jacobian"] = store_jacobian.to_json();
  for (const auto& j : load_jacobians) {
    jdata["load_jacobians"].push_back({
      {"name", j.first}
      , {"jacobian", j.second.to_json()}
    });
  }
  return jdata;
}

std::string LoopNode::MakeVarName(Function f, int var_index, int stage_index, int vector_size, int depth) {
  std::ostringstream var_name;
  var_name << f.name();
  var_name << ".s" << stage_index;
  var_name << ".v" << var_index;
  if (vector_size > 1) {
    var_name << ".vec" << vector_size;
  }
  var_name << "." << depth;
  return var_name.str();
}

LoopNode::LoopNode(Function f, int var_index, int stage_index, int64_t extent, int vector_size, const BlockNode* parent, int depth, bool parallel)
  : func{f}
  , var_name{MakeVarName(f, var_index, stage_index, vector_size, depth)}
  , var{Variable::make(Int(32), var_name)}
  , var_index{var_index}
  , stage_index{stage_index}
  , extent{extent}
  , vector_size{vector_size}
  , parent{parent}
  , parallel{parallel}
  , body{make_unique<BlockNode>()}
{
  body->parent = this;
}

void LoopNode::dump(int indent_level) const {
  std::cout << indent(indent_level);
  std::cout << "for " << var << " in [0, " << extent << ")";

  if (parallel) {
    std::cout << " parallel";
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
  jdata["var_index"] = var_index;
  jdata["extent"] = extent;
  jdata["vector_size"] = vector_size;
  jdata["parallel"] = parallel;
  jdata["block"] = body->to_json();
  return jdata;
}

} // namespace Internal
} // namespace Halide
