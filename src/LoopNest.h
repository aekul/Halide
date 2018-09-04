#ifndef HALIDE_INTERNAL_LOOP_NEST_H
#define HALIDE_INTERNAL_LOOP_NEST_H

#include "LoopNestFeaturizer.h"
#include "PipelineFeatures.h"

#include <string>

#include <map>
#include <json.hpp>

using json = nlohmann::json;

namespace Halide {
namespace Internal {

using std::string;
using std::vector;
using std::map;
  
struct LoopNode;
struct BlockNode;
struct LoopLevelNode {
  virtual void dump(int indent_level = 0) const = 0;
  virtual json to_json() const = 0;
  std::string indent(int indent_level) const {
    return std::string(2 * indent_level, ' ');
  }
};

struct AllocNode : LoopLevelNode {
  std::string name;
  int64_t size;
  int bytes_per_point;
  std::vector<int64_t> region;
  const BlockNode* parent = nullptr;

  void dump(int indent_level = 0) const override;
  json to_json() const override;
};

struct BlockNode : LoopLevelNode {
  std::vector<std::unique_ptr<LoopLevelNode>> children;
  std::set<std::string> stored_here;
  const LoopNode* parent = nullptr;

  void add_child(std::unique_ptr<LoopLevelNode> child);

  void add_child(std::unique_ptr<AllocNode> child);

  void dump(int indent_level = 0) const override;
  json to_json() const override;
};

struct ComputeNode : LoopLevelNode {
  Function func;
  Expr arg;
  std::vector<Expr> values;
  const BlockNode* parent;
  LoopNestPipelineFeatures features;
  LoopNestFeaturizer::Jacobian store_jacobian;
  vector<std::pair<std::string, LoopNestFeaturizer::Jacobian>> load_jacobians;

  ComputeNode(Function func, const Expr& arg, const std::vector<Expr>& values, const BlockNode* parent);

  void featurize();
  std::vector<std::pair<std::string, bool>> get_loop_stack() const;
  void dump(int indent_level = 0) const override;
  json to_json() const override;
};

struct LoopNode : LoopLevelNode {
  Function func;
  Expr var;
  int var_index;
  int stage_index;
  int64_t extent;
  int vector_size;
  const BlockNode* parent = nullptr;
  bool parallel;
  std::unique_ptr<BlockNode> body;

  static std::string MakeVarName(Function f, int var_index, int stage_index, int vector_size, int depth);

  LoopNode(Function f, int var_index, int stage_index, int64_t extent, int vector_size, const BlockNode* parent, int depth, bool parallel);

  void dump(int indent_level = 0) const override;
  json to_json() const override;
};

} // namespace Internal
} // namespace Halide

#endif // HALIDE_INTERNAL_LOOP_NEST_H
