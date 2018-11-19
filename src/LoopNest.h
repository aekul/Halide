#ifndef HALIDE_INTERNAL_LOOP_NEST_H
#define HALIDE_INTERNAL_LOOP_NEST_H

#include "LoopNestFeaturizer.h"
#include "PipelineFeatures.h"
#include "ScheduleFeatures.h"
#include "Func.h"

#include <string>

#include <map>
#include <json.hpp>

using json = nlohmann::json;

namespace Halide {
namespace Internal {

using std::string;
using std::vector;
using std::map;

// For testing
struct PipelineLoop {
  std::string func_name;
  std::string var_name;
  int stage_index;
  bool parallel;
  bool vectorized;
  bool unrolled;
  std::set<std::string> compute_here; 
  std::set<std::string> store_here; 

  PipelineLoop(const std::string& func_name, const std::string& var_name, int stage_index, bool parallel, bool vectorized, bool unrolled)
    : func_name{func_name}
    , var_name{var_name}
    , stage_index{stage_index}
    , parallel{parallel}
    , vectorized{vectorized}
    , unrolled{unrolled}
  {}

  void print(int depth = 0);
  vector<std::shared_ptr<PipelineLoop>> children;

  bool match(const PipelineLoop& other);

  static std::vector<std::shared_ptr<PipelineLoop>> create(const std::vector<std::string>& lines);
};

  
struct LoopNode;
struct BlockNode;
struct LoopLevelNode {
  virtual void dump(int indent_level = 0) const = 0;
  virtual json to_json() const = 0;
  virtual std::vector<std::shared_ptr<PipelineLoop>> create_pipeline_loop_nest() const = 0;
  virtual std::set<std::string> get_compute_funcs() const {
    return {};
  }

  virtual std::set<std::string> get_store_funcs() const {
    return {};
  }
  virtual ~LoopLevelNode() {}
  std::string indent(int indent_level) const {
    return std::string(2 * indent_level, ' ');
  }

  std::string basic_name(const std::string& name) const;
};

struct AllocNode : LoopLevelNode {
  std::string name;
  int64_t size;
  int bytes_per_point;
  bool should_store_on_stack;
  std::vector<int64_t> region;
  const BlockNode* parent = nullptr;

  void dump(int indent_level = 0) const override;
  json to_json() const override;
  std::vector<std::shared_ptr<PipelineLoop>> create_pipeline_loop_nest() const override;
  std::set<std::string> get_store_funcs() const override;
};

struct BlockNode : LoopLevelNode {
  std::vector<std::unique_ptr<LoopLevelNode>> children;
  std::set<std::string> stored_here;
  const LoopNode* parent = nullptr;

  void add_child(std::unique_ptr<LoopLevelNode> child);

  void add_child(std::unique_ptr<AllocNode> child);

  void dump(int indent_level = 0) const override;
  std::vector<std::shared_ptr<PipelineLoop>> create_pipeline_loop_nest() const override;
  json to_json() const override;

  bool matches_pipeline_loop_nest(const std::vector<Function> &outputs);
  std::set<std::string> get_compute_funcs() const override;
  std::set<std::string> get_store_funcs() const override;
};

struct ComputeNode : LoopLevelNode {
  Function func;
  Expr arg;
  std::vector<Expr> values;
  const BlockNode* parent;
  LoopNestPipelineFeatures features;
  LoopNestFeaturizer::Jacobian store_jacobian;
  vector<std::pair<std::string, LoopNestFeaturizer::Jacobian>> load_jacobians;
  ScheduleFeatures schedule_features;
  PipelineFeatures pipeline_features;

  ComputeNode(Function func, const Expr& arg, const std::vector<Expr>& values, const BlockNode* parent, const ScheduleFeatures& schedule_features, const PipelineFeatures& pipeline_features);

  void featurize();
  std::vector<const LoopNode*> get_loop_stack() const;
  std::vector<std::shared_ptr<PipelineLoop>> create_pipeline_loop_nest() const override;
  std::set<std::string> get_compute_funcs() const override;
  void dump(int indent_level = 0) const override;
  json to_json() const override;
};

struct LoopNode : LoopLevelNode {
  Function func;
  std::string var_name;
  Expr var;
  int var_index;
  int stage_index;
  int64_t extent;
  int vector_size;
  const BlockNode* parent = nullptr;
  bool parallel;
  bool unrolled;
  std::unique_ptr<BlockNode> body;
  TailStrategy tail_strategy;

  static std::string MakeVarName(Function f, int stage_index, int depth, VarOrRVar var, bool parallel);

  LoopNode(Function f, int var_index, int stage_index, int64_t extent, int vector_size, const BlockNode* parent, int depth, bool parallel, TailStrategy tail_strategy, VarOrRVar var, bool unrolled);

  std::vector<std::shared_ptr<PipelineLoop>> create_pipeline_loop_nest() const override;
  void dump(int indent_level = 0) const override;
  json to_json() const override;
};

} // namespace Internal
} // namespace Halide

#endif // HALIDE_INTERNAL_LOOP_NEST_H
