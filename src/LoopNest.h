#ifndef HALIDE_INTERNAL_LOOP_NEST_H
#define HALIDE_INTERNAL_LOOP_NEST_H

#include "LoopNestFeaturizer.h"
#include "PipelineFeatures.h"
#include "ScheduleFeatures.h"
#include "Simplify.h"
#include "Substitute.h"
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

struct OutputSize {
  Function func{};
  std::vector<int> strides{};
  std::vector<int> mins{};
  std::vector<int> extents{};

  OutputSize() = default;

  OutputSize(const Function& func) 
    : func{func}
  {}

  void add(int stride, int min, int extent) {
    strides.push_back(stride);
    mins.push_back(min);
    extents.push_back(extent);
  }

  int size() const {
    return strides.size();
  }
};

struct Compute {
  std::string func;
  Expr index;
  Expr value;
};

// For testing
struct PipelineLoop {
  std::string func_name;
  std::string var_name;
  bool parallel;
  bool vectorized;
  bool unrolled;
  std::set<std::string> compute_here{}; 
  std::vector<Compute> compute_here_{}; 
  std::set<std::string> store_here{}; 
  Expr min{};
  Expr extent{};

  PipelineLoop(const std::string& func_name, const std::string& var_name, bool parallel, bool vectorized, bool unrolled)
    : func_name{func_name}
    , var_name{var_name}
    , parallel{parallel}
    , vectorized{vectorized}
    , unrolled{unrolled}
  {}

  void print(int depth = 0);
  vector<std::shared_ptr<PipelineLoop>> children;

  bool match(const PipelineLoop& other);

  static std::vector<std::shared_ptr<PipelineLoop>> create(const std::vector<std::string>& lines);
};

std::string basic_name(std::string name);
void basic_name_test();

template <typename T>
std::vector<double> log2_one_plus(const std::vector<T>& input) {
  std::vector<double> result;
  result.reserve(input.size());
  for (const auto& x : input) {
    result.push_back(std::log2(1 + x));
  }
  return result;
}

template <typename T>
std::vector<double> log2(const std::vector<T>& input) {
  std::vector<double> result;
  result.reserve(input.size());
  for (const auto& x : input) {
    result.push_back(std::log2(x));
  }
  return result;
}
  
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

  virtual int64_t get_non_unique_bytes_read() const {
    return 0;
  }

  virtual int64_t get_sum_of_allocs() const {
    return 0;
  }
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
  int64_t get_sum_of_allocs() const override;
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

  std::set<std::string> get_compute_funcs() const override;
  std::set<std::string> get_store_funcs() const override;

  int64_t get_non_unique_bytes_read() const override;
  int64_t get_sum_of_allocs() const override;
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
  int64_t non_unique_bytes_read_per_point;

  ComputeNode(Function func, const Expr& arg, const std::vector<Expr>& values, const BlockNode* parent, const ScheduleFeatures& schedule_features, const PipelineFeatures& pipeline_features, int64_t non_unique_bytes_read_per_point);

  void featurize();
  std::vector<const LoopNode*> get_loop_stack() const;
  std::vector<std::shared_ptr<PipelineLoop>> create_pipeline_loop_nest() const override;
  std::set<std::string> get_compute_funcs() const override;
  void dump(int indent_level = 0) const override;
  json to_json() const override;
  int64_t get_non_unique_bytes_read() const override;
};

struct LoopNode : LoopLevelNode {
  Function func;
  std::string var_name;
  Expr var;
  int stage_index;
  int64_t extent;
  int vector_size;
  const BlockNode* parent = nullptr;
  bool parallel;
  bool unrolled;
  std::unique_ptr<BlockNode> body;
  TailStrategy tail_strategy;
  int product_of_outer_loops;
  int64_t unique_bytes_read;

  static std::string MakeVarName(Function f, int stage_index, int depth, VarOrRVar var, bool parallel);

  LoopNode(Function f, int stage_index, int64_t extent, int vector_size, const BlockNode* parent, int depth, bool parallel, TailStrategy tail_strategy, VarOrRVar var, bool unrolled, int product_of_outer_loops, int64_t unique_bytes_read);

  std::vector<std::shared_ptr<PipelineLoop>> create_pipeline_loop_nest() const override;
  void dump(int indent_level = 0) const override;
  json to_json() const override;
  int64_t get_non_unique_bytes_read() const override;
  int64_t get_sum_of_allocs() const override;
};

struct LoweredFuncToLoopNest : IRVisitor {
  using IRVisitor::visit;

  bool finished_preamble = false; 
  std::vector<std::string> productions;
  std::vector<std::shared_ptr<PipelineLoop>> root_loops;
  std::vector<std::shared_ptr<PipelineLoop>> loops;
  std::map<std::string, Expr> table;

  LoweredFuncToLoopNest(const std::map<std::string, OutputSize>& output_sizes) {
    for (const auto& output_size : output_sizes) {
      for (int i = 0; i < output_size.second.size(); i++) {
        const auto& func_name = output_size.first;
        table[func_name + ".min." + std::to_string(i)] = output_size.second.mins[i];
        table[func_name + ".extent." + std::to_string(i)] = output_size.second.extents[i];
        table[func_name + ".stride." + std::to_string(i)] = output_size.second.strides[i];
      }
    }
  }

  void visit(const IfThenElse *op) override {
    // Visit body of bounds check if statements but skip others e.g.
    // GuardWithIfs
    if (!finished_preamble) {
      op->then_case.accept(this);
    }
  }

  void visit(const LetStmt *op) override {
    //std::cout << op->name << std::endl;
    //std::cout << op->value << std::endl;
    if (table.count(op->name) == 0) {
      table[op->name] = simplify(substitute(table, op->value));
    }
    //std::cout << table[op->name] << "\n" << std::endl;

    op->body.accept(this);
  }

  void visit(const Let *op) override {
    std::cout << op->name << std::endl;
  }

  void visit(const ProducerConsumer *op) override {
    if (op->is_producer) {
      finished_preamble = true;
    }
    productions.push_back(op->name);
    op->body.accept(this);
    productions.pop_back();
  }

  void visit(const For *op) override {
    const auto& func_name = op->name.substr(0, op->name.find("."));
    std::shared_ptr<PipelineLoop> l = std::make_shared<PipelineLoop>(
      func_name
      , op->name
      , op->for_type == ForType::Parallel 
      , op->for_type == ForType::Vectorized
      , op->for_type == ForType::Unrolled 
    );

    l->min = simplify(substitute(table, op->min));
    l->extent = simplify(substitute(table, op->extent));
    
    if (loops.size() == 0) {
      root_loops.push_back(l);
    } else {
      loops.back()->children.push_back(l);
    }
    
    loops.push_back(l);
    //table[op->name] = (op->name - op->min) * table[op->name.substr(0, op->name.find(".")) + ".extent." + var_index];
    op->body.accept(this);
    loops.pop_back();
  }

  void visit(const Store *op) override {
    loops.back()->compute_here_.push_back(Compute{
      op->name
      , simplify(substitute(table, op->index))
      , simplify(substitute(table, op->value))
    });
  }

  void create(const std::vector<Function>& outputs) {

  }
};

struct LoopNestRoot {
  BlockNode block;
  std::map<std::string, OutputSize> output_sizes;

  std::vector<std::shared_ptr<PipelineLoop>> get_lowered_loops(const std::vector<Function> &outputs);
  bool matches_pipeline_loop_nest(const std::vector<Function> &outputs);
  bool matches_lowered_loop_nest(const std::vector<Function> &outputs);
  void dump() const;
};

} // namespace Internal
} // namespace Halide

#endif // HALIDE_INTERNAL_LOOP_NEST_H
