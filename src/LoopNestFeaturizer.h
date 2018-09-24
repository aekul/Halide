#ifndef HALIDE_INTERNAL_LOOP_NEST_FEATURIZER_H
#define HALIDE_INTERNAL_LOOP_NEST_FEATURIZER_H

#include "IRVisitor.h"
#include "ModulusRemainder.h"
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
  
struct LoopNestFeaturizer : public IRVisitor {
  using IRVisitor::visit;

  struct DerivativeResult {
    bool exists;
    int64_t numerator, denominator;

    json to_json() const {
      json jdata;
      jdata["exists"] = exists;
      jdata["numerator"] = numerator;
      jdata["denominator"] = denominator; 
      return jdata;
    }

    void operator+=(const DerivativeResult &other) {
      if (!exists || !other.exists) {
        exists = false;
        return;
      }
      int64_t l = lcm(denominator, other.denominator);
      numerator *= l / denominator;
      denominator *= l / denominator;
      numerator += other.numerator * (l / other.denominator);
      int64_t g = gcd(numerator, denominator);
      numerator /= g;
      denominator /= g;
    }

    bool is_whole_number() const {
      return exists && (numerator % denominator == 0);
    }

    bool is_one() const {
      return exists && (numerator == denominator);
    }

    bool is_zero() const {
      return exists && (numerator == 0);
    }

    bool is_small_integer() const {
      return exists && (numerator == denominator ||
        numerator == denominator * 2 ||
        numerator == denominator * 3 ||
        numerator == denominator * 4);
    }
  };

  struct Jacobian {
    vector<DerivativeResult> derivatives;
    LoopNestPipelineFeatures::ScalarType scalar_type;
    LoopNestPipelineFeatures::AccessType access_type;

    std::size_t size() const {
      return derivatives.size();
    }

    json to_json() const;
  };

  LoopNestFeaturizer(
    Function &func
    , LoopNestPipelineFeatures &features
    , const std::vector<const LoopNode*>& loop_nodes
    , Jacobian& store_jacobian
    , vector<std::pair<std::string, Jacobian>>& load_jacobians
  );

  int &op_bucket(LoopNestPipelineFeatures::OpType op_type, Type scalar_type);
  LoopNestPipelineFeatures::ScalarType classify_type(Type t);

  void visit(const Variable *op) override;
  void visit(const IntImm *op) override;
  void visit(const UIntImm *op) override;
  void visit(const FloatImm *op) override;
  void visit(const Add *op) override;
  void visit(const Sub *op) override;
  void visit(const Mul *op) override;
  void visit(const Mod *op) override;
  void visit(const Div *op) override;
  void visit(const Min *op) override;
  void visit(const Max *op) override;
  void visit(const EQ *op) override;
  void visit(const NE *op) override;
  void visit(const LT *op) override;
  void visit(const LE *op) override;
  void visit(const GT *op) override;
  void visit(const GE *op) override;
  void visit(const And *op) override;
  void visit(const Or *op) override;
  void visit(const Not *op) override;
  void visit(const Select *op) override;
  void visit(const Let *op) override;
  void visit(const Call *op) override;
  void visit_store_args(Type t, Expr arg);

  // Take the derivative of an integer index expression. If it's
  // a rational constant, return it, otherwise return a sentinel
  // value.
  DerivativeResult differentiate(const Expr &e, const string &v);

  Jacobian visit_memory_access(Type t, Expr arg, LoopNestPipelineFeatures::AccessType type);

private:
  Function &func;
  LoopNestPipelineFeatures &features;
  const std::vector<const LoopNode*>& loops; // from innermost outwards
  Jacobian& store_jacobian;
  vector<std::pair<std::string, Jacobian>>& load_jacobians;
  map<std::string, Expr> let_replacements;
};

} // namespace Internal
} // namespace Halide

#endif // HALIDE_INTERNAL_LOOP_NEST_FEATURIZER_H
