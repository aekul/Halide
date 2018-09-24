#include "CSE.h"
#include "ExprUsesVar.h"
#include "LoopNest.h"
#include "LoopNestFeaturizer.h"
#include "Simplify.h"
#include "Substitute.h"

namespace Halide {
namespace Internal {

using std::vector;

LoopNestFeaturizer::LoopNestFeaturizer(
  Function &func
  , LoopNestPipelineFeatures &features
  , const std::vector<const LoopNode*>& loops
  , LoopNestFeaturizer::Jacobian& store_jacobian
  , vector<std::pair<std::string, Jacobian>>& load_jacobians
)
  : func{func}
  , features{features}
  , loops{loops}
  , store_jacobian{store_jacobian}
  , load_jacobians{load_jacobians}
{
  internal_assert(loops.size() > 0);
}

int &LoopNestFeaturizer::op_bucket(LoopNestPipelineFeatures::OpType op_type, Type scalar_type) {
  int type_bucket = (int)classify_type(scalar_type);
  features.types_in_use[type_bucket] = true;
  return features.op_histogram[(int)op_type][type_bucket];
}

LoopNestPipelineFeatures::ScalarType LoopNestFeaturizer::classify_type(Type t) {
  if (t.is_float() && t.bits() > 32) {
    return LoopNestPipelineFeatures::ScalarType::Double;
  } else if (t.is_float()) {
    return LoopNestPipelineFeatures::ScalarType::Float;
  } else if (t.bits() == 1) {
    return LoopNestPipelineFeatures::ScalarType::Bool;
  } else if (t.bits() <= 8) {
    return LoopNestPipelineFeatures::ScalarType::UInt8;
  } else if (t.bits() <= 16) {
    return LoopNestPipelineFeatures::ScalarType::UInt16;
  } else if (t.bits() <= 32) {
    return LoopNestPipelineFeatures::ScalarType::UInt32;
  } else {
    return LoopNestPipelineFeatures::ScalarType::UInt64;
  }
}

void LoopNestFeaturizer::visit(const Variable *op) {
  if (op->param.defined()) {
    op_bucket(LoopNestPipelineFeatures::OpType::Param, op->type)++;
  } else {
    op_bucket(LoopNestPipelineFeatures::OpType::Variable, op->type)++;
  }
}

void LoopNestFeaturizer::visit(const IntImm *op) {
  op_bucket(LoopNestPipelineFeatures::OpType::Const, op->type)++;
}

void LoopNestFeaturizer::visit(const UIntImm *op) {
  op_bucket(LoopNestPipelineFeatures::OpType::Const, op->type)++;
}

void LoopNestFeaturizer::visit(const FloatImm *op) {
  op_bucket(LoopNestPipelineFeatures::OpType::Const, op->type)++;
}

void LoopNestFeaturizer::visit(const Add *op) {
  op_bucket(LoopNestPipelineFeatures::OpType::Add, op->type)++;
  IRVisitor::visit(op);
}

void LoopNestFeaturizer::visit(const Sub *op) {
  op_bucket(LoopNestPipelineFeatures::OpType::Sub, op->type)++;
  IRVisitor::visit(op);
}

void LoopNestFeaturizer::visit(const Mul *op) {
  op_bucket(LoopNestPipelineFeatures::OpType::Mul, op->type)++;
  IRVisitor::visit(op);
}

void LoopNestFeaturizer::visit(const Mod *op) {
  op_bucket(LoopNestPipelineFeatures::OpType::Mod, op->type)++;
  IRVisitor::visit(op);
}

void LoopNestFeaturizer::visit(const Div *op) {
  op_bucket(LoopNestPipelineFeatures::OpType::Div, op->type)++;
  IRVisitor::visit(op);
}

void LoopNestFeaturizer::visit(const Min *op) {
  op_bucket(LoopNestPipelineFeatures::OpType::Min, op->type)++;
  IRVisitor::visit(op);
}

void LoopNestFeaturizer::visit(const Max *op) {
  op_bucket(LoopNestPipelineFeatures::OpType::Max, op->type)++;
  IRVisitor::visit(op);
}

void LoopNestFeaturizer::visit(const EQ *op) {
  op_bucket(LoopNestPipelineFeatures::OpType::EQ, op->type)++;
  IRVisitor::visit(op);
}

void LoopNestFeaturizer::visit(const NE *op) {
  op_bucket(LoopNestPipelineFeatures::OpType::NE, op->type)++;
  IRVisitor::visit(op);
}

void LoopNestFeaturizer::visit(const LT *op) {
  op_bucket(LoopNestPipelineFeatures::OpType::LT, op->type)++;
  IRVisitor::visit(op);
}

void LoopNestFeaturizer::visit(const LE *op) {
  op_bucket(LoopNestPipelineFeatures::OpType::LE, op->type)++;
  IRVisitor::visit(op);
}

void LoopNestFeaturizer::visit(const GT *op) {
  // Treat as a flipped LT
  op_bucket(LoopNestPipelineFeatures::OpType::LT, op->type)++;
  IRVisitor::visit(op);
}

void LoopNestFeaturizer::visit(const GE *op) {
  op_bucket(LoopNestPipelineFeatures::OpType::LE, op->type)++;
  IRVisitor::visit(op);
}

void LoopNestFeaturizer::visit(const And *op) {
  op_bucket(LoopNestPipelineFeatures::OpType::And, op->type)++;
  IRVisitor::visit(op);
}

void LoopNestFeaturizer::visit(const Or *op) {
  op_bucket(LoopNestPipelineFeatures::OpType::Or, op->type)++;
  IRVisitor::visit(op);
}

void LoopNestFeaturizer::visit(const Not *op) {
  op_bucket(LoopNestPipelineFeatures::OpType::Not, op->type)++;
  IRVisitor::visit(op);
}

void LoopNestFeaturizer::visit(const Select *op) {
  op_bucket(LoopNestPipelineFeatures::OpType::Select, op->type)++;
  IRVisitor::visit(op);
}

void LoopNestFeaturizer::visit(const Let *op) {
  op_bucket(LoopNestPipelineFeatures::OpType::Let, op->type)++;
  let_replacements[op->name] = op->value;
  IRVisitor::visit(op);
}

void LoopNestFeaturizer::visit(const Call *op) {
  IRVisitor::visit(op);
  LoopNestFeaturizer::Jacobian j;
  if (op->call_type == Call::Halide) {
    // args should be flattened to a single element
    user_assert(op->args.size() == 1);
    if (op->name == func.name()) {
      j = visit_memory_access(op->type, op->args.at(0), LoopNestPipelineFeatures::AccessType::LoadSelf);
      op_bucket(LoopNestPipelineFeatures::OpType::SelfCall, op->type)++;
    } else {
      j = visit_memory_access(op->type, op->args.at(0), LoopNestPipelineFeatures::AccessType::LoadFunc);
      op_bucket(LoopNestPipelineFeatures::OpType::FuncCall, op->type)++;
    }
  } else if (op->call_type == Call::Extern || op->call_type == Call::PureExtern) {
    op_bucket(LoopNestPipelineFeatures::OpType::ExternCall, op->type)++;
    return;
  } else if (op->call_type == Call::Image) {
    // args should be flattened to a single element
    user_assert(op->args.size() == 1);
    j = visit_memory_access(op->type, op->args.at(0), LoopNestPipelineFeatures::AccessType::LoadImage);
    op_bucket(LoopNestPipelineFeatures::OpType::ImageCall, op->type)++;
  }

  load_jacobians.push_back({op->name, j});
}

// Take the derivative of an integer index expression. If it's
// a rational constant, return it, otherwise return a sentinel
// value.
LoopNestFeaturizer::DerivativeResult LoopNestFeaturizer::differentiate(const Expr &e, const string &v) {
  if (!expr_uses_var(e, v)) {
    return {true, 0, 1};
  } else if (e.as<Variable>()) {
    return {true, 1, 1};
  } else if (const Add *op = e.as<Add>()) {
    auto a = differentiate(op->a, v);
    a += differentiate(op->b, v);
    return a;
  } else if (const Sub *op = e.as<Sub>()) {
    auto a = differentiate(op->a, v);
    auto b = differentiate(op->b, v);
    b.numerator = -b.numerator;
    a += b;
    return a;
  } else if (const Mul *op = e.as<Mul>()) {
    if (const int64_t *ib = as_const_int(op->b)) {
      auto a = differentiate(op->a, v);
      a.numerator *= *ib;
      return a;
    } else {
      return {false, 0, 0};
    }
  } else if (const Div *op = e.as<Div>()) {
    if (const int64_t *ib = as_const_int(op->b)) {
      auto a = differentiate(op->a, v);
      a.denominator *= *ib;
      return a;
    } else {
      return {false, 0, 0};
    }
  } else {
    // TODO: min, max?
    return {false, 0, 0};
  }
}

LoopNestFeaturizer::Jacobian LoopNestFeaturizer::visit_memory_access(Type t, Expr arg, LoopNestPipelineFeatures::AccessType type) {
  // Compute matrix of partial derivatives of args w.r.t. loop params
  LoopNestFeaturizer::Jacobian j;
  j.derivatives.resize(loops.size());

  bool is_strided = true;
  bool is_vector = true;
  std::vector<int> is_scalar;
  bool is_pointwise = true;
  //bool is_broadcast = false;
  //bool is_slice = true;
  bool is_constant = true;

  int stride = 1;
  for (size_t i = 0; i < loops.size(); i++) {
    arg = substitute(let_replacements, arg);
    j.derivatives[i] = differentiate(arg, loops[i]->var_name);
    auto deriv = j.derivatives[i];

    is_pointwise &= deriv.is_whole_number() && (deriv.numerator / deriv.denominator) == stride;
    stride *= loops[i]->extent;

    // Loop arg is vectorized
    if (loops[i]->vector_size > 1) {
      is_vector &= deriv.is_one();
    }

    // Innermost loop
    if (i == 0) {
      is_strided &= deriv.is_small_integer();
    }

    is_scalar.push_back(deriv.is_zero());
    is_constant &= deriv.is_zero();

    //is_broadcast |= deriv.is_zero();
    //is_slice &= deriv.is_one();
  }


  auto type_class = classify_type(t);

  bool is_gather_scatter = !is_vector && !is_strided && !is_scalar[0];

  features.pointwise_accesses[(int)type][(int)type_class] += is_pointwise;
  //features.transpose_accesses[(int)type][(int)type_class] += is_transpose;
  //features.broadcast_accesses[(int)type][(int)type_class] += is_broadcast;
  //features.slice_accesses[(int)type][(int)type_class] += is_slice;
  features.vectorizable_accesses[(int)type][(int)type_class] += is_vector;
  features.strided_accesses[(int)type][(int)type_class] += is_strided;
  features.scalar_accesses[(int)type][(int)type_class] += is_scalar[0];
  features.constant_accesses[(int)type][(int)type_class] += is_constant;
  features.gather_scatter_accesses[(int)type][(int)type_class] += is_gather_scatter;

  j.scalar_type = type_class;
  j.access_type = type;

  return j;
}

void LoopNestFeaturizer::visit_store_args(Type t, Expr arg) {
  arg = common_subexpression_elimination(simplify(arg)); // Get things into canonical form
  store_jacobian = visit_memory_access(t, arg, LoopNestPipelineFeatures::AccessType::Store);
}

json LoopNestFeaturizer::Jacobian::to_json() const {
  json jdata;
  jdata["scalar_type"] = scalar_type;
  jdata["access_type"] = access_type;
  for (const auto& derivative : derivatives) {
    jdata["derivatives"].push_back(derivative.to_json());
  }
  return jdata;
}

} // namespace Internal
} // namespace Halide
