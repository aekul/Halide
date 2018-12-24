/*
    Env vars used (directly or indirectly):

    TODO(someone): document all these

    HL_AUTO_SCHEDULE_TIME_LIMIT
    HL_BEAM_SIZE
    HL_CYOS
    HL_FEATURE_FILE -> output
    HL_MACHINE_PARAMS
    HL_PERMIT_FAILED_UNROLL
    HL_RANDOM_DROPOUT
    HL_SCHEDULE_FILE
    HL_SEED
    HL_USE_MANUAL_COST_MODEL
    HL_WEIGHTS_DIR

*/
#include <set>
#include <queue>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <iostream>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <sstream>

#include "Halide.h"
#include "halide_benchmark.h"
#include "CostModel.h"
#include "Featurization.h"
#include "FunctionDAG.h"
#include "LoopNest.h"
#include "PerfectHashMap.h"
#include "Errors.h"

#include <json.hpp>

using json = nlohmann::json;

namespace Halide {
namespace Internal {

namespace {

using std::string;
using std::vector;
using std::map;
using std::set;
using std::pair;

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

std::string expr2str(Expr e) {
  std::stringstream s;
  s << e;
  return s.str();
}


uint64_t get_dropout_threshold() {
    string random_dropout_str = get_env_variable("HL_RANDOM_DROPOUT");
    if (!random_dropout_str.empty()) {
        return atoi(random_dropout_str.c_str());
    } else {
        return 100;
    }
}

static uint64_t random_dropout_threshold = 100;

bool random_dropout() {
    static bool init =
        []() {random_dropout_threshold = get_dropout_threshold(); return true;}();
    (void)init;
    uint64_t r = rand();
    bool drop_it = (r % 100) >= random_dropout_threshold;
    return drop_it;
}

vector<vector<int64_t>> generate_tilings(const vector<int64_t> &s, int d, int factor, bool allow_splits, int vector_dim, int vector_size) {
    vector<vector<int64_t>> result;
    if (d == -1) {
        result.push_back(vector<int64_t>());
    } else {
        vector<vector<int64_t>> v;
        v = generate_tilings(s, d - 1, factor, allow_splits, vector_dim, vector_size);
        // If we're already generated tons of tiling configs for the
        // inner loops, search the outer loops with coarser
        // granularity.
        while (v.size() > (size_t)factor * 100) {
            factor *= 2;
        }

        for (auto t : v) {
            bool is_full = false, is_one = false;
            // Skip trivial tilings
            if ((size_t)d == s.size() - 1) {
                is_one = is_full = true;
                for (int i = 0; i < d; i++) {
                    is_one &= (t[i] == 1);
                    is_full &= (t[i] == s[i]);
                }
            }
            t.push_back(0);
            if (!allow_splits) {
                if (!is_one) {
                    t.back() = 1;
                    result.push_back(t);
                }
                if (s[d] != 1 && !is_full && is_one && (d != vector_dim)) {
                    t.back() = s[d];
                    result.push_back(t);
                }
            } else {
                int max_inner = 0;
                int first_inner = (d == vector_dim) ? vector_size : 1;
                for (int inner = first_inner; inner < s[d]; inner *= factor) {
                    int outer = (s[d] + inner - 1) / inner;
                    if (is_one && outer == 1) continue;
                    if (is_full && outer == s[d]) continue;
                    // Stop when we hit inner sizes that would do too much recompute
                    if (inner > first_inner && inner * outer * 7 > s[d] * 8) break;
                    max_inner = inner;
                    t.back() = outer;
                    result.push_back(t);
                }
                for (int outer = 1; outer <= s[d]; outer *= factor) {
                    int inner = (s[d] + outer - 1) / outer;
                    if (is_one && outer == 1) continue;
                    if (is_full && outer == s[d]) continue;
                    // Stop when we get into the regime covered by the loop above.
                    if (outer > 1 && inner < max_inner * 2) break;
                    // Or when the wasted compute gets too bad.
                    if (inner * outer * 7 > s[d] * 8) break;
                    t.back() = outer;
                    result.push_back(t);
                }

                // The sequence above (in terms of the inner loop) goes 1 2 4 8 16 ...
                // but 3 is an important inner tiling factor for matrix multiply ops.
                int inner3 = (d == vector_dim) ? 3*vector_size : 3;
                int outer3 = (s[d] + inner3 - 1) / inner3;
                if (factor == 2 && inner3 < s[d] && outer3 < s[d] && outer3 > 1) {
                    if (inner3 * outer3 * 7 <= s[d] * 8) {
                        t.back() = outer3;
                        result.push_back(t);
                    }
                }
            }
        }
    }
    return result;
}

template<typename T>
using NodeMap = PerfectHashMap<FunctionDAG::Node, T>;

template<typename T>
using StageMap = PerfectHashMap<FunctionDAG::Node::Stage, T>;

std::string var_base_name(const std::string& name, const std::string& var, int stage_index) {
    std::string result{name + "." + var + ".min"};
    if (stage_index > 0) {
        result += + ".s" + std::to_string(stage_index);
    }
    return result;
};

std::string var_base_name(const std::string& name, const Var& var, int stage_index) {
    return var_base_name(name, var.name(), stage_index);
};

std::string var_base_name(Function f, const Var& var, int stage_index) {
    return var_base_name(f.name(), var.name(), stage_index);
};


// We're going to do a tree search over possible schedules to find an
// optimal one. A tree search requires a state, and a function that
// gives you children of the state (with costs). The following struct
// represents the state, which is a partial schedule.
//
// A partial schedule is a tree. Each node is some portion of the for
// loop nest of some Func. If there are no children, it's the
// innermost set of loops. If there are children, it's a loop over
// tiles of that Func.
struct LoopNest {
    mutable RefCount ref_count;

    // The size of the outer loop, and the split factor used to create
    // the inner loop. Put another way, the number of tiles, and the
    // size of each tile.
    vector<int64_t> size, split_factor;

    // The nodes inside the loop body
    vector<IntrusivePtr<const LoopNest>> children;

    // Funcs inlined into this inner loop, and the number of times they are called. Only valid if children is empty.
    NodeMap<int64_t> inlined;

    // Funcs realized inside this inner loop
    set<const FunctionDAG::Node *> store_at;

    // The total bounds required of the given Func for one
    // representative iteration of this loop. Computed lazily and
    // cached. entries are immutable so that bounds are shared across
    // different instances.
    mutable NodeMap<Bound> bounds;

    const FunctionDAG::Node *node = nullptr;
    const FunctionDAG::Node::Stage *stage = nullptr;
    int stage_idx = 0;

    // Is this the innermost loop of this func?
    bool innermost = false;

    // Are we permitted to tile this loop?
    bool tileable = false;

    // Is this the parallel outer loop?
    bool parallel = false;

    // What dimension is this Func vectorized over, in terms of the args of the Func?
    int vector_dim = -1;

    // Which loop corresponds to the innermost storage dimension and will be vectorized. -1 means none of them.
    int vectorized_loop_index = -1;

    void copy_from(const LoopNest &n) {
        size = n.size;
        children = n.children;
        inlined = n.inlined;
        store_at = n.store_at;
        bounds = n.bounds;
        node = n.node;
        stage = n.stage;
        stage_idx = n.stage_idx;
        innermost = n.innermost;
        tileable = n.tileable;
        parallel = n.parallel;
        vector_dim = n.vector_dim;
        vectorized_loop_index = n.vectorized_loop_index;
    };

    static void hash_combine(uint64_t &h, uint64_t next) {
        // From boost
        h ^= (next + 0x9e3779b9 + (h<<6) + (h>>2));
    }

    // Hash the loop structure and sizes up to a fixed depth
    void structural_hash(uint64_t &h, int depth, int parallelism) const {
        if (depth < 0) return;

        // Which Funcs are store_at this level?
        for (const auto *n : store_at) {
            hash_combine(h, n->id);
        }

        hash_combine(h, -1);

        // Which Funcs are compute_at this level?
        for (const auto &c : children) {
            hash_combine(h, c->stage->id);
        }

        // Add a barrier to ensure that moving something from the last
        // compute_at to the first inlined doesn't result in the same
        // hash.
        hash_combine(h, -1);

        // Which Funcs are inlined at this level?
        for (auto it = inlined.begin(); it != inlined.end(); it++) {
            hash_combine(h, it.key()->id);
        }

        hash_combine(h, -1);

        if (depth > 0) {
            // What are their loop sizes?
            for (const auto &c : children) {
                for (int64_t s : c->size) {
                    if (depth == 1) {
                        // Just take the most significant bit: is it more
                        // or less than the parallelism factor.
                        s = s >= parallelism ? 1 : 0;
                    }
                    hash_combine(h, s);
                }
            }
        }

        if (innermost) {
            // Which dimension are we vectorized over?
            hash_combine(h, vectorized_loop_index);
        }

        if (depth > 1) {
            // Descend into children
            for (const auto &c : children) {
                c->structural_hash(h, depth - 2, parallelism);
            }
        }
    }

    struct ArgReplacer : public IRMutator2 {
        using IRMutator2::visit;

        std::map<std::string, Expr>& root_store_offsets;
        std::map<std::string, std::map<std::string, Expr>>& store_at_bounds;
        std::map<std::string, int> strides;
        const NodeMap<int64_t>& inlined;

        ArgReplacer(std::map<std::string, Expr>& root_store_offsets, std::map<std::string, std::map<std::string, Expr>>& store_at_bounds, std::map<std::string, int> strides, const NodeMap<int64_t>& inlined)
            : root_store_offsets{root_store_offsets}
            , store_at_bounds{store_at_bounds}
            , strides{strides}
            , inlined{inlined}
        {}

        vector<Expr> mutate_args(const vector<Expr>& args) {
            vector<Expr> mutated_args;
            for (const auto& a : args) {
                mutated_args.push_back(mutate(a));
            }
            return mutated_args;
        }

        Expr create_new_call(const Call *op, const vector<Expr>& new_args) {
            std::string name = op->name;
            vector<Expr> args;
            for (size_t i = 0; i < new_args.size(); i++) {
                Expr min;
                Expr stride;

                if (op->call_type == Call::Image && op->image.defined()) {
                    min = op->image.min(i);
                    stride = op->image.stride(i);
                } else {
                    // Input images may not have a defined func (their args
                    // should be _0, _1, _2, etc.)
                    std::string var_name = op->func.defined() ? Function(op->func).args()[i] : "_" + std::to_string(i);
                    std::string key = var_base_name(name, var_name, 0);
                    if (store_at_bounds.count(name) > 0 && store_at_bounds[name].count(var_name) > 0 && strides.count(key) > 0) {
                        min = store_at_bounds[name][var_name];
                        stride = strides[key];
                    } else if (root_store_offsets.count(key) > 0) {
                        min = root_store_offsets[key];
                        stride = strides[key];
                    } else {
                        min = 0;
                        stride = strides.at(key);
                        user_assert(false);
                    }
                }

                // Apply offset to get store location e.g. if x is in [5, 1000],
                // then x will be stored at [0, 995] so each store should apply
                // a -5 offset
                Expr arg = new_args[i] - min;
                arg *= stride;

                if (args.empty()) {
                    args.push_back(arg);
                } else {
                    args[0] = simplify(common_subexpression_elimination(arg + args[0]));
                }
            }
            user_assert(args.size() == 1);
            return Call::make(op->type, op->name, mutate_args(args), op->call_type, op->func, op->value_index, op->image, op->param);
        };

        Expr visit(const Call *op) override {
            vector<Expr> new_args = op->args;

            bool is_inlined = false;
            for (auto it = inlined.begin(); it != inlined.end(); it++) {
                const auto *f = it.key();
                if (f->func.name() == op->name) {
                    is_inlined = true;
                    break;
                }
            }            

            if (!is_inlined) {
                if (op->call_type == Call::PureExtern || op->call_type == Call::PureIntrinsic) {
                  return Call::make(op->type, op->name, mutate_args(new_args), op->call_type, op->func, op->value_index, op->image, op->param);
                }

                return create_new_call(op, new_args);
            }

            if (op->call_type == Call::Image) {
                return create_new_call(op, new_args);
            }

            Function f{op->func};
            Expr inlined_result = f.values()[op->value_index];
            for (size_t i = 0; i < new_args.size(); i++) {
                inlined_result = substitute(f.args()[i], new_args[i], inlined_result);
            }

            // visit inlined_result in case it contains Calls that need
            // replacing
            return mutate(inlined_result);
        }
    };

    bool is_scheduled(const FunctionDAG::Node* n) const {
        if (node == n) { 
            return true;
        }

        for (const auto& child : children) {
            if (child->is_scheduled(n)) {
                return true;
            }
        }

        return false;
    }

    bool is_stored_in_tree(const FunctionDAG::Node* n) const {
        if (store_at.count(n) > 0) { 
            return true;
        }

        for (const auto& child : children) {
            if (child->is_stored_in_tree(n)) {
                return true;
            }
        }

        return false;
    }

    bool is_stored_or_inlined_in_tree(const FunctionDAG::Node* n) const {
        if (store_at.count(n) > 0 || inlined.contains(n)) { 
            return true;
        }

        for (const auto& child : children) {
            if (child->is_stored_or_inlined_in_tree(n)) {
                return true;
            }
        }

        return false;
    }

    mutable map<const FunctionDAG::Node *, std::shared_ptr<SymbolicBound>> symbolic_bounds;

    struct StageScheduleState;
    struct ScheduleData;

    struct CollectVars : public IRVisitor {
        using IRVisitor::visit;

        std::set<std::string> vars;

        void visit(const Variable *var) override {
            vars.insert(var->name);
        }
    };

    int64_t get_non_unique_bytes_read_per_point(const FunctionDAG::Node * f) const {
        int64_t non_unique_bytes_read_per_point = 0;
        for (const auto *e : f->get_incoming_edges()) {
            if (inlined.contains(e->producer)) {
                non_unique_bytes_read_per_point += get_non_unique_bytes_read_per_point(e->producer);
                continue;
            }
            non_unique_bytes_read_per_point += e->calls * e->producer->bytes_per_point;
        }
        return non_unique_bytes_read_per_point; 
    }

    set<const FunctionDAG::Node*> get_producers() const {
        if (is_root()) {
            return {};
        }

        set<const FunctionDAG::Node*> nodes;

        vector<const FunctionDAG::Node*> pending;
        pending.push_back(node);
        while (pending.size() > 0) {
            const auto& n = pending.back();
            pending.pop_back();

            const auto &next = n->get_incoming_edges();

            for (const auto *e : next) {
                if (inlined.contains(e->producer)) {
                    pending.push_back(e->producer);
                    continue;
                }

                // Already processed this node
                //if (nodes.count(e->producer)) {
                    //continue;
                //}

                nodes.insert(e->producer);
            }

            for (auto& child : children) {
                for (auto& child_n : child->get_producers()) {
                    nodes.insert(child_n);
                }
            }
        }

        return nodes;
    }

    int64_t get_bytes_loaded_from_external_producers(const LoopNest *parent, set<std::string>& nested_producers) const {
        int64_t bytes_loaded = 0;

        if (!node || !parent) {
            return bytes_loaded;
        }

        set<const FunctionDAG::Node*> nodes = get_producers();

        for (const auto& n : nodes) {
            int64_t footprint = n->bytes_per_point;

            // Skip funcs that are realized inside this loop nest
            if (is_stored_in_tree(n)) {
                nested_producers.insert(n->func.name());
                continue; 
            }

            const auto &bounds = parent->get_bounds(n);

            for (int i = 0; i < node->func.dimensions(); i++) {
                auto p = bounds->region_required(i);

                footprint *= p.second - p.first + 1;
            }

            bytes_loaded += footprint;
        }

        return bytes_loaded;
    }

    void create_loop_nest(const FunctionDAG &dag,
                          const MachineParams &params,
                          const LoopNest *parent,
                          int indent_level,
                          int depth,
                          int node_depth,
                          BlockNode* block,
                          std::map<std::string, Expr>& store_at_bounds,
                          std::map<std::string, Expr> compute_bounds,
                          std::map<std::string, int>& strides,
                          std::map<std::string, double> parallelism,
                          double num_cores,
                          const StageMap<ScheduleFeatures>& features,
                          StageMap<std::unique_ptr<LoopNest::StageScheduleState>>& vars_map,
                          LoopNest::ScheduleData& schedule_data,
                          std::map<std::string, OutputSize>& output_sizes,
                          int product_of_outer_loops,
                          std::map<std::string, AllocNode*> allocs,
                          std::set<std::string>& alloced,
                          std::map<std::string, Expr> compute_offsets,
                          std::map<std::string, std::map<std::string, Expr>>& store_offsets,
                          std::map<std::string, Expr> compute_mins,
                          std::map<std::string, Expr> current_mins,
                          std::map<std::string, Expr> root_store_offsets) const {

        auto get_orig_var_name = [&](const StageScheduleState::FuncVar& fv) {
            const auto &symbolic_loop = stage->loop;
            for (size_t i = 0; i < symbolic_loop.size(); i++) {
                if (symbolic_loop[i].var == fv.orig.name()) {
                    return symbolic_loop[i].var;
                }
            }
            user_assert(false);
            return symbolic_loop.back().var;
        };

        auto get_inner_extent = [](const StageScheduleState& func_vars, int index) -> int64_t {
            int64_t extent = 1;
            for (int i = index - 1; i >= 0; i--) {
                if (func_vars.vars[index].orig.name() == func_vars.vars[i].orig.name()) {
                    extent *= func_vars.vars[i].extent;
                }
            }

            return extent;
        };

        auto create_alloc_node = [&](const FunctionDAG::Node *f, const std::string& name) {
            std::unique_ptr<AllocNode> alloc = make_unique<AllocNode>();
            alloc->parent = block;
            alloc->name = name;
            alloc->bytes_per_point = f->bytes_per_point;
            alloc->size = 1;
            alloc->should_store_on_stack = schedule_data.store_on_stack_set.count(f) > 0;
            allocs[name] = alloc.get();
            block->add_child(std::move(alloc));
        };

        auto add_alloc_node = [&](const FunctionDAG::Node *f, const std::string& name) {
            AllocNode* alloc = allocs[name];
            int stride = 1;
            // get_uncached_bounds if the bounds of this func have not yet been
            // computed (this typically happens for funcs that have not been
            // scheduled and/or are not direct producers of funcs that have been
            // scheduled)
            auto* bound_nest = parent ? parent : this;
            Bound b = bounds.contains(f) ? bound_nest->get_bounds(f) : bound_nest->get_uncached_bounds(f, false);
            std::shared_ptr<SymbolicBound> symbolic_bound = bound_nest->get_symbolic_bounds(f, false);
            for (int j = 0; j < f->func.dimensions(); j++) {
                auto extent = b->region_computed(j).second - b->region_computed(j).first + 1;
                alloc->region.push_back(extent);
                alloc->size *= extent;

                std::string key = var_base_name(name, f->func.args()[j], 0);
                strides[key] = stride;

                //if (f->is_output) {
                  //const auto& func_name = f->func.name();
                  //if (output_sizes.count(func_name) == 0) {
                    //output_sizes.emplace(std::make_pair(func_name, f->func));
                  //}
                  //const int64_t *imin = as_const_int(compute_bounds[key]);
                  //internal_assert(imin) << compute_bounds[key] << "\n";
                  //output_sizes[func_name].add(stride, *imin, extent);
                //}

                stride *= extent;
            }

            alloced.insert(name);
        };

        auto initialize_offsets = [&](const FunctionDAG::Node *f, const std::string& name) {
            std::shared_ptr<SymbolicBound> symbolic_bound = get_symbolic_bounds(f, false);
            for (int j = 0; j < f->func.dimensions(); j++) {
                std::string key = var_base_name(name, f->func.args()[j], 0);
                root_store_offsets[key] = simplify(substitute(compute_mins, symbolic_bound->region_computed[j].first));
                compute_mins[key] = root_store_offsets[key];
            }

            for (int s = 1; s < (int)symbolic_bound->loops.size(); s++) {
                for (int j = 0; j < (int)symbolic_bound->loops[s].size(); j++) {
                    const auto& loop = f->stages[s].loop[j];
                    std::string key = var_base_name(name, loop.var, s);
                    compute_mins[key] = simplify(substitute(compute_mins, symbolic_bound->loops[s][j].first));
                }
            }
        };

        if (is_root()) {
            // Initialize compute mins for all funcs/stages
            for (int i = 0, N = dag.nodes.size(); i < N; i++) {
                const auto& f = &dag.nodes[i];

                initialize_offsets(f, f->func.name());

                if (f->func.name().find("_im") == std::string::npos) {
                  continue;
                }

                // The random pipeline generator creates wrapper funcs (e.g. "input_im")
                // that call input images (e.g. "input", the actual input image) 
                // i.e. input_im(x, y, ...) = input(x, y, ...).
                // The actual input image is not part of the FunctionDAG so manually check if
                // the wrapper ("*_im") is part of the DAG. If it is, add an alloc node for the image

                // Remove "_im"
                int len = f->func.name().size() - 3;
                auto base_name = f->func.name().substr(0, len);
                create_alloc_node(f, base_name);
                add_alloc_node(f, base_name);

                initialize_offsets(f, base_name);
            }

        }

        std::set<std::string> nested_producers;
        int64_t bytes_loaded_from_external_producers = 0;
        if (innermost) {
            bytes_loaded_from_external_producers = get_bytes_loaded_from_external_producers(parent, nested_producers);
        }

        auto add_loop_node = [&](StageScheduleState& func_vars, int i) {
            const auto& fv = func_vars.vars[i]; 
            auto var_name = get_orig_var_name(fv);
            int inner_extent = get_inner_extent(func_vars, i);
            std::unique_ptr<LoopNode> loop_node = make_unique<LoopNode>(
                node->func
                , stage_idx
                , fv.extent
                , fv.vectorized
                , block
                , depth
                , fv.parallel
                , fv.tail_strategy
                , fv.var
                , fv.unrolled
                , product_of_outer_loops
                , bytes_loaded_from_external_producers
                , nested_producers
            );

            product_of_outer_loops += std::log2(fv.extent);

            std::string key = var_base_name(node->func, var_name, stage_idx);
            auto var_min = loop_node->var * IntImm::make(Int(32), inner_extent);

            if (current_mins.count(var_name) == 0) {
                current_mins[var_name] = compute_mins[key];
            }

            // First loop in production of this stage
            const auto& fn_name = node->func.name();
            if (store_offsets.count(fn_name) == 0) {
                store_offsets[fn_name] = {};
            }

            if (store_offsets[fn_name].count(var_name) == 0) {
                if (compute_offsets.count(var_name) > 0) {
                    store_offsets[fn_name][var_name] = compute_offsets[var_name];
                } else {
                    store_offsets[fn_name][var_name] = 0;
                }
                store_offsets[fn_name][var_name] += current_mins[var_name];
            }

            // First loop in production of this stage
            if (compute_offsets.count(var_name) == 0) {
                compute_offsets[var_name] = simplify(var_min);
            } else {
                compute_offsets[var_name] = simplify(var_min + compute_offsets[var_name]);
            }

            BlockNode* child_block = loop_node->body.get();
            block->add_child(std::move(loop_node));
            block = child_block;

            indent_level++;
            depth++;

            func_vars.vars.pop_back();
        };

        if (stage && vars_map.contains(stage)) {
            StageScheduleState& func_vars = *(vars_map.get(stage));

            // This is the first loop nest for this stage: compute the necessary
            // allocation size for it
            const auto& name = node->func.name();
            if (alloced.count(name) == 0 && allocs.count(name) != 0) {
                add_alloc_node(node, name);
            }

            for (int i = func_vars.vars.size() - 1; i >= 0; i--) {
                const auto& fv = func_vars.vars[i]; 
                add_loop_node(func_vars, i);

                auto it = schedule_data.loop_levels.find(this);
                if (it != schedule_data.loop_levels.end() && it->second == fv.var.name()) {
                    break; 
                }
            }
        }

        // Alloc in reverse topological order
        for (int i = 0, N = dag.nodes.size(); i < N; i++) {
            const auto& f = &dag.nodes[i];

            // In case a particular func has not been scheduled; store_at 
            // bounds are needed when computing the bounds of calls to that func
            bool store_root = is_root() && !is_stored_or_inlined_in_tree(&dag.nodes[i]);

            if (store_at.count(f) == 0 && !store_root) { 
                continue;
            }

            create_alloc_node(f, f->func.name());
            if (!is_scheduled(f)) {
                add_alloc_node(f, f->func.name());
            }
        } 

        for (int i = children.size() - 1; i >= 0; i--) {
            children[i]->create_loop_nest(dag, params, this, indent_level, depth + 1, node_depth + 1, block, store_at_bounds, compute_bounds, strides, parallelism, num_cores, features, vars_map, schedule_data, output_sizes, product_of_outer_loops, allocs, alloced, compute_offsets, store_offsets, compute_mins, current_mins, root_store_offsets);
        }


        if (innermost) {
            StageScheduleState& func_vars = *(vars_map.get(stage));
            // Add remaining loops
            for (int i = func_vars.vars.size() - 1; i >= 0; i--) {
                add_loop_node(func_vars, i);
            }

            Definition def = node->func.definition();
            if (stage_idx > 0) def = node->func.updates()[stage_idx - 1];

            std::vector<Expr> args;
            std::vector<Expr> values;

            Expr arg = 0;
            user_assert(def.args().size() > 0);

            CollectVars collect_vars;
            def.accept(&collect_vars);
            
          
            std::map<std::string, Expr> arg_compute_bounds;
            for (const auto& var : collect_vars.vars) {
                if (compute_offsets.count(var) == 0 || current_mins.count(var) == 0) {
                    continue;
                }
              
                arg_compute_bounds[var] = compute_offsets[var] + current_mins[var];
            }

            for (int i = 0, N = def.args().size(); i < N; i++) {
                auto a = node->func.args()[i];
                std::string store_key = var_base_name(node->func, a, 0);

                arg += (substitute(arg_compute_bounds, def.args()[i]) - store_offsets[node->func.name()][a]) * strides[store_key];
                arg = simplify(common_subexpression_elimination(arg));
            }

            // TODO: support tuples
            user_assert(def.values().size() == 1);
            ArgReplacer replacer{root_store_offsets, store_offsets, strides, inlined};
            for (auto value : def.values()) {
                value = replacer.mutate(value);
                value = substitute(arg_compute_bounds, value);
                values.push_back(simplify(common_subexpression_elimination(value)));
            }

            auto compute = make_unique<ComputeNode>(
                node->func
                , replacer.mutate(arg)
                , values
                , block
                , features.get(stage)
                , node->stages[stage_idx].features
                , get_non_unique_bytes_read_per_point(node)
                , node->bytes_per_point
            );
            block->add_child(std::move(compute));
        }
    }

    size_t funcs_realized_or_inlined() const {
        size_t count = inlined.size() + store_at.size();
        for (auto c : children) {
            count += c->funcs_realized_or_inlined();
        }
        return count;
    }

    struct Sites {
        const LoopNest *compute = nullptr, *store = nullptr, *produce = nullptr, *innermost = nullptr;
        bool inlined = false;
    };

    void get_sites(StageMap<Sites> &sites,
                   const LoopNest *parent = nullptr) const {
        for (auto c : children) {
            c->get_sites(sites, this);
        }
        if (parent && node != parent->node) {
            auto &s = sites.get_or_create(stage);
            s.compute = parent;
            s.produce = this;
        }
        for (auto f : store_at) {
            for (const auto &s : f->stages) {
                sites.get_or_create(&s).store = this;
            }
        }
        for (auto it = inlined.begin(); it != inlined.end(); it++) {
            sites.get_or_create(&(it.key()->stages[0])).inlined = true;
        }
        if (innermost) {
            sites.get_or_create(stage).innermost = this;
        }
    }

    void compute_features(const MachineParams &params,
                          const StageMap<Sites> &sites,
                          int64_t instances,
                          int64_t parallelism,
                          const LoopNest *parent,
                          const LoopNest &root,
                          int64_t *working_set,
                          StageMap<ScheduleFeatures> *features) const {

        int64_t working_set_here = 0;

        int64_t loop_instances = 1, parallel_tasks = 1;
        bool in_impure = false;
        for (int idx = (int)size.size() - 1; idx >= 0; idx--) {
            size_t i = size[idx];
            loop_instances *= i;
            if (stage->loop[idx].pure && !in_impure) {
                if (parallel || (parent->is_root() && parallel_tasks < params.parallelism)) {
                    // Either we've picked our parallel tiling, or
                    // it's not yet determined. Assume we'll not split
                    // any loops and just stop after we hit the
                    // required number of cores
                    parallel_tasks *= i;
                    if (!parallel && parallel_tasks > params.parallelism * 8) {
                        // We would split this loop
                        parallel_tasks = params.parallelism * 8;
                    }
                }
            } else if (i != 1) {
                in_impure = true;
            }
        }

        int64_t subinstances = instances * loop_instances;

        for (const auto *node : store_at) {
            // Figure out the features at the store_at level
            const auto &bounds = get_bounds(node);

            for (size_t s = 0; s < node->stages.size(); s++) {
                // TODO: Lift invariants from this loop. Most of it's the same for every stage.
                internal_assert(!node->is_input);
                ScheduleFeatures &feat = features->get_or_create(&(node->stages[s]));

                feat.num_realizations = subinstances;

                feat.points_computed_per_realization = 1;
                feat.num_scalars = feat.num_vectors = subinstances;
                bool vectorized = false;
                for (int i = 0; i < (int)node->stages[s].loop.size(); i++) {
                    const auto &p = bounds->loops(s, i);
                    int64_t extent = p.second - p.first + 1;
                    feat.points_computed_per_realization *= extent;
                    if (i == sites.get(&(node->stages[s])).produce->vectorized_loop_index) {
                        // Assumes that we're not going to split
                        // things such that non-native-width
                        // vectorization is a problem, except for the
                        // tail.
                        feat.num_vectors *= extent / node->stages[s].vector_size;
                        feat.num_scalars *= extent % node->stages[s].vector_size;
                        vectorized = true;
                    } else {
                        feat.num_vectors *= extent;
                        feat.num_scalars *= extent;
                    }
                }
                if (!vectorized) {
                    feat.num_vectors = 0;
                }
                feat.points_computed_total = feat.points_computed_per_realization * feat.num_realizations;

                feat.bytes_at_realization = node->bytes_per_point;
                for (int i = 0; i < node->func.dimensions(); i++) {
                    const auto &p = bounds->region_computed(i);
                    feat.bytes_at_realization *= (p.second - p.first) + 1;
                }
                int64_t innermost_storage_extent = 1;
                int v = sites.get(&(node->stages[s])).produce->vector_dim;
                if (v >= 0) {
                    innermost_storage_extent = (bounds->region_computed(v).second -
                                                bounds->region_computed(v).first + 1);
                }
                feat.innermost_bytes_at_realization = node->bytes_per_point * innermost_storage_extent;

                if (!is_root()) {
                    feat.bytes_at_task = feat.bytes_at_realization;
                    feat.innermost_bytes_at_task = feat.innermost_bytes_at_realization;
                }
            }
        }

        if (is_root()) {
            for (auto c : children) {
                c->compute_features(params, sites, subinstances, parallelism, this, root, &working_set_here, features);
            }

            // Figure out the root-level features for every Func
            for (auto it = features->begin(); it != features->end(); it++) {
                const auto *stage = it.key();
                const auto *node = stage->node;
                auto &feat = it.value();
                const auto &root_bounds = root.get_bounds(node);

                feat.bytes_at_root = node->bytes_per_point;
                for (int i = 0; i < node->func.dimensions(); i++) {
                    const auto &p = root_bounds->region_computed(i);
                    feat.bytes_at_root *= (p.second - p.first) + 1;
                }

                // What innermost storage extent means for inlined
                // Funcs is unclear, because we haven't selected which
                // storage dimension is innermost.
                auto *p = sites.get(stage).produce;
                if (p) {
                    int64_t innermost_storage_extent = 1;
                    int v = p->vector_dim;
                    if (v >= 0) {
                        innermost_storage_extent = root_bounds->region_computed(v).second - root_bounds->region_computed(v).first + 1;
                    }
                    feat.innermost_bytes_at_root = node->bytes_per_point * innermost_storage_extent;
                } else {
                    feat.innermost_bytes_at_root = 0;
                }

                feat.points_computed_minimum = 1;
                int s = stage - &node->stages[0];
                for (int i = 0; i < (int)stage->loop.size(); i++) {
                    const auto &p = root_bounds->loops(s, i);
                    feat.points_computed_minimum *= (p.second - p.first + 1);
                }

                if (node->stages.size() == 1 && !node->is_output) {
                    int64_t points_computed_minimum_if_inlined = 0;
                    for (auto *e : node->outgoing_edges) {
                        points_computed_minimum_if_inlined += features->get(e->consumer).points_computed_minimum * e->calls;
                    }
                    feat.points_computed_minimum = std::min(feat.points_computed_minimum, points_computed_minimum_if_inlined);
                }
            }

            return;
        }

        int64_t subparallelism = parallel_tasks * parallelism;

        // Figure out the features at the compute_at level
        internal_assert(!stage->node->is_input);
        ScheduleFeatures &feat = features->get_or_create(stage);

        if (innermost) {
            if (vectorized_loop_index >= 0 && vectorized_loop_index < size.size()) {
                feat.vector_size = size[vectorized_loop_index];
            } else {
                feat.vector_size = 1;
            }
            if (feat.vector_size == 1) {
                // They're all scalars
                feat.num_scalars += feat.num_vectors;
                feat.num_vectors = 0;
            }
        } else {
            // These will get progressively overwritten as we visit the children
            feat.innermost_loop_extent = size.empty() ? 1 : size[0];
            if (vectorized_loop_index >= 0 && vectorized_loop_index < size.size()) {
                feat.innermost_pure_loop_extent = size[vectorized_loop_index];
            } else {
                feat.innermost_pure_loop_extent = 1;
            }
        }

        const bool at_task = parent->is_root();
        const bool at_production = parent->node != node;
        const bool at_pure_production = at_production && stage_idx == 0;

        if (at_task) {
            if (parallel) {
                const auto &bounds = get_bounds(node);
                feat.bytes_at_task = node->bytes_per_point;
                int64_t innermost_storage_extent = 1;
                for (int i = 0; i < node->func.dimensions(); i++) {
                    int64_t outer = 1;
                    for (int l = 0; l < stage->loop.size(); l++) {
                        if (stage->loop[l].var == node->func.args()[i]) {
                            outer = size[l];
                            break;
                        }
                    }
                    const auto &p = bounds->region_computed(i);
                    int64_t extent = (p.second - p.first) + 1;
                    extent /= outer;
                    feat.bytes_at_task *= extent;
                    if (i == vector_dim) {
                        innermost_storage_extent = extent;
                    }
                }
                feat.innermost_bytes_at_task = node->bytes_per_point * innermost_storage_extent;
            } else {
                // How this loop will be parallelized is not yet
                // determined. Use optimistic values for the features.
                feat.bytes_at_task = (feat.bytes_at_realization + params.parallelism - 1) / params.parallelism;
                feat.innermost_bytes_at_task = std::min(feat.bytes_at_task, feat.innermost_bytes_at_realization);
            }
        }

        if (at_production) {
            feat.num_productions = instances;
            feat.inner_parallelism = parallel_tasks;
            feat.outer_parallelism = parallelism;
            feat.native_vector_size = stage->vector_size;

            const auto &bounds = parent->get_bounds(node);

            feat.bytes_at_production = node->bytes_per_point;
            for (int i = 0; i < node->func.dimensions(); i++) {
                const auto &p = bounds->region_computed(i);
                feat.bytes_at_production *= (p.second - p.first) + 1;
            }
            int64_t innermost_storage_extent = 1;
            if (vector_dim >= 0) {
                innermost_storage_extent = bounds->region_computed(vector_dim).second - bounds->region_computed(vector_dim).first + 1;
            }
            feat.innermost_bytes_at_production = node->bytes_per_point * innermost_storage_extent;
        }

        // Recurse inwards
        for (auto c : children) {
            c->compute_features(params, sites, subinstances, subparallelism, this, root, &working_set_here, features);
        }

        if (at_production) {
            for (const auto *node : store_at) {
                working_set_here += features->get(&(node->stages[0])).bytes_at_production;
            }
            // TODO: This seems like it would mask off allocations just inside an inner loop
            feat.working_set = working_set_here;
        }

        *working_set += working_set_here;

        int64_t bytes_loaded = 0, lines_loaded = 0, allocation_bytes_loaded = 0, vectors_loaded = 0, scalars_loaded = 0, elements_loaded = 0;
        if (innermost || at_production) {
            // Pick the site at which we will compute the footprint relationship
            const auto *consumer_store_site = innermost ? parent : sites.get(&(node->stages[0])).store;
            int64_t consumer_instances = innermost ? instances : feat.num_realizations;
            if (consumer_instances == 0) {
                root.dump(" ");
            }
            internal_assert(consumer_instances != 0) << node->func.name() << " " << innermost << " " << instances << " " << feat.num_realizations << "\n";

            vector<const FunctionDAG::Node::Stage *> pending;
            pending.push_back(stage);
            while (!pending.empty()) {
                const auto &next = pending.back()->incoming_edges;
                pending.pop_back();
                for (const auto *e : next) {
                    if (!sites.contains(&(e->producer->stages[0]))) {
                        // Not yet scheduled. Optimistically treat it as free.
                        continue;
                    }

                    const auto &site = sites.get(&(e->producer->stages[0]));

                    if (site.inlined) {
                        // Recursively examine the inputs
                        pending.push_back(&(e->producer->stages[0]));
                        continue;
                    }

                    const auto *producer_compute_site = site.compute;
                    const auto *producer_store_site = site.store;
                    const auto &bounds = consumer_store_site->get_bounds(e->producer);
                    const auto &producer_compute_bounds = producer_compute_site->get_bounds(e->producer);
                    const auto &producer_store_bounds = producer_store_site->get_bounds(e->producer);
                    int64_t footprint = e->producer->bytes_per_point;
                    int64_t vector_footprint = 1;
                    int64_t compute_footprint = footprint;
                    int64_t store_footprint = footprint;
                    int64_t line_footprint = 1;
                    int64_t compute_line_footprint = 1;
                    int64_t store_line_footprint = 1;

                    bool dense_vector_loads = true;

                    if (e->producer->is_input) {
                        internal_assert(producer_store_site->is_root());
                        internal_assert(producer_compute_site->is_root());
                    }

                    for (int i = 0; i < e->producer->func.dimensions(); i++) {
                        auto p = bounds->region_required(i);
                        auto compute_p = producer_compute_bounds->region_computed(i);
                        auto store_p = producer_store_bounds->region_required(i);

                        internal_assert(store_p.first <= store_p.second) << store_p.first << " " << store_p.second << "\n";
                        internal_assert(compute_p.first <= compute_p.second) << compute_p.first << " " << compute_p.second << "\n";

                        int64_t extent = p.second - p.first + 1;
                        int64_t compute_extent = compute_p.second - compute_p.first + 1;
                        int64_t store_extent = store_p.second - store_p.first + 1;
                        footprint *= extent;
                        compute_footprint *= compute_extent;
                        store_footprint *= store_extent;

                        bool dense = i == (e->producer->is_input ? 0 : site.produce->vector_dim);

                        if (dense) {
                            dense_vector_loads = extent >= feat.vector_size;
                            // TODO: This is not exactly correct. The
                            // footprint can be larger than a vector
                            // without the loads being contiguous
                            // vector loads - e.g. consider a lookup
                            // into an 8-element LUT.
                            vector_footprint *= (extent + stage->vector_size - 1) / stage->vector_size;
                        } else {
                            line_footprint *= extent;
                            compute_line_footprint *= compute_extent;
                            store_line_footprint *= store_extent;
                            vector_footprint *= extent;
                        }
                    }

                    if (dense_vector_loads) {
                        vectors_loaded += vector_footprint;
                    } else {
                        scalars_loaded += footprint / e->producer->bytes_per_point;
                    }
                    elements_loaded += footprint / e->producer->bytes_per_point;

                    int64_t store_instances_per_consumption = 1;

                    if (!e->producer->is_input) {
                        const auto &producer_feat = features->get_or_create(&(e->producer->stages[0]));

                        if (producer_feat.num_realizations) {
                            // The producer's realization is nested inside this Func's realization
                            const int64_t producer_store_instances = producer_feat.num_realizations;
                            if (producer_store_instances > consumer_instances) {
                                store_instances_per_consumption = producer_store_instances / consumer_instances;
                            }
                        }
                    }

                    allocation_bytes_loaded += compute_footprint;

                    if (store_instances_per_consumption > 1) {
                        // The producer is nested inside the consumer
                        bytes_loaded += store_footprint; // * store_instances_per_consumption;
                        // Due to folding, the actual buffer size is smaller than the bounds at the store level
                        lines_loaded += store_line_footprint; // * store_instances_per_consumption;
                    } else {
                        // The consumer is consuming some portion of a larger producer computed earlier
                        bytes_loaded += footprint;
                        lines_loaded += line_footprint;
                    }
                }
            }
        }

        if (at_production) {
            // Properties of the realization, but the values are
            // computable at the production site because that's where
            // the consumers are.
            internal_assert(bytes_loaded >= 0) << "Negative bytes loaded: " << bytes_loaded << "\n";
            feat.unique_bytes_read_per_realization = bytes_loaded;
            feat.allocation_bytes_read_per_realization = allocation_bytes_loaded;
            feat.unique_lines_read_per_realization = lines_loaded;

            if (!at_pure_production) {
                // Also pessimistically assume this update definition relies on the entirety of the produced region so far.
                // TODO: This overbills scatters, or writes to a restriction region.
                internal_assert(bytes_loaded >= 0) << "Negative bytes at production: " << feat.bytes_at_production << "\n";
                feat.unique_bytes_read_per_realization += feat.bytes_at_production;
                feat.unique_lines_read_per_realization++; // It's accessed contiguously (TODO: This is fishy. Should probably be lines_at_production)
                feat.allocation_bytes_read_per_realization += feat.bytes_at_production;
            }
        }

        if (innermost) {
            feat.points_computed_per_production = subinstances / feat.num_productions;
            feat.vector_loads_per_vector = vectors_loaded;
            feat.scalar_loads_per_vector = scalars_loaded;
            feat.scalar_loads_per_scalar = (elements_loaded + subinstances - 1) / subinstances;
        }

        // Track features for inlined Funcs
        for (auto it = inlined.begin(); it != inlined.end(); it++) {
            const auto *f = it.key();
            internal_assert(f);
            auto &inlined_feat = features->get_or_create(&(f->stages[0]));
            inlined_feat.inlined_calls += it.value() * subinstances;
            inlined_feat.num_vectors += it.value() * feat.num_vectors;
            inlined_feat.num_scalars += it.value() * feat.num_scalars;
            inlined_feat.native_vector_size = (int64_t)(stage->vector_size);
            if (inlined_feat.vector_size > 0) {
                inlined_feat.vector_size = std::min(inlined_feat.vector_size, (int64_t)stage->vector_size);
            } else {
                inlined_feat.vector_size = feat.vector_size;
            }
            if (inlined_feat.innermost_pure_loop_extent > 0) {
                inlined_feat.innermost_pure_loop_extent = std::min(inlined_feat.innermost_pure_loop_extent,
                                                                   feat.innermost_pure_loop_extent);
            } else {
                inlined_feat.innermost_pure_loop_extent = feat.innermost_pure_loop_extent;
            }
            inlined_feat.inner_parallelism = 1;
            inlined_feat.outer_parallelism = parallelism;
        }
    }

    bool is_root() const {
        return node == nullptr;
    }

    std::shared_ptr<SymbolicBound> get_symbolic_bounds(const FunctionDAG::Node *f, bool ignore_outside_consumers = true) const {
        std::shared_ptr<SymbolicBound> symbolic_bound = std::make_shared<SymbolicBound>();
        // Compute the region required
        if (f->is_output && is_root()) {
            internal_assert(f->outgoing_edges.empty()) << "Outputs that access other outputs not yet supported\n";
            // It's an output.
            // Get the bounds estimate
            map<string, pair<int64_t, int64_t>> estimates;
            for (auto b : f->func.schedule().estimates()) {
                int64_t i_min = *as_const_int(b.min);
                int64_t i_extent = *as_const_int(b.extent);
                estimates[b.var] = {i_min, i_min + i_extent - 1};
            }
            // Use the bounds estimate
            for (int i = 0; i < f->func.dimensions(); i++) {
                auto it = estimates.find(f->func.args()[i]);
                user_assert(it != estimates.end())
                    << "Need an estimate on dimension " << i << " of \"" << f->func.name() << "\"";
                symbolic_bound->region_required.push_back({
                    IntImm::make(Int(32), it->second.first)
                    , IntImm::make(Int(32), it->second.second)
                });
            }
        } else {
            internal_assert(!f->outgoing_edges.empty())
                << "No consumers of " << f->func.name()
                << " at loop over " << (is_root() ? "root" : node->func.name()) << '\n';

            for (const auto *e : f->outgoing_edges) {
                // Ignore consumers outside of this loop nest
                if (!computes(e->consumer->node) && ignore_outside_consumers) {
                    continue;
                }
                const auto &c_bounds = get_bounds(e->consumer->node, ignore_outside_consumers);
                const auto *consumer_loop = &(c_bounds->loops(e->consumer->index, 0)); // For the concrete sizes of the loop
                e->expand_footprint_symbolic(consumer_loop, symbolic_bound.get());
            }
        }

        f->required_to_computed_symbolic(symbolic_bound.get());
        symbolic_bound->loops.resize(f->stages.size());
        for (int i = 0; i < (int)f->stages.size(); i++) {
            f->loop_nest_for_region_symbolic(i, symbolic_bound.get());
        }

        return symbolic_bound;
    }

    const Bound &set_bounds(const FunctionDAG::Node *f, BoundContents *b) const {
        return bounds.emplace(f, b);
    }

    const Bound &get_bounds(const FunctionDAG::Node *f, bool ignore_outside_consumers = true) const {
        // debug(0) << "get_bounds of " << f.name() << " in loop over " << (is_root() ? "root" : func.name()) << '\n';
        if (bounds.contains(f)) {
            const Bound &b = bounds.get(f);
            // debug(0) << "Getting bounds of " << f->func.name() << " at site:\n";
            // dump("  ");
            b->validate();
            return b;
        }

        const Bound &b = set_bounds(f, get_uncached_bounds(f, ignore_outside_consumers));
        b->validate();
        return b;
    }

    BoundContents* get_uncached_bounds(const FunctionDAG::Node *f, bool ignore_outside_consumers = true) const {
        auto bound = f->make_bound();
        // Compute the region required
        if (f->is_output && is_root()) {
            internal_assert(f->outgoing_edges.empty()) << "Outputs that access other outputs not yet supported\n";
            // It's an output.
            // Use the bounds estimate
            for (int i = 0; i < f->func.dimensions(); i++) {
                bound->region_required(i) = f->estimated_region_required[i];
            }
        } else {
            internal_assert(!f->outgoing_edges.empty())
                << "No consumers of " << f->func.name()
                << " at loop over " << (is_root() ? "root" : node->func.name()) << '\n';
            std::pair<int64_t, int64_t> init {INT64_MAX, INT64_MIN};
            for (int i = 0; i < f->func.dimensions(); i++) {
                bound->region_required(i) = init;
            }
            for (const auto *e : f->outgoing_edges) {
                // Ignore consumers outside of this loop nest
                if (!computes(e->consumer->node) && ignore_outside_consumers) {
                    // debug(0) << "Skipping edge from " << e->producer->func.name() << " to " << e->consumer->func.name() << "\n";
                    continue;
                }
                // debug(0) << "Expanding footprint along edge " << e->producer->func.name() << " -> " << e->consumer->func.name() << "\n";
                const auto &c_bounds = get_bounds(e->consumer->node, ignore_outside_consumers);
                const auto *consumer_loop = &(c_bounds->loops(e->consumer->index, 0)); // For the concrete sizes of the loop
                e->expand_footprint(consumer_loop, &(bound->region_required(0)));
            }
        }

        f->required_to_computed(&(bound->region_required(0)), &(bound->region_computed(0)));

        for (int i = 0; i < (int)f->stages.size(); i++) {
            f->loop_nest_for_region(i, &(bound->region_computed(0)), &(bound->loops(i, 0)));
        }

        bound->validate();
        return bound;
    }

    void dump(string prefix) const {
        if (!is_root()) {
            debug(0) << prefix << node->func.name();
            prefix += " ";
        }
        for (size_t i = 0; i < size.size(); i++) {
            debug(0) << " " << size[i];
            if (innermost && i == vectorized_loop_index) {
                debug(0) << 'v';
            }
        }
        debug(0) << " (" << vectorized_loop_index << ", " << vector_dim << ")";
        if (tileable) {
            debug(0) << " t";
        }
        if (innermost) {
            debug(0) << " *\n";
        } else if (parallel) {
            debug(0) << " p\n";
        } else {
            debug(0) << '\n';
        }
        for (auto p : store_at) {
            debug(0) << prefix << "realize: " << p->func.name() << '\n';
        }
        for (size_t i = children.size(); i > 0; i--) {
            children[i-1]->dump(prefix);
        }
        for (auto it = inlined.begin(); it != inlined.end(); it++) {
            debug(0) << prefix << "inlined: " << it.key()->func.name() << " " << it.value() << '\n';
        }
        /*
          for (auto p : bounds) {
          debug(0) << prefix << "bounds: " << p.first.name();
          for (auto d : p.second.region) {
          debug(0) << " [" << d.first << ", " << d.second << "]";
          }
          debug(0) << '\n';
          }
        */
    }

    bool calls(const FunctionDAG::Node *f) const {
        for (const auto &c : children) {
            if (c->calls(f)) return true;
        }
        for (const auto *e : f->outgoing_edges) {
            if (e->consumer == stage) {
                return true;
            }
            if (inlined.contains(e->consumer->node)) {
                return true;
            }
        }
        return false;
    }

    int64_t max_inlined_calls() const {
        int64_t result = 0;
        for (auto it = inlined.begin(); it != inlined.end(); it++) {
            result = std::max(result, it.value());
        }
        for (const auto &c : children) {
            result = std::max(result, c->max_inlined_calls());
        }
        return result;
    }

    bool accesses_input_buffer() const {
        for (const auto &c : children) {
            if (c->accesses_input_buffer()) return true;
        }
        if (is_root()) return false;

        auto check = [&](const FunctionDAG::Node::Stage *s) {
            for (const auto *e : s->incoming_edges) {
                if (e->producer->is_input) return true;
            }

            for (int t = 0; t < (int)PipelineFeatures::ScalarType::NumScalarTypes; t++) {
                if (s->features.op_histogram[(int)PipelineFeatures::OpType::ImageCall][t] > 0) return true;
            }
            return false;
        };

        if (check(stage)) return true;
        for (auto it = inlined.begin(); it != inlined.end(); it++) {
            if (check(&(it.key()->stages[0]))) return true;
        }
        return false;
    }

    bool computes(const FunctionDAG::Node *f) const {
        if (f == node) {
            return true;
        }
        if (inlined.contains(f)) {
            return true;
        }
        for (const auto &c : children) {
            if (c->computes(f)) return true;
        }
        return false;
    }

    void inline_func(const FunctionDAG::Node *f) {
        // Inline it into the children
        for (size_t i = 0; i < children.size(); i++) {
            if (children[i]->calls(f)) {
                std::unique_ptr<LoopNest> new_child{new LoopNest};
                new_child->copy_from(*children[i]);
                new_child->inline_func(f);
                children[i] = new_child.release();
            }
        }

        // Inline it here if there are any direct calls
        if (innermost) {
            int64_t calls = 0;
            for (const auto *e : f->outgoing_edges) {
                if (inlined.contains(e->consumer->node)) {
                    calls += inlined.get(e->consumer->node) * e->calls;
                }
                if (e->consumer == stage) {
                    calls += e->calls;
                }
            }
            if (calls) {
                inlined.insert(f, calls);
            }
        }
    }

    void compute_here(const FunctionDAG::Node *f, bool tileable, int v) {
        const auto &bounds = get_bounds(f);

        for (int s = (int)f->stages.size() - 1; s >= 0; s--) {
            LoopNest *node = new LoopNest;
            node->node = f;
            node->stage_idx = s;
            node->stage = &f->stages[s];
            node->innermost = true;
            node->vectorized_loop_index = -1;
            // TODO: rvars are not tileable
            node->tileable = tileable;
            // Set up a bound for the inside of the
            // loop. computed/required is still the full region, but
            // the loop nest will be a single representative point.
            auto single_point = bounds->make_copy();
            size_t loop_dim = f->stages[s].loop.size();
            node->size.resize(loop_dim);

            int64_t total_extent = 1;
            int64_t vector_size = 1;
            for (size_t i = 0; i < loop_dim; i++) {
                const auto &l = bounds->loops(s, i);
                // Initialize the loop nest
                node->size[i] = l.second - l.first + 1;
                total_extent *= node->size[i];
                // Pick a representative loop iteration for the inner
                // loop. With the way tiling is done below, it needs
                // to be the first loop iteration.
                single_point->loops(s, i) = {l.first, l.first};

                internal_assert(l.second >= l.first) << i << " " << l.second << " " << l.first << "\n";

                if (f->func.dimensions() &&
                    node->size[i] >= node->stage->vector_size &&
                    f->stages[s].loop[i].var == f->func.args()[v]) {
                    node->vectorized_loop_index = (int)i;
                    vector_size = (int64_t)(node->stage->vector_size);
                    single_point->loops(s, i).second += vector_size - 1;
                    node->size[i] += vector_size - 1;
                    node->size[i] /= vector_size;
                }
            }
            // Leave region required blank inside the computation of a Func
            node->set_bounds(f, std::move(single_point));
            node->vector_dim = v;

            if (node->vectorized_loop_index >= 0) {
                // Split off the single vector as an inner loop nest.
                node->innermost = false;

                LoopNest *one_vector = new LoopNest;
                one_vector->node      = node->node;
                one_vector->stage     = node->stage;
                one_vector->stage_idx = node->stage_idx;
                one_vector->tileable  = false;
                one_vector->vectorized_loop_index = node->vectorized_loop_index;
                one_vector->vector_dim = v;
                one_vector->size.resize(loop_dim, 1);
                one_vector->innermost = true;
                auto b = node->get_bounds(f)->make_copy();
                // Set the region computed inside this node to be the first vector lane
                b->loops(s, node->vectorized_loop_index).second = b->loops(s, node->vectorized_loop_index).first;
                one_vector->set_bounds(f, b);
                one_vector->size[node->vectorized_loop_index] = vector_size;

                node->children.emplace_back(one_vector);
            }
            children.emplace_back(node);
        }
    }

    // Return all possible ways to parallelize this loop
    vector<IntrusivePtr<const LoopNest>> parallelize_in_tiles(const MachineParams &params, const LoopNest *parent) const {
        // For now we use a single fixed strategy
        int64_t total_pure_extent = 1;
        bool any_impure = false;
        for (size_t i = 0; i < stage->loop.size(); i++) {
            if (stage->loop[i].pure) {
                total_pure_extent *= size[i];
            } else if (size[i] > 1) {
                any_impure = true;
            }
        }

        vector<IntrusivePtr<const LoopNest>> result;
        if (total_pure_extent < params.parallelism * 2 && !any_impure) {
            // No splits to be made
            LoopNest *child = new LoopNest;
            child->copy_from(*this);
            child->parallel = true;
            result.emplace_back(child);
            return result;
        }

        // Split this loop and move factors to the inner loop
        LoopNest *inner = new LoopNest, *outer = new LoopNest;
        inner->node      = outer->node      = node;
        inner->stage     = outer->stage     = stage;
        inner->stage_idx = outer->stage_idx = stage_idx;
        inner->tileable  = outer->tileable  = tileable;
        inner->vector_dim = outer->vector_dim = vector_dim;
        inner->vectorized_loop_index = outer->vectorized_loop_index = vectorized_loop_index;
        outer->size = size;
        outer->innermost = false;
        outer->parallel = true;
        outer->tileable = true;

        // First make an inner loop representing a 1x1x1... tile
        inner->size.resize(size.size(), 1);
        inner->innermost = innermost;
        inner->children = children;
        inner->inlined = inlined;
        inner->bounds = bounds;
        inner->store_at = store_at;

        auto b = inner->get_bounds(node)->make_copy();

        // Then move factors from the outer loop to the inner loop
        auto parent_bounds = parent->get_bounds(node);

        // We want this many parallel tasks remaining in the outer loop
        int64_t parallelism_required = params.parallelism * 6;
        // TODO: times some factor to be searched over. 6 is nice
        // because you can further tile it by 2 or 3.

        // So far we've found nothing
        int64_t parallelism_found = 1;

        // End at -1, which will be the vectorized loop
        for (int i = (int)stage->loop.size() - 1; i >= -1; i--) {
            int l = (i == -1) ? vectorized_loop_index : i;
            if (l == -1) break; // There's no vectorized loop
            if (i == vectorized_loop_index) continue; // We will handle the vectorized loop last

            int64_t outer_extent = 1;
            if (!stage->loop[l].pure) {
                // Not parallelizeable. We must move this inwards.
                outer_extent = 1;
            } else if (i == -1) {
                if (parallelism_found < params.parallelism) {
                    // Things are dire. We need to parallelize across
                    // the innermost storage dimension. Do it
                    // minimally.
                    outer_extent = std::min(outer->size[l], (params.parallelism + parallelism_found - 1) / parallelism_found);
                }
            } else if (outer->size[l] * parallelism_found < parallelism_required * 2) {
                outer_extent = outer->size[l];
            } else {
                // Pick some number of loop iterations per parallel tasks
                outer_extent = std::max((int64_t)1, parallelism_required / parallelism_found);
            }

            inner->size[l] = (outer->size[l] + outer_extent - 1) / outer_extent;
            outer->size[l] = outer_extent;
            const auto &p = parent_bounds->loops(stage_idx, l);
            int64_t min = p.first;
            int64_t extent = p.second - min + 1;
            extent = (extent + outer_extent - 1) / outer_extent;
            b->loops(stage_idx, l) = {min, min + extent - 1};

            parallelism_found *= outer_extent;
        }
        outer->set_bounds(node, b);

        outer->children.emplace_back(inner);
        result.emplace_back(outer);
        return result;
    }

    // Return all possible ways to compute f in tiles.
    vector<IntrusivePtr<const LoopNest>> compute_in_tiles(const FunctionDAG::Node *f,
                                                          const LoopNest *parent,
                                                          const MachineParams &params,
                                                          int v,
                                                          bool in_realization) const {
        internal_assert(f);

        vector<IntrusivePtr<const LoopNest>> result;

        // Some pruning to not waste time on terrible states
        if (parent) {
            // Don't descend into loops that break our ability to
            // vectorize if we could have vectorized one level up.
            const auto &p = get_bounds(f)->region_computed(v);
            const auto &p_parent = parent->get_bounds(f)->region_computed(v);
            int64_t e = p.second - p.first + 1;
            int64_t ep = p_parent.second - p_parent.first + 1;
            if (ep >= f->vector_size && e < f->vector_size) return result;
        }

        // Figure out which child we can fuse this into
        int child = -1;
        bool called_by_multiple_children = false;
        for (int i = 0; i < (int)children.size(); i++) {
            if (children[i]->calls(f)) {
                if (child != -1) {
                    called_by_multiple_children = true;
                }
                child = i;
            }
        }

        const int vector_size = is_root() ? 1 : stage->vector_size;

        // HACK (when true)
        const bool force_only_output_compute_root = false;

        if ((!is_root() || f->is_output || !force_only_output_compute_root) &&
            !innermost &&
            (!in_realization || size.empty() || vector_dim == -1 || size[vector_dim] == 1)) {
            // Place the computation inside this loop
            std::unique_ptr<LoopNest> r{new LoopNest};
            r->copy_from(*this);
            r->compute_here(f, true, v);
            if (!in_realization) {
                r->store_at.insert(f);
            } else {
                r->tileable = false;
            }
            result.emplace_back(r.release());
        }

        if (f->is_output) {
            // Not permitted to compute at tiles of some consumer
            return result;
        }

        if (tileable) {
            // Generate a list of tile sizes to try
            auto tilings = generate_tilings(size, (int)(size.size() - 1), 2, !in_realization, vectorized_loop_index, innermost ? vector_size : 1);

            if (tilings.size() > 1000) {
                debug(0) << "Warning: lots of tilings: " << tilings.size() << "\n";
            }

            for (auto t : tilings) {
                if (parent->is_root()) {
                    const auto &l = stage->loop;
                    // Skip root-level tilings that would leave too
                    // many cores idle, and root-level tilings that
                    // would force serialization of dimensions we have
                    // decided to parallelize over in an earlier pass.
                    int total = 1;
                    size_t idx = 0;
                    for (auto s : t) {
                        if (l[idx].pure) {
                            total *= s;
                        }
                        idx++;
                    }
                    const double tasks_per_core = (double)total / params.parallelism;
                    const double idle_cores = std::ceil(tasks_per_core) / tasks_per_core;
                    if (idle_cores > 1.1) continue;
                }

                // Tile this loop and place the computation at some coarser granularity
                LoopNest *inner = new LoopNest, *outer = new LoopNest;
                inner->node      = outer->node      = node;
                inner->stage     = outer->stage     = stage;
                inner->stage_idx = outer->stage_idx = stage_idx;
                inner->tileable  = outer->tileable  = tileable;
                inner->vector_dim = outer->vector_dim = vector_dim;
                inner->vectorized_loop_index = outer->vectorized_loop_index = vectorized_loop_index;
                outer->size = size;
                outer->innermost = false;
                outer->parallel = parallel;
                inner->parallel = false;

                // First make an inner loop representing a 1x1x1... tile
                inner->size.resize(size.size(), 1);
                inner->innermost = innermost;
                inner->children = children;
                inner->inlined = inlined;
                inner->bounds = bounds;
                inner->store_at = store_at;


                {
                    auto b = inner->get_bounds(node)->make_copy();

                    // Then move factors from the outer loop to the inner loop
                    auto parent_bounds = parent->get_bounds(node);

                    for (size_t i = 0; i < t.size(); i++) {
                        int factor = t[i];
                        inner->size[i] = (outer->size[i] + factor - 1) / factor;
                        outer->size[i] = factor;
                        const auto &p = parent_bounds->loops(stage_idx, i);
                        int64_t min = p.first;
                        int64_t extent = p.second - min + 1;
                        extent = (extent + factor - 1) / factor;
                        b->loops(stage_idx, i) = {min, min + extent - 1};
                    }

                    // Region_{computed/required} on outer is now
                    // wrong, but it doesn't matter because consumers
                    // only look at the loops in get_bounds. Still,
                    // this is weird.

                    if (false) {// HACK
                        // Set those values to something clearly recognizable as non-meaningful.
                        for (int i = 0; i < node->func.dimensions(); i++) {
                            // The schedule depends on these!!! Chaos! Madness!
                            b->region_required(i).first = 2020202;
                            b->region_required(i).second = -2020202;
                            b->region_computed(i).first = 2020202;
                            b->region_computed(i).second = -2020202;
                        }
                    }

                    outer->set_bounds(node, b);
                }

                if (!in_realization) {
                    outer->store_at.insert(f);
                }
                outer->children.emplace_back(inner);

                bool may_slide = (!in_realization &&
                                  f->stages.size() == 1);
                if (may_slide) {
                    // Store here, but compute further in. Currently
                    // don't have to worry about the constraints this
                    // places on parallelism, as we forced all the
                    // parallelism to the outer loop.
                    auto opts = inner->compute_in_tiles(f, outer, params, v, true);
                    for (IntrusivePtr<const LoopNest> &n : opts) {
                        LoopNest *store_at_outer_compute_further_in = new LoopNest;
                        store_at_outer_compute_further_in->copy_from(*outer);
                        store_at_outer_compute_further_in->children.pop_back();
                        store_at_outer_compute_further_in->children.emplace_back(std::move(n));
                        result.emplace_back(store_at_outer_compute_further_in);
                    }
                }

                // Site the computation inside the outer loop
                outer->compute_here(f, true, v);
                outer->tileable &= !in_realization;
                result.emplace_back(outer);
            }
        }

        if (child >= 0 && !called_by_multiple_children && !in_realization) {
            // Push the Func further inwards in the loop nest

            // See if it's appropriate to slide over this loop
            const vector<int64_t> &child_size = children[child]->size;
            int num_ones = 0;
            for (auto s : child_size) {
                num_ones += (s == 1) ? 1 : 0;
            }
            // Can't slide at the root level, or no parallelism
            bool may_slide = !is_root();
            // Only slide over single-dimensional loops
            may_slide &= num_ones == ((int)child_size.size() - 1);
            // Don't slide funcs with update stages
            may_slide &= f->stages.size() == 1;

            // Don't slide over a split vector dimension (why?)
            may_slide &= (children[child]->vectorized_loop_index == -1 ||
                          child_size[children[child]->vectorized_loop_index] == 1);
            for (int store_here = 0; store_here < 2; store_here++) {
                if (store_here && !may_slide) {
                    // We place all our parallel loops at the root
                    // level, so this would constrain parallelism.
                    continue;
                }
                auto opts = children[child]->compute_in_tiles(f, this, params, v, store_here);
                for (IntrusivePtr<const LoopNest> &n : opts) {
                    // (Only valid if one child calls f) Push the
                    // computation into the child. Possibly leaving
                    // the storage out here.
                    LoopNest *r = new LoopNest;
                    r->copy_from(*this);
                    if (store_here) {
                        r->store_at.insert(f);
                    }
                    r->children[child] = n;
                    result.emplace_back(r);
                }
            }
        }

        return result;
    }

    // Note that StageScheduleState is movable-but-not-copyable thanks to its ostringstream member.
    struct StageScheduleState {
        double num_cores = 0; // How much parallelism do we need to exploit with this Func?
        int vector_dim = -1; // Which storage dimension is vectorized? We need to reorder it innermost.
        struct FuncVar {
            VarOrRVar orig;
            VarOrRVar var;
            string accessor;
            int64_t extent = 0;
            bool outermost = false, parallel = false, exists = false, pure = false;
            bool unrolled = false;
            bool vectorized = false;
            TailStrategy tail_strategy = TailStrategy::Auto;
            FuncVar() : orig(Var()), var(Var()) {}
        };
        vector<FuncVar> vars; // In order from innermost to outermost. Each group of d is one tiling.
        std::ostringstream schedule_source;
    };

    struct ScheduleData {
        set<const FunctionDAG::Node*> store_on_stack_set;
        map<const LoopNest*, std::string> loop_levels;
    };

    struct TEST {
        StageMap<std::unique_ptr<LoopNest::StageScheduleState>> state_map;
        ScheduleData schedule_data;
    };

    void apply(LoopLevel here,
               StageMap<std::unique_ptr<StageScheduleState>> &state_map,
               double num_cores,
               int depth,
               const LoopNest *parent,
               const LoopNest *compute_site,
               StageScheduleState::FuncVar* level,
               StageScheduleState* vars_parent,
               ScheduleData& schedule_data,
               bool obtain_loop_nest_only = false) const {
        if (is_root()) {
            for (auto &c : children) {
                if (!obtain_loop_nest_only) {
                    Func(c->node->func).compute_root();
                }
                c->apply(LoopLevel::root(), state_map, num_cores, 1, this, c.get(), level, vars_parent, schedule_data, obtain_loop_nest_only);
                if (c->stage_idx == 0) {
                    auto &state = state_map.get(c->stage);
                    state->schedule_source << "\n    .compute_root()";
                    // TODO: Omitting logic for printing store_root() assumes everything store_root is also compute root
                }
            }
        } else {
            if (parent && parent->node != node) {
                compute_site = this;
            }

            const auto &symbolic_loop = stage->loop;
            const auto &parent_bounds = parent->get_bounds(node);
            if (!state_map.contains(stage)) {
                StageScheduleState *state = new StageScheduleState;
                state->num_cores = num_cores;
                state->vector_dim = vector_dim;
                for (size_t i = 0; i < symbolic_loop.size(); i++) {
                    StageScheduleState::FuncVar fv;
                    const auto &l = symbolic_loop[i];
                    fv.var = VarOrRVar(l.var, !l.pure);
                    fv.orig = fv.var;
                    fv.accessor = l.accessor;
                    const auto &p = parent_bounds->loops(stage_idx, i);
                    fv.extent = p.second - p.first + 1;
                    fv.outermost = true;
                    fv.parallel = parent->is_root() && l.pure;
                    fv.exists = true;
                    fv.pure = l.pure;
                    state->vars.push_back(fv);
                }
                state_map.emplace(stage, std::unique_ptr<StageScheduleState>(state));
            }
            auto &state = *(state_map.get(stage));

            // The getter for grabbing Func handles is reverse topological order
            Stage s = Func(node->func);
            if (stage_idx > 0) {
                s = Func(node->func).update(stage_idx - 1);
            }

            if (stage_idx == 0 && parent->node != node) {
                // Pick a memory type
                double bytes = node->bytes_per_point;
                for (int i = 0; i < node->func.dimensions(); i++) {
                    const auto &p = parent_bounds->region_computed(i);
                    bytes *= p.second - p.first + 1;
                }
                if (bytes < 64000 && depth > 2) {
                    // If it's probably a small allocation, and it's
                    // made more than once, use stack-scoped
                    // storage. Otherwise let the compiler pick heap
                    // or stack as it likes.
                    if (!obtain_loop_nest_only) {
                        Func(node->func).store_in(MemoryType::Stack);
                    }
                    state.schedule_source << "\n    .store_in(MemoryType::Stack)";
                    schedule_data.store_on_stack_set.insert(node);
                }
            }

            // Pick a tail strategy for any splits of pure vars. RVars always use guardwithif
            auto pure_var_tail_strategy = TailStrategy::Auto;
            if (!compute_site->accesses_input_buffer() && !node->is_output) {
                // Roundup is lowest overhead, provided it doesn't
                // expand the bounds read on the input or written on
                // the output. However, you can only really use it on
                // pure stages that don't access the input anywhere in
                // their loop nest.
                pure_var_tail_strategy = TailStrategy::RoundUp;
            } else if (stage_idx == 0) {
                // Pure stages that access the input use shiftinwards
                pure_var_tail_strategy = TailStrategy::ShiftInwards;
            } else {
                // For pure vars in update stages that access the
                // input, it's not safe to round up or redundantly
                // recompute
                pure_var_tail_strategy = TailStrategy::GuardWithIf;
            }

            if (!size.empty()) {
                if (innermost) {
                    if (vectorized_loop_index >= 0) {
                        auto &v = state.vars[vectorized_loop_index];
                        internal_assert(v.exists);
                        // Is the result of a split
                        state.schedule_source
                            << "\n    .vectorize(" << v.var.name() << ")";
                        if (!obtain_loop_nest_only) {
                            s.vectorize(v.var);
                        }
                        v.vectorized = true;
                    }
                } else {
                    // Grab the innermost loop for this node
                    const LoopNest *innermost_loop = this, *child = nullptr;
                    while (!innermost_loop->innermost) {
                        for (const auto &c : innermost_loop->children) {
                            if (c->node == node) {
                                if (!child) {
                                    child = c.get();
                                }
                                innermost_loop = c.get();
                                break;
                            }
                        }
                    }

                    // Do the implied splits
                    vector<StageScheduleState::FuncVar> new_inner;
                    for (size_t i = 0; i < symbolic_loop.size(); i++) {
                        StageScheduleState::FuncVar v;
                        StageScheduleState::FuncVar &parent = state.vars[i];

                        int64_t factor = (parent.extent + size[i] - 1) / size[i];

                        if (child && innermost_loop->size[i] > factor) {
                            factor = innermost_loop->size[i];
                        }

                        if (!parent.exists || factor == 1) {
                            v.exists = false;
                            v.extent = 1;
                        } else if (size[i] == 1 && !(child && child->innermost && i == vectorized_loop_index)) {
                            // Not split in this dimension
                            v = parent;
                            v.parallel = false;
                            parent.exists = false;
                            parent.extent = 1;
                        } else {
                            VarOrRVar inner(Var(parent.var.name() + "i"));
                            if (parent.var.is_rvar) {
                                inner = RVar(parent.var.name() + "i");
                            }

                            auto tail_strategy = pure_var_tail_strategy;
                            // If it's an RVar, or not the outermost split and we're in an update, we need a guard with if instead.
                            if (parent.var.is_rvar || (stage_idx != 0 && !parent.outermost)) {
                                tail_strategy = TailStrategy::GuardWithIf;
                            }
                            if (!obtain_loop_nest_only) {
                                s.split(parent.var, parent.var, inner, (int)factor, tail_strategy);
                            }
                            state.schedule_source
                                << "\n    .split("
                                << parent.var.name() << ", "
                                << parent.var.name() << ", "
                                << inner.name() << ", "
                                << factor << ", "
                                << "TailStrategy::" << tail_strategy << ")";
                            v = parent;
                            parent.extent = size[i];
                            v.var = inner;
                            v.accessor.clear();
                            v.extent = factor;
                            v.parallel = false;
                            v.outermost = false;
                        }
                        new_inner.push_back(v);
                    }

                    if (child->innermost) {
                        // Maybe do some unrolling

                        int64_t product_of_pure_loops = 1;
                        for (size_t i = 0; i < symbolic_loop.size(); i++) {
                            if (symbolic_loop[i].pure) {
                                product_of_pure_loops *= state.vars[i].extent;
                            }
                        }

                        // Temporary hack until we can actually model
                        // which loops are constant size. The other part
                        // of this hack is that we changed the unrolling
                        // pass to not complain if things are not
                        // constant.
                        bool all_pure_loops_constant_size = true;

                        if (product_of_pure_loops <= 16 && all_pure_loops_constant_size) {
                            // There's a hope we can fit anything compute-at this level into registers if we fully unroll
                            // TODO: 16 should be the number of vector registers in the architecture
                            std::stable_sort(state.vars.begin(), state.vars.begin() + symbolic_loop.size(),
                                             [](const StageScheduleState::FuncVar &a, const StageScheduleState::FuncVar &b) {
                                                 return a.pure && !b.pure;
                                             });

                            for (size_t i = 0; i < symbolic_loop.size(); i++) {
                                if (state.vars[i].pure && state.vars[i].exists && state.vars[i].extent > 1) {
                                    if (!obtain_loop_nest_only) {
                                        s.unroll(state.vars[i].var);
                                    }
                                    state.schedule_source << "\n    .unroll(" << state.vars[i].var.name() << ")";
                                    state.vars[i].unrolled = true;
                                }
                            }
                        }
                    }

                    bool found = false;
                    for (const auto &v : state.vars) {
                        if (!v.exists) continue;
                        here = LoopLevel(node->func, v.var);
                        schedule_data.loop_levels[this] = v.var.name();
                        found = true;
                        break;
                    }
                    if (!found) {
                        here = LoopLevel(node->func, Var::outermost());
                    }
                    // internal_assert(found) << "Could not find appropriate compute_at location for children of " << node->func.name() << "\n";
                    state.vars.insert(state.vars.begin(), new_inner.begin(), new_inner.end());
                }
            }
            if (innermost) {
                internal_assert(store_at.empty());
                internal_assert(children.empty());
                return;
            }


            for (auto f : store_at) {
                if (!obtain_loop_nest_only) {
                    Func(f->func).store_at(here);
                }
            }
            for (auto s : size) {
                num_cores /= s;
            }
            here.lock();
            string loop_level;
            if (here.is_root()) {
                loop_level = "_root()";
            } else {
                loop_level = "_at(" + here.func() + ", " + here.var().name() + ")";
            }
            for (auto &c : children) {
                if (c->node != node && !obtain_loop_nest_only) {
                    Func(c->node->func).compute_at(here);
                }
                c->apply(here, state_map, num_cores, depth + 1, this, compute_site, level, &state, schedule_data, obtain_loop_nest_only);
                if (c->node != node && c->stage_idx == 0) {
                    auto &state = *(state_map.get(c->stage));
                    state.schedule_source << "\n    .compute" << loop_level;
                }
            }
            for (auto f : store_at) {
                bool computed_here = false;
                for (auto &c : children) {
                    if (c->node == f) {
                        computed_here = true;
                        break;
                    }
                }
                if (!computed_here) {
                    auto &state = *(state_map.get(&(f->stages[0])));
                    state.schedule_source << "\n    .store" << loop_level;
                }
            }
        }
    }

};

}

template<>
RefCount &ref_count<LoopNest>(const LoopNest *t) {return t->ref_count;}

template<>
void destroy<LoopNest>(const LoopNest *t) {delete t;}

namespace {

struct State {
    mutable RefCount ref_count;
    IntrusivePtr<const LoopNest> root;
    IntrusivePtr<const State> parent;
    double cost = 0;
    int num_funcs_scheduled = 0;
    bool penalized = false;

    State() = default;
    State(const State &) = delete;
    State(State &&) = delete;
    void operator=(const State &) = delete;
    void operator=(State &&) = delete;

    static int cost_calculations;

    uint64_t structural_hash(int depth, int parallelism) const {
        uint64_t h = num_funcs_scheduled;
        internal_assert(root.defined());
        root->structural_hash(h, depth, parallelism);
        return h;
    }

    void compute_featurization(const FunctionDAG &dag, const MachineParams &params, StageMap<ScheduleFeatures> *features) {
        StageMap<LoopNest::Sites> sites;
        sites.make_large(dag.nodes[0].stages[0].max_id);
        features->make_large(dag.nodes[0].stages[0].max_id);
        internal_assert(root.defined());
        root->get_sites(sites);

        // For the input nodes, the compute and store sites are root,
        // and the produce and innermost sites are unset (nullptr)
        for (const auto &n : dag.nodes) {
            if (n.is_input) {
                auto &s = sites.get_or_create(&(n.stages[0]));
                s.compute = root.get();
                s.store = root.get();
            }
        }

        root->compute_features(params, sites, 1, 1, nullptr, *root, nullptr, features);
    }

    void save_featurization(const FunctionDAG &dag, const MachineParams &params, const std::string &feature_file) {
        StageMap<ScheduleFeatures> features;
        compute_featurization(dag, params, &features);

        std::ofstream binfile(feature_file, std::ios::binary | std::ios_base::trunc);
        for (const auto &n : dag.nodes) {
            if (n.is_input) continue;
            for (size_t stage_idx = n.stages.size(); stage_idx > 0; stage_idx--) {
                const auto &s = n.stages[stage_idx - 1];
                const size_t num_schedule_features = sizeof(ScheduleFeatures) / sizeof(int64_t);
                const size_t num_pipeline_features = sizeof(PipelineFeatures) / sizeof(int);
                const auto &sched_feat = features.get(&s);
                const int64_t *sched_ints = (const int64_t *)(&sched_feat);
                const int *pipe_ints = (const int *)(&s.features);

                float buf[num_schedule_features + num_pipeline_features];
                // Save them as floats
                for (size_t i = 0; i < num_schedule_features; i++) {
                    buf[i] = sched_ints[i];
                }

                for (size_t i = 0; i < num_pipeline_features; i++) {
                    buf[i + num_schedule_features] = pipe_ints[i];
                }

                binfile.write((const char *)buf, sizeof(buf));
            }
        }
        binfile.close();
        internal_assert(!binfile.fail()) << "Failed to write " << feature_file;
    }

    bool calculate_cost(const FunctionDAG &dag, const MachineParams &params, CostModel *cost_model, LoopNestRoot& loop_nest, bool verbose = false, json *json_dump = nullptr) {
        StageMap<ScheduleFeatures> features;
        compute_featurization(dag, params, &features);

        json jdata;
        if (json_dump) {
            std::map<std::string, Expr> store_at_bounds;
            std::map<std::string, Expr> compute_bounds;
            std::map<std::string, Expr> compute_offsets;
            std::map<std::string, std::map<std::string, Expr>> store_offsets;
            std::map<std::string, Expr> compute_mins;
            std::map<std::string, Expr> current_mins;
            std::map<std::string, Expr> root_store_offsets;
            std::map<std::string, int> strides;
            std::map<std::string, double> parallelism;
            std::map<std::string, AllocNode*> allocs;
            std::set<std::string> alloced;

            auto vars_and_schedule_data = apply_schedule(dag, params, true);

            root->create_loop_nest(dag, params, nullptr, 0, 0, 0, &loop_nest.block,
                store_at_bounds, compute_bounds, strides, parallelism,
                params.parallelism, features, vars_and_schedule_data.first,
                vars_and_schedule_data.second, loop_nest.output_sizes, 0, allocs, alloced, compute_offsets, store_offsets, compute_mins, current_mins, root_store_offsets);

            jdata["loop"] = loop_nest.block.to_json();

            (*json_dump)["features"] = jdata;
        }

        cost = 0;

        if (verbose) {
            for (auto it = features.begin(); it != features.end(); it++) {
                auto &stage = *(it.key());
                const auto &feat = it.value();
                debug(0) << "Schedule features for " << stage.stage.name() << "\n";
                feat.dump();
            }
        }

        // use either deep network or linear model to predict cost
        if (cost_model) {

            // Perform any quick rejection tests before enqueuing this
            for (auto it = features.begin(); it != features.end(); it++) {
                if (!it.key()->node->func.is_wrapper()) { // It's OK to repeatedly stage data
                    auto &feat = it.value();
                    if (feat.points_computed_total + feat.inlined_calls > 10 * feat.points_computed_minimum) {
                        cost = 1e50;
                        return true;
                    }
                }
            }

            // Avoid code size explosion from recursive inlining.
            if (root->max_inlined_calls() >= 256) {
                cost = 1e50;
                return true;
            }

            int num_stages = (int)features.size();

            const size_t schedule_feat_size = sizeof(ScheduleFeatures) / sizeof(int64_t);

            Runtime::Buffer<float> schedule_features;

            // Won't actually run anything until we call evaluate_costs...
            cost_model->enqueue(num_stages, &schedule_features, &cost);

            // index of current stage whose features we are reading
            int stage = 0;
            // load schedule features into input buffer
            for (const auto &n : dag.nodes) {
                if (n.is_input) continue; // Inputs are computed outside of the pipeline and don't count.
                if (stage >= num_stages) break;
                for (auto it = n.stages.rbegin(); it != n.stages.rend(); it++) {
                    internal_assert(features.contains(&*it)) << n.func.name() << "\n";
                    const auto &feat = features.get(&*it);
                    const int64_t *sched_stats = (const int64_t *)(&feat);
                    for (size_t i = 0; i < schedule_feat_size; i++) {
                        schedule_features(i, stage) = sched_stats[i];
                    }

                    stage += 1;
                }
            }
            internal_assert(stage == num_stages);
        } else {
            // We have no throughput predictor.
            for (auto it = features.begin(); it != features.end(); it++) {
                auto &stage = *(it.key());
                const auto &feat = it.value();
                // Reject silly schedules. They're not even useful for
                // training data, as they potentially take the age of
                // the universe to benchmark.
                if (feat.points_computed_total + feat.inlined_calls > 1000*feat.points_computed_minimum) return false;
                if (feat.inlined_calls >= 64) return false;
                
                double compute_cost = 0;
                const int *pipeline_feat = (const int *)(&stage.features.op_histogram[0][0]);
                double per_element_compute_cost = 0;
                for (size_t i = 0; i < sizeof(stage.features.op_histogram) / sizeof(int); i++) {
                    per_element_compute_cost += pipeline_feat[i];
                }

                if (feat.inlined_calls > 0) {
                    const double per_element_compute_cost_of_memcpy = 1 + 2*stage.node->func.dimensions();
                    per_element_compute_cost = std::max(0.0, per_element_compute_cost - per_element_compute_cost_of_memcpy);
                }
                
                // Assume that narrow types are cheaper because they
                // vectorize wider, and just count the number of
                // vectors computed.
                compute_cost = per_element_compute_cost * (feat.num_vectors + feat.num_scalars);

                // Figure out vector overcompute
                const int native_vector_size = feat.native_vector_size;
                const double idle_simd_lanes = (double)native_vector_size / feat.vector_size;
                compute_cost *= idle_simd_lanes;

                {
                    // Few parallel tasks may be a bad idea due to
                    // waiting for the long pole to finish.  Say
                    // we have a huge number of tasks relative to
                    // cores. We'd expect their start times to
                    // eventually become evenly spaced, which
                    // means we get a little triangle of idle
                    // cores with total area 0.5 * task_size *
                    // num_cores at the end. This bloats the total
                    // amount of work by:
                    //   (0.5 * task_size * num_cores + task_size * num_tasks) / (task_size * num_tasks)
                    // = (0.5 * num_cores + num_tasks) / num_tasks

                    internal_assert(feat.inner_parallelism > 0 && feat.outer_parallelism > 0);

                    const double num_tasks = feat.inner_parallelism;
                    const double num_cores = (double)params.parallelism / feat.outer_parallelism;
                    double idle_core_wastage = (0.5 * num_cores + num_tasks) / num_tasks;

                    // Evaluated at num_tasks = num_cores, this
                    // gives a ridiculous 1.5x multiplier. Our
                    // argument doesn't hold because the tasks
                    // start synchronized. Just cap it at 20%
                    // wastage.
                    idle_core_wastage = std::min(idle_core_wastage, 1.2);

                    if (verbose) {
                        debug(0) << "idle_core_wastage_1 = " << idle_core_wastage << "\n";
                    }

                    // Cores can also be idle if the number of
                    // tasks is small and not a multiple of the
                    // number of cores. E.g. 9 tasks on 8 cores
                    // takes about the same amount of time as 16
                    // tasks.
                    idle_core_wastage *= std::ceil(num_tasks / num_cores) * (num_cores / num_tasks);

                    compute_cost *= idle_core_wastage;

                    if (verbose) {
                        debug(0) << "idle_core_wastage_2 = " << idle_core_wastage << "\n";
                    }
                }

                double cold_cache_misses = 0, cost_of_cold_miss = 0, capacity_cache_misses = 0, cost_of_capacity_miss = 0;
                if (feat.inlined_calls == 0) {
                    // Estimate the number of cold cache misses on the data that this reads from and their cost
                    // Cost dominated by lines not bytes due to streaming prefetchers
                    cold_cache_misses = (feat.unique_lines_read_per_realization +
                                         feat.unique_bytes_read_per_realization * 1e-3);

                    cold_cache_misses *= feat.num_realizations;
                    //int64_t footprint = std::min(feat.allocation_bytes_read_per_realization, feat.bytes_read_per_realization);
                    int64_t footprint = feat.allocation_bytes_read_per_realization;
                    //cost_of_miss = std::sqrt(footprint) * 40 * 5e-3;
                    cost_of_cold_miss = footprint * 40 * 1e-4;

                    // Now estimate the number of capacity-related cache misses using the total number of bytes read.

                    // We have a number of unique bytes read. Call the
                    // cache level large enough to fit it L(n+1). The
                    // next cache level in is Ln. How many misses will
                    // we incur in Ln? If we load randomly within the
                    // footprint, we'll miss some constant fraction of
                    // the time. The cost of such a miss is the cost
                    // of going out to cache level L(n+1). Note that
                    // *cold* misses, by contrast, go out to the cache
                    // level that fits the entire source allocation,
                    // not just the footprint accessed of it.
                    capacity_cache_misses = feat.num_vectors * (feat.vector_loads_per_vector + feat.scalar_loads_per_vector);
                    capacity_cache_misses += feat.num_scalars * feat.scalar_loads_per_scalar;
                    capacity_cache_misses *= 1e-2;
                    cost_of_capacity_miss = feat.unique_bytes_read_per_realization * 40 * 1e-4;

                    // We'll assume multiway caches work well and ignore the other 'C' (conflict cache misses).
                }

                double memory_load_cost = cold_cache_misses * cost_of_cold_miss + capacity_cache_misses * cost_of_capacity_miss;

                double cache_misses = 0, cost_of_miss = 0;
                if (feat.inlined_calls == 0) {
                    // Estimate the number of cache misses on the data that this writes to and their cost
                    int64_t lines_written_per_realization = feat.inner_parallelism * (feat.bytes_at_task / feat.innermost_bytes_at_task);
                    cache_misses = 1e1 * lines_written_per_realization + feat.bytes_at_realization * 1e-2;
                    cache_misses *= feat.num_realizations;
                    //cost_of_miss = std::sqrt(feat.bytes_at_production) * 40 * 5e-3;
                    cost_of_miss = feat.bytes_at_production * 40 * 2e-6;
                }

                double memory_store_cost = cache_misses * cost_of_miss;

                // Penalize writing partial cache lines. Assume a cache line is two simd vectors.
                const double native_cache_line_size = 2 * idle_simd_lanes; // two full vectors
                const double cache_line_wastage = std::max(1.0, native_cache_line_size / feat.innermost_pure_loop_extent);
                memory_store_cost *= cache_line_wastage;

                // Malloc aint free. Small allocations should go on the stack, but this isn't totally reliable.
                double cost_of_mallocs = feat.num_realizations * 1e2;

                // Penalize working sets that start to fall out of cache
                double ws = 1e-6 * feat.working_set;
                double cost_of_working_set = ws * ws * ws * 40 * feat.num_realizations;

                if (verbose) {
                    debug(0) << "Cost model for " << stage.stage.name() << " "
                             << compute_cost << " + "
                             << memory_load_cost << " + "
                             << memory_store_cost << " + "
                             << cost_of_mallocs << " + "
                             << cost_of_working_set << '\n';
                }

                cost += compute_cost + memory_load_cost + memory_store_cost + cost_of_mallocs + cost_of_working_set;
            }
        }
        cost_calculations++;
        return true;
    }

    IntrusivePtr<State> make_child() const {
        State *s = new State;
        s->parent = this;
        s->root = root;
        s->cost = cost;
        s->num_funcs_scheduled = num_funcs_scheduled;
        return s;
    }

    void generate_children(const FunctionDAG &dag,
                           const MachineParams &params,
                           CostModel *cost_model,
                           std::function<void(IntrusivePtr<State> &&)> &accept_child) const {
        internal_assert(root.defined() && root->is_root());

        if (num_funcs_scheduled == 2*(int)dag.nodes.size()) {
            return;
        }

        int next_node = num_funcs_scheduled / 2;
        int phase = num_funcs_scheduled % 2;

        // Enumerate all legal ways to schedule the next Func
        const FunctionDAG::Node *node = &dag.nodes[next_node];
        for (const auto *e : node->outgoing_edges) {
            internal_assert(root->computes(e->consumer->node))
                << "Partially scheduled code doesn't compute " << e->consumer->name
                << ", which is one of the consumers of " << node->func.name();
        }

        if (node->is_input) {
            // We don't need to schedule nodes that represent inputs,
            // and there are no other decisions to be made about them
            // at this time.
            // debug(0) << "Skipping over scheduling input node: " << node->func.name() << "\n";
            auto child = make_child();
            child->num_funcs_scheduled++;
            accept_child(std::move(child));
            return;
        }

        if (!node->outgoing_edges.empty() && !root->calls(node)) {
            debug(0) << "In state:\n";
            dump();
            debug(0) << node->func.name() << " is consumed by:\n";
            for (const auto *e : node->outgoing_edges) {
                debug(0) << e->consumer->name << "\n";
                debug(0) << "Which in turn consumes:\n";
                for (const auto *e2 : e->consumer->incoming_edges) {
                    debug(0) << "  " << e2->producer->func.name() << "\n";
                }
            }
            internal_error << "Pipeline so far doesn't use next Func: " << node->func.name() << '\n';
        }

        int num_children = 0;

        if (phase == 0) {
            // Injecting realizations
            {
                // 1) Inline it
                if (node->stages.size() == 1 && !node->is_output) {
                    auto child = make_child();
                    LoopNest *new_root = new LoopNest;
                    new_root->copy_from(*root);
                    new_root->inline_func(node);
                    child->root = new_root;
                    child->num_funcs_scheduled++;
                    // TODO: filter children here instead of calculating the cost of children we don't want.
                    LoopNestRoot loop_nest;
                    if (child->calculate_cost(dag, params, cost_model, loop_nest)) {
                        internal_assert(child->root->computes(node)) << "Failed to inline " << node->func.name() << '\n';
                        num_children++;
                        accept_child(std::move(child));
                    } else {
                        // Discarding state....
                    }
                }
            }

            // Construct a list of plausible dimensions to vectorize over
            // TODO: Pre-prune the list of sane dimensions to
            // vectorize a Func over to reduce branching factor.
            vector<int> vector_dims;
            for (int v = 0; v < node->func.dimensions(); v++) {
                const auto &p = root->get_bounds(node)->region_computed(v);
                if (p.second - p.first + 1 >= node->vector_size) {
                    vector_dims.push_back(v);
                }
            }
            if (vector_dims.empty()) {
                vector_dims.push_back(0);
            }


            // HACK: May only vectorize across x, if there is one
            
            for (int v = 0; v < node->func.dimensions(); v++) {
                if (node->func.args()[v] == "x") {
                    vector_dims.clear();
                    vector_dims.push_back(v);
                    break;
                }
            }
            

            // 2) Realize it somewhere
            for (int vector_dim : vector_dims) {
                // Outputs must be vectorized over their innermost
                // dimension, because we don't have control of the
                // storage. TODO: Go inspect to see which dimension has a
                // stride==1 constraint instead of assuming 0.
                if (vector_dim > 0 && (node->is_output || node->is_input)) break;

                auto tile_options = root->compute_in_tiles(node, nullptr, params, vector_dim, false);
                for (IntrusivePtr<const LoopNest> &n : tile_options) {
                    auto child = make_child();
                    child->root = std::move(n);
                    child->num_funcs_scheduled++;
                    LoopNestRoot loop_nest;
                    if (child->calculate_cost(dag, params, cost_model, loop_nest)) {
                        internal_assert(child->root->computes(node)) << "Failed to inject realization of " << node->func.name() << '\n';
                        num_children++;
                        accept_child(std::move(child));
                    }
                }
            }
        } else {
            // Deciding on parallel tasks

            auto child = make_child();
            LoopNest *new_root = new LoopNest;
            new_root->copy_from(*root);

            for (int i = 0; i < (int)root->children.size(); i++) {
                if (root->children[i]->node == node) {
                    // For now assume that parallelize in tiles returns a single option
                    new_root->children[i] =
                        new_root->children[i]->parallelize_in_tiles(params, root.get())[0];
                }
            }

            child->root = new_root;
            child->num_funcs_scheduled++;
            LoopNestRoot loop_nest;
            if (child->calculate_cost(dag, params, cost_model, loop_nest)) {
                num_children++;
                accept_child(std::move(child));
            }
        }

        if (num_children == 0) {
            debug(0) << "Warning: Found no legal way to schedule "
                     << node->func.name() << " in the following State:\n";
            dump();
            internal_error << "Aborted";
        }
    }

    void dump() const {
        debug(0) << "State with cost " << cost << ":\n";
        root->dump("");
        debug(0) << schedule_source;
    }

    json json_dump() const {
        json jdata;
        jdata["cost"] = cost;
        return jdata;
    }

    string schedule_source;

    pair<StageMap<std::unique_ptr<LoopNest::StageScheduleState>>, LoopNest::ScheduleData> apply_schedule(const FunctionDAG &dag, const MachineParams &params, bool obtain_loop_nest_only = false) {
        //TEST t;
        //StageMap<std::unique_ptr<LoopNest::StageScheduleState>>& state_map = t.state_map;
        //LoopNest::ScheduleData& schedule_data = t.schedule_data;
        StageMap<std::unique_ptr<LoopNest::StageScheduleState>> state_map;
        LoopNest::ScheduleData schedule_data;
        root->apply(LoopLevel::root(), state_map, params.parallelism, 0, nullptr, nullptr, nullptr, nullptr, schedule_data, obtain_loop_nest_only);

        std::ostringstream src;

        // Print handles for all the Funcs
        int i = (int)(dag.nodes.size() - 1);
        for (const auto &n : dag.nodes) {
            if (!n.is_input) {
                src << "Func " << n.func.name() << " = get_pipeline().get_func(" << i << ");\n";
            }
            i--;
        }

        // Gather all Vars and RVars so that we can declare them in the emitted source
        map<string, string> vars, rvars;
        for (auto &p : state_map) {
            for (auto &v : p.second->vars) {
                if (v.exists) {
                    if (v.var.is_rvar) {
                        rvars.emplace(v.var.name(), v.accessor);
                    } else {
                        vars.emplace(v.var.name(), v.accessor);
                    }
                }
            }
        }
        if (!vars.empty()) {
            string prefix = "Var ";
            for (const auto &p : vars) {
                if (p.second.empty()) {
                    src << prefix << p.first << "(\"" << p.first << "\")";
                } else {
                    src << prefix << p.first << "(" << p.second << ")";
                }
                prefix = ", ";
            }
            src << ";\n";
        }
        if (!rvars.empty()) {
            string prefix = "RVar ";
            for (const auto &p : rvars) {
                if (p.second.empty()) {
                    src << prefix << p.first << "(\"" << p.first << "\")";
                } else {
                    src << prefix << p.first << "(" << p.second << ")";
                }
                prefix = ", ";
            }
            src << ";\n";
        }

        for (auto &p : state_map) {
            if (p.first->node->is_input) continue;

            Stage stage(p.first->stage);

            // Do all the reorders and pick which vars to
            // parallelize.
            vector<VarOrRVar> vars;
            int64_t parallel_tasks = 1;
            vector<VarOrRVar> parallel_vars;
            bool any_parallel_vars = false, any_parallel_rvars = false;
            bool parallelism_exhausted = false;
            for (auto it = p.second->vars.rbegin(); it != p.second->vars.rend(); it++) {
                if (parallelism_exhausted) {
                    it->parallel = false;
                    continue;
                }
                if (!it->exists || it->extent == 1) continue;
                if (!it->parallel) {
                    parallelism_exhausted = true;
                    continue;
                }
                any_parallel_rvars |= it->var.is_rvar;
                any_parallel_vars |= !it->var.is_rvar;
                parallel_tasks *= it->extent;
                parallel_vars.push_back(it->var);
                it->unrolled = false;
            }

            vector<LoopNest::StageScheduleState::FuncVar> func_vars;
            for (auto &v : p.second->vars) {
                if (v.exists) {
                    func_vars.push_back(v);
                }
            }

            if (p.second->vars.size() > 1) {
                p.second->schedule_source << "\n    .reorder(";
                bool first = true;
                for (auto &v : p.second->vars) {
                    if (v.exists) {
                        vars.push_back(v.var);
                        if (!first) {
                            p.second->schedule_source << ", ";
                        }
                        first = false;
                        p.second->schedule_source << v.var.name();
                    }
                }
                p.second->schedule_source << ")";
                if (!obtain_loop_nest_only) {
                    stage.reorder(vars);
                }
            }

            p.second->vars = func_vars;

            // Halide doesn't let you fuse an RVar with a Var, even if
            // they are both pure.
            bool can_fuse = false; // !(any_parallel_vars && any_parallel_rvars);

            if (can_fuse) {
                for (size_t i = 1; i < parallel_vars.size(); i++) {
                    // Outermost, and next outermost. Preserve the inner
                    // name to not invalidate any compute_ats.
                    p.second->schedule_source << "\n    .fuse(" << parallel_vars[i].name()
                                              << ", " << parallel_vars[i-1].name()
                                              << ", " << parallel_vars[i].name() << ")";
                    if (!obtain_loop_nest_only) {
                        stage.fuse(parallel_vars[i], parallel_vars[i-1], parallel_vars[i]);
                    }
                }
                if (!parallel_vars.empty()) {
                    p.second->schedule_source << "\n    .parallel(" << parallel_vars.back().name() << ")";
                    if (!obtain_loop_nest_only) {
                        stage.parallel(parallel_vars.back());
                    }
                }
            } else {
                for (const auto &v : parallel_vars) {
                    p.second->schedule_source << "\n    .parallel(" << v.name() << ")";
                    if (!obtain_loop_nest_only) {
                        stage.parallel(v);
                    }
                }
            }

            // Reorder the vector dimension innermost
            if (p.first->index == 0 && p.second->vector_dim > 0) {
                vector<Var> storage_vars = Func(p.first->node->func).args();
                for (int i = p.second->vector_dim; i > 0; i--) {
                    std::swap(storage_vars[i], storage_vars[i-1]);
                }
                p.second->schedule_source << "\n    .reorder_storage(";
                bool first = true;
                for (auto v : storage_vars) {
                    if (!first) {
                        p.second->schedule_source << ", ";
                    }
                    first = false;
                    p.second->schedule_source << v.name();
                }
                p.second->schedule_source << ")";
                if (!obtain_loop_nest_only) {
                    Func(p.first->node->func).reorder_storage(storage_vars);
                }
            }

            // Dump the schedule source string
            src << p.first->name
                << p.second->schedule_source.str()
                << ";\n";
        }
        schedule_source = src.str();
        bool in_quotes = false;
        for (auto &c : schedule_source) {
            in_quotes ^= (c == '"');
            if (!in_quotes && c == '$') c = '_';
        }

        return {std::move(state_map), schedule_data};
    }
};



int State::cost_calculations = 0;

}

template<>
RefCount &ref_count<State>(const State *t) {return t->ref_count;}

template<>
void destroy<State>(const State *t) {delete t;}

namespace {

// A priority queue of states, sorted according to increasing
// cost. Never shrinks, to avoid reallocations.
// Can't use std::priority_queue because it doesn't support unique_ptr.
class StateQueue {
private:
    struct CompareStates {
        bool operator()(const IntrusivePtr<State> &a, const IntrusivePtr<State> &b) const {
            return a->cost > b->cost;
        }
    };

    std::vector<IntrusivePtr<State>> storage;
    size_t sz = 0;
public:
    void emplace(IntrusivePtr<State> &&s) {
        if (sz >= storage.size()) {
            storage.resize(std::max(sz * 2, (size_t)64));
        }
        internal_assert(sz < storage.size()) << sz << " " << storage.size() << "\n";
        storage[sz] = std::move(s);
        sz++;
        std::push_heap(storage.begin(), storage.begin() + sz, CompareStates{});
    }

    IntrusivePtr<State> pop() {
        internal_assert(sz <= storage.size()) << sz << " " << storage.size() << "\n";
        std::pop_heap(storage.begin(), storage.begin() + sz, CompareStates{});
        sz--;
        return std::move(storage[sz]);
    }

    const IntrusivePtr<State> &top() {
        return storage[0];
    }

    bool empty() const {
        return sz == 0;
    }

    size_t size() const {
        return sz;
    }

    void swap(StateQueue &other) {
        storage.swap(other.storage);
        std::swap(sz, other.sz);
    }

    IntrusivePtr<State> operator[](int idx) const {
        return storage[idx];
    }

    void resort() {
        std::make_heap(storage.begin(), storage.begin() + sz, CompareStates{});
    }

    void clear() {
        for (size_t i = 0; i < sz; i++) {
            storage[i] = IntrusivePtr<State>{};
        }
        sz = 0;
    }
};

void configure_pipeline_features(const FunctionDAG &dag,
                                 const MachineParams &params,
                                 CostModel *cost_model) {
    cost_model->reset();
    const int pipeline_feat_size = 56 * 7;
    static_assert(sizeof(PipelineFeatures) - 7 * sizeof(int) ==
                  sizeof(int) * pipeline_feat_size,
                  "Incorrect size for pipeline features");
    int num_stages = 0;
    for (const auto &n : dag.nodes) {
        if (!n.is_input) num_stages += (int)n.stages.size();
    }    
    Runtime::Buffer<float> pipeline_features(56, 7, num_stages);
    int stage = 0;
    for (const auto &n : dag.nodes) {
        if (n.is_input) continue;
        for (auto it = n.stages.rbegin(); it != n.stages.rend(); it++) {
            const auto &s = *it;
            const int *pipeline_feats = (const int *)(&(s.features)) + 7;
            // skip the first 7 features
            for (int i = 0; i < pipeline_feat_size; i++) {
                int x = i/7;
                int y = i%7;
                pipeline_features(x, y, stage) = pipeline_feats[i];
            }
            stage += 1;
        }
    }
    internal_assert(stage == num_stages);
    cost_model->set_pipeline_features(pipeline_features, params.parallelism);
}

IntrusivePtr<State> optimal_schedule_pass(FunctionDAG &dag,
                                          vector<Function> outputs,
                                          const MachineParams &params,
                                          CostModel *cost_model,
                                          int beam_size,
                                          int pass_idx,
                                          std::unordered_set<uint64_t> &permitted_hashes) {

    if (cost_model) {
        configure_pipeline_features(dag, params, cost_model);
    }

    StateQueue q, pending;

    {
        IntrusivePtr<State> initial{new State};
        initial->root = new LoopNest;
        q.emplace(std::move(initial));
    }

    // A progress bar.
    uint32_t counter = 0;
    bool draw_progress_bar = isatty(2);
    auto tick = [&](double progress) {
        if (!draw_progress_bar) return;
        counter++;
        const int bits = 11;
        if (counter & ((1 << bits) - 1)) return;
        progress *= 78;
        debug(0) << '[';
        for (int j = 0; j < 78; j++) {
            if (j < progress) {
                debug(0) << '.';
            } else if (j - 1 < progress) {
                debug(0) << "/-\\|"[(counter >> bits) % 4];
            } else {
                debug(0) << ' ';
            }
        }
        debug(0) << ']';
        for (int j = 0; j < 80; j++) {
            debug(0) << '\b';
        }
    };

    int expanded;

    std::function<void(IntrusivePtr<State> &&)> enqueue_new_children =
        [&](IntrusivePtr<State> &&s) {

        // debug(0) << "\n** Generated child: ";
        // s->dump();
        // s->calculate_cost(dag, params, nullptr, true);

        internal_assert(s->num_funcs_scheduled == s->parent->num_funcs_scheduled + 1);

        int progress = s->num_funcs_scheduled * beam_size + expanded;
        size_t max_progress = dag.nodes.size() * beam_size;
        tick(double(progress) / max_progress);
        s->penalized = false;

        q.emplace(std::move(s));
    };

    for (int i = 0; ; i++) {
        std::unordered_map<uint64_t, int> hashes;
        q.swap(pending);

        internal_assert(!pending.empty());

        if ((int)pending.size() > beam_size * 10000) {
            debug(0) << "Warning: Huge number of states generated (" << pending.size() << ").\n";
        }

        expanded = 0;
        while (expanded < beam_size && !pending.empty()) {

            IntrusivePtr<State> state {pending.pop()};

            if (beam_size > 1) {
                // Apply cost penalties to the queue according to
                // structural uniqueness.
                if (!state->penalized) {
                    uint64_t h1 = state->structural_hash(pass_idx + 1, params.parallelism);
                    uint64_t h0 = state->structural_hash(pass_idx - 1, params.parallelism);
                    int penalty = ++hashes[h1];
                    if (pass_idx > 0 && !permitted_hashes.count(h0)) {
                        // It's possible to get yourself into a state
                        // where the only things in the beam that match
                        // the hash were quick-rejected due to details not
                        // captured in the hash, so we apply a huge
                        // penalty, but leave the impermissible state in
                        // the beam.
                        // debug(0) << "\nImpermissible hash " << pass_idx << " at " << state->num_funcs_scheduled << " " << h0 << ":\n";
                        // state->dump();
                        penalty += 10;
                    }
                    if (penalty > 1) {
                        state->penalized = true;
                        state->cost *= penalty;
                        // After penalizing this state, it's no longer the
                        // best, defer it.
                        if (!pending.empty() && state->cost > pending.top()->cost) {
                            pending.emplace(std::move(state));
                            continue;
                        }
                    }
                }
            }

            if (pending.size() > 1 && random_dropout()) {
                // debug(0) << "Dropping state\n";
                continue;
            }

            if (state->num_funcs_scheduled == 2*(int)dag.nodes.size()) {
                debug(0) << '\n';

                if (false) {
                    debug(0) << "Optimal state?\n";
                    state->dump();

                    debug(0) << "\nRest of queue:\n";
                    while (!pending.empty()) {
                        pending.pop()->dump();
                    }
                }

                auto best = state;

                // Bless the reasonable stuff in the beam as permissible states to visit again
                int blessed = 0;
                while (state->cost <= 1.2 * best->cost && blessed < beam_size) {
                    const State *s = state.get();
                    while (s) {
                        uint64_t h1 = s->structural_hash(pass_idx, params.parallelism);
                        permitted_hashes.insert(h1);
                        s = s->parent.get();
                    }
                    if (pending.empty()) break;
                    state = pending.pop();
                    blessed++;
                }

                return best;
            }

            /*
            if (state->num_funcs_scheduled > 0 &&
                dag.nodes[state->num_funcs_scheduled].func.name() == "downsampled_x") {
            */
            if (false) {
                debug(0) << "\n\n**** Beam: (" << expanded << "):\n";
                state->dump();
            }

            /*
              debug(0) << "Expanding state:";
              state->dump();
              state->calculate_cost(dag, params, nullptr, true);
            */

            state->generate_children(dag, params, cost_model, enqueue_new_children);
            expanded++;
        }

        // Drop the other states unconsidered.
        pending.clear();

        if (cost_model) {
            // Now evaluate all the costs and re-sort them in the priority queue
            cost_model->evaluate_costs();
            q.resort();
        }

        string cyos_str = get_env_variable("HL_CYOS");
        if (cyos_str == "1") {
            // Manually discard everything in the queue except for the user-chosen option
            // Print user choices.
            debug(0) << "\n--------------------\n";
            debug(0) << "Select a schedule:\n";
            for (int choice_label = (int)q.size() - 1; choice_label >= 0; choice_label--) {
                auto state = q[choice_label];
                debug(0) << "\n[" << choice_label << "]:\n";
                state->dump();
                LoopNestRoot loop_nest;
                state->calculate_cost(dag, params, cost_model, loop_nest, true);
            }
            cost_model->evaluate_costs();

            // Select next partial schedule to expand.
            int selection = -1;
            while (selection < 0 || selection >= (int)q.size()) {
                debug(0) << "\nEnter selection: ";
                std::cin >> selection;
            }

            auto selected = q[selection];
            selected->dump();
            q.clear();
            q.emplace(std::move(selected));
        }
    }
}

IntrusivePtr<State> optimal_schedule(FunctionDAG &dag,
                                     vector<Function> outputs,
                                     const MachineParams &params,
                                     CostModel *cost_model,
                                     int beam_size) {

    IntrusivePtr<State> best;

    std::unordered_set<uint64_t> permitted_hashes;
    int num_passes = (beam_size == 1) ? 1 : 5;

    string cyos_str = get_env_variable("HL_CYOS");
    if (cyos_str == "1") {
        num_passes = 1;
    }

    for (int i = 0; i < num_passes; i++) {
        auto pass = optimal_schedule_pass(dag, outputs, params, cost_model, beam_size, i, permitted_hashes);
        debug(0) << "\nPass " << i << " result:\n";
        pass->dump();

        if (i == 0 || pass->cost < best->cost) {
            best = pass;
        }
    }

    debug(0) << "Best cost: " << best->cost << "\n";

    return best;
}

}

std::string generate_schedules_new(const std::vector<Function> &outputs,
                                   const Target &target,
                                   const MachineParams &params) {

    State::cost_calculations = 0;
    string seed_str = get_env_variable("HL_SEED");
    int seed = (int)time(NULL);
    if (!seed_str.empty()) {
        seed = atoi(seed_str.c_str());
    }
    debug(0) << "Dropout seed = " << seed << '\n';
    srand(seed);

    string beam_size_str = get_env_variable("HL_BEAM_SIZE");
    size_t beam_size = 20;
    if (!beam_size_str.empty()) {
        beam_size = atoi(beam_size_str.c_str());
    }

    string time_limit_str = get_env_variable("HL_AUTO_SCHEDULE_TIME_LIMIT");
    double time_limit = 0;
    if (!time_limit_str.empty()) {
        time_limit = atof(time_limit_str.c_str());
    }

    string weights_dir = get_env_variable("HL_WEIGHTS_DIR");

    string randomize_weights_str = get_env_variable("HL_RANDOMIZE_WEIGHTS");
    bool randomize_weights = randomize_weights_str == "1";

    string weights_server_hostname = get_env_variable("HL_WEIGHTS_SERVER_HOSTNAME");

    string weights_server_port_str = get_env_variable("HL_WEIGHTS_SERVER_PORT");
    int weights_server_port = 0;
    if (!weights_server_port_str.empty()) {
        weights_server_port = atoi(weights_server_port_str.c_str());
    }

    string weights_server_experiment_id_str = get_env_variable("HL_WEIGHTS_SERVER_EXPERIMENT_ID");
    int weights_server_experiment_id = 0;
    if (!weights_server_experiment_id_str.empty()) {
        weights_server_experiment_id = atoi(weights_server_experiment_id_str.c_str());
    }

    FunctionDAG dag(outputs, params, target);

    dag.dump();

    std::unique_ptr<CostModel> cost_model;
    if (get_env_variable("HL_USE_MANUAL_COST_MODEL") != "1") {
        cost_model = CostModel::make_default(weights_dir, randomize_weights, weights_server_hostname, weights_server_port, weights_server_experiment_id);
    }
    
    IntrusivePtr<State> optimal;

    json jdata;
    jdata["schedule_seed"] = seed;
    jdata["beam_size"] = beam_size;
    jdata["autoschedule_timelimit"] = time_limit;

    if (time_limit) {
        // Use a fixed running time
        auto start = std::chrono::steady_clock::now();
        for (size_t beam_size = 1; ; beam_size *= 2) {
            auto s = optimal_schedule(dag, outputs, params, cost_model.get(), beam_size);
            if (beam_size == 1 || s->cost < optimal->cost) {
                optimal = s;
            }
            auto t = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t - start).count();
            if (elapsed > time_limit / 2) {
                break;
            }
        }
    } else {
        // Use a fixed beam size
        optimal = optimal_schedule(dag, outputs, params, cost_model.get(), beam_size);
    }

    debug(0) << "Cost evaluated this many times: " << State::cost_calculations << '\n';

    debug(0) << "** Optimal schedule:\n";

    string json_path = get_env_variable("HL_JSON_DUMP");

    // Just to get the debugging prints to fire
    LoopNestRoot loop_nest;
    if (json_path.empty()) {  // do not store json dump
        optimal->calculate_cost(dag, params, cost_model.get(), loop_nest, true);
    } else {
        optimal->calculate_cost(dag, params, cost_model.get(), loop_nest, true, &jdata);
    }

    // Apply the schedules
    optimal->apply_schedule(dag, params);
    loop_nest.dump();

    string loop_nest_test_mode = get_env_variable("HL_LOOP_NEST_TEST_MODE");
    if (atoi(loop_nest_test_mode.c_str())) {
        internal_assert(loop_nest.matches_pipeline_loop_nest(outputs));
    }

    // Print out the schedule
    optimal->dump();

    string schedule_file = get_env_variable("HL_SCHEDULE_FILE");
    if (!schedule_file.empty()) {
        debug(0) << "Writing schedule to " << schedule_file << "...\n";
        std::ofstream f(schedule_file);
        f << "// --- BEGIN machine-generated schedule\n"
          << optimal->schedule_source
          << "// --- END machine-generated schedule\n";
        f.close();
        internal_assert(!f.fail()) << "Failed to write " << schedule_file;
    }

    if (!json_path.empty()) {
        jdata["optimal_schedule"] = optimal->json_dump();
        jdata["optimal_schedule"]["cost_evaluations"] = State::cost_calculations;
        std::ofstream json_file(json_path, std::ios::binary);
        std::vector<std::uint8_t> msgpack_data = json::to_msgpack(jdata);
        json_file.write(reinterpret_cast<char*>(msgpack_data.data()), msgpack_data.size() * sizeof(std::uint8_t));
    }

    // Print out the predicted runtime of each Func, so we can compare them to a profile
    // optimal->print_predicted_runtimes(params);

    string feature_file = get_env_variable("HL_FEATURE_FILE");
    if (!feature_file.empty()) {
        optimal->save_featurization(dag, params, feature_file);
    }

    return "";
}

// Register this as the autoscheduler
struct AutoScheduler {
    AutoScheduler() {
        debug(0) << "Registering autoscheduler...\n";
        Pipeline::set_custom_auto_scheduler(*this);
    }

    string operator()(Pipeline p, const Target &target, const MachineParams &params) {
        std::vector<Function> outputs;
        for (Func f : p.outputs()) {
            outputs.push_back(f.function());
        }
        return generate_schedules_new(outputs, target, params);
    }
} auto_scheduler;

}
}
