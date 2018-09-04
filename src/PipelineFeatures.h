#ifndef HALIDE_INTERNAL_PIPELINE_FEATURES_H
#define HALIDE_INTERNAL_PIPELINE_FEATURES_H

#include <string>

#include <json.hpp>

using json = nlohmann::json;

namespace Halide {
namespace Internal {
  
struct PipelineFeatures {
    // A featurization of the compute done by a Func, to
    // feed the neural network.

    enum class OpType {
        Const,
        Cast,
        Variable,
        Param,
        Add, Sub, Mod, Mul, Div, Min, Max,
        EQ, NE, LT, LE,
        And, Or, Not,
        Select,
        ImageCall,
        FuncCall,
        SelfCall,   // Recursive calls from a Func to itself
        ExternCall, // Math intrinsics, typically
        Let,        // Depends on what CSE has decided to do, but a good indication of register pressure
        NumOpTypes,
    };

    enum class ScalarType {
        Bool,
        UInt8,  // includes Int8
        UInt16, // includes Int16
        UInt32, // includes Int32 (TODO: is this a good idea? index math is a different sort of beast)
        UInt64, // Includes Int64
        Float,
        Double,
        NumScalarTypes
    };

    // Not a super-useful feature, but helps avoid printing huge numbers of zeros while debugging things
    int types_in_use[(int)ScalarType::NumScalarTypes];

    int op_histogram[(int)OpType::NumOpTypes][(int)ScalarType::NumScalarTypes];

    enum class AccessType {
        LoadFunc,
        LoadSelf,
        LoadImage,
        Store,
        NumAccessTypes
    };

    // Finer granularity call/store node properties. These are a
    // function of the matrix of derivatives of each arg to a
    // call w.r.t the loop variables of the Stage. Each row of
    // the matrix corresponds to one of the call arguments. In
    // each case we illustrate such a call, assuming that the
    // variables of this Func are x, y, z, and that the
    // dimension vectorized over is the first (x).
    int pointwise_accesses[(int)AccessType::NumAccessTypes][(int)ScalarType::NumScalarTypes],  // Square identity matrix. f(x - 2, y + 8, z + param)
        transpose_accesses[(int)AccessType::NumAccessTypes][(int)ScalarType::NumScalarTypes],         // Square permutation matrix. f(y + 1, z - 3, x)
        broadcast_accesses[(int)AccessType::NumAccessTypes][(int)ScalarType::NumScalarTypes],         // Each row sums to 1. Each column sums to 1 or 0. f(y, x)
        slice_accesses[(int)AccessType::NumAccessTypes][(int)ScalarType::NumScalarTypes],             // Each row sums to 1 or 0. Each column sums to 1. f(z, y, x, 4)
        vectorizable_accesses[(int)AccessType::NumAccessTypes][(int)ScalarType::NumScalarTypes], // First (vectorized) col is 1, 0, 0, ... f(x+y, z*y, y/z)
        strided_accesses[(int)AccessType::NumAccessTypes][(int)ScalarType::NumScalarTypes],      // First col is [(int)2,3,4], 0, 0, ...        f(3*x + 1, z/8, y/z)
        scalar_accesses[(int)AccessType::NumAccessTypes][(int)ScalarType::NumScalarTypes],       // First col is all zero                  f(y, 2, z*8)
        gather_scatter_accesses[(int)AccessType::NumAccessTypes][(int)ScalarType::NumScalarTypes];            // Not one of the three categories above  f(x, x, sqrt(y))

    // TODO: We should possibly feed these Jacobians directly
    // to the net rather than computing the properties above.

    // TODO: strided captures downsamples. What about upsamples?

    // TODO: It's weird that we've already selected a
    // dimension to be vectorized over - that should be part
    // of the scheduling search space instead.

    void dump() const {
        for (int i = 0; i < (int)ScalarType::NumScalarTypes; i++) {
            const char *type_names[] = {"Bool", "UInt8", "UInt16", "UInt32", "UInt64", "Float", "Double"};
            // Skip printing for types not used
            if (!types_in_use[i]) continue;


            debug(0) << "    Featurization for type " << type_names[i] << '\n'
                     << "     Op histogram:\n"
                     << "      Constant:   " << op_histogram[(int)OpType::Const][i] << '\n'
                     << "      Cast:       " << op_histogram[(int)OpType::Cast][i] << '\n'
                     << "      Variable:   " << op_histogram[(int)OpType::Variable][i] << '\n'
                     << "      Param:      " << op_histogram[(int)OpType::Param][i] << '\n'
                     << "      Add:        " << op_histogram[(int)OpType::Add][i] << '\n'
                     << "      Sub:        " << op_histogram[(int)OpType::Sub][i] << '\n'
                     << "      Mod:        " << op_histogram[(int)OpType::Mod][i] << '\n'
                     << "      Mul:        " << op_histogram[(int)OpType::Mul][i] << '\n'
                     << "      Div:        " << op_histogram[(int)OpType::Div][i] << '\n'
                     << "      Min:        " << op_histogram[(int)OpType::Min][i] << '\n'
                     << "      Max:        " << op_histogram[(int)OpType::Max][i] << '\n'
                     << "      EQ:         " << op_histogram[(int)OpType::EQ][i] << '\n'
                     << "      NE:         " << op_histogram[(int)OpType::NE][i] << '\n'
                     << "      LT:         " << op_histogram[(int)OpType::LT][i] << '\n'
                     << "      LE:         " << op_histogram[(int)OpType::LE][i] << '\n'
                     << "      And:        " << op_histogram[(int)OpType::And][i] << '\n'
                     << "      Or:         " << op_histogram[(int)OpType::Or][i] << '\n'
                     << "      Not:        " << op_histogram[(int)OpType::Not][i] << '\n'
                     << "      Select:     " << op_histogram[(int)OpType::Select][i] << '\n'
                     << "      ImageCall:  " << op_histogram[(int)OpType::ImageCall][i] << '\n'
                     << "      FuncCall:   " << op_histogram[(int)OpType::FuncCall][i] << '\n'
                     << "      SelfCall:   " << op_histogram[(int)OpType::SelfCall][i] << '\n'
                     << "      ExternCall: " << op_histogram[(int)OpType::ExternCall][i] << '\n'
                     << "      Let:        " << op_histogram[(int)OpType::Let][i] << '\n'
                     << "     Memory access patterns. Columns are calls to other Funcs, self-calls, input image access, and stores\n"
                     << "      Pointwise:      " << pointwise_accesses[0][i] << ' ' << pointwise_accesses[1][i] << ' ' << pointwise_accesses[2][i] << ' ' << pointwise_accesses[3][i] << '\n'
                     << "      Transpose:      " << transpose_accesses[0][i] << ' ' << transpose_accesses[1][i] << ' ' << transpose_accesses[2][i] << ' ' << transpose_accesses[3][i] << '\n'
                     << "      Broadcast:      " << broadcast_accesses[0][i] << ' ' << broadcast_accesses[1][i] << ' ' << broadcast_accesses[2][i] << ' ' << broadcast_accesses[3][i] << '\n'
                     << "      Slice:          " << slice_accesses[0][i] << ' ' << slice_accesses[1][i] << ' ' << slice_accesses[2][i] << ' ' << slice_accesses[3][i] << '\n'
                     << "      Vectorizable:   " << vectorizable_accesses[0][i] << ' ' << vectorizable_accesses[1][i] << ' ' << vectorizable_accesses[2][i] << ' ' << vectorizable_accesses[3][i] << '\n'
                     << "      Strided:        " << strided_accesses[0][i] << ' ' << strided_accesses[1][i] << ' ' << strided_accesses[2][i] << ' ' << strided_accesses[3][i] << '\n'
                     << "      Scalar:         " << scalar_accesses[0][i] << ' ' << scalar_accesses[1][i] << ' ' << scalar_accesses[2][i] << ' ' << scalar_accesses[3][i] << '\n'
                     << "      Gather/Scatter: " << gather_scatter_accesses[0][i] << ' ' << gather_scatter_accesses[1][i] << ' ' << gather_scatter_accesses[2][i] << ' ' << gather_scatter_accesses[3][i] << '\n';
        }
    }

    json json_dump() const {
        json jdata;
        const char *type_names[] = {"Bool", "UInt8", "UInt16", "UInt32", "UInt64", "Float", "Double"};
        for (int i = 0; i < (int)ScalarType::NumScalarTypes; i++) {

            json jtype;

            // Histogram of ops
            json jhistogram;
            jhistogram["constant"] = op_histogram[(int)OpType::Const][i];
            jhistogram["cast"] = op_histogram[(int)OpType::Cast][i];
            jhistogram["variable"] = op_histogram[(int)OpType::Variable][i];
            jhistogram["param"] = op_histogram[(int)OpType::Param][i];
            jhistogram["add"] = op_histogram[(int)OpType::Add][i];
            jhistogram["sub"] = op_histogram[(int)OpType::Sub][i];
            jhistogram["mod"] = op_histogram[(int)OpType::Mod][i];
            jhistogram["mul"] = op_histogram[(int)OpType::Mul][i];
            jhistogram["div"] = op_histogram[(int)OpType::Div][i];
            jhistogram["min"] = op_histogram[(int)OpType::Min][i];
            jhistogram["max"] = op_histogram[(int)OpType::Max][i];
            jhistogram["eq"] = op_histogram[(int)OpType::EQ][i];
            jhistogram["ne"] = op_histogram[(int)OpType::NE][i];
            jhistogram["lt"] = op_histogram[(int)OpType::LT][i];
            jhistogram["le"] = op_histogram[(int)OpType::LE][i];
            jhistogram["and"] = op_histogram[(int)OpType::And][i];
            jhistogram["or"] = op_histogram[(int)OpType::Or][i];
            jhistogram["not"] = op_histogram[(int)OpType::Not][i];
            jhistogram["select"] = op_histogram[(int)OpType::Select][i];
            jhistogram["image_call"] = op_histogram[(int)OpType::ImageCall][i];
            jhistogram["func_call"] = op_histogram[(int)OpType::FuncCall][i];
            jhistogram["self_call"] = op_histogram[(int)OpType::SelfCall][i];
            jhistogram["extern_call"] = op_histogram[(int)OpType::ExternCall][i];
            jhistogram["let"] = op_histogram[(int)OpType::Let][i];
            jtype["op_histogram"] = jhistogram;

            // Memory access patterns. Columns are 
            //  - calls to other Funcs,
            //  - self-calls
            //  - input image access
            //  - and stores
            json jmemory;
            jmemory["pointwise"] = { 
              pointwise_accesses[0][i],
              pointwise_accesses[1][i],
              pointwise_accesses[2][i],
              pointwise_accesses[3][i] };
            jmemory["transpose"] = { 
              transpose_accesses[0][i],
              transpose_accesses[1][i],
              transpose_accesses[2][i],
              transpose_accesses[3][i] };
            jmemory["broadcast"] = { 
              broadcast_accesses[0][i],
              broadcast_accesses[1][i],
              broadcast_accesses[2][i],
              broadcast_accesses[3][i] };
            jmemory["slice"] = { 
              slice_accesses[0][i],
              slice_accesses[1][i],
              slice_accesses[2][i],
              slice_accesses[3][i] };
            jmemory["vectorizable"] = { 
              vectorizable_accesses[0][i],
              vectorizable_accesses[1][i],
              vectorizable_accesses[2][i],
              vectorizable_accesses[3][i] };
            jmemory["strided"] = { 
              strided_accesses[0][i],
              strided_accesses[1][i],
              strided_accesses[2][i],
              strided_accesses[3][i] };
            jmemory["scalar"] = { 
              scalar_accesses[0][i],
              scalar_accesses[1][i],
              scalar_accesses[2][i],
              scalar_accesses[3][i] };
            jmemory["gather_scatter"] = { 
              gather_scatter_accesses[0][i],
              gather_scatter_accesses[1][i],
              gather_scatter_accesses[2][i],
              gather_scatter_accesses[3][i] };
            jtype["memory_access_patterns"] = jmemory;

            jdata[type_names[i]] = jtype;
        } // Loop over types

        return jdata;
    }
};

struct LoopNestPipelineFeatures {
    // A featurization of the compute done by a Func, to
    // feed the neural network.

    enum class OpType {
        Const,
        Cast,
        Variable,
        Param,
        Add, Sub, Mod, Mul, Div, Min, Max,
        EQ, NE, LT, LE,
        And, Or, Not,
        Select,
        ImageCall,
        FuncCall,
        SelfCall,   // Recursive calls from a Func to itself
        ExternCall, // Math intrinsics, typically
        Let,        // Depends on what CSE has decided to do, but a good indication of register pressure
        NumOpTypes,
    };

    enum class ScalarType {
        Bool,
        UInt8,  // includes Int8
        UInt16, // includes Int16
        UInt32, // includes Int32 (TODO: is this a good idea? index math is a different sort of beast)
        UInt64, // Includes Int64
        Float,
        Double,
        NumScalarTypes
    };

    // Not a super-useful feature, but helps avoid printing huge numbers of zeros while debugging things
    int types_in_use[(int)ScalarType::NumScalarTypes];

    int op_histogram[(int)OpType::NumOpTypes][(int)ScalarType::NumScalarTypes];

    enum class AccessType {
        LoadFunc,
        LoadSelf,
        LoadImage,
        Store,
        NumAccessTypes
    };

    void dump() const {
        for (int i = 0; i < (int)ScalarType::NumScalarTypes; i++) {
            const char *type_names[] = {"Bool", "UInt8", "UInt16", "UInt32", "UInt64", "Float", "Double"};
            // Skip printing for types not used
            if (!types_in_use[i]) continue;


            debug(0) << "    Featurization for type " << type_names[i] << '\n'
                     << "     Op histogram:\n"
                     << "      Constant:   " << op_histogram[(int)OpType::Const][i] << '\n'
                     << "      Cast:       " << op_histogram[(int)OpType::Cast][i] << '\n'
                     << "      Variable:   " << op_histogram[(int)OpType::Variable][i] << '\n'
                     << "      Param:      " << op_histogram[(int)OpType::Param][i] << '\n'
                     << "      Add:        " << op_histogram[(int)OpType::Add][i] << '\n'
                     << "      Sub:        " << op_histogram[(int)OpType::Sub][i] << '\n'
                     << "      Mod:        " << op_histogram[(int)OpType::Mod][i] << '\n'
                     << "      Mul:        " << op_histogram[(int)OpType::Mul][i] << '\n'
                     << "      Div:        " << op_histogram[(int)OpType::Div][i] << '\n'
                     << "      Min:        " << op_histogram[(int)OpType::Min][i] << '\n'
                     << "      Max:        " << op_histogram[(int)OpType::Max][i] << '\n'
                     << "      EQ:         " << op_histogram[(int)OpType::EQ][i] << '\n'
                     << "      NE:         " << op_histogram[(int)OpType::NE][i] << '\n'
                     << "      LT:         " << op_histogram[(int)OpType::LT][i] << '\n'
                     << "      LE:         " << op_histogram[(int)OpType::LE][i] << '\n'
                     << "      And:        " << op_histogram[(int)OpType::And][i] << '\n'
                     << "      Or:         " << op_histogram[(int)OpType::Or][i] << '\n'
                     << "      Not:        " << op_histogram[(int)OpType::Not][i] << '\n'
                     << "      Select:     " << op_histogram[(int)OpType::Select][i] << '\n'
                     << "      ImageCall:  " << op_histogram[(int)OpType::ImageCall][i] << '\n'
                     << "      FuncCall:   " << op_histogram[(int)OpType::FuncCall][i] << '\n'
                     << "      SelfCall:   " << op_histogram[(int)OpType::SelfCall][i] << '\n'
                     << "      ExternCall: " << op_histogram[(int)OpType::ExternCall][i] << '\n'
                     << "      Let:        " << op_histogram[(int)OpType::Let][i] << '\n';
        }
    }

    json json_dump() const {
        json jdata;
        const char *type_names[] = {"Bool", "UInt8", "UInt16", "UInt32", "UInt64", "Float", "Double"};
        for (int i = 0; i < (int)ScalarType::NumScalarTypes; i++) {

            json jtype;

            // Histogram of ops
            json jhistogram;
            jhistogram["constant"] = op_histogram[(int)OpType::Const][i];
            jhistogram["cast"] = op_histogram[(int)OpType::Cast][i];
            jhistogram["variable"] = op_histogram[(int)OpType::Variable][i];
            jhistogram["param"] = op_histogram[(int)OpType::Param][i];
            jhistogram["add"] = op_histogram[(int)OpType::Add][i];
            jhistogram["sub"] = op_histogram[(int)OpType::Sub][i];
            jhistogram["mod"] = op_histogram[(int)OpType::Mod][i];
            jhistogram["mul"] = op_histogram[(int)OpType::Mul][i];
            jhistogram["div"] = op_histogram[(int)OpType::Div][i];
            jhistogram["min"] = op_histogram[(int)OpType::Min][i];
            jhistogram["max"] = op_histogram[(int)OpType::Max][i];
            jhistogram["eq"] = op_histogram[(int)OpType::EQ][i];
            jhistogram["ne"] = op_histogram[(int)OpType::NE][i];
            jhistogram["lt"] = op_histogram[(int)OpType::LT][i];
            jhistogram["le"] = op_histogram[(int)OpType::LE][i];
            jhistogram["and"] = op_histogram[(int)OpType::And][i];
            jhistogram["or"] = op_histogram[(int)OpType::Or][i];
            jhistogram["not"] = op_histogram[(int)OpType::Not][i];
            jhistogram["select"] = op_histogram[(int)OpType::Select][i];
            jhistogram["image_call"] = op_histogram[(int)OpType::ImageCall][i];
            jhistogram["func_call"] = op_histogram[(int)OpType::FuncCall][i];
            jhistogram["self_call"] = op_histogram[(int)OpType::SelfCall][i];
            jhistogram["extern_call"] = op_histogram[(int)OpType::ExternCall][i];
            jhistogram["let"] = op_histogram[(int)OpType::Let][i];
            jtype["op_histogram"] = jhistogram;

            jdata[type_names[i]] = jtype;
        } // Loop over types

        return jdata;
    }
};

} // namespace Internal
} // namespace Halide

#endif // HALIDE_INTERNAL_PIPELINE_FEATURES_H
