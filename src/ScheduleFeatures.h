#ifndef HALIDE_INTERNAL_SCHEDULE_FEATURES_H
#define HALIDE_INTERNAL_SCHEDULE_FEATURES_H

#include <string>

#include <json.hpp>

using json = nlohmann::json;

namespace Halide {
namespace Internal {
  
// The schedule-dependent portion of the featurization of a stage
struct ScheduleFeatures {
    int64_t num_realizations = 0; // Product of outer loops at store_at site
    int64_t num_productions = 0;  // Product of outer loops at compute_at site
    int64_t points_computed_per_realization = 0; // Number of times the innermost stmt happens per store_at
    int64_t points_computed_per_production = 0;  // Number of times the innermost stmt happens per compute_at
    int64_t points_computed_total = 0;
    // points_computed_total
    //  == num_realizations * points_computed_per_realization
    //  ~= num_productions * points_computed_per_production
    // Only approximately equal because of the simplifications made
    // regarding the modelling of sliding window

    int64_t points_computed_minimum = 0; // The minimum number of points that are actually required to be computed to produce a correct output.

    int64_t innermost_loop_extent = 0; // Trip count of innermost loop
    int64_t innermost_pure_loop_extent = 0; // Trip count of the loop that's going to be vectorized
    int64_t inner_parallelism = 0; // The number of parallel jobs used in the production of this Func. 1 unless the Func is compute_root.
    int64_t outer_parallelism = 0; // The number of times this Func could be realized in parallel. 1 when the Func is compute_root.

    int64_t bytes_at_realization = 0; // Size of the region computed at the store_at site, measured in bytes
    int64_t bytes_at_production = 0; // Size of the region computed at the compute_at site, measured in bytes
    int64_t bytes_at_root = 0; // The same at root, regardless of where it's actually scheduled
    int64_t innermost_bytes_at_realization = 0;
    int64_t innermost_bytes_at_production = 0;
    int64_t innermost_bytes_at_root = 0;

    int64_t bytes_read_per_tile = 0; // Number of bytes loaded from alnputs per tile (TODO: Not particularly useful without knowing how many times this runs)

    int64_t inlined_calls = 0; // For inlined Funcs, how many calls are made to this Func total

    // Logically these features should be grouped earlier, but the convnet currently doesn't know about them
    int64_t bytes_read_per_realization = 0; // Number of bytes loaded from all inputs per production
    int64_t lines_read_per_realization = 0; // Number of contiguous segments of memory loaded from all inputs per production
    int64_t allocation_bytes_read_per_realization = 0; // The sum of the sizes of the allocations accessed per production. Gives a hint as to the likely locality of it.

    int64_t working_set = 0; // The sum of the sizes of the allocations within the production of this Func. Probably a good thing if it fits in cache.

    int64_t vector_size = 0; // The vectorization factor (#simd lanes) to be used to compute this stage. Wasted work if innermost_pure_loop is not a multiple of this, or if it's smaller than the stage's native vector size (which is in the pipeline features).

    int64_t rounded_innermost_pure_loop_extent = 0; // Innermost pure loop extend rounded up to the next multiple of the vector size

    int native_vector_size; // The native vector size for the narrowest type used.

    void dump() const {
        debug(0) << "    num_realizations:                      " << num_realizations << '\n'
                 << "    num_productions:                       " << num_productions << '\n'
                 << "    points_computed_per_realization:       " << points_computed_per_realization << '\n'
                 << "    points_computed_per_production:        " << points_computed_per_production << '\n'
                 << "    points_computed_total:                 " << points_computed_total << '\n'
                 << "    points_computed_minimum:               " << points_computed_minimum << '\n'
                 << "    innermost_loop_extent:                 " << innermost_loop_extent << '\n'
                 << "    innermost_pure_loop_extent:            " << innermost_pure_loop_extent << '\n'
                 << "    inner_parallelism:                     " << inner_parallelism << '\n'
                 << "    outer_parallelism:                     " << outer_parallelism << '\n'
                 << "    bytes_at_realization:                  " << bytes_at_realization << '\n'
                 << "    bytes_at_production:                   " << bytes_at_production << '\n'
                 << "    bytes_at_root:                         " << bytes_at_root << '\n'
                 << "    innermost_bytes_at_realization:        " << innermost_bytes_at_realization << '\n'
                 << "    innermost_bytes_at_production:         " << innermost_bytes_at_production << '\n'
                 << "    innermost_bytes_at_root:               " << innermost_bytes_at_root << '\n'
                 << "    bytes_read_per_tile:                   " << bytes_read_per_tile << '\n'
                 << "    inlined_calls:                         " << inlined_calls << '\n'
                 << "    bytes_read_per_realization:            " << bytes_read_per_realization << '\n'
                 << "    lines_read_per_realization:            " << lines_read_per_realization << '\n'
                 << "    allocation_bytes_read_per_realization: " << allocation_bytes_read_per_realization << '\n'
                 << "    working_set:                           " << working_set << '\n'
                 << "    vector_size:                           " << vector_size << '\n'
                 << "    rounded_innermost_pure_loop_extent     " << rounded_innermost_pure_loop_extent << '\n'
                 << "    native_vector_size:                    " << vector_size << '\n';
    }

    json json_dump(bool as_vector=true) const {
        json jdata;
        if (as_vector) {
            jdata["features"] = {
                num_realizations,
                num_productions,
                points_computed_per_realization,
                points_computed_per_production,
                points_computed_total,
                points_computed_minimum,
                innermost_loop_extent,
                innermost_pure_loop_extent,
                inner_parallelism,
                outer_parallelism,
                bytes_at_realization,
                bytes_at_production,
                bytes_at_root,
                innermost_bytes_at_realization,
                innermost_bytes_at_production,
                innermost_bytes_at_root,
                bytes_read_per_tile,
                inlined_calls,
                bytes_read_per_realization,
                lines_read_per_realization,
                allocation_bytes_read_per_realization,
                working_set,
                vector_size,
                rounded_innermost_pure_loop_extent,
                native_vector_size
            };
        } else {
            jdata["num_realizations"]                      = num_realizations;
            jdata["num_productions"]                       = num_productions;
            jdata["points_computed_per_realization"]       = points_computed_per_realization;
            jdata["points_computed_per_production"]        = points_computed_per_production;
            jdata["points_computed_total"]                 = points_computed_total;
            jdata["points_computed_minimum"]               = points_computed_minimum;
            jdata["innermost_loop_extent"]                 = innermost_loop_extent;
            jdata["innermost_pure_loop_extent"]            = innermost_pure_loop_extent;
            jdata["inner_parallelism"]                     = inner_parallelism;
            jdata["outer_parallelism"]                     = outer_parallelism;
            jdata["bytes_at_realization"]                  = bytes_at_realization;
            jdata["bytes_at_production"]                   = bytes_at_production;
            jdata["bytes_at_root"]                         = bytes_at_root;
            jdata["innermost_bytes_at_realization"]        = innermost_bytes_at_realization;
            jdata["innermost_bytes_at_production"]         = innermost_bytes_at_production;
            jdata["innermost_bytes_at_root"]               = innermost_bytes_at_root;
            jdata["bytes_read_per_tile"]                   = bytes_read_per_tile;
            jdata["inlined_calls"]                         = inlined_calls;
            jdata["bytes_read_per_realization"]            = bytes_read_per_realization;
            jdata["lines_read_per_realization"]            = lines_read_per_realization;
            jdata["allocation_bytes_read_per_realization"] = allocation_bytes_read_per_realization;
            jdata["working_set"]                           = working_set;
            jdata["vector_size"]                           = vector_size;
            jdata["rounded_innermost_pure_loop_extent"]    = rounded_innermost_pure_loop_extent;
            jdata["native_vector_size"]                    = native_vector_size;
        }
        return jdata;
    }
}; // ScheduleFeatures

} // namespace Internal
} // namespace Halide

#endif // HALIDE_INTERNAL_SCHEDULE_FEATURES_H
