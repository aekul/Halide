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
    int64_t unique_bytes_read_per_realization = 0; // Number of unique bytes loaded from all inputs per production
    int64_t unique_lines_read_per_realization = 0; // Number of unique contiguous segments of memory loaded from all inputs per production
    int64_t allocation_bytes_read_per_realization = 0; // The sum of the sizes of the allocations accessed per production. Gives a hint as to the likely locality of it.

    int64_t working_set = 0; // The sum of the sizes of the allocations within the production of this Func. Probably a good thing if it fits in cache.

    int64_t vector_size = 0; // The vectorization factor (#simd lanes) to be used to compute this stage. Wasted work if innermost_pure_loop is not a multiple of this, or if it's smaller than the stage's native vector size (which is in the pipeline features).

    int64_t rounded_innermost_pure_loop_extent = 0; // Innermost pure loop extend rounded up to the next multiple of the vector size

    int native_vector_size; // The native vector size for the narrowest type used.

    int64_t non_unique_bytes_read_per_realization = 0; // Number of bytes read per realization, counting reloads of the same memory.

    int64_t total_element_compute_cost = 0;
    int64_t compute_cost_inlined = 0;
    double vector_overcompute_factor = 0;
    double idle_core_wastage = 0;
    double load_cold_cache_misses = 0;
    double cost_of_cold_miss = 0;
    double capacity_cache_misses = 0;
    double cost_of_capacity_miss = 0;
    double memory_load_cost = 0;
    double store_cache_misses = 0;
    double store_cost_of_miss = 0;
    double memory_store_cost = 0;
    double cache_line_wastage = 0;
    double cost_of_mallocs = 0;
    double cost_of_working_set = 0;
    double compute_cost = 0;
    double total_cost = 0;

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
                 << "    unique_bytes_read_per_realization:     " << unique_bytes_read_per_realization << '\n'
                 << "    unique_lines_read_per_realization:     " << unique_lines_read_per_realization << '\n'
                 << "    allocation_bytes_read_per_realization: " << allocation_bytes_read_per_realization << '\n'
                 << "    working_set:                           " << working_set << '\n'
                 << "    vector_size:                           " << vector_size << '\n'
                 << "    rounded_innermost_pure_loop_extent     " << rounded_innermost_pure_loop_extent << '\n'
                 << "    native_vector_size:                    " << vector_size << '\n'
                 << "    non_unique_bytes_read_per_realization: " << non_unique_bytes_read_per_realization << '\n'
                 << "    total_element_compute_cost:            " << total_element_compute_cost << '\n'
                 << "    compute_cost_inlined:                  " << compute_cost_inlined << '\n'
                 << "    vector_overcompute_factor:             " << vector_overcompute_factor << '\n'
                 << "    idle_core_wastage:                     " << idle_core_wastage << '\n'
                 << "    load_cold_cache_misses                 " << load_cold_cache_misses << '\n'
                 << "    cost_of_cold_miss                      " << cost_of_cold_miss << '\n'
                 << "    capacity_cache_misses                  " << capacity_cache_misses << '\n'
                 << "    cost_of_capacity_miss                  " << cost_of_capacity_miss << '\n'
                 << "    memory_load_cost                       " << memory_load_cost << '\n'
                 << "    store_cache_misses:                    " << store_cache_misses << '\n'
                 << "    store_cost_of_miss:                    " << store_cost_of_miss << '\n'
                 << "    memory_store_cost:                     " << memory_store_cost << '\n'
                 << "    cache_line_wastage:                    " << cache_line_wastage << '\n'
                 << "    cost_of_mallocs:                       " << cost_of_mallocs << '\n'
                 << "    cost_of_working_set:                   " << cost_of_working_set << '\n'
                 << "    compute_cost:                          " << compute_cost << '\n'
                 << "    total_cost:                            " << total_cost << '\n';
    }

    std::vector<int64_t> to_vector() const {
        return {
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
            unique_bytes_read_per_realization,
            unique_lines_read_per_realization,
            allocation_bytes_read_per_realization,
            working_set,
            vector_size,
            rounded_innermost_pure_loop_extent,
            native_vector_size,
            non_unique_bytes_read_per_realization
        };
    }

    // json json_dump(bool as_vector=true) const {
    //     json jdata;
    //     if (as_vector) {
    //         jdata["features"] = {
    //             num_realizations,
    //             num_productions,
    //             points_computed_per_realization,
    //             points_computed_per_production,
    //             points_computed_total,
    //             points_computed_minimum,
    //             innermost_loop_extent,
    //             innermost_pure_loop_extent,
    //             inner_parallelism,
    //             outer_parallelism,
    //             bytes_at_realization,
    //             bytes_at_production,
    //             bytes_at_root,
    //             innermost_bytes_at_realization,
    //             innermost_bytes_at_production,
    //             innermost_bytes_at_root,
    //             bytes_read_per_tile,
    //             inlined_calls,
    //             unique_bytes_read_per_realization,
    //             unique_lines_read_per_realization,
    //             allocation_bytes_read_per_realization,
    //             working_set,
    //             vector_size,
    //             rounded_innermost_pure_loop_extent,
    //             native_vector_size,
    //             non_unique_bytes_read_per_realization,
    //
    //             total_element_compute_cost,
    //             compute_cost_inlined, 
    //             vector_overcompute_factor,
    //             idle_core_wastage,
    //             load_cold_cache_misses,
    //             cost_of_cold_miss,
    //             capacity_cache_misses,
    //             cost_of_capacity_miss,
    //             memory_load_cost,
    //             store_cache_misses,
    //             store_cost_of_miss,
    //             memory_store_cost,
    //             cache_line_wastage,
    //             cost_of_mallocs,
    //             cost_of_working_set,
    //             compute_cost,
    //             total_cost
    //         };
    //     } else {
    //         jdata["num_realizations"]                      = num_realizations;
    //         jdata["num_productions"]                       = num_productions;
    //         jdata["points_computed_per_realization"]       = points_computed_per_realization;
    //         jdata["points_computed_per_production"]        = points_computed_per_production;
    //         jdata["points_computed_total"]                 = points_computed_total;
    //         jdata["points_computed_minimum"]               = points_computed_minimum;
    //         jdata["innermost_loop_extent"]                 = innermost_loop_extent;
    //         jdata["innermost_pure_loop_extent"]            = innermost_pure_loop_extent;
    //         jdata["inner_parallelism"]                     = inner_parallelism;
    //         jdata["outer_parallelism"]                     = outer_parallelism;
    //         jdata["bytes_at_realization"]                  = bytes_at_realization;
    //         jdata["bytes_at_production"]                   = bytes_at_production;
    //         jdata["bytes_at_root"]                         = bytes_at_root;
    //         jdata["innermost_bytes_at_realization"]        = innermost_bytes_at_realization;
    //         jdata["innermost_bytes_at_production"]         = innermost_bytes_at_production;
    //         jdata["innermost_bytes_at_root"]               = innermost_bytes_at_root;
    //         jdata["bytes_read_per_tile"]                   = bytes_read_per_tile;
    //         jdata["inlined_calls"]                         = inlined_calls;
    //         jdata["unique_bytes_read_per_realization"]            = unique_bytes_read_per_realization;
    //         jdata["unique_lines_read_per_realization"]            = unique_lines_read_per_realization;
    //         jdata["allocation_bytes_read_per_realization"] = allocation_bytes_read_per_realization;
    //         jdata["working_set"]                           = working_set;
    //         jdata["vector_size"]                           = vector_size;
    //         jdata["rounded_innermost_pure_loop_extent"]    = rounded_innermost_pure_loop_extent;
    //         jdata["native_vector_size"]                    = native_vector_size;
    //         jdata["non_unique_bytes_read_per_realization"] = non_unique_bytes_read_per_realization;
    //         jdata["total_element_compute_cost"] = total_element_compute_cost;
    //         jdata["compute_cost_inlined"] = compute_cost_inlined;
    //         jdata["vector_overcompute_factor"] = vector_overcompute_factor;
    //         jdata["load_cold_cache_misses"] = load_cold_cache_misses;
    //         jdata["cost_of_cold_miss"] = cost_of_cold_miss;
    //         jdata["capacity_cache_misses"] = capacity_cache_misses;
    //         jdata["cost_of_capacity_miss"] = cost_of_capacity_miss;
    //         jdata["memory_load_cost"] = memory_load_cost;
    //         jdata["store_cache_misses"] = store_cache_misses;
    //         jdata["store_cost_of_miss"] = store_cost_of_miss;
    //         jdata["memory_store_cost"] = memory_store_cost;
    //         jdata["cache_line_wastage"] = cache_line_wastage;
    //         jdata["cost_of_mallocs"] = cost_of_mallocs;
    //         jdata["cost_of_working_set"] = cost_of_working_set;
    //         jdata["compute_cost"] = compute_cost;
    //         jdata["total_cost"] = total_cost;
    //     }
    //     return jdata;
    // }
}; // ScheduleFeatures

} // namespace Internal
} // namespace Halide

#endif // HALIDE_INTERNAL_SCHEDULE_FEATURES_H
