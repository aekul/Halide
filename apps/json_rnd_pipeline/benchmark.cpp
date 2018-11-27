#include <cmath>
#include <cstdint>
#include <cstdio>
#include <atomic>
#include <fstream>

#include "HalideBuffer.h"
#include "halide_benchmark.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

#include "random_pipeline.h"

#include <json.hpp>

using json = nlohmann::json;

int main(int argc, char *argv[])
{
    if (argc != 2) {
        printf("please provide path of .json file to write the benchmark results.\n");
        return 1;
    }

    // TODO(mgharbi): specifcy output size as cli args
    Buffer<float> output(2000, 2000, 3);

    for (int y = 0; y < output.height(); y++) {
        for (int x = 0; x < output.width(); x++) {
            for (int c = 0; c < output.channels(); c++) {
                output(x, y, c) = rand() & 0xfff;
            }
        }
    }

    // Get the input size
    Buffer<float> input;
    Buffer<uint8_t> uint8_weights;
    Buffer<uint16_t> uint16_weights;
    Buffer<uint32_t> uint32_weights;
    Buffer<int8_t> int8_weights;
    Buffer<int16_t> int16_weights;
    Buffer<int32_t> int32_weights;
    Buffer<float> float32_weights;

    assert(input.is_bounds_query());
    assert(input.is_bounds_query());
    assert(uint8_weights.is_bounds_query());
    assert(uint16_weights.is_bounds_query());
    assert(uint32_weights.is_bounds_query());
    assert(int8_weights.is_bounds_query());
    assert(int16_weights.is_bounds_query());
    assert(int32_weights.is_bounds_query());
    assert(float32_weights.is_bounds_query());

    random_pipeline(input,
                    uint8_weights,
                    uint16_weights,
                    uint32_weights,
                    int8_weights,
                    int16_weights,
                    int32_weights,
                    float32_weights,
                    output);
    input.allocate();
    input.fill(0.0f);

    uint8_weights.allocate();
    uint8_weights.fill(0.0f);

    uint16_weights.allocate();
    uint16_weights.fill(0.0f);

    uint32_weights.allocate();
    uint32_weights.fill(0.0f);

    int8_weights.allocate();
    int8_weights.fill(0.0f);

    int16_weights.allocate();
    int16_weights.fill(0.0f);

    int32_weights.allocate();
    int32_weights.fill(0.0f);

    float32_weights.allocate();
    float32_weights.fill(0.0f);

    printf("Input size: %d %d %d\n", input.width(), input.height(), input.channels());

    double best = benchmark([&]() {
        random_pipeline(input,
                        uint8_weights,
                        uint16_weights,
                        uint32_weights,
                        int8_weights,
                        int16_weights,
                        int32_weights,
                        float32_weights,
                        output);
    });

    best *= 1e3;  // in ms

    // in milliseconds
    printf("Time: %g ms\n", best);

    char* json_path = argv[1];
    printf("saving timing to %s\n", json_path);

    json jdata;
    jdata["time"] = best;

    std::ofstream json_file(json_path, std::ios::binary);
    std::vector<std::uint8_t> msgpack_data = json::to_msgpack(jdata);
    json_file.write(reinterpret_cast<char*>(msgpack_data.data()), msgpack_data.size() * sizeof(std::uint8_t));

    return 0;
}
