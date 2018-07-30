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
    assert(input.is_bounds_query());
    random_pipeline(input, output);
    input.allocate();
    input.fill(0.0f);

    printf("Input size: %d %d %d\n", 
           input.width(), input.height(), input.channels());

    double best = benchmark([&]() {
        random_pipeline(input, output);
    });

    best *= 1e3;  // in ms

    // in milliseconds
    printf("Time: %g ms\n", best);

    char* json_path = argv[1];
    printf("saving timing to %s\n", json_path);

    json jdata;
    jdata["time"] = best;

    std::ofstream json_file(json_path);
    json_file << std::setw(2) << jdata << std::endl;
    json_file.close();

    return 0;
}
