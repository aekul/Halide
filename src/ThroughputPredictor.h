#ifndef HALIDE_INTERNAL_THROUGHPUT_PREDICTOR_H
#define HALIDE_INTERNAL_THROUGHPUT_PREDICTOR_H

#include <string>

#include <map>
#include <json.hpp>
#include <zmq.hpp>

using json = nlohmann::json;

namespace Halide {
namespace Internal {
namespace AutoScheduleModel {
  
class ThroughputPredictor
{
public:
  ThroughputPredictor(std::string url="tcp://localhost:5555");
  // virtual ~ThroughputPredictor();

  void enqueue(json features, double* cost);
  bool join();

  // TODO: asynchronous requests

private:
  zmq::context_t context;
  zmq::socket_t socket;

  int query_id_;
  std::map <int, double*> cost_map_;
};

} // namespace AutoScheduleModel
} // namespace Internal
} // namespace Halide

#endif // HALIDE_INTERNAL_THROUGHPUT_PREDICTOR_H
