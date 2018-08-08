#include "ThroughputPredictor.h"

#include <iostream>

namespace Halide {
namespace Internal {
namespace AutoScheduleModel {

ThroughputPredictor::ThroughputPredictor(std::string url) :
  context(1), socket(context, ZMQ_REQ), query_id_(0)
{
  socket.connect(url);
}

void ThroughputPredictor::enqueue(json features, double* cost) {
  cost_map_[query_id_] = cost;
  *cost = 2.0;
  // std::cout << "PREDICTOR: enqueue \n";
  // Launch prediction requests
  std::string data = features.dump();
  zmq::message_t request(data.size());
  memcpy(request.data(), data.c_str(), data.size());

  std::cout << "PREDICTOR: sending " << data;
  socket.send(request);

  zmq::message_t reply;
  socket.recv(&reply);
  std::string smessage(static_cast<char*>(reply.data()), reply.size());
  
  json response = json::parse(smessage.data());
  std::cout << " received " << response << "\n";
  *cost = response["cost"];

  // TODO: launch another thread to receive and update the queries cost
  // query_id_;
}

bool ThroughputPredictor::join() {
  if (true) {  // TODO: join all requests, make sure we have the costs
    return true;
  }

  return false;
}



} // namespace AutoScheduleModel
} // namespace Internal
} // namespace Halide
