#include "ThroughputPredictor.h"

#include <iostream>

namespace Halide {
namespace Internal {
namespace AutoScheduleModel {

ThroughputPredictor::ThroughputPredictor(std::string url) :
  context(1), socket(context, ZMQ_REQ), query_id_(0), pipeline_id_(0)
{
  socket.connect(url);
}


void ThroughputPredictor::send_dag(const json &jdata) {
  // TODO: enum for request types
  json request_payload;
  request_payload["request_type"] = 0;
  request_payload["pipe_id"] = ++pipeline_id_;
  request_payload["dag"] = jdata["dag"];
  std::string data = request_payload.dump();
  zmq::message_t request(data.size());
  memcpy(request.data(), data.c_str(), data.size());

  socket.send(request);

  zmq::message_t reply;
  socket.recv(&reply);
  std::string smessage(static_cast<char*>(reply.data()), reply.size());
  std::cout << " received " << smessage << "\n";
}

void ThroughputPredictor::enqueue(const json &features, double* cost) {
  // cost_map_[query_id_] = cost;
  // TODO: need to send graph info
  
  // Launch prediction requests
  json request_payload;
  request_payload["request_type"] = 1;
  request_payload["query_id"] = query_id_;
  request_payload["pipe_id"] = pipeline_id_;
  request_payload["features"] = features;
  std::string data = request_payload.dump();
  zmq::message_t request(data.size());
  memcpy(request.data(), data.c_str(), data.size());

  socket.send(request);

  zmq::message_t reply;
  socket.recv(&reply);
  std::string smessage(static_cast<char*>(reply.data()), reply.size());
  
  json response = json::parse(smessage.data());
  std::cout << " received " << response << "\n";
  *cost = response["cost"];

  // TODO: launch another thread to receive and update the queries cost
  query_id_++;
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
