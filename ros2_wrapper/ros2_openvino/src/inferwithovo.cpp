//
// Created by rrh4fe on 7/8/21.
//

#include "rclcpp/rclcpp.hpp"
#include "image_transport/image_transport.h"
#include "cv_bridge/cv_bridge.h"

# include "ovoinference.h"

class InferWithOVONode
{
public:
    InferWithOVONode(rclcpp::Node::SharedPtr node, const std::string &model_path, const std::string &device)
    :it(node), infer(std::make_shared<ovoinference>(model_path, device))
    {
        std::vector<size_t> dims = infer->checkdims();
        iH = dims[0];
        iW = dims[1];

        RCLCPP_INFO(node->get_logger(), "The model expects input of dimension of '%d' x '%d'",iH, iW);

        sub = it.subscribe("/camera/color/image_raw",10, &InferWithOVONode::segmentCallback, this);
        pub = it.advertise("/seg_map",1);
    }


private:

    image_transport::ImageTransport it;
    image_transport::Subscriber sub;
    image_transport::Publisher pub;
    std::shared_ptr<ovoinference>infer;
    cv_bridge::CvImagePtr cv_ptr;
    int iH, iW;

    void segmentCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
    {
        cv_ptr = cv_bridge::toCvCopy(msg);
        cv_ptr->image = infer->Segment(cv_ptr->image);
        pub.publish(cv_ptr->toImageMsg());
    }
};

int main(int argc, char * argv[])

{
    rclcpp::init(argc, argv);
    const std::string model_path = "/home/rrh4fe//openvino_models/ir/bisenet_v1.xml";
    const std::string device = "MULTI:MYRIAD.1.1-ma2480,MYRIAD.1.5-ma2480,GPU,CPU";
    auto node_ = rclcpp::Node::make_shared("inferwithovo_node");
    auto infer_ = std::make_shared<InferWithOVONode>(node_, model_path, device);
    rclcpp::spin(node_);
    rclcpp::shutdown();
    return 0;
}