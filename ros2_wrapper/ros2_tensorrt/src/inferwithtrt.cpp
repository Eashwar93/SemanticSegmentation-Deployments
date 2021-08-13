#include "rclcpp/rclcpp.hpp"
#include "image_transport/image_transport.h"
#include "cv_bridge/cv_bridge.h"

#include "trtinference.h"

#include <iostream>
#include "opencv2/opencv.hpp"

class InferWithTRTNode
{
public:
    InferWithTRTNode(rclcpp::Node::SharedPtr node, std::string &model_path)
            :it(node), infer(std::make_shared<trtinference>(model_path))
    {
        std::vector<int> dims = infer->checkdims();
        iH = dims[0];
        iW = dims[1];


        RCLCPP_INFO(node->get_logger(), "The model expects input of dimension of '%d' x '%d'",iH, iW);

        sub = it.subscribe("/camera/color/image_raw",1, &InferWithTRTNode::segmentCallback, this);
        pub = it.advertise("/seg_map",1);


    }
private:

    //Member variables
    image_transport::ImageTransport it;
    image_transport::Subscriber sub;
    image_transport::Publisher pub;
    std::shared_ptr<trtinference>infer;
    cv_bridge::CvImagePtr cv_ptr;
    int iH, iW;

    //TRT inferencer

    //Segmentation Callback using tensorRT
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
    std::string model_path = "/home/rrh4fe/deployment_pipeline/inferwithtrt/example/monorail_model_fp16.trt";
    auto node_ = rclcpp::Node::make_shared("inferwithtrt_node");
    auto infer_ = std::make_shared<InferWithTRTNode>(node_, model_path);
    rclcpp::spin(node_);
    rclcpp::shutdown();
    return 0;
}
