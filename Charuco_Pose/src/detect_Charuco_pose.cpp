#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/objdetect/charuco_detector.hpp"
#include <opencv2/aruco/charuco.hpp>
#include "aruco_utility.hpp"
//! [charucohdr]o

using namespace cv;
using namespace std;

static bool readStringList(const string &filename, vector<string> &l)
{
    l.clear();
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
        return false;
    FileNode n = fs.getFirstTopLevelNode();
    if (n.type() != FileNode::SEQ)
        return false;
    FileNodeIterator it = n.begin(), it_end = n.end();
    for (; it != it_end; ++it)
        l.push_back((string)*it);
    return true;
}

int main(int argc, char **argv)
{
    Size boardSize;           // The size of the board -> Number of items by width and height
    float squareSize;         // The size of a square in your defined unit (point, millimeter,etc).
    float markerSize;         // The size of a marker in your defined unit (point, millimeter,etc).
    string arucoDictName;     // The Name of ArUco dictionary which you use in ChArUco pattern
    string arucoDictFileName; // The Name of file which contains ArUco dictionary for ChArUco pattern
    string images_xml;
    string output;
    string calib_filename;

    // read xml file
    string filename = argv[1];
    FileStorage fs_i(filename, FileStorage::READ);
    fs_i["BoardSize_Width"]>>boardSize.width;
    fs_i["BoardSize_Height"]>>boardSize.height;
    fs_i["Square_Size"]>>squareSize;
    fs_i["Marker_Size"]>>markerSize;
    fs_i["ArUco_Dict_Name"]>>arucoDictName;
    fs_i["Images_xml"]>>images_xml;
    fs_i["Write_outputFileName"]>>output;
    fs_i["Calib_Filename"]>>calib_filename;
    fs_i.release();

    // create aruco board
    auto dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_100);
    cv::Ptr<cv::aruco::CharucoBoard> board = new cv::aruco::CharucoBoard({boardSize.width, boardSize.height}, squareSize, markerSize, dictionary);
    cv::Ptr<cv::aruco::DetectorParameters> params = cv::makePtr<cv::aruco::DetectorParameters>();
    
    // read camera parameters
    cv::Mat cameraMatrix, distCoeffs;
    bool readOk = readCameraParameters(calib_filename, cameraMatrix, distCoeffs);
    // read names of a list of images from the input xml file
    vector<string> imageList;
    readStringList(images_xml, imageList);
    // open output fes
    FileStorage fs_o(output, FileStorage::WRITE);
    for (const auto &imageName : imageList)
    {
        // Load image
        cv::Mat image = cv::imread(imageName);
        cv::Mat imageCopy;
        image.copyTo(imageCopy);
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners;
        cv::aruco::detectMarkers(imageCopy, cv::makePtr<cv::aruco::Dictionary>(board->getDictionary()), markerCorners, markerIds, params);
        // if at least one marker detected
        if (markerIds.size() > 0)
        {
            cv::aruco::drawDetectedMarkers(imageCopy, markerCorners, markerIds);
            std::vector<cv::Point2f> charucoCorners;
            std::vector<int> charucoIds;
            cv::aruco::interpolateCornersCharuco(markerCorners, markerIds, imageCopy, board, charucoCorners, charucoIds, cameraMatrix, distCoeffs);
            // if at least one charuco corner detected
            if (charucoIds.size() > 0)
            {
                cv::Scalar color = cv::Scalar(255, 0, 0);
                //! [detcor]
                cv::aruco::drawDetectedCornersCharuco(imageCopy, charucoCorners, charucoIds, color);
                //! [detcor]
                cv::Vec3d rvec, tvec;
                //! [pose]
                bool valid = cv::aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds, board, cameraMatrix, distCoeffs, rvec, tvec);
                if (valid)
                    cv::drawFrameAxes(imageCopy, cameraMatrix, distCoeffs, rvec, tvec, 50);
                fs_o << "rvec" << rvec;
                fs_o << "tvec" << tvec;
            }
        }
        cv::imshow("out", imageCopy);
        char key = (char)cv::waitKey(0);
    }
    return 0;
}