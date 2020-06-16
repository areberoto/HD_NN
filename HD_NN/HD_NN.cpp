/*  Handwritten Digits Neural Network implemented in C++
    Author: Alberto Alvarez & Octavio Torres
    Date:   20th May, 2020
*/

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "Network.h"
#include "MNIST_DS.h"
#include "Matrix.h"

using std::cout;
using std::cin;
using std::endl;

using namespace cv;

int argmax(Matrix& mtx);

int main()
{
    cout << "\tSHALLOW NEURAL NETWORK FOR HANDWRITTEN DIGITS RECOGNITION" << endl;
    int num_images{ 50 };
    int* sizes = new int[3] {784, 30, 10};

    /*Network NN{ sizes };   
    cout << "\nTraining..." << endl;
    NN.SGD(30, 100, 3.0f);*/

    Network classifier{ sizes };
    classifier.loadWeightsBiases();

    MNIST_DS test{ false };

    Matrix ima{};
    Matrix lab{};
    Matrix result{};

    Mat image = Mat::zeros(500, 1400, CV_8UC3);
    for (size_t k{ 0 }; k < num_images; k++) {

        ima = test.getImage(k);
        lab = test.getLabel(k);
        int y = argmax(lab);
        result = classifier.feedforward(ima);
        int r = argmax(result);

        for (size_t i{ 0 }; i < 28; i++) {
            for (size_t j{ 0 }; j < 28; j++)
                circle(image, Point(j + (k * 28), i), 0, Scalar(ima[i * 28 + j], ima[i * 28 + j], ima[i * 28 + j]), 0);
        }

        if(y == r)
            putText(image, std::to_string(r), Point(k * 28, 56), FONT_HERSHEY_DUPLEX, 1.0, Scalar(0, 255, 0), 1);
        else
            putText(image, std::to_string(r), Point(k * 28, 56), FONT_HERSHEY_DUPLEX, 1.0, Scalar(0, 0, 255), 1);
    }
    
    imshow("Display Window", image);
    waitKey(0);

    return 0;
}

int argmax(Matrix& mtx) {
    double max{ mtx[0] };
    int index{ 0 };
    for (size_t i{ 1 }; i < mtx.getSize(); i++) {
        if (max < mtx[i]) {
            max = mtx[i];
            index = i;
        }
    }
    return index;
}
