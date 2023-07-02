#include <iostream>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;
using namespace std::chrono;

const char *inputImagePath = "../input.jpg";
const char *outputImagePath = "output-cpu.jpg";

/*
    Function that takes loaded image as an array of type uint8_t and converts it to an array of integers
    Since uint8_t has BGR format, 2 - k part is needed to convert it to RGB
*/

int *imgToArray(uint8_t *pixelPtr, int sizeRows, int sizeCols, int sizeDepth)
{
    int *pixels = (int *)malloc((sizeRows * sizeCols * sizeDepth) * sizeof(int));
    for (int i = 0; i < sizeRows; i++)
    {
        for (int j = 0; j < sizeCols; j++)
        {
            for (int k = 0; k < sizeDepth; k++)
            {
                // converting BGR to RGB colors
                pixels[i * sizeCols * sizeDepth + j * sizeDepth + k] =
                    (int)pixelPtr[i * sizeCols * sizeDepth + j * sizeDepth + 2 - k];
            }
        }
    }
    return pixels;
}

/*
    Function that takes output image as an array of integers and converts it to an array of uint8_t
*/

void arrayToImg(std::vector<int> &pixels, uint8_t *pixelPtr, int sizeRows, int sizeCols, int sizeDepth)
{
    for (int i = 0; i < sizeRows; i++)
    {
        for (int j = 0; j < sizeCols; j++)
        {
            for (int k = 0; k < sizeDepth; k++)
            {
                pixelPtr[i * sizeCols * sizeDepth + j * sizeDepth + k] =
                    (uint8_t)pixels[i * sizeCols * sizeDepth + j * sizeDepth + (sizeDepth - 1 - k)];
            }
        }
    }
    return;
}

/*
    Function that takes input image pixels and performs Gaussian blur on every single pixel
    Every pixel has a new value that's computed by convoluting Gaussian kernel to 5x5 matrix around that pixel
*/

std::vector<int> gaussianBlur(int *pixels, std::vector<std::vector<double>> &kernel, double kernelConst, int sizeRows, int sizeCols, int sizeDepth)
{
    std::vector<int> pixelsBlur(sizeRows * sizeCols * sizeDepth);
    for (int i = 0; i < sizeRows; i++)
    {
        for (int j = 0; j < sizeCols; j++)
        {
            for (int k = 0; k < sizeDepth; k++)
            {
                double sum = 0;
                double sumKernel = 0;
                for (int y = -2; y <= 2; y++)
                {
                    for (int x = -2; x <= 2; x++)
                    {
                        if ((i + x) >= 0 && (i + x) < sizeRows && (j + y) >= 0 && (j + y) < sizeCols)
                        {
                            double channel = (double)pixels[(i + x) * sizeCols * sizeDepth + (j + y) * sizeDepth + k];
                            sum += channel * kernelConst * kernel[x + 2][y + 2];
                            sumKernel += kernelConst * kernel[x + 2][y + 2];
                        }
                    }
                }
                pixelsBlur[i * sizeCols * sizeDepth + j * sizeDepth + k] = (int)(sum / sumKernel);
            }
        }
    }
    return pixelsBlur;
}

/*
    Function that converts given blurred image to grayscale
    Size of blurred image is reduced by 3 since every pixel is now presented with only one entry in the array
    Grayscale value for each pixel is found by summing up it's RGB values and dividing it by 3
*/

std::vector<int> rgbToGrayscale(std::vector<int> &pixels, int sizeRows, int sizeCols, int sizeDepth)
{
    std::vector<int> pixelsGray(sizeRows * sizeCols);
    for (int i = 0; i < sizeRows; i++)
    {
        for (int j = 0; j < sizeCols; j++)
        {
            int sum = 0;
            for (int k = 0; k < sizeDepth; k++)
            {
                sum = sum + pixels[i * sizeCols * sizeDepth + j * sizeDepth + k];
            }
            pixelsGray[i * sizeCols + j] = (int)(sum / sizeDepth);
        }
    }
    return pixelsGray;
}

/*
    Function that performs non maximum suppresion on each pixel
    Actions are performed based on current pixel gradient direction

    Example:

    If current pixel has gradient direction 0 or 180 degrees, we have to check pixel that's bellow and above it
    If any of these pixels have greater gradient value than current one, current one's gradient value will be set to 0
*/

void performNonMaximumSuppresion(double *G, std::vector<int> &theta, int sizeCols, int sizeRows, std::vector<int> &pixels, double largestG)
{
    for (int i = 1; i < sizeRows - 1; i++)
    {
        for (int j = 1; j < sizeCols - 1; j++)
        {
            if (theta[i * sizeCols + j] == 0 || theta[i * sizeCols + j] == 180)
            {
                if (G[i * sizeCols + j] < G[i * sizeCols + j - 1] || G[i * sizeCols + j] < G[i * sizeCols + j + 1])
                {
                    G[i * sizeCols + j] = 0;
                }
            }
            else if (theta[i * sizeCols + j] == 45 || theta[i * sizeCols + j] == 225)
            {
                if (G[i * sizeCols + j] < G[(i + 1) * sizeCols + j + 1] || G[i * sizeCols + j] < G[(i - 1) * sizeCols + j - 1])
                {
                    G[i * sizeCols + j] = 0;
                }
            }
            else if (theta[i * sizeCols + j] == 90 || theta[i * sizeCols + j] == 270)
            {
                if (G[i * sizeCols + j] < G[(i + 1) * sizeCols + j] || G[i * sizeCols + j] < G[(i - 1) * sizeCols + j])
                {
                    G[i * sizeCols + j] = 0;
                }
            }
            else
            {
                if (G[i * sizeCols + j] < G[(i + 1) * sizeCols + j - 1] || G[i * sizeCols + j] < G[(i - 1) * sizeCols + j + 1])
                {
                    G[i * sizeCols + j] = 0;
                }
            }

            pixels[i * sizeCols + j] = (int)(G[i * sizeCols + j] * (255.0 / largestG));
        }
    }
};

/*
    Function that peforms double thresholding based on provided parameters for lower and higher threshold
    Double thresholding is performed on every pixel of the image as it follows:

    If current pixel's gradient is lower than lower threshold value multiplied by largest gradient, set that pixel's
    gradient to 0 since it's considered as `weak` edge -> supressing weaker edges

    If current pixel's gradient is higher or equal than higher threshold value multiplied by largest gradient,
    it's value remains untouched since that pixel is considered as `strong` edge -> preserving strong edges

    If current pixel's gradient is greater than lower threshold value multiplied by largest gradient and lower than
    higher threshold value multipled by largest gradient, we're checking neighbor pixels gradient values as it follows:

        * If any of the neighboring pixels has gradient value greater than higher threshold value multiplied by largest gradient,
            gradient value of current pixel will be set to higher threshold value multiplied by largest gradient and whole function
            will be performed again with new values

*/

void performDoubleThresholding(double *G, std::vector<int> &theta, int sizeCols, int sizeRows, std::vector<int> &pixels, double largestG)
{
    double lowerThreshold = 0.03;
    double higherThreshold = 0.1;
    bool changes;
    do
    {
        changes = false;
        for (int i = 1; i < sizeRows - 1; i++)
        {
            for (int j = 1; j < sizeCols - 1; j++)
            {
                if (G[i * sizeCols + j] < (lowerThreshold * largestG))
                {
                    G[i * sizeCols + j] = 0;
                }
                else if (G[i * sizeCols + j] >= (higherThreshold * largestG))
                {
                    continue;
                }
                else if (G[i * sizeCols + j] < (higherThreshold * largestG))
                {
                    G[i * sizeCols + j] = 0;
                    for (int x = -1; x <= 1; x++)
                    {
                        bool breakNestedLoop = false;
                        for (int y = -1; y <= 1; y++)
                        {
                            if (x == 0 && y == 0)
                            {
                                continue;
                            }
                            if (G[(i + x) * sizeCols + (j + y)] >= (higherThreshold * largestG))
                            {
                                G[i * sizeCols + j] = (higherThreshold * largestG);
                                changes = true;
                                breakNestedLoop = true;
                                break;
                            }
                        }
                        if (breakNestedLoop)
                        {
                            break;
                        }
                    }
                }
                pixels[i * sizeCols + j] = (int)(G[i * sizeCols + j] * (255.0 / largestG));
            }
        }
    } while (changes);
}

std::vector<int> cannyFilter(std::vector<int> &pixels, int sizeRows, int sizeCols, int sizeDepth)
{
    std::vector<int> pixelsCanny(sizeRows * sizeCols);
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    double *G = new double[sizeRows * sizeCols];
    std::vector<int> theta(sizeRows * sizeCols);
    double largestG = 0;

    // perform canny edge detection on everything but the edges
    for (int i = 1; i < sizeRows - 1; i++)
    {
        for (int j = 1; j < sizeCols - 1; j++)
        {
            // find gx and gy for each pixel
            double gxValue = 0;
            double gyValue = 0;
            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= 1; y++)
                {
                    gxValue = gxValue + (gx[1 - x][1 - y] * (double)(pixels[(i + x) * sizeCols + j + y]));
                    gyValue = gyValue + (gy[1 - x][1 - y] * (double)(pixels[(i + x) * sizeCols + j + y]));
                }
            }

            // calculate G and theta
            G[i * sizeCols + j] = std::sqrt(std::pow(gxValue, 2) + std::pow(gyValue, 2));
            double atanResult = atan2(gyValue, gxValue) * 180.0 / 3.14159265;
            theta[i * sizeCols + j] = (int)(180.0 + atanResult);

            if (G[i * sizeCols + j] > largestG)
            {
                largestG = G[i * sizeCols + j];
            }

            // setting the edges
            if (i == 1)
            {
                G[i * sizeCols + j - 1] = G[i * sizeCols + j];
                theta[i * sizeCols + j - 1] = theta[i * sizeCols + j];
            }
            else if (j == 1)
            {
                G[(i - 1) * sizeCols + j] = G[i * sizeCols + j];
                theta[(i - 1) * sizeCols + j] = theta[i * sizeCols + j];
            }
            else if (i == sizeRows - 1)
            {
                G[i * sizeCols + j + 1] = G[i * sizeCols + j];
                theta[i * sizeCols + j + 1] = theta[i * sizeCols + j];
            }
            else if (j == sizeCols - 1)
            {
                G[(i + 1) * sizeCols + j] = G[i * sizeCols + j];
                theta[(i + 1) * sizeCols + j] = theta[i * sizeCols + j];
            }

            // setting the corners
            if (i == 1 && j == 1)
            {
                G[(i - 1) * sizeCols + j - 1] = G[i * sizeCols + j];
                theta[(i - 1) * sizeCols + j - 1] = theta[i * sizeCols + j];
            }
            else if (i == 1 && j == sizeCols - 1)
            {
                G[(i - 1) * sizeCols + j + 1] = G[i * sizeCols + j];
                theta[(i - 1) * sizeCols + j + 1] = theta[i * sizeCols + j];
            }
            else if (i == sizeRows - 1 && j == 1)
            {
                G[(i + 1) * sizeCols + j - 1] = G[i * sizeCols + j];
                theta[(i + 1) * sizeCols + j - 1] = theta[i * sizeCols + j];
            }
            else if (i == sizeRows - 1 && j == sizeCols - 1)
            {
                G[(i + 1) * sizeCols + j + 1] = G[i * sizeCols + j];
                theta[(i + 1) * sizeCols + j + 1] = theta[i * sizeCols + j];
            }

            // round to the nearest 45 degrees
            theta[i * sizeCols + j] = round(theta[i * sizeCols + j] / 45) * 45;
        }
    }

    performNonMaximumSuppresion(G, theta, sizeCols, sizeRows, pixelsCanny, largestG);

    performDoubleThresholding(G, theta, sizeCols, sizeRows, pixelsCanny, largestG);

    return pixelsCanny;
};

void cannyEdgeDetection(uint8_t *inputImage, int width, int height, int channels)
{
    int *pixels = imgToArray(inputImage, height, width, channels);

    // GAUSSIAN_BLUR:

    std::vector<std::vector<double>> kernel = {{2.0, 4.0, 5.0, 4.0, 2.0},
                                               {4.0, 9.0, 12.0, 9.0, 4.0},
                                               {5.0, 12.0, 15.0, 12.0, 5.0},
                                               {4.0, 9.0, 12.0, 9.0, 4.0},
                                               {2.0, 4.0, 5.0, 4.0, 2.0}};
    double kernelConst = (1.0 / 159.0);
    std::vector<int> pixelsBlur = gaussianBlur(pixels, kernel, kernelConst, height, width, channels);

    // GRAYSCALE:

    std::vector<int> pixelsGray = rgbToGrayscale(pixelsBlur, height, width, channels);

    // CANNY_FILTER:

    std::vector<int> pixelsCanny = cannyFilter(pixelsGray, height, width, 1);

    uint8_t *outputImage = (unsigned char *)malloc(width * height);
    arrayToImg(pixelsCanny, outputImage, height, width, 1);

    stbi_write_jpg(outputImagePath, width, height, 1, outputImage, 100);

    free(outputImage);
}

int main()
{
    int width, height, channels;

    // Load the input image
    uint8_t *inputImage = stbi_load(inputImagePath, &width, &height, &channels, STBI_rgb);
    if (inputImage == NULL)
    {
        printf("Failed to load image: %s\n", inputImagePath);
        return -1;
    }

    // if (width < 1400 || height < 1400)
    // {
    //     printf("Choose image with greater resolution\n");
    //     return -1;
    // }

    if (channels != 3)
    {
        printf("Images is not in RGB format\n");
        return -1;
    }

    auto start = high_resolution_clock::now();

    cannyEdgeDetection(inputImage, width, height, channels);

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(stop - start);
    std::cout << "Time of execution on CPU is: " << duration.count() << " ms" << std::endl;

    return 0;
}