#pragma once

#include <exception>
#include <string>

/**
 * @class ImageFormatException
 * @brief Custom exception class for handling image format errors.
 *
 * The ImageFormatException class is used to represent errors that occur
 * due to invalid image formats or mismatched properties during image-related
 * processing operations. This class extends the standard `std::exception`
 * class to provide additional context through its error message.
 */
class ImageFormatException : public std::exception {
    /**
     * A string variable that holds the error message associated with an
     * instance of the ImageFormatException class. This message provides
     * detailed information about the nature of the image format exception
     * encountered.
     */
    std::string message;
    /**
     * Constructs an ImageFormatException with a specific error message.
     *
     * @param message The error message describing the cause of the exception.
     * @return An instance of ImageFormatException initialized with the given message.
     */
public:
    ImageFormatException(const std::string& message): message(message) {
    }

};