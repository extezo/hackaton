import helpers
import cv2
import random
import numpy as np

# Загрузка данных из учебных и проверочных данных в  константы IMAGE_LIST и
def load_data():
    IMAGE_DIR_TRAINING = "data/train/"
    IMAGE_DIR_TEST = "data/val/"
    IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)
    TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)
    return IMAGE_LIST,TEST_IMAGE_LIST

# приведение входного изображения к стандартному виду
def standardize_input(image):
    standard_im = np.copy(image)

    ## TODO: Выполните необходимые преобразования изображения для стандартизации (обрезка, поворот, изменение размера)

    standard_im = cv2.resize(standard_im, (40, 40))
    return standard_im

# Перекодировка из текстового названия в массив данных
def one_hot_encode(label):
    one_hot_encoded = []
    if label == "none":
        one_hot_encoded = [0, 0, 0, 0, 0, 0, 0, 0]
    elif label == "pedistrain":
        one_hot_encoded = [1, 0, 0, 0, 0, 0, 0, 0]
    elif label == "no_drive":
        one_hot_encoded = [0, 1, 0, 0, 0, 0, 0, 0]
    elif label == "stop":
        one_hot_encoded = [0, 0, 1, 0, 0, 0, 0, 0]
    elif label == "way_out":
        one_hot_encoded = [0, 0, 0, 1, 0, 0, 0, 0]
    elif label == "no_entry":
        one_hot_encoded = [0, 0, 0, 0, 1, 0, 0, 0]
    elif label == "road_works":
        one_hot_encoded = [0, 0, 0, 0, 0, 1, 0, 0]
    elif label == "parking":
        one_hot_encoded = [0, 0, 0, 0, 0, 0, 1, 0]
    elif label == "a_unevenness":
        one_hot_encoded = [0, 0, 0, 0, 0, 0, 0, 1]

    return one_hot_encoded

# приведение набора изображений к стандартному виду
def standardize(image_list):
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)

        # Append the image, and it's one hot encoded label to the full, processed list of image data
        standard_list.append((standardized_im, one_hot_label))

    return standard_list



# совокупность функций классификации
def predict_label(rgb_image):
    ## TODO: обьедините ваши функции классификации в одну программу или напишите код внутри этой функции

    predicted_label = []

    ## Пример:
    predicted_label=[1, 0, 0, 0, 0, 0, 0, 0]

    return predicted_label


def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []
    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:
        # Get true data
        im = image[0]
        true_label = image[1]

        assert (len(true_label) == 8), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = predict_label(im)
        assert (len(predicted_label) == 8), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels
        if (predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))

    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels

def main():
    ## загрузка учебных изображений
    IMAGE_LIST,TEST_IMAGE_LIST = load_data()


    # #отображение загруженного изображения и его метки (наименования знака)
    test_im = IMAGE_LIST[1520][0]
    test_im_label=IMAGE_LIST[1520][1]

    # cv2.imshow("test_im",test_im)
    # print(test_im_label)
    # cv2.waitKey(0)

    ## Отображение изображения, приведенного к стандартному виду (размер 32х32) и его метки (массива чисел)
    STANDARDIZED_LIST = standardize(IMAGE_LIST)
    standart_test_im = STANDARDIZED_LIST[1520][0]
    standart_test_im_label=STANDARDIZED_LIST[1520][1]

    # cv2.imshow("standart_test_im",standart_test_im)
    # print(standart_test_im_label)
    # cv2.waitKey(0)

    ## Проверка на тестовых данных

    STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)
    random.shuffle(STANDARDIZED_TEST_LIST)

    standart_test_result_im = STANDARDIZED_TEST_LIST[100][0]
    standart_test_result_im_label = STANDARDIZED_TEST_LIST[100][1]

    # cv2.imshow("standart_test_result_im", standart_test_result_im)
    # print(standart_test_result_im_label)
    # cv2.waitKey(0)

    ## сравнение учебных и тестовых изображений
    # поиск неклассифицированных изображений в тестовой выборке
    MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)
    # вычисление точности
    total = len(STANDARDIZED_TEST_LIST)
    num_correct = total - len(MISCLASSIFIED)
    accuracy = num_correct / total

    print('Точность: ' + str(accuracy))
    print("Число не распознанных изображений = " + str(len(MISCLASSIFIED)) + ' из ' + str(total))

if __name__ == '__main__':
    main()
