import numpy as np
import struct
import cv2


def Scalar(a, b):
    result = 0
    for i, v in enumerate(a):
        result += v[0] * b[i][0]
    return result


def length(a):
    result = 0
    for i in a:
        result += i[0] * i[0]
    return result


def gram_schmidt(key):
    K = np.array(key)
    vectors = []
    column = K.shape[1]
    row = K.shape[0]
    for i in range(column):
        vector = []
        for j in range(row):
            vector.append([K[j][i]])
        vectors.append(vector)

    u = [vectors[0]]

    for i in range(1, len(vectors)):
        projection = np.array(vectors[i], dtype=np.float64)
        for j in range(i):
            scalar = Scalar(u[j], vectors[i])
            distance = length(u[j])
            coefficient = scalar / distance
            projection -= coefficient * np.array(u[j], dtype=np.float64)
        u.append(projection)
        if length(projection) <= 0.000001:
            return False
    return True


def number_to_bin(f):
    b = struct.pack('!f', f)

    return ''.join(['{:08b}'.format(b) for b in b])


def bin_to_number(b):
    byte = [b[i:i+8] for i in range(0, len(b), 8)]
    byte = [int(b, 2) for b in byte]
    return struct.unpack('!f', bytes(byte))[0]


def Encrypt(K, text):
    values = []
    vectors = []
    vector = []
    K = np.array(K)
    n = K.shape[1]
    K_inv = np.linalg.inv(K)
    count = 0
    for character in text:
        if count == n:
            vectors.append(vector)
            count = 0
            vector = []
        vector.append([ord(character)])
        count += 1
    while count < n:
        vector.append([ord(" ")])
        count += 1
    vectors.append(vector)

    for v in vectors:
        result = np.dot(K_inv, np.array(v))
        for value in result:
            values.append(value[0])

    return values


def getLetterOrders(binTexts, K):
    K = np.array(K)
    n = K.shape[1]
    result = []
    index = 0
    while index < len(binTexts):
        vector = []
        for i in range(index, index + n):
            vector.append([bin_to_number(binTexts[i])])
        computation = np.dot(K, np.array(vector))

        for values in computation:
            result.append(int(np.round(values[0])))

        index += n
    return result


def hideInTheImage(path, binTexts, encrypt_path):
    image = cv2.imread(path)
    text = ''.join(binTexts)
    index = 0
    for i in range(len(image)):
        for j in range(len(image[i])):
            if index >= len(text):
                _bin = bin(image[i][j][2])
                _bin = _bin[:-1] + "1"
                image[i][j][2] = eval(_bin)
                break
            bin1 = bin(image[i][j][0])
            bin2 = bin(image[i][j][1])
            bin3 = bin(image[i][j][2])
            bin1 = bin1[:-2] + text[index] + text[index + 1]
            bin2 = bin2[:-2] + text[index + 2] + text[index + 3]
            bin3 = bin3[:-1] + "0"
            image[i][j][0] = eval(bin1)
            image[i][j][1] = eval(bin2)
            image[i][j][2] = eval(bin3)
            index += 4

    cv2.imwrite(encrypt_path, image)

    return True


def GetNumbersFromImage(path):
    image = cv2.imread(path)
    text = ""
    result = []
    for i in range(len(image)):
        for j in range(len(image[i])):
            bin3 = bin(image[i][j][2])
            if bin3[len(bin3) - 1] == "1":
                break
            bin1 = bin(image[i][j][0])[-2:]
            bin2 = bin(image[i][j][1])[-2:]
            text += bin1 + bin2
    for i in range(0, len(text), 32):
        result.append(text[i: i+32])

    return result
