from keras.models import model_from_json
import cv2
import numpy

with open('src/mnist_model.json', 'r') as json:
    clf = model_from_json(json.read())
clf.load_weights('src/model.h5')
clf.compile(loss='categorical_crossentropy',
            optimizer='ADAM',
            metrics=['accuracy'])

path = "src/digits.jpg"
im_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

_, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
_, ctrs, _ = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

im = cv2.imread(path)
for x, y, h, w in (cv2.boundingRect(ctr) for ctr in ctrs):
    # Make the rectangular region around the digit
    leng = int(max(w, h) * 1.47)
    dx = (leng - h) // 2
    dy = (leng - w) // 2
    x1, x2 = max(x - dx, 0), x + h + dx
    y1, y2 = max(y - dy, 0), y + w + dy
    roi = im_gray[y1: y2, x1: x2]

    cv2.rectangle(im, (x, y), (x + h, y + w), (0, 255, 0), 3)

    roi = 1.0 - (cv2.resize(roi, (28, 28)) / 255.0)
    roi = numpy.expand_dims(roi, axis=0).reshape(1, 1, 28, 28)

    digit = int(clf.predict_classes(roi)[0])

    # Draw the rectangle and put digit
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(im, str(digit), (x - dx // 2, y + w // 2), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 38, 0), 1)

cv2.imwrite("src/result.jpg", im)
