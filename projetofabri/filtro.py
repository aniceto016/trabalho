import cv2
import imutils


cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)


image = cv2.imread('C:\\Users\\anice001\\Documents\\GitHub\\projetofabri\\image\\goku.png', cv2.IMREAD_UNCHANGED)


faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:

	ret, frame = cap.read()
	if ret == False: break
	frame = imutils.resize(frame, width=640)

	
	faces = faceClassif.detectMultiScale(frame, 1.1, 4)

	for (x, y, w, h) in faces:
		
		resized_image = imutils.resize(image, width=w)
		linhas_image = resized_image.shape[0]
		col_image = w

		
		altura = linhas_image // 4

		dif = 0

	
		if y + altura - linhas_image >= 0:

			
			n_frame = frame[y + altura - linhas_image : y + altura,
				x : x + col_image]
		else:
			
			dif = abs(y + altura - linhas_image) 
			
			n_frame = frame[0 : y + altura,
				x : x + col_image]

		
		mask = resized_image[:, :, 3]
		mask_inv = cv2.bitwise_not(mask)
			
		
		black = cv2.bitwise_and(resized_image, resized_image, mask=mask)
		black = black[dif:, :, 0:3]
		bg_frame = cv2.bitwise_and(n_frame, n_frame, mask=mask_inv[dif:,:])

		
		result = cv2.add(black, bg_frame)
		if y + altura - linhas_image >= 0:
			frame[y + altura - linhas_image : y + altura, x : x + col_image] = result

		else:
			frame[0 : y + altura, x : x + col_image] = result
		
	cv2.imshow('frame',frame)

	k = cv2.waitKey(7) & 0xFF
	if k == 10:
		break
cap.release()
cv2.destroyAllWindows()