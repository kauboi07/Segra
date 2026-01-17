from face_utils import recognize_face

user, distance = recognize_face(r"C:\Users\123co\Downloads\1396149.jpg")

print("Matched User:", user)
print("Distance:", distance)
