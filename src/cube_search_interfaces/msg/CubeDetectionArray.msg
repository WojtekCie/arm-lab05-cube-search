# Wynik przetworzenia jednej klatki obrazu (publikowany takze bez detekcji,
# bo camera_pose sluzy do aktualizacji mapy pokrycia).
# stamp = stempel obrazu, frame_id = ramka globalna (domyslnie map).
std_msgs/Header header
# Poza camera_rgb_frame w ramce header.frame_id w chwili wykonania zdjecia.
geometry_msgs/Pose camera_pose
CubeDetection[] detections
