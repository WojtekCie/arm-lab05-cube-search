# Zfuzjonowana (potwierdzona) pozycja kostki.
int32 id
# Srodek kostki; header.frame_id = ramka globalna (domyslnie map = swiat Gazebo).
geometry_msgs/PointStamped position
# Liczba obserwacji uzytych w estymacie.
int32 observations
# Rozrzut obserwacji wokol estymaty [m].
float32 std_dev
