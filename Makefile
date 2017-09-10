all:
	g++ -lopencv_imgcodecs `pkg-config --libs opencv` fisheye2.cpp -o fisheye2
