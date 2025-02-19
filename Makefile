LDFLAGS=$(shell pkg-config --cflags --libs opencv4)

main: main.cpp
	g++ main.cpp -o run $(LDFLAGS)

clean:
	rm -f run