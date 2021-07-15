TARGET = conv
SOURCE = conv.cu

NVCC = nvcc
NVCCFLAGS += -O3 -Xcompiler -fopenmp -lcuda

$(TARGET):$(SOURCE)
	$(NVCC) $(SOURCE) -o $(TARGET) $(NVCCFLAGS)

.PHONY:clean
clean:
	-rm -rf $(TARGET)
