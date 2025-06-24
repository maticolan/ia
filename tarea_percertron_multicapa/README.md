Diseñamos una red neuronal de tres capas: una capa de entrada con 784 neuronas (para las imágenes del dataset MNIST), 
una capa oculta con 128 neuronas y una capa de salida con 10 neuronas, una por cada dígito posible. Usamos la función 
sigmoide como activación, ya que su derivada era más sencilla de trabajar y era más fácil utilizarla durante la retropropagación.

Luego desarrollamos dos kernels en CUDA para paralelizar el entrenamiento. En el forward_kernel, asignamos un thread a 
cada neurona de la capa oculta y de la capa de salida, y asi calcular las activaciones de forma simultánea. 
En el backpropagation_kernel, utilizamos memoria compartida (shared) para almacenar los datos de error y que sea más rápido Finalmente, 
cada thread se encargó de actualizar en paralelo los pesos y sesgos y asi poder obtener un buen aprendizaje en cada época.
