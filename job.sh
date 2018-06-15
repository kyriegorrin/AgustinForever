#!/bin/bash
export PATH=/Soft/cuda/8.0.61/bin:$PATH

### Directivas para el gestor de colas
# Asegurar que el job se ejecuta en el directorio actual
#$ -cwd
# Asegurar que el job mantiene las variables de entorno del shell lamador
#$ -V
# Cambiar el nombre del job
#$ -N MultiGPU 
# Cambiar el shell
#$ -S /bin/bash



# Para comprobar que funciona no es necesario usar matrices muy grandes
# Con N = 1024 es suficiente
./bitonic_1GPU.exe

# Con matrices muy grandes no es recomendable comprobar el resultado
#nvprof --print-gpu-trace ./kernel4GPUs.exe 8192 N
#nvprof --print-gpu-trace ./kernel4GPUs.exe 4096 N
#for i in {1..10}
#do
#echo "Welcome $i times"
#./kernel4GPUs.exe  8192 N
#done

#./kernel4GPUs.exe  512 N
#./kernel4GPUs.exe 1024 N
#./kernel4GPUs.exe 2048 N
#./kernel4GPUs.exe 4096 N
#./kernel4GPUs.exe 8192 N
#nvprof ./kernel4GPUs.exe 8192 N

