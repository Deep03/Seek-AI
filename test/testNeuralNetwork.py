inputs=[1,2,3,4]
weights=[[2.3,4.5,5.6,1.2],
         [2.4,5.6,7.1,1.3],
         [0.2,-3.2,-4.6,1.7],
         [2.4,5.2,4.1,3.2]]
bias=[1,3,5,2]

layer_outputs=[]
for neuron_weights,neuron_biases in zip(weights,bias):
    neuron_outputs=0
    for n_inputs,weight in zip(inputs,neuron_weights):
        neuron_outputs=neuron_outputs+(n_inputs*weight)
    neuron_outputs=neuron_outputs+neuron_biases
    layer_outputs.append(neuron_outputs)

print(layer_outputs)
